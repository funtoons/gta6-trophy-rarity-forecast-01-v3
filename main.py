#!/usr/bin/env python3
"""GTA VI trophy rarity forecast -- v3 adaptive ensemble.

Forecasts, for each of the 55 GTA VI PS5 trophies, five quantiles of its
Exophase earn rate (%) at the case target, from the house's reads before the
checkpoint. Python 3.13, standard library only, offline, deterministic:
no randomness, no clock, no filesystem paths in the output.

Method (per trophy, all in logit space):
  * Plateau-collapse: consecutive identical reads are one refresh, not
    fresh data, so they are merged before any fit.
  * Four median estimators: (a) persistence (last rate), (b) log-time
    linear trend of logit(rate) vs log(1 + hours since release), recency
    weighted, (c) earners/players decomposition -- log earners and log
    players extrapolated separately, rate = earners / players,
    (d) real-time linear trend of logit(rate) vs hours.
  * Ensemble weights learned by leave-future-out backtest: each estimator
    is fit on history up to T-24h and scored on the last 24h of reads
    (a 12h window barely separates the shapes, so the split is a full
    day); weights are inverse-error to the fourth power, pooled per
    trophy group (story / grind / platinum+100%) and shrunk toward the
    global weights. The ensemble median may also be shrunk toward
    persistence, and the lower tail widened with horizon, because a
    player flood can crash rates one-sidedly (launch night saw players
    638 -> 3868 in a single refresh).
  * The persistence weight adapts to regime: recent-slope acceleration
    lowers it, dilution (players outgrowing earners) raises it, bounded
    to +/-0.15 from the backtest weight.
  * Structural priors applied softly: story-chain monotonicity
    (Ch1 >= Ch2 >= Ch3 >= Ch4 >= final), platinum <= every other trophy,
    100% <= final mission; medians clamped to logit(last) +/- clamp.
  * Spread from ensemble disagreement, slope standard error times
    extrapolation distance, backtest residual and a horizon term, with a
    floor; the upper tail widens when the trend sits above persistence
    (upside risk) and the lower tail when dilution threatens.
  * Edge cases (1 read, empty history, all-null rates, 0% / 100%) fall
    back to persistence plus the sigma floor.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Tunable parameters (defaults; the sweep in v3-scratch varies these).
PARAMS = {
    "half_life_h": 8.0,       # recency half-life for WLS weights (sweep: 8
                              # beats 12/24/48 -- the launch-night regime turns
                              # fast, old reads mislead the trend fits)
    "backtest_h": 24.0,       # leave-future-out window (hours); long enough
                              # that the estimators' shapes actually differ
    "backtest_power": 6.0,    # exponent on inverse backtest error (sweep:
                              # 6 beats 1/2/3/4 -- the backtest must pick a
                              # winner sharply, not average the shapes)
    "persist_shrink": 0.0,    # shrink of the ensemble median toward
                              # persistence (0 = none); guards the flood case
    "k_downtail": 0.0,        # extra lower-tail width per sqrt(day): over a
                              # long horizon a player flood can crash rates
                              # (launch night saw players 638 -> 3868 in one
                              # refresh), a one-sided dilution risk
    "tail_extra": 0.0,        # extra tail width on the at-risk side
    "w_default": (0.35, 0.25, 0.25, 0.15),  # persist, logtrend, decomp, realtime
    "shrink_global": 0.5,     # group weight shrunk this far toward global
    "persist_adj_max": 0.15,  # max regime adjustment of persistence weight
    "clamp_logit": 2.0,       # median may move at most this far from last rate
    "sigma_floor": 0.30,
    "sigma_cap": 1.25,
    "sigma_base": 0.35,       # irreducible spread component
    "k_horizon": 0.10,        # horizon term per sqrt(day)
    "resid_w": 0.5,           # weight of backtest residual in sigma
}

STORY_IDS = (3, 4, 5, 6, 7)   # Ch1, Ch2, Ch3, Ch4, final mission
PLAT_ID = 1
FULL_ID = 14                  # 100% completion (gold)


def parse_time(s):
    """ISO 8601 UTC -> aware datetime (accepts trailing Z)."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(rate):
    """logit of a percent in [0, 100], clamped to the printed grid."""
    p = clamp(rate, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    p = 1.0 / (1.0 + math.exp(-clamp(l, -12.0, 12.0)))
    return clamp(p * 100.0, 0.0, 100.0)


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.

    Returns (intercept, slope, slope_se, resid_rms) or None when the fit
    is degenerate (fewer than 2 distinct x, or zero total weight).
    """
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    if sw <= 0.0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    s2 = sum(w * r * r for w, r in zip(ws, resid)) / sw
    # Effective-sample correction for the slope standard error.
    se = math.sqrt(s2 / sxx) if sxx > 0 else 0.0
    rms = math.sqrt(sum(r * r for r in resid) / n)
    return (a, b, se, rms)


def recency_weights(times_h, t_ref, half_life):
    return [0.5 ** max(0.0, (t_ref - t) / half_life) for t in times_h]


class Series:
    """One trophy's collapsed history, in hours since release."""

    def __init__(self, reads, idx, t0):
        # reads: list of (time_h, players, rate or None), oldest first.
        pts = []
        for t_h, players, rate in reads:
            if rate is None:
                continue
            pts.append((t_h, players, rate))
        # Plateau-collapse: consecutive identical (rate, players) reads
        # are one Exophase refresh; keep the first time of the run.
        collapsed = []
        for pt in pts:
            if collapsed and collapsed[-1][1] == pt[1] and collapsed[-1][2] == pt[2]:
                continue
            collapsed.append(pt)
        self.pts = collapsed
        self.last_rate = pts[-1][2] if pts else None
        self.last_players = pts[-1][1] if pts else None
        self.last_t = pts[-1][0] if pts else None
        self.t0 = t0

    def __len__(self):
        return len(self.pts)


def fit_estimators(pts, t_target_h, params):
    """Fit the four median estimators on points, predict at target.

    pts: list of (t_h, players, rate). Returns dict name -> (logit, se)
    with se the slope standard error where defined, else 0.0.
    """
    out = {}
    if not pts:
        return out
    hl = params["half_life_h"]
    t_ref = pts[-1][0]
    ts = [p[0] for p in pts]
    ws = recency_weights(ts, t_ref, hl)
    lg = [logit_pct(p[2]) for p in pts]
    xs_log = [math.log1p(max(0.0, t)) for t in ts]

    # (a) persistence
    out["persist"] = (logit_pct(pts[-1][2]), 0.0)

    # (b) log-time linear trend
    fit = wls(xs_log, lg, ws)
    if fit is not None:
        a, b, se, _ = fit
        out["logtrend"] = (a + b * math.log1p(max(0.0, t_target_h)), se)

    # (c) earners / players decomposition
    earn = [max(p[2] / 100.0 * p[1], 0.05) if p[1] else None for p in pts]
    play = [p[1] if p[1] else None for p in pts]
    if all(e is not None for e in earn) and all(v is not None for v in play):
        le = [math.log(e) for e in earn]
        lp = [math.log(v) for v in play]
        fe = wls(xs_log, le, ws)
        fp = wls(xs_log, lp, ws)
        if fe is not None and fp is not None:
            xt = math.log1p(max(0.0, t_target_h))
            pe = fe[0] + fe[1] * xt
            pp = fp[0] + fp[1] * xt
            rate = math.exp(pe - pp) * 100.0
            out["decomp"] = (logit_pct(rate), 0.0)

    # (d) real-time linear trend
    fit_r = wls(ts, lg, ws)
    if fit_r is not None:
        a, b, se, _ = fit_r
        out["realtime"] = (a + b * t_target_h, se)
    return out


def backtest_errors(series_list, t_target_h, params):
    """Leave-future-out errors per estimator, pooled over all trophies.

    Returns (global_err, group_err): dicts estimator -> mean abs logit
    error, global over every trophy and per group. Empty when the
    history is too short to split.
    """
    bt = params["backtest_h"]
    errs = {"persist": [], "logtrend": [], "decomp": [], "realtime": []}
    gerrs = {}
    for series, group in series_list:
        pts = series.pts
        if len(pts) < 3:
            continue
        t_last = pts[-1][0]
        split = t_last - bt
        pre = [p for p in pts if p[0] <= split]
        post = [p for p in pts if p[0] > split]
        if len(pre) < 2 or not post:
            continue
        for t_h, players, rate in post:
            est = fit_estimators(pre, t_h, params)
            y = logit_pct(rate)
            for name in errs:
                if name in est:
                    errs[name].append(abs(est[name][0] - y))
                    gerrs.setdefault(group, {}).setdefault(name, []).append(
                        abs(est[name][0] - y))
    def mean(v):
        return sum(v) / len(v) if v else None
    glob = {k: mean(v) for k, v in errs.items()}
    grp = {g: {k: mean(v) for k, v in d.items()} for g, d in gerrs.items()}
    return glob, grp


def weights_from_errors(glob, grp, group, params):
    """Inverse-error weights, group pooled and shrunk to global."""
    names = ("persist", "logtrend", "decomp", "realtime")
    default = params["w_default"]

    def inv(err_dict):
        ws = []
        for i, name in enumerate(names):
            e = err_dict.get(name) if err_dict else None
            if e is None:
                ws.append(default[i])
            else:
                ws.append((1.0 / (e + 0.05)) ** params.get("backtest_power", 1.0))
        s = sum(ws)
        return [w / s for w in ws] if s > 0 else list(default)

    wg = inv(glob)
    if group in grp:
        wgrp = inv(grp[group])
        sh = params["shrink_global"]
        return [sh * a + (1.0 - sh) * b for a, b in zip(wg, wgrp)]
    return wg


def regime_adjustment(series, weights, params):
    """Adjust the persistence weight for the current regime.

    Acceleration (recent 12h slope above the previous 24h slope) lowers
    persistence; dilution (players growing faster than earners over the
    recent window) raises it. Bounded to +/-persist_adj_max.
    """
    pts = series.pts
    if len(pts) < 4:
        return weights
    t_last = pts[-1][0]
    recent = [p for p in pts if p[0] > t_last - 12.0]
    older = [p for p in pts if t_last - 36.0 < p[0] <= t_last - 12.0]
    adj = 0.0
    if len(recent) >= 2 and len(older) >= 2:
        def slope(sub):
            xs = [math.log1p(max(0.0, p[0])) for p in sub]
            ys = [logit_pct(p[2]) for p in sub]
            ws = [1.0] * len(sub)
            f = wls(xs, ys, ws)
            return f[1] if f else 0.0
        accel = slope(recent) - slope(older)
        # accel is logit per log-hour; scale to a bounded adjustment.
        adj -= clamp(accel * 0.5, -1.0, 1.0) * params["persist_adj_max"]
    # Dilution: compare log-player growth vs log-earner growth, recent.
    if len(recent) >= 2:
        t0, t1 = recent[0][0], recent[-1][0]
        if t1 > t0 and recent[0][1] and recent[-1][1]:
            gp = math.log(recent[-1][1] / recent[0][1])
            e0 = max(recent[0][2] / 100.0 * recent[0][1], 0.05)
            e1 = max(recent[-1][2] / 100.0 * recent[-1][1], 0.05)
            ge = math.log(e1 / e0)
            if gp > ge + 0.02:
                adj += params["persist_adj_max"] * 0.5
            elif ge > gp + 0.02:
                adj -= params["persist_adj_max"] * 0.5
    adj = clamp(adj, -params["persist_adj_max"], params["persist_adj_max"])
    w = list(weights)
    w[0] = clamp(w[0] + adj, 0.05, 0.85)
    s = sum(w)
    return [x / s for x in w]


def group_of(tid):
    if tid in STORY_IDS:
        return "story"
    if tid in (PLAT_ID, FULL_ID):
        return "plat"
    return "grind"


def forecast_case(case, params=None):
    """Return the forecast rows [{"id": n, "q": [...]}] for one case."""
    params = dict(PARAMS, **(params or {}))
    trophies = case["trophies"]
    history = case.get("history", [])
    t_at = parse_time(case["at"])
    t_target = parse_time(case["target"])
    release = case.get("game", {}).get("release_date", "2026-11-19")
    t0 = parse_time(release + "T00:00:00Z") if len(release) == 10 else parse_time(release)
    horizon_days = max(0.0, (t_target - t_at).total_seconds() / 86400.0)
    t_target_h = (t_target - t0).total_seconds() / 3600.0

    # Build per-trophy series. history rows are dicts with slot/at/
    # players_tracked/rates aligned with trophies order.
    reads = []
    for row in history:
        try:
            t_h = (parse_time(row["at"]) - t0).total_seconds() / 3600.0
        except Exception:
            continue
        players = row.get("players_tracked")
        rates = row.get("rates") or []
        reads.append((t_h, players, rates))
    reads.sort(key=lambda r: r[0])

    series_list = []
    for idx, tr in enumerate(trophies):
        per = [(t_h, players, rates[idx] if idx < len(rates) else None)
               for (t_h, players, rates) in reads]
        s = Series(per, idx, t0)
        series_list.append((s, group_of(tr["id"])))

    glob_err, grp_err = backtest_errors(series_list, t_target_h, params)

    medians = {}
    sigmas = {}
    for (series, group), tr in zip(series_list, trophies):
        tid = tr["id"]
        if series.last_rate is not None and series.last_rate >= 99.995:
            medians[tid] = 100.0
            sigmas[tid] = 0.0
            continue
        est = fit_estimators(series.pts, t_target_h, params)
        names = ("persist", "logtrend", "decomp", "realtime")
        if not est:
            # No usable history at all: neutral 1% median, wide spread.
            med_logit = logit_pct(1.0)
            sigma = 1.0
            medians[tid] = inv_logit_pct(med_logit)
            sigmas[tid] = sigma
            continue
        w = weights_from_errors(glob_err, grp_err, group, params)
        w = regime_adjustment(series, w, params)
        avail = [n for n in names if n in est]
        wsum = sum(w[names.index(n)] for n in avail)
        wnorm = {n: w[names.index(n)] / wsum for n in avail}
        med_logit = sum(wnorm[n] * est[n][0] for n in avail)
        # Shrink toward persistence: trend extrapolation over a 4.5-day
        # horizon is the riskiest move the ensemble makes.
        if params["persist_shrink"] > 0.0 and "persist" in est:
            s = params["persist_shrink"]
            med_logit = (1.0 - s) * med_logit + s * est["persist"][0]
        # Clamp the median move from the last rate.
        if series.last_rate is not None:
            lo = logit_pct(series.last_rate) - params["clamp_logit"]
            hi = logit_pct(series.last_rate) + params["clamp_logit"]
            med_logit = clamp(med_logit, lo, hi)
        # Sigma: disagreement + slope SE x extrapolation + backtest
        # residual + horizon term, on top of an irreducible base.
        vals = [est[n][0] for n in avail]
        wts = [wnorm[n] for n in avail]
        mean_v = sum(a * b for a, b in zip(vals, wts))
        disagree = math.sqrt(sum(w * (v - mean_v) ** 2 for w, v in zip(wts, vals)))
        se_term = 0.0
        if "logtrend" in est:
            xt = math.log1p(max(0.0, t_target_h))
            x_last = math.log1p(max(0.0, series.pts[-1][0])) if series.pts else xt
            se_term = est["logtrend"][1] * abs(xt - x_last)
        resid = 0.0
        if glob_err.get("logtrend") is not None:
            resid = params["resid_w"] * glob_err["logtrend"]
        horiz = params["k_horizon"] * math.sqrt(horizon_days)
        sigma = math.sqrt(params["sigma_base"] ** 2 + disagree ** 2
                          + se_term ** 2 + resid ** 2 + horiz ** 2)
        sigma = clamp(sigma, params["sigma_floor"], params["sigma_cap"])
        medians[tid] = inv_logit_pct(med_logit)
        sigmas[tid] = sigma
        # Remember the (median, persistence) gap for asymmetric tails.
        series._trend_gap = med_logit - est.get("persist", (med_logit, 0.0))[0]

    # Structural priors, applied softly to medians only.
    # Story chain: Ch1 >= Ch2 >= Ch3 >= Ch4 >= final (rates).
    chain = [t for t in STORY_IDS if t in medians]
    run_min = None
    for tid in chain:
        v = medians[tid]
        if run_min is None or v < run_min:
            run_min = v
        else:
            medians[tid] = run_min
    # Platinum <= every other trophy; 100% <= final mission.
    if PLAT_ID in medians:
        others = [v for k, v in medians.items() if k != PLAT_ID]
        if others:
            medians[PLAT_ID] = min(medians[PLAT_ID], min(others))
    if FULL_ID in medians and 7 in medians:
        medians[FULL_ID] = min(medians[FULL_ID], medians[7])

    rows = []
    for (series, group), tr in zip(series_list, trophies):
        tid = tr["id"]
        if medians[tid] >= 99.995:
            rows.append({"id": tid, "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
            continue
        med_logit = logit_pct(medians[tid])
        sigma = sigmas[tid]
        gap = getattr(series, "_trend_gap", 0.0)
        up = 1.0 + params["tail_extra"] if gap > 0 else 1.0
        down = 1.0 + params["tail_extra"] if gap < 0 else 1.0
        down *= 1.0 + params.get("k_downtail", 0.0) * math.sqrt(horizon_days)
        qs = []
        for z in Z:
            s = sigma * (up if z > 0 else down if z < 0 else 1.0)
            qs.append(round(inv_logit_pct(med_logit + z * s), 4))
        # Enforce non-decreasing after rounding.
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"id": tid, "q": qs})
    return rows


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        out = {"trophies": rows}
        with open(os.path.join(out_dir, case_id + ".out"), "w",
                  encoding="utf-8") as fh:
            json.dump(out, fh, separators=(",", ":"), sort_keys=False)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
