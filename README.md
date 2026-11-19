# GTA VI Trophy Rarity Forecast — v3 (make.fun)

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/funtoons/gta6-trophy-rarity-forecast-01-v3/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/funtoons/gta6-trophy-rarity-forecast-01-v3/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for each of the 55 GTA VI PS5 trophies, five quantiles of its
Exophase earn rate (%) at 2026-11-26T00:00:00Z.

Referral: [make.fun/r/mqqhsens](https://make.fun/r/mqqhsens)

## Program

`main.py` — Python 3.13, standard library only, offline, deterministic
(no randomness, no clock, no network). Reads every `*.in` case in
`$MAKEFUN_INPUT_DIR` (else `/input`), writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (else `/output`).

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Method

Adaptive ensemble in logit space, per trophy:

1. **Plateau-collapse** — consecutive identical reads are one Exophase
   refresh, not fresh data; they are merged before any fit.
2. **Four median estimators** — (a) persistence (last rate),
   (b) log-time linear trend of logit(rate) vs log(1 + hours since
   release), recency-weighted (half-life 8h), (c) earners/players
   decomposition (log earners and log players extrapolated separately),
   (d) real-time linear trend (for real-time logistic adoption).
3. **Learned ensemble weights** — leave-future-out backtest (fit to
   T−24h, score on the last 24h of reads), inverse-error to the 6th
   power, pooled per trophy group (story / grind / platinum+100%) and
   shrunk toward the global weights.
4. **Regime-adaptive persistence weight** — acceleration lowers it,
   dilution (players outgrowing earners) raises it, bounded ±0.15.
5. **Structural priors (soft)** — story-chain monotonicity
   (Ch1 ≥ Ch2 ≥ Ch3 ≥ Ch4 ≥ final), platinum ≤ every other trophy,
   100% ≤ final mission; medians clamped to logit(last) ± 2.
6. **Adaptive spread** — from ensemble disagreement, slope standard
   error × extrapolation distance, backtest residual and a horizon
   term; floor 0.30 logit.

## Validation (fresh synthetic suite, 9 worlds)

Mean pinball loss (logit scale) over 9 fresh synthetic worlds
(2-day history, 4.52-day horizon, 55 trophies, refreshes every 2–5h):

| world | v3 | persistence | trend |
|---|---:|---:|---:|
| logtime | 0.0999 | 0.2612 | 0.2214 |
| realtime-logistic | 0.2671 | 0.4541 | 0.3619 |
| catchup-fast | 0.1201 | 0.2996 | 0.2316 |
| catchup-slow | 0.1047 | 0.2345 | 0.2224 |
| stall | 0.0872 | 0.2157 | 0.2234 |
| dilution-wave | 0.1764 | 0.3037 | 0.2534 |
| late-flood | 1.5139 | 0.8764 | 1.0609 |
| viral-surge | 0.0870 | 0.2618 | 0.2217 |
| mixed-grind | 0.1689 | 0.3086 | 0.2496 |
| **mean** | **0.2917** | **0.3573** | **0.3385** |

v3 beats both baselines on the mean and in 8 of 9 worlds. It loses
only **late-flood** (players ×6 after the checkpoint, earners flat):
no signal in history can predict that shock — even persistence, the
best estimator there, scores 0.88. The headline target of 0.2038
(v2's mean on its own suite) was **not** reached, entirely because of
that one world (without it, v3's mean is 0.149). Details, sweep tables
and caveats: `gta6-trophy-rarity-forecast-01-v3-progress.md` in the
parent workspace (not shipped in this repo).

Practice cases p00/p01: valid (55 rows, quantiles non-decreasing,
0–100), byte-identical across two runs, ~0.04 s per case.
