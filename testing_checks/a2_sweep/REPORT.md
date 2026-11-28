# a2_sweep: independent re-check of web/numbers.json

Scripts: `run.py` (everything, about 25 s on 12 cores, cached in `results.pkl`), `extra.py` (runaway edge, 200 seeds). Machine-readable output: `numbers_summary.json`, `extra_summary.json`. Figures in `figures/`. Nothing outside this folder was edited.

## Method
Same recipe as `scripts/web_numbers.py`: `ex.fitted_settings()` (settings/sp500_calibrated.json, a1 = 1.0844, burn 1000), histories of 19,329 days (the S&P record), `ex.run_mix`, `ex.summary` (Hill 5%, excess kurtosis, memory = autocorrelation of |r| at lag 20, worst day in sd, big days > 4 sd), `ex.crash_counts` / `ex.big_falls` / `ex.crash_check` / `ex.crash_table`. Trader mix (z, x) from the `paths` that `run_mix` fills (the same arrays as `ex.model_market`; I checked that r is identical), plus `model_market` up to the runaway day for histories that ran away.

Seeds. Saved: 1000-1039 (sweep: first 10). No file in herding/, scripts/, tests/, web/ uses these: **fresh 700000-700039** (mixes, crash check), **710000-710039** (sweep, 40 seeds per a1 instead of 10), **720000-720199** (runaway edge). Used elsewhere: 0-9 and other small seeds, 1000-1039, 2000-2019, 3000-3199, 4000-4400, 5000, 60000-62039.

## Part A: saved numbers reproduce exactly
I reran everything, not only a sample: all 40 seeds x 3 mixes, and the sweep (12 a1 values x 10 seeds = 120 cells).

| Check | Result |
|---|---|
| fundamentalists kurtosis, tail, memory (40 seeds) | max abs difference 0.0 |
| herding tail, kurtosis, memory, worst (40 seeds) | max abs difference 0.0 |
| runaway days, herding and chartists (all 40 seeds) | identical lists (chartists: 40 of 40 run away) |
| crash falls per size and every >= 8 sd fall, 40 histories | identical |
| sweep, 120 cells: runaway cell and day | 0 mismatches |
| sweep tail, memory, kurtosis | max abs difference 5e-11 (float round-off; JSON round-trip) |

Verdict: **reproduced**. No difference found.

## Part B: fresh seeds (700000-700039), fitted S&P settings
Median (5th-95th percentile) over 40 histories. "Saved" = numbers.json (rerun gives the same). p = Mann-Whitney, fresh vs saved. Big days per 1,000 is not in numbers.json; I recomputed it on the saved seeds (identical reruns).

### Herding mix (0 of 40 ran away in both sets)
| Measure | Saved | Fresh | S&P 500 | Fresh median in saved 90%? | MW p | Verdict |
|---|---|---|---|---|---|---|
| Tail exponent | 3.11 (2.69-3.71) | 3.21 (2.74-3.80) | 2.92 | yes | 0.41 | reproduced |
| Excess kurtosis | 5.53 (3.16-12.2) | 5.46 (3.24-13.7) | 25.2 | yes | 0.65 | reproduced |
| Volatility memory | 0.169 (0.119-0.227) | 0.173 (0.111-0.236) | 0.195 | yes | 0.55 | reproduced |
| Worst day (sd) | -9.3 (-16.9 to -7.4) | -9.7 (-14.1 to -6.8) | -23.0 | yes | 0.87 | reproduced |
| Big days per 1,000 | 4.84 (3.33-6.20) | 4.71 (3.35-6.53) | 5.59 | yes | 0.92 | reproduced |

### Fundamentalists only (0 ran away)
| Measure | Saved | Fresh | MW p | Verdict |
|---|---|---|---|---|
| Excess kurtosis | 0.009 (-0.043 to 0.051) | 0.010 (-0.028 to 0.053) | 0.37 | reproduced; bell curve holds |
| Tail exponent | 6.59 (6.14-7.00) | 6.56 (6.20-6.83) | 0.27 | reproduced |
| Memory | 0.002 | 0.000 | 0.30 | reproduced (no clustering) |
| Worst day (sd) | -3.90 | -3.99 | 0.44 | reproduced |
| Big days per 1,000 | 0.05 | 0.05 | 0.42 | reproduced |

### Chartists without herding
40 of 40 ran away in saved and fresh seeds: **reproduced**.

### Where the S&P 500 sits relative to the herding model (both sets)
Inside the model's middle 90%: tail (2.92), memory (0.195), big days (5.6 per 1,000). **Outside**: kurtosis 25.2 vs a 90% band up to 13.7 (saved 12.2), and worst day -23.0 sd vs a band to -14.1 (saved -16.9). Exceeding the S&P: kurtosis >= 25.2 in 1 of 40 fresh and 2 of 40 saved histories; worst day <= -23.05 in 1 of 40 fresh and 2 of 40 saved (saved min -92.6, kurtosis max 5,258: one explosive history). So the model gets the S&P's typical tail and clustering but its typical history is not as extreme in its single worst day as the S&P (about 1 in 20 histories is).

### Crash check (survivors pooled, 40 x 19,329 days = 3,068 years per set)
| Measure | Saved (rerun = json) | Fresh |
|---|---|---|
| Years between 8-sd falls | 34.9 (88 falls; 32 of 40 histories had one; bootstrap 90% 28-44) | 33.7 (91 falls; 31 of 40; bootstrap 26-45) |
| Bell curve, 8 sd | 6e12 years | same |
| S&P 500, 8 sd | 11.0 years | 11.0 |
| Years between falls as big as Black Monday (23.0 sd) | 1,534 (2 falls) | 3,068 (1 fall) |
| Bell curve, Black Monday size | 5e114 years | same |
| crash_check: 8 sd within 50 years | True | True |
| crash_check: Black Monday size within 50 years | False | False |
| crash_check passed | False | False |

Other sizes (pooled years between falls, saved / fresh / S&P): 4 sd 1.70 / 1.63 / 1.26; 5 sd 4.19 / 3.96 / 2.26; 6 sd 8.7 / 8.9 / 3.5; 8 sd 34.9 / 33.7 / 11.0; 10 sd 128 / 93 / 25.6; 12 sd 170 / 256 / 38. Verdict: **reproduced** (8 sd stable to 3%). The Black Monday count (2 vs 1) is a 1-2 event number: reproduced in order of magnitude only (thousands of years), and either way the model's falls are several times rarer than the S&P's at 5 sd and above (S&P is about 2 times more frequent at 5 sd and up to 4-7 times at 10-12 sd).

## Herding-strength sweep (fresh seeds 710000-710039, 40 per a1; saved sweep has 10 seeds)
Medians are over survivors; n runaways out of 40. z = average share of chartists (trend chasers; value traders = 1 - z); |x| = average absolute mood of the chartists. Saved columns: median tail from numbers.json (10 seeds, runaways out of 10).

| a1 | runaways | tail | kurtosis | memory | z (trend chasers) | value traders | mean abs x | saved tail (runaways of 10) |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 40/40 | n/a | n/a | n/a | (to runaway) 0.138 | | 0.26 | n/a (10) |
| 0.2 | 38/40 | 2.67 (2 survivors, noise) | 10.4 | 0.264 | 0.049 | 0.951 | 0.40 | n/a (10) |
| 0.4 | 37/40 | 2.67 (3) | 11.8 | 0.306 | 0.047 | 0.953 | 0.44 | 2.24 (9) |
| 0.6 | 34/40 | 2.46 (6) | 30.2 | 0.315 | 0.054 | 0.946 | 0.46 | 2.45 (9) |
| 0.8 | 14/40 | 2.56 | 14.5 | 0.282 | 0.044 | 0.956 | 0.55 | 2.77 (5) |
| 1.0 | 3/40 | 2.96 | 8.29 | 0.204 | 0.039 | 0.961 | 0.63 | 2.99 (0) |
| **1.0844 (fitted)** | 0/40 | 3.30 | 4.91 | 0.163 | 0.036 | 0.964 | 0.66 | 3.07 (0) |
| 1.2 | 0/40 | 3.64 | 3.69 | 0.121 | 0.033 | 0.967 | 0.72 | 3.57 (0) |
| 1.4 | 0/40 | 4.17 | 1.78 | 0.082 | 0.031 | 0.969 | 0.80 | 3.94 (0) |
| 1.6 | 0/40 | 4.54 | 1.27 | 0.058 | 0.029 | 0.971 | 0.86 | 4.68 (0) |
| 1.8 | 0/40 | 4.62 | 0.99 | 0.049 | 0.028 | 0.972 | 0.90 | 4.62 (0) |
| 2.0 | 0/40 | 4.55 | 0.77 | 0.039 | 0.028 | 0.972 | 0.93 | 4.53 (0) |

(Finer points 0.9-1.5 are in `numbers_summary.json` and the figure; e.g. a1 = 0.9: 6/40 ran away, tail 2.71; 0.95: 1/40, 2.72; 1.05: 0/40, 3.16.) At a1 = 0.2-0.6 the median is over only 2-6 survivors, mostly calm histories, so those points are selection-biased and not meaningful. The saved and fresh sweeps agree where both have survivors (differences in tail up to about 0.2-0.3 at 0.8 and 1.08 are within sampling noise of 10 seeds). Runaway counts at 0.8: 5/10 saved vs 14/40 fresh (35%).

Findings:
- **Stronger herding makes tails thinner, not fatter**, once the price stays near its value: tail 2.96 (a1 1.0) to 4.55 (a1 2.0); kurtosis 8.3 to 0.8; memory 0.20 to 0.04. All three vanish smoothly as a1 grows. Mood |x| rises toward 1 (the crowd leans all one way) while the trend-chaser share z falls (0.039 to 0.028): strong herding makes the chartists agree and stay put, so there are fewer bursts of switching.
- **The trend-chaser share is small everywhere**: about 3-5% of the 500 traders, 96% value traders at the fitted a1.
- **Where the S&P tail (2.92) sits**: the median tail crosses 2.92 at a1 about 0.99 (fresh; saved sweep 0.93). The fitted a1 = 1.084 gives median tail 3.1-3.3, slightly thinner than the S&P.
- **Runaway edge**: the fitted a1 is only just on the stable side. Over 200 further fresh seeds (extra.py), runaways: a1 0.9: 23/200 (11.5%), 0.95: 20/200, 1.0: 12/200 (6%), 1.05: 4/200, **fitted 1.0844: 2/200 (1%)**, 1.10: 3/200, 1.15: 0/200. The S&P's own tail of 2.92 needs a1 about 1.0, where 6% of histories run away, and weaker herding (fatter tails and more clustering, a1 < 0.8) runs away in most histories (a1 = 0.6: 34/40). The model has fat tails only in the narrow band just above the point where it breaks.

## Part C: figures (`figures/`)
- `fig04_fresh_proposal_checks.png` vs figures/fig04_proposal_checks.png. Key numbers old (RESULTS.md / saved): no-chartist kurtosis 0.01, herding tail 3.10-3.11. New: 0.01 and 3.21. Same shape.
- `fig10_fresh_herding_dial.png` (6 panels: runaway share, kurtosis, tail, memory, trend-chaser share z, mood |x|; saved 10-seed sweep dashed): the fresh curve overlays the saved one.
- `fig_crash_wait_fresh.png`: bell curve, model (fresh and saved rerun), S&P 500; the two model lines nearly coincide up to 10 sd, both under the S&P line (model waits are about 1.3-7 times longer), both far under the bell curve.

## Conclusion
1. **The saved numbers are exactly reproducible** (Part A), and **the fresh out-of-sample seeds reproduce every saved measure**: every fresh median is inside the saved middle 90% and no Mann-Whitney test is below 0.05 (smallest p 0.27). No sign of cherry-picked seeds.
2. **Project question.** With bell-curve news the fundamentalists-only market is a bell curve (kurtosis 0.01, tail 6.6, no memory), and adding herding chartists gives fat tails (Hill 3.2, S&P 2.92; excess kurtosis 5.5), volatility memory (0.17 vs the S&P's 0.195), the right number of 4-sd days (4.7 vs 5.6 per 1,000), and 8-sd falls every 34 years (S&P: every 11; bell curve: trillions). So herding alone can produce the S&P's tail exponent, clustering and wild days in kind. It matches the size of the S&P's extremes less well: kurtosis 5.5 vs 25 and worst day -9.7 sd vs -23 sd, and falls of 5 sd or more come 2-7 times less often than in the S&P (more the bigger the fall). Falls the size of Black Monday are about thousands of years apart in the model, so the crash check's second half fails.
3. **Caveat that matters.** The fat tails come only from a narrow window of herding strength just above a runaway edge. Weaker herding (the regime that would give still fatter tails and more memory) runs away as an endless bubble in most histories; stronger herding thins the tails. At the fitted a1 about 1% of histories run away, and 6% at the a1 that best hits the S&P's tail. Also the "herding makes tails fatter" reading should be put carefully: relative to no chartists, yes; but within the herding model, more herding means thinner tails.
4. No measure contradicts numbers.json. The shifts that exist are statistical noise (kurtosis band, Black Monday count) and are labelled above. The only claims in the numbers I would flag are the single-event ones (Black Monday every 1,534 years rests on 2 falls; fresh gives 1 fall, 3,068 years) and sweep points with fewer than about 10 survivors.
