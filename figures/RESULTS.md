# Results: what each figure shows

**Model version:** the rebuild of November 2026. The model's equations are in
[`notes/SPEC.md`](../notes/SPEC.md), each with its source. Figures come from
`python3 make_figures.py` (about 1–2 minutes on 8 cores). Every model check uses
**held-out seeds** (1000–1039) that neither fit ever saw.

## Two fitted settings

The Nature paper gives no parameter values, and the paper that does (Lux &
Marchesi 2000) is not in hand. So the settings are **fitted**, not copied
(`python3 -m herding.calibrate`):

| | Fitted to | Score | Key settings |
|---|---|---|---|
| **Paper-matched** (orange), `settings/paper.json` | the paper's own numbers: α 2.64, H 0.48, H\|r\| 0.85, back to a bell curve by 25 days | 0.06 (in units of the paper's error bars, squared) | a1 0.54, a2 1.58, a3 2.21, v1 1.91, v2 1.16; chartist push T_c = 0.48, fundamentalist pull T_f = 17.1 |
| **S&P-calibrated** (blue), `settings/spx.json` | the S&P 500's 11-number fingerprint, 1950–2026 | 2.94 (mean squared gap, in units of the S&P's own uncertainty) | a1 1.08, a2 0.13, a3 0.76, v1 2.02, v2 0.21; T_c = 6.07, T_f = 7.77 |

The news is the same in both: Gaussian, 0.5% a day, no memory.

| Colour | Means |
|---|---|
| Black | real market data |
| Blue | S&P-calibrated model |
| Orange | paper-matched model |
| Green | news (the input), or the model with no chartists |

Vocabulary: **kurtosis** = how fat the tails are (0 = bell curve). **α** = tail
exponent (smaller = fatter; real markets about 3). **H** = DFA memory exponent
(0.5 = no memory, near 1 = calm and wild spells that last). **Percentile** = the
share of model histories below the real value; under 5% or over 95% = unusual.

---

## Fig 1 — The price follows its value, but returns are wilder than the news (paper's Fig 1)
`fig1_snapshot.png`. 4,000 days of one held-out history per setting.
- The price (a) tracks its fundamental value closely, as in the paper.
- The news (c) has kurtosis 0.0: a plain bell curve. The returns (b) have bursts
  (kurtosis 2.9 in this stretch), and the bursts line up with a high share of
  chartists (d).
- The S&P-calibrated model keeps few chartists most of the time (d, right), so
  this stretch is calm; its bursts are rarer and are shown in Fig 5.

## Fig 2 — Fat tails over a day, a bell curve over weeks (paper's Fig 2)
`fig2_tails.png`.
- **Paper-matched:** tail α (the paper's log-log fit to the largest 30% of moves)
  is **2.62 on average over 8 held-out runs of 100,000 days** (paper: 2.64 ± 0.08).
  The run shown gives 2.84. Kurtosis falls from 68 at 1 day to 1.8 at 25 days:
  the paper's **cross-over to the bell curve**.
- **S&P-calibrated:** the median kurtosis falls from 5.5 (1 day) to 1.6 (25 days).
- **Real S&P 500:** kurtosis 25 at 1 day, but still **3.3 at 25 days**. The real
  market keeps fat tails at long horizons more than either model does (1987
  matters a lot here).
- The news collapses onto the bell curve at every horizon, as in the paper.

## Fig 3 — No memory of direction, long memory of size (paper's Fig 3)
`fig3_memory.png`. Detrended fluctuation analysis (Peng et al. 1994).

| | Paper | Paper-matched (mean of 8 held-out runs) | S&P 500 | S&P-calibrated (median of 20) |
|---|---|---|---|---|
| H of returns | 0.48 | 0.47 | 0.50 | **0.39** |
| H of \|returns\| | 0.85 | 0.81 | 0.90 | 0.88 |
| H of news / \|news\| | 0.49 / 0.51 | 0.50 / 0.49 | — | — |

The long memory in move sizes is created by the traders: the news has none.
The S&P-calibrated model's returns **reverse** a little from day to day
(H 0.39), while the S&P's do not (0.50). This is its clearest miss (Fig 7).

## Fig 4 — The proposal's checks
`fig4_checks.png`.
- **No chartists → bell curve:** median kurtosis **0.01** over 40 histories,
  Hill α 6.6 (thin tails). Check passed.
- **With herding → α near 3:** Hill α median **3.10** (S&P-calibrated) and **2.90**
  (paper-matched); S&P 500: 2.92. The inverse cubic law holds. Check passed.
- Kurtosis: S&P-calibrated median 5.5 vs the S&P's 25. The S&P's number is
  dominated by one day (Black Monday); kurtosis is that sensitive.

## Fig 5 — The mechanism (paper p. 500)
`fig5_mechanism.png`. S&P-calibrated, one 100,000-day held-out run.
- Average move size climbs steadily with the share of chartists: 0.6× the
  average when chartists are rare, 3.5× when they are 20–25% of traders.
- Crowded markets are rare: chartists are under 5% of traders on about 77% of
  days.
- The wildest stretch (a) shows a burst that **ends on its own** when the
  chartist share falls back, as the paper describes.
- The paper's efficient-market claims (`analysis/efficiency_claims.py`, 4 × 100,000 days):

| Claim | Paper-matched | S&P-calibrated |
|---|---|---|
| Price equals value on average | mean gap +0.0000 | +0.0004 |
| No persistent mispricing | over 10% off on 0.02% of days, at most 4 days in a row | 0.47% of days, at most 29 days |
| Volatility rises with the chartist share | 0.59× (least crowded tenth) to 1.41× (most crowded) | 0.51× to 1.49× |

## Fig 6 — Kirman's ants (the herding check from the proposal)
`fig6_kirman.png`. 40 million steps per case. The simulated time at each state
matches Kirman's exact formula: total variation distance **0.020** (herding) and
**0.004** (no herding). The same recruitment rule moves traders in and out of the
chartist group in Lux–Marchesi.

## Fig 7 — Is the S&P 500 a typical history of the model?
`fig7_typical.png`. 40 held-out S&P-calibrated histories, each 19,329 days long.
- **9 of 11 measures typical.** Daily sd 90%, kurtosis 95%, Hill α 32%,
  next-day correlation 92%, clustering 82% (1 day) and 60% (50 days), H|r| 78%,
  5-day kurtosis 57%, 25-day kurtosis 92%.
- **Unusual:** H of returns (100%: model returns reverse more) and the
  paper-style α (0%: the S&P's tail is fatter in the middle-to-large range).
- One of the 40 histories has a freak event (kurtosis about 5,000): a bubble that
  burst.

## Fig 8 — Markets the model never saw
`fig8_out_of_sample.png`. Percentile of each real market among 20 model
histories of the same length. 49 of 220 cells are unusual. The 20 single stocks
have **fatter tails than an index-calibrated model** (kurtosis 75–100%, Hill α
mostly 0–15%), which is expected: companies have their own jumps (earnings,
lawsuits) on top of the market's. The S&P since 1928 is unusual on memory (H
and H|r| both at 100%) and the paper-style α (0%), with clustering at the 95%
edge: the 1930s were far wilder and longer-lasting than anything since.

## Fig 9 — How robust is it? (paper: "an open set of parameter values")
`fig9_robustness.png`. 400 random settings, one 20,000-day run each.
- **44%** run away into an endless bubble.
- Of the 222 that stay stable: **65%** give fat tails *and* long memory, 4% fat
  tails only, 32% thin tails.
- The paper is right that good settings form a whole region, not a single point.
  But the region is not "almost everything": stability is the main limit.

## Fig 10 — Turning the herding dial
`fig10_herding.png`. Only a1 (copying the majority mood) is changed.
- **Paper-matched:** kurtosis peaks at a1 = 0.6 (fitted 0.54) and **collapses to a
  bell curve for a1 ≥ 1**. More herding does *not* simply mean more crashes. Below
  1, moods keep flipping, and the crowd swings hardest near the critical point.
  Above 1, the chartists lock into one mood, so they push the price steadily
  instead of in bursts.
- **S&P-calibrated:** a1 below 0.8 makes most histories run away (endless
  bubbles). These settings only work in a narrow band of herding strength.

---

## Limitations (honest list)
1. **Parameters are fitted, not the paper's.** The paper's parameter tables
   (Lux & Marchesi 2000) were not available. The remembered "set I" gives a
   price that runs away after about 28,500 days in our build (`tests/test_model.py`).
2. **Two source disagreements** were settled by choice (SPEC.md sec. 3): the
   fundamentalist demand uses (p_f − p)/p, and tick rates are scaled so that eq. 25
   holds.
3. **Daily reversal:** the S&P-calibrated model's returns reverse a little
   (H 0.39 vs 0.50).
4. **No bounce-back after big days:** after a 3-std day the S&P tends to reverse
   (next-day correlation −0.19). The model doesn't (+0.02; S&P at the 0th
   percentile, `analysis/crash_shape.py`). The old model's slow-motion slides
   (+0.31) are gone.
5. **Stability:** the S&P-calibrated settings run away in 2 of 40 histories of
   100,000 days (about 400 years each). The paper-matched settings: 0 of 40
   (`analysis/stability.py`).
6. **Long-horizon tails:** the real S&P keeps fat tails at 25 days (kurtosis 3.3);
   both models return to the bell curve faster.
7. **Fits rejected bursting bubbles:** both fits treated any 2.7× bubble as a
   failure, which favours calm settings. The finished model keeps bubbles that
   burst on their own (SPEC.md, "Runaways").
8. **The time step is no longer a problem:** results at dt = 0.01 and 0.005 agree
   within about one seed-to-seed spread on every measure (`analysis/dt_check.py`).
