# Results: what each figure shows

**Model version:** the rebuild of November 2026. The model's equations are in
[`notes/SPEC.md`](../notes/SPEC.md), each with its source. Figures come from
`python3 make_figures.py` (about 2 minutes on 8 cores). Every model check uses
**held-out seeds** that neither fit ever saw (mostly 1000–1039; Fig 9 and Fig 11 use
other unused seeds, named there).

## Two fitted settings

The Nature paper gives no parameter values, and the paper that does (Lux &
Marchesi 2000) is not in hand. So the settings are **fitted**, not copied
(`python3 -m herding.calibrate`):

| | Fitted to | Score | Key settings |
|---|---|---|---|
| **Paper-matched** (orange), `settings/paper.json` | the paper's own numbers: α 2.64, H 0.48, H\|r\| 0.85, back to a bell curve by 25 days | training 0.06 (8 training seeds, 50,000 days); held-out 2.0 (8 held-out seeds, 100,000 days; 1.7 over 40). Squared gaps in units of the project's tolerances (α ± 0.15, H ± 0.03, H\|r\| ± 0.03), which are wider than the paper's error bars (± 0.077, 0.003, 0.010; see below); almost all of the held-out score is H\|r\| | a1 0.54, a2 1.58, a3 2.21, v1 1.91, v2 1.16; chartist push T_c = 0.48, fundamentalist pull T_f = 17.1 |
| **S&P-calibrated** (blue), `settings/spx.json` | the S&P 500's 11-number fingerprint, 1950–2026 | 2.94 (mean squared gap, in units of the S&P's own uncertainty) | a1 1.08, a2 0.13, a3 0.76, v1 2.02, v2 0.21; T_c = 6.07, T_f = 7.77 |

The news is the same in both: Gaussian, 0.5% a day, no memory.

**The paper-matched fit does not reproduce H of |returns|.** Its score of 0.06 is a
*training* score. Measured the way the figures are (held-out seeds, 100,000-day
runs), the score is about 2.0, and nearly all of it is one number: H of |returns| is
**0.81 ± 0.03** (mean ± spread between runs) against the paper's **0.85 ± 0.01**. That
gap is about 4 of the paper's error bars, so H|r| is not reproduced; α (2.62) and
H (0.47) are, within the project's tolerances. Model outputs and settings are
unchanged; this is only how the score is reported.

**The tolerances are the project's, not the paper's.** The score divides each gap by
a yardstick this project chose: α ± 0.15, H ± 0.03, H|r| ± 0.03. The paper's own
error bars are much smaller: α ± 0.077, H ± 0.003, H|r| ± 0.010 (Figs 2 and 3). Those
are regression errors from one long run, smaller than how much the numbers change
from one run of the model to the next (H|r| varies by about 0.03 between runs), so
no setting could match them on every seed. Measured against the paper's bars, H
(0.47 vs 0.48) is also several error bars off; only α is inside them.

**Recalibration (November 2026).** Both fits were rerun with a wider search
(3,000 random settings instead of 1,200) and a new **stability screen**: every
finalist is run for 100,000 days on 20 extra seeds (2000–2019), and settings that
run away lose. Then old and new settings were compared on the same 8 training
seeds and the same screen. **Both fitted settings above were kept**:
- **Paper-matched:** the new fit scored 0.55, the old 0.06 (lower is better); neither
  ever ran away. The old one stays.
- **S&P-calibrated:** the new fit scored better in training (1.51 vs 2.94) and never
  ran away (the old one: 2 of 20). **But on held-out seeds the S&P was typical on
  only 6 of 11 measures** (old: 9 of 11). The new fit had thin tails (kurtosis 4.0
  vs the S&P's 25; 25-day kurtosis 0.4 vs 3.3) and almost no herding (a1 0.18, chartists
  trading 20 times less). Its training score barely noticed, because the S&P's
  kurtosis is so uncertain (it hinges on Black Monday) that missing it costs almost
  nothing. A model of herding that does not herd answers a different question, so
  the old settings stay (the rejected fit: `scratch/recalibration/`, not in the repository).
- **Lesson:** the S&P score treats every measure by its own uncertainty, so it can
  trade away fat tails, the very thing this project is about. Weighing the tails more
  by hand was considered and not done: published fits of these models (Winker, Gilli &
  Jeleskovic 2007; Franke 2009) weight each measure by its own uncertainty, as this score
  does, and a hand-picked weight would have no basis.
- Rerunning `make_figures.py` with the kept settings reproduced figures 1–10 pixel for pixel.

**Overnight recalibration (November 2026).** Three more fits were tried, to test the
search method and the choice of data (`scratch/overnight/refits.py`, not in the repository).
Each ended with the same stability screen (20 runs of 100,000 days, seeds 2000–2019).
The judging rule was fixed **before** any result was seen: on 40 held-out histories
(seeds 1000–1039) against the S&P 1950– fingerprint, a new fit wins only if the S&P
is typical on more of the 11 measures (or on as many, with a lower held-out loss),
**and** it runs away in no more of the 20 long runs.

| Fit | What was different | Typical (of 11) | Held-out loss | Runaways (of 20) |
|---|---|---|---|---|
| **Current S&P settings (kept)** | | **9** | 4.15 | 2 |
| Differential evolution | a population search instead of random search + Nelder-Mead | 8 | 2.78 | 0 |
| S&P 1928– | fitted to the S&P from 1928 (1929 and the 1930s included) | 8 | 11.10 | 0 |

None ran away in the 40 held-out histories. The paper fit was also redone by
differential evolution: training score 0.012 (saved: 0.06), but on 4 held-out seeds of
50,000 days 1.39 (saved: 0.81), so it learned its training seeds rather than the paper.

What they teach:
- **The search was not badly stuck.** Differential evolution found settings of the same
  kind (strong herding, a1 1.18) and only a little better in training (2.72 vs 2.94).
  Its best training score (2.47) ran away in 7 of 20 long runs, and the screen threw it
  out. It has the lowest held-out loss and no runaways, but the S&P's kurtosis is above
  all 40 of its histories, so it is typical on one measure fewer.
- **The 1930s are a different market:** fitted to them, the model's daily sd is too
  large for 1950–2026 (the S&P is below every history), hence the large loss.
- **H of returns is unusual (100th percentile) in every fit:** the model's returns
  reverse from day to day more than the S&P's, whatever the settings.

By the rule fixed in advance, **the current settings were kept** (`settings/` is unchanged).

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
`fig1_snapshot.png`. 4,000 days of one held-out history per setting, drawn as in the
paper's Fig 1: price solid, value dotted and **shifted down** so both lines show (by 20
and 35 points; the legend says how much), and the output (b) and input (c) on the same
y scale.
- The price (a) tracks its fundamental value closely, as in the paper.
- The news (c) has kurtosis 0.0: a plain bell curve. The returns (b) have bursts
  (kurtosis 2.9 in this stretch), and the bursts line up with a high share of
  chartists (d).
- The S&P-calibrated model keeps few chartists most of the time (d, right), so
  this stretch is calm (returns kurtosis 0.9, news −0.1); its bursts are rarer and
  are shown in Fig 5.
- **New third column: the real S&P 500**, its last 4,000 trading days (price and daily
  returns, on its own y scale). Kurtosis **14.5**: wilder than either model's stretch.
  A real market's news and chartist share cannot be measured, so c and d are model-only.

## Fig 2 — Fat tails over a day, a bell curve over weeks (paper's Fig 2)
`fig2_tails.png`. Drawn as in the paper's Fig 2: each horizon τ = 1, 5, 15, 25 days
is a set of markers in the paper's symbols (▲ + ◆ ▼), and the bell curve is the solid
line. New in every panel: the paper's tail fit itself (dotted, a straight line through
the largest 30% of daily moves, labelled with its α) and a dashed line with the paper's
slope (α = 2.64) from the same starting point, so only the slopes are compared. The
fitted α: 2.84 (a, the run shown), 2.63 (b, all 40 histories pooled; median of the
histories 2.64), 2.35 (c, S&P).
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
`fig3_memory.png`. Detrended fluctuation analysis (Peng et al. 1994), drawn as in the
paper's Fig 3: the output as triangles with a solid fitted line, the news as plus signs
with a dotted (a) or dashed (b) fitted line. New: a grey dashed line with the paper's
slope (H 0.85 in a, 0.48 in b, 0.85 on the S&P's |returns| in c) and a dotted "no
memory" line (H 0.5) on the S&P's returns in c, each starting at the first point of the
curve it is compared with.

| | Paper | Paper-matched (mean of 8 held-out runs) | S&P 500 | S&P-calibrated (median of the 40 Fig 7 histories) |
|---|---|---|---|---|
| H of returns | 0.48 | 0.47 | 0.50 | **0.39** |
| H of \|returns\| | 0.85 | 0.81 | 0.90 | 0.87 |
| H of news / \|news\| | 0.49 / 0.51 | 0.50 / 0.49 | — | — |

The figure itself shows **one run of each**, so its printed values differ a little
from these averages: paper-matched H 0.45, H|r| 0.78, news 0.47 (seed 1000);
S&P-calibrated H 0.38, H|r| 0.92 (one history). One run varies; the table is the
better summary.

The long memory in move sizes is created by the traders: the news has none.
The paper-matched H of |returns| (0.81 ± 0.03 between runs) is below the paper's 0.85 ± 0.01: not reproduced, though still far above the news's 0.49.
The S&P-calibrated model's returns **reverse** a little from day to day
(H 0.39), while the S&P's do not (0.50). This is its clearest miss (Fig 7).

## Fig 4 — The proposal's checks
`fig4_checks.png`.
- **No chartists → bell curve:** median kurtosis **0.01** over 40 histories (`main.py`
  prints 0.04: one history, so a little noisier),
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
`fig6_kirman.png`. 40 million steps per case, δ = 0.1 (so "switch alone *or* be recruited"
adds up to at most 1 and the ant story holds literally). The simulated time at each state
matches Kirman's exact formula: total variation distance **0.007** (herding) and
**0.006** (no herding). The same recruitment rule moves traders in and out of the
chartist group in Lux–Marchesi.

## Fig 7 — Is the S&P 500 a typical history of the model?
`fig7_typical.png`. 40 held-out S&P-calibrated histories, each 19,329 days long.
- **9 of 11 measures typical.** Daily sd 90%, kurtosis 95%, Hill α 32%,
  next-day correlation 92%, clustering 82% (1 day) and 60% (50 days), H|r| 78%,
  5-day kurtosis 57%, 25-day kurtosis 92%.
- **Unusual:** H of returns (100%: model returns reverse more) and the
  paper-style α (0%: the S&P's 2.35 is below all 40 histories, median 2.64; its tail
  is fatter in the middle-to-large range).
- "9 of 11" is fragile: kurtosis sits exactly on the 95% edge (38 of 40 histories
  below), and sd, next-day correlation and 25-day kurtosis are at 90–92%. With 40
  histories one more or fewer moves a measure across the line.
- One of the 40 histories has a freak event (kurtosis about 5,000): a bubble that
  burst.

## Fig 8 — Markets the model never saw
`fig8_out_of_sample.png`. Percentile of each real market among 20 model
histories of the same length: the two indexes against seeds 1000–1019, the 20
stocks against their own seeds 1020–1039 (until November 2026 a length check was
off by one day and the stocks reused the index seeds; fixing it changed 49 unusual
cells to 43). **43 of 220 cells are unusual**: 6 of 20 for the indexes, 37 of 200 for
the stocks.
- **Stocks:** kurtosis is high (percentiles 70–100, median 95; 7 of 20 unusual), as
  expected: companies have their own jumps (earnings, lawsuits) on top of the
  market's. But their Hill tail exponent is typical (5–50, none unusual). The
  clearest miss is H of returns: 11 of 20 stocks sit at 100%, because the model's
  returns reverse from day to day and stocks' do not.
- **S&P since 1928:** unusual on memory (H and H|r| both 100%) and the paper-style α
  (0%), with clustering and 25-day kurtosis at the 95% edge: the 1930s were far
  wilder and longer-lasting than anything since.
- **Nasdaq-100 since 1985:** unusual on 3 of 10: 50-day clustering, H and 25-day
  kurtosis (all 100%).

## Fig 9 — How robust is it? (paper: "an open set of parameter values")
`fig9_robustness.png`. 400 random settings, one 20,000-day run each. The star in a is
the paper's own point (α 2.64, H|r| 0.85).
- **44%** run away into an endless bubble.
- Of the 222 that stay stable: **65%** give fat tails *and* long memory, 4% fat
  tails only, 32% thin tails.
- The paper is right that good settings form a whole region, not a single point.
  But the region is not "almost everything": stability is the main limit.

## Fig 10 — Turning the herding dial
`fig10_herding.png`. Only a1 (copying the majority mood) is changed.
- **Paper-matched:** kurtosis peaks at a1 = 0.6 (fitted 0.54) and **collapses to a
  bell curve for a1 ≥ 1**. More herding does *not* simply mean more crashes. Two
  things happen together above 1 (one history, 20,000 days, a1 0.6 → 1.2): the
  chartists lock into one mood (mood flips fall from 0.22 to 0.05 a day), *and* there
  are far fewer chartists (23% → 8% of traders). Either could thin the tails; this
  figure cannot tell which. Linking the peak to the paper's "criticality" is a
  hypothesis: the paper's critical point is a share of chartists, not a herding strength.
- **S&P-calibrated:** there is a **lower limit**, not a band: below a1 ≈ 0.8 most
  histories run away (12 of 12 at a1 ≤ 0.2, 11 of 12 at 0.4–0.6, 7 of 12 at 0.8); from
  a1 = 1.2 all survive, and the tails thin out as a1 grows (kurtosis 3.8 at 1.2, 1.8
  at 1.4). Points with fewer than 5 surviving histories are not drawn (they were
  medians of a single history before November 2026).

---

## Fig 11 — Policy experiments (an extension, not in the paper)
`fig11_policy.png`. S&P-calibrated settings; code in `herding/policy.py`; the rules in
`notes/SPEC.md` ("Policy experiments").

**Circuit breakers.** The US rules since 2013 (NYSE Rule 7.12): trading pauses for
15 minutes when the market is 7% or 13% below the previous close, **but only before
3:25 pm**; at 20% it closes for the day at any time. Scaled ×0.91 to the model's
volatility. Each of 200 histories is run twice, with and without breakers; the two
runs are identical until the first pause, so that day is a fair comparison (180
histories had one; none ran away).
- **No detectable effect.** Change from the previous close, with minus without
  breakers: **−0.16** percentage points on the halt day (95% interval −0.42 to +0.08),
  **−0.4** after 5 days, counting the halt day (−1.3 to +0.5), **−0.7** after 20 days
  (−2.0 to +0.7). All three intervals include zero; they rule out effects larger
  than about 2 percentage points, but cannot show the effect is exactly zero.
- On the halt day breakers did better in 101 histories and worse in 79, with no
  measurable difference on average.
- **A modeling choice that leans toward "no effect":** while trading is paused, the
  buy and sell orders of that time are dropped, not queued up for the reopening.
- Pauses fire on about 4 days per 77-year history; a full-day close almost never.
- With the 3:25 pm cutoff left out (the first version of this experiment), the
  results were the same within the intervals (−0.1, +0.4, −0.5).

**A policy surprise** (like an unexpected rate hike): the fundamental value drops
10% at once, on a random day, in 400 histories. Measured: the price's largest fall
within 10 days, in units of the surprise (1 = the price fell exactly as much as the
value), next to a placebo (the same history without the surprise: ordinary wobble).

| Chartists the day before | Histories | Largest fall (median) | Placebo (median) | After 10 days |
|---|---|---|---|---|
| under 5% | 311 | 1.10 | 1.06 | 1.03 |
| 5–10% | 70 | 1.27 | 1.15 | 1.08 |
| 10–20% | 18 | 1.48 | 1.22 | 1.30 |

(One more history had over 20% chartists; one history is too few for a row.)

- **The more crowded the market, the more a surprise overshoots.** In a calm market
  a 10% surprise moves the price about 10% (a 4% overshoot beyond the wobble). With
  10–20% chartists the overshoot beyond the wobble is about 26%. These gaps are the
  difference of the two medians in the table, a shortcut: a history's largest fall and
  its placebo's are not taken on the same day. Pairing each history with its own placebo
  and taking the median of the differences gives 0.02 (95% interval 0.01 to 0.03) under 5%,
  0.11 (95% interval 0.06 to 0.17) at 5–10%, and 0.29 (95% interval 0.14 to 0.41) at
  10–20% chartists (bootstrap over histories, `policy.median_overshoot`). The crowded row
  rests on only 18 histories, so "about a quarter" is a rough size, not a precise one;
  what holds firmly is the direction (more crowding, more overshoot). After 10 days the
  price has fallen about 1.3 times the surprise (about 3 percentage points more than the
  value), in 18 histories. There is no placebo for the 10-day number, so part of that
  extra fall may be ordinary wobble (the placebo for the largest fall is 1.22 in this row).
- **In this model the same surprise has a bigger effect when it lands in a crowded
  market.** Only one size (a 10% drop) was tested, so this does not say how timing
  compares with the size of the surprise.

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
5. **Stability:** the S&P-calibrated settings run away in 2 of 40 held-out histories
   of 100,000 days (about 400 years each; `analysis/stability.py`) and 2 of 20 on
   the stability-screen seeds. The paper-matched settings: 0 of 40 and 0 of 20.
   (The rejected stability-first refit: 0 of 40, but it lost the fat tails.)
6. **Long-horizon tails:** the real S&P keeps fat tails at 25 days (kurtosis 3.3);
   both models return to the bell curve faster.
7. **The S&P score can trade away fat tails** (see "Recalibration" above): a
   stability-first refit fitted better by its own score but lost the tails, so it
   was not used.
8. **The S&P fit rejected bursting bubbles:** it treats any day with the price 2.7×
   away from its value as a failure (the S&P never was), which favours calm settings.
   The paper fit only rejects bubbles that last 250 days, like the finished model,
   which keeps bubbles that burst on their own (SPEC.md, "Runaways").
9. **The time step is no longer a problem:** results at dt = 0.01 and 0.005 agree
   within about one seed-to-seed spread on every measure (`analysis/dt_check.py`).
