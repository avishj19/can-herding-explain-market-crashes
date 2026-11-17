# Results: what each figure shows

**Model version:** the rebuild of November 2026. The model's equations are in
[`docs/SPEC.md`](../docs/SPEC.md), each with its source. Figures come from
`python3 scripts/make_figures.py` (about 2 minutes on 8 cores). Every model check uses
**held-out seeds** that neither fit ever saw (mostly 1000–1039; Fig 9 and Fig 11 use
other unused seeds, named there).

## Two fitted settings

The Nature paper gives no parameter values, and the paper that does (Lux &
Marchesi 2000) is not in hand. So the settings are **fitted**, not copied
(`python3 -m herding.calibrate`):

| | Fitted to | Score | Key settings |
|---|---|---|---|
| **Paper-matched** (orange), `settings/paper_matched.json` | the paper's own numbers: α 2.64, H 0.48, H\|r\| 0.85, back to a bell curve by 25 days | training 0.06 (8 training seeds, 50,000 days); held-out 2.0 (8 held-out seeds, 100,000 days; 1.7 over 40). Squared gaps in units of the project's tolerances (α ± 0.15, H ± 0.03, H\|r\| ± 0.03), which are wider than the paper's error bars (± 0.077, 0.003, 0.010; see below); almost all of the held-out score is H\|r\| | a1 0.54, a2 1.58, a3 2.21, v1 1.91, v2 1.16; chartist push T_c = 0.48, fundamentalist pull T_f = 17.1 |
| **S&P-calibrated** (blue), `settings/sp500_calibrated.json` | the S&P 500's 11-number fingerprint, 1950–2026 | 2.94 (mean squared gap, in units of the S&P's own uncertainty) | a1 1.08, a2 0.13, a3 0.76, v1 2.02, v2 0.21; T_c = 6.07, T_f = 7.77 |

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

**Full refit with today's code (6 November 2026, afternoon).** After the stress-test fixes and
the folder reorganization, both fits were rerun from scratch with the standard method
(`python3 -m herding.calibrate`, 1,200 random settings, 8 refined, the stability screen), into
`scratch/recal_nov6/` instead of `settings/`, and judged by the same rule, fixed before the results:
- **S&P:** the search found the saved settings again (training 2.94, 2 of 20 runaways), but
  the screen chose another candidate: training 3.26, **0 of 20 runaways**. On the 40 held-out
  histories the S&P is typical on **8 of 11** measures (saved: 9), with held-out loss 4.13
  (saved: 4.15). It made 25-day kurtosis more typical (S&P at the 50th percentile; saved: 92nd)
  but lost next-day correlation (S&P above all 40 histories; saved: 92nd). One fewer typical measure, so **the saved settings stay**. It is the
  third fit with no runaways that is typical on one measure fewer: stability and the last
  typical measure keep trading off.
- **Paper:** training 0.11 (saved 0.06), held-out loss **3.04** against the saved **0.81**: it
  learned its training seeds. **The saved settings stay.**
- The saved settings' judge numbers came out exactly as before (9 of 11, 4.15; 0.81), and
  regenerating every figure from the new folders gave byte-identical PNGs.

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
[`fig01_price_follows_value.png`](fig01_price_follows_value.png). 4,000 days of one held-out history per setting, drawn as in the
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
[`fig02_fat_tails_day_vs_weeks.png`](fig02_fat_tails_day_vs_weeks.png). Drawn as in the paper's Fig 2: each horizon τ = 1, 5, 15, 25 days
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
[`fig03_memory_of_move_size.png`](fig03_memory_of_move_size.png). Detrended fluctuation analysis (Peng et al. 1994), drawn as in the
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
[`fig04_proposal_checks.png`](fig04_proposal_checks.png).
- **No chartists → bell curve:** median kurtosis **0.01** over 40 histories (`main.py`
  prints 0.04: one history, so a little noisier),
  Hill α 6.6 (thin tails). Check passed.
- **With herding → α near 3:** Hill α median **3.10** (S&P-calibrated) and **2.90**
  (paper-matched); S&P 500: 2.92. The inverse cubic law holds. Check passed.
- Kurtosis: S&P-calibrated median 5.5 vs the S&P's 25. The S&P's number is
  dominated by one day (Black Monday); kurtosis is that sensitive.

## Fig 5 — The mechanism (paper p. 500)
[`fig05_how_herding_makes_bursts.png`](fig05_how_herding_makes_bursts.png). S&P-calibrated, one 100,000-day held-out run.
- Average move size climbs steadily with the share of chartists: 0.6× the
  average when chartists are rare, 3.5× when they are 20–25% of traders.
- Crowded markets are rare: chartists are under 5% of traders on about 77% of
  days.
- The wildest stretch (a) shows a burst that **ends on its own** when the
  chartist share falls back, as the paper describes.
- The paper's efficient-market claims (`scripts/analysis/efficient_market_claims.py`, 4 × 100,000 days):

| Claim | Paper-matched | S&P-calibrated |
|---|---|---|
| Price equals value on average | mean gap +0.0000 | +0.0004 |
| No persistent mispricing | over 10% off on 0.02% of days, at most 4 days in a row | 0.47% of days, at most 29 days |
| Volatility rises with the chartist share | 0.59× (least crowded tenth) to 1.41× (most crowded) | 0.51× to 1.52× |

## Fig 6 — Kirman's ants (the herding check from the proposal)
[`fig06_kirman_ants_check.png`](fig06_kirman_ants_check.png). 40 million steps per case, δ = 0.1 (so "switch alone *or* be recruited"
adds up to at most 1 and the ant story holds literally). The simulated time at each state
matches Kirman's exact formula: total variation distance **0.007** (herding) and
**0.006** (no herding). The same recruitment rule moves traders in and out of the
chartist group in Lux–Marchesi.

## Fig 7 — Is the S&P 500 a typical history of the model?
[`fig07_is_sp500_typical.png`](fig07_is_sp500_typical.png). 40 held-out S&P-calibrated histories, each 19,329 days long.
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
[`fig08_markets_model_never_saw.png`](fig08_markets_model_never_saw.png). Percentile of each real market among 20 model
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
[`fig09_robustness_random_settings.png`](fig09_robustness_random_settings.png). 400 random settings, one 20,000-day run each. The star in a is
the paper's own point (α 2.64, H|r| 0.85).
- **6%** (26) are not run: their time step is too coarse. The fundamentalists' pull
  in one step, β·N·γ·dt (`model.step_pull`), is 2 or more, so every step would overshoot
  the value by more than the last and the run would blow up even with no chartists: a
  flaw of the simulation, not a bubble. (Until November 2026 these were counted as
  runaways; 18 of them had run away.)
- **40%** (160) run away into an endless bubble (43% of the 374 that can be trusted).
- Of the 214 that stay stable: **66%** give fat tails *and* long memory, 4% fat
  tails only, 30% thin tails.
- The paper is right that good settings form a whole region, not a single point.
  But the region is not "almost everything": stability is the main limit.

## Fig 10 — Turning the herding dial
[`fig10_herding_strength_dial.png`](fig10_herding_strength_dial.png). Only a1 (copying the majority mood) is changed.
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
[`fig11_circuit_breakers_and_policy.png`](fig11_circuit_breakers_and_policy.png). S&P-calibrated settings; code in `herding/policy.py`; the rules in
`docs/SPEC.md` ("Policy experiments").

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

## Fig 12 — Political parties (an extension, not in the paper)
[`fig12_political_parties.png`](fig12_political_parties.png). Code in `herding/politics.py`; the
numbers below come from `python3 scripts/analysis/president_party_gap.py`. The president's party is taken from inauguration
dates (12 changes of party from 1928 to 2026). The "presidential puzzle" (Santa-Clara &
Valkanov 2003; Pástor & Veronesi 2020) says US stocks return more under Democratic presidents,
with no matching difference in risk.

The 1928 file has 11 wrong closes (a 3–10% "move" undone the next day while the Dow barely
moved, e.g. 1935-04-16: S&P −10.0%, Dow +0.4%). They were found by checking every day from 1928
to 1952 against the Dow Jones average (MeasuringWorth) and are left out when the data is loaded
(`market_data.BAD_CLOSES`, November 2026). The numbers below are from after that fix.

The model has no politics in it, so it can serve as a yardstick for luck. Every gap is
measured in units of the series' own daily sd, so that markets of different volatility can be
compared. Two yardsticks:
- **the model:** 40 S&P-calibrated histories, cut at the real inauguration dates;
- **the real returns with the party calendar slid along:** whole quarters, at least a year away
  from the real calendar (386 fake calendars for 1928–, 299 for 1950–). This keeps the market's own
  calm and wild spells.

| S&P 500 | 1928–2026 | 1950–2026 |
|---|---|---|
| Return gap, Democrats − Republicans | **+6.8% a year** | +5.3% a year |
| ... percentile among shifted calendars / model | **97% / 100%** | 95% / 100% |
| Volatility D ÷ R | 0.87 (26% / 10%) | 0.86 (13% / 10%) |
| Fat tails (kurtosis) D − R | −11.2 (20% / 10%) | −28 (13% / 0%) |
| Crash days (below −4 sd) per 1,000, D − R | −2.2 (30% / 15%) | −1.6 (27% / 22%) |

- **The return gap is real but borderline.** Against the fair yardstick (shifted calendars) it
  sits at the 95th–97th percentile. With only 12 changes of party, a few eras carry much of it:
  leaving out the years 1929–1932 (the crash, under Hoover) shrinks it to 3.7% a year (leaving out
  exactly Hoover's term, March 1929 to March 1933: 3.1%), and leaving out 2008 (Bush) as well
  shrinks it to 2.4%.
- **The model is the wrong yardstick for average returns.** Its returns reverse from day to day
  (H 0.39, the miss in Fig 7), so its long-run averages wobble too little, and every real gap
  looks extreme against it. For volatility, tails and crashes the two yardsticks agree on the
  S&P, but the model's yardstick is too narrow for those as well (see "Is the luck test itself
  sound?" below), so the shifted calendar is the one to trust.
- **No party difference in crashes, volatility or fat tails.** Every one sits inside both
  yardsticks. The −28 kurtosis gap since 1950 is mostly one day: without Black Monday 1987 (under a
  Republican president) it is −8.3.
- **What this means for the project question:** the crash-making mechanism (herding) shows no
  sign of depending on the party in power. Parties may shift average returns, as Pástor &
  Veronesi argue, through investors' fear, which this model does not have. Party is correlation,
  not cause: this test cannot say what drives the gap.

## Fig 12b — Term by term: which presidencies can the model explain? (extension)

Code: `python3 scripts/analysis/president_terms.py` (no figure; merged from three agents' analyses). Fig 12 pooled the
days of each party. Here each of the 18 terms since 1928 is checked on its own: the S&P 500 and
40 politics-free model histories (rescaled to the S&P's daily sd and mean) are cut at the same
days. For seven measures we find the S&P's **percentile**, the share of histories below it
(50 = typical; 0 or 100 = none of the 40 did that). A measure outside 5–95 is a **flag**: the
model does not explain it.

**Luck check.** Letting each model history play the S&P against the other 39 gives 0.63 flags
per term, a median of 10 over 18 terms (95th percentile 20). The S&P gets **31**, reached by
only 2% of luck runs. So some misses are real.

**By party.** Democrats: 10 flags in 8 terms (1.25 per term). Republicans: 21 in 10 (2.10). If we
shuffle the party labels over the 18 terms, a gap this big turns up 37% of the time (39%
without the return measure), so it may well be luck. Hoover (5) and Bush 2001 (6) give 11 of
the 21 Republican flags; without them Republicans have 1.25 per term.

**By kind of event.** Three groups, each with both parties in it:
- *Long crises, too wild:* Hoover, Roosevelt, Bush 2001. Hoover had 37 crash days; no model
  history had more than 8 over the same days.
- *Calm eras, too quiet:* Eisenhower, Kennedy, Johnson, Nixon (volatility percentile 0–2). All
  6 "too calm" flags fall in 1946–1980; only 1 of the 19 "too wild" flags does.
- *One shock day in a calm term:* 1962 (Kennedy), 1987 (Reagan, a 20.5% fall in a day), 2025
  (Trump): kurtosis at the 95th–100th percentile. (Truman's 1946 "shock" was a wrong close: with
  it left out his kurtosis falls from 12.9 to 7.0 and his term has no flags.)

**What it means.** The misses follow the era and the kind of event, not the party. Herding
makes calm and wild spells, but they don't last long enough: the real market stayed wild for
years and calm for a decade. And real shocks can strike a calm market out of the blue, like
outside news; the model's news is a mild bell curve, so its big days come from herding.

**Limitations.** Only 40 histories, so percentiles move in steps of 2.5. The model's returns
reverse a little from day to day (H 0.39, Fig 7), so its multi-year returns spread too little
and return flags come too easily. Terms run from 291 to 3,015 days; long terms show misses more
easily. Flags sit close to the 5/95 lines: leaving out the 11 wrong closes (Fig 12) moved Clinton's
and Obama's returns from the 95th to the 98th percentile (two new flags), because every model
history is rescaled to the whole series' spread, which the wrong closes had inflated.

Table: S&P value (percentile among the 40). Falls are price falls over 1 or 20 trading days.
Crash days: below mean − 4 sd. α: Hill tail exponent, terms of 1,000+ days only (smaller =
fatter tails).

| Term | Party | Years | Days | Return/yr | Volatility | Kurtosis | Worst day | Worst 20 days | Crash days | α | Flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Coolidge | R | 1928–1929 | 291 | +32.9% (100) | 15.2% (45) | 1.7 (80) | 3.8% (68) | 10.3% (65) | 0 (39) | — | **1** |
| Hoover | R | 1929–1933 | 1,001 | −37.4% (0) | 41.2% (100) | 3.2 (75) | 12.9% (100) | 42.2% (100) | 37 (100) | 3.10 (12) | **5** |
| Roosevelt | D | 1933–1945 | 3,015 | +7.3% (60) | 24.3% (98) | 9.1 (98) | 9.1% (78) | 26.1% (98) | 31 (98) | 2.93 (15) | **4** |
| Truman | D | 1945–1953 | 1,937 | +8.0% (80) | 14.2% (10) | 7.0 (90) | 7.1% (68) | 20.0% (75) | 4 (65) | 2.64 (8) | **0** |
| Eisenhower | R | 1953–1961 | 2,016 | +10.4% (75) | 11.3% (2) | 6.5 (90) | 6.6% (57) | 9.3% (5) | 1 (30) | 3.26 (28) | **1** |
| Kennedy | D | 1961–1963 | 715 | +6.4% (48) | 12.2% (0) | 13.5 (100) | 6.7% (90) | 15.5% (85) | 1 (72) | — | **2** |
| Johnson | D | 1963–1969 | 1,271 | +7.0% (57) | 8.9% (0) | 4.4 (78) | 2.8% (2) | 9.5% (8) | 0 (18) | 3.61 (38) | **2** |
| Nixon | R | 1969–1974 | 1,401 | −4.0% (0) | 13.0% (2) | 2.2 (48) | 3.1% (2) | 13.9% (57) | 0 (19) | 3.79 (45) | **3** |
| Ford | R | 1974–1977 | 619 | +9.8% (68) | 16.3% (55) | 1.4 (65) | 3.7% (52) | 14.5% (82) | 0 (35) | — | **0** |
| Carter | D | 1977–1981 | 1,009 | +6.4% (52) | 12.6% (5) | 1.5 (48) | 3.0% (5) | 13.0% (52) | 0 (25) | 4.47 (65) | **0** |
| Reagan | R | 1981–1989 | 2,024 | +9.4% (82) | 18.0% (48) | 86.9 (100) | 20.5% (98) | 29.6% (95) | 5 (70) | 3.00 (18) | **2** |
| Bush | R | 1989–1993 | 1,011 | +10.4% (72) | 13.4% (5) | 4.3 (78) | 6.1% (72) | 13.7% (60) | 1 (46) | 3.65 (45) | **0** |
| Clinton | D | 1993–2001 | 2,021 | +14.0% (98) | 15.7% (20) | 5.3 (82) | 6.9% (57) | 13.9% (40) | 3 (46) | 3.77 (55) | **1** |
| Bush | R | 2001–2009 | 2,009 | −5.7% (0) | 21.5% (98) | 9.9 (98) | 9.0% (90) | 28.2% (100) | 14 (99) | 2.49 (2) | **6** |
| Obama | D | 2009–2017 | 2,015 | +12.2% (98) | 17.3% (57) | 4.7 (80) | 6.7% (70) | 22.1% (92) | 5 (75) | 3.34 (28) | **1** |
| Trump | R | 2017–2021 | 1,006 | +13.0% (88) | 20.6% (78) | 20.8 (95) | 12.0% (95) | 30.9% (98) | 6 (88) | 2.14 (0) | **2** |
| Biden | D | 2021–2025 | 1,005 | +11.4% (75) | 16.5% (45) | 1.9 (57) | 4.3% (40) | 12.1% (48) | 0 (20) | 4.00 (48) | **0** |
| Trump | R | 2025–2026 | 448 | +13.2% (88) | 16.3% (42) | 16.8 (100) | 6.0% (78) | 12.1% (65) | 2 (74) | — | **1** |

## Fig 13–15 — Parties at scale: 130 years, 24 markets (extension, November 2026)

Code: `python3 scripts/party_figures.py` (the three figures), `python3 scripts/analysis/party_scale.py`
(the numbers; about 3 minutes), tests in `tests/test_party_figures.py`, `tests/test_party_scale.py`
and `tests/test_politics_heavy.py`. Fig 12 used the S&P since 1928 and 40 model histories. Here
the test is run on the **Dow Jones average since 1896** (35,247 trading days, 15 changes of
party; from MeasuringWorth), the S&P, the Nasdaq-100 and the 20 stocks, against **1,000** model
histories (seeds 60000–60999; 20 ran away and were left out).

- [`fig13_parties_130_years.png`](fig13_parties_130_years.png): $1 in the Dow in 1896 became
  $1,639 by 2026 (price only: no dividends, no inflation). Invested only under Democratic
  presidents it became $106; only under Republican presidents, $15. But without Clinton the
  Democratic $1 ends at $33, and without Hoover the Republican $1 ends at $91.
- [`fig14_every_presidency.png`](fig14_every_presidency.png): every presidency since 1896 side by
  side. Best yearly return: Coolidge (R, +26%); worst: Hoover (R, −36%). Wildest: Hoover;
  calmest: Johnson (D). Both parties have rows at both ends.
- [`fig15_is_the_gap_luck.png`](fig15_is_the_gap_luck.png): the real gap against every shifted
  party calendar, and the gap in every 40-year window of the Dow: from −4.2% to +13.9% a year,
  and negative (better under Republicans) for windows ending about 1975–1995.

| Market | Years | Party changes | Gap D − R a year | 95% range (bootstrap) | Beats % of shifted calendars |
|---|---|---|---|---|---|
| Dow Jones | 1896–2026 | 15 | +3.5% | −2.7% to +10.2% | 91% |
| S&P 500 | 1928–2026 | 12 | +6.8% | −1.0% to +15.3% | 97% |
| S&P 500 | 1950–2026 | 11 | +5.3% | −1.2% to +11.7% | 95% |
| Nasdaq-100 | 1985–2026 | 6 | +11.7% | −1.3% to +26.2% | 91% |

(The 95% range resamples the history in 250-day blocks, 2,000 times. "A year" is 252 trading days;
the Dow traded about 295 days a year before 1952, so its gap per calendar year is nearer 3.8%.)

- **The gap shrinks with more history and could be zero.** Every one of the 24 ranges includes
  0. With 32 more years of data the gap halves (6.8% → 3.5%) and is no longer unusual (91%).
- **Single presidencies carry it.** Without Hoover the Dow gap is 0.8%. Without Bush 2001 the
  Nasdaq gap is 0.9%. For 9 of the 20 stocks, leaving out one presidency flips its sign.
- **Across everything, about what luck gives.** Of 96 tests (24 markets × 4 measures), 8 are
  outside 5–95% against shifted calendars; luck alone gives about 10. (The tests are not
  independent: the stocks share one calendar and move together.)

### Is the luck test itself sound? (`tests/test_politics_heavy.py`)
A test can be wrong too, so the method was tried on 1,100 made-up markets that have realistic
calm and wild spells but **no** party effect at all (two generators, described in the test file).
- **The shifted calendar is nearly fair, slightly generous.** It called a no-effect market
  "unusual" 9–12% of the time on every measure; a fair test gives 10%. Rerun with 24 other sets
  of random numbers (overnight stress test, Nov 2026), it averaged 11.7% (every measure
  11.2–12.0%): about 1 market in 8.5 instead of 1 in 10. So a percentile just outside 5–95
  against shifted calendars is a little weaker evidence than it looks.
- **It cannot see small gaps.** With only 12–15 changes of party, a true gap of 5% a year is
  missed about 60% of the time, and a true 7% gap about 35% of the time. So "not unusual" does
  not prove there is no gap; it means the data cannot tell.
- **The model is too narrow a yardstick on every measure, not only returns.** It called the
  no-effect markets "unusual" 29–50% of the time (25–48% averaged over 24 other seed sets). Its calm and wild spells are shorter than the
  made-up markets' (and the real market's, Fig 12b), so its gaps wobble too little.
- The same tests found and fixed four input-handling bugs in `politics.py` (empty or missing
  dates, unsorted dates, `min_days=0`); none affected a result.

## News, elections, and "logical" reactions (extension, November 2026)

Code: `python3 scripts/analysis/news_and_reactions.py`, tests in `tests/test_news_and_reactions.py`,
events in `data/big_days_events.csv`. **A reaction is called logical when the price moves about
as much as the news does that day.** In the model the news is known exactly (the daily change
of the fundamental value), so this can be measured (40 histories of 20,000 days).

| In the model | With herding traders | Without them |
|---|---|---|
| Share of day-to-day movement explained by that day's news | 27% | 69% |
| On the biggest 1% of days: size of the move ÷ size of the news | 9.2× | 1.3× |
| ... share of those days that had big news (top 5%) | 10% | 77% |

- **With herding, the biggest days are not about the news.** They move about 9 times as much as
  the news, and only 1 in 10 had big news. They happen when chartists are many (13% of traders,
  against 3.6% on an average day).
- Without herding the price follows the news; the missing 31% is the model's random trading
  noise (with that noise switched off, news explains 88%).

**Presidential elections as news (Dow, 33 elections, 1896–2024).** The move from the last close
before election day to the first close after:
- **Elections are bigger days than usual:** 13 of 33 are above the 90th percentile of nearby
  days; luck gives about 3.
- **Direction:** up after 13 of 18 Republican wins (average +1.1%), up after 5 of 15 Democratic
  wins (average −0.4%). With 15–18 events per group, and many ways to slice them, this is
  suggestive, not proven. It also runs opposite to the Democratic gap over whole terms above.
- Whether the White House changed party made no measurable difference.
- This cannot say whether the reactions were logical: nobody knows the true size of the news.

**The Dow's 20 biggest days and their documented causes** (`data/big_days_events.csv`, each with
its source): 10 had clear news or a policy decision that day (e.g. 15 Mar 1933, the first day
after the bank holiday; 13 Oct 2008, bank rescues), 4 had no clear news (19 Oct 1987; 28 and
29 Oct 1929), 4 were rebounds after a crash, and 2 could not be confirmed. So about half had
news, between the model with herding (10%) and without (77%). 17 of the 20 fall in four crisis
clusters (1929–33, 1987, 2008, 2020), the clustering that herding produces. The labels rest on
one or two secondary sources per day and are judgement calls.

## Fig 16–18 — Outside news: commodities, interest rates, the dollar and the Fed (extension, November 2026)

Code: `python3 scripts/outside_news_figures.py` (module `herding/factors.py`), tests in
`tests/test_factors.py`. **The question:** some news reaches every trader at the same moment,
and a fully logical trader acts on it with no herding at all: an oil shock, a jump in interest
rates, a Fed decision. If such news carried the S&P's fat tails and clustering, herding would
not be needed to explain them. Lux & Marchesi assume the news is a calm bell curve; this checks
the real outside news.

**New data** (`data/factors/`, `data/fed_decisions.csv`; all end 2026-10-30, rebuilt byte for
byte by `data/download_market_data.py`): Brent oil (FRED, 1987–; Brent because WTI closed
below zero on 2020-04-20), gold and copper futures (Yahoo, 2000–), the broad dollar (FRED,
2006–), the 10-year Treasury yield (FRED, 1962–), the Fed funds target (FRED), and every FOMC
decision since February 1994 from the Federal Reserve's own calendars: 262 scheduled
decisions and the 9 surprise moves between meetings. Check against a published number: on
scheduled FOMC days 1994–2011 the S&P rose 33.1 basis points on average, against 1.4 on other
days, which is what Lucca & Moench (2015) report (about 33 and 1).

**Fig 16 — how the S&P moves with them** (2006–2026):

| | correlation with the S&P | S&P beta | on the S&P's worst 5% of days |
|---|---|---|---|
| copper | +0.29 | +0.20 | +0.28 |
| 10-year interest rate | +0.28 | +0.06 per point | +0.18 |
| oil (Brent) | +0.20 | +0.09 | +0.21 |
| gold | +0.05 | +0.05 | +0.05 |
| US dollar | −0.29 | −1.03 | −0.28 |

- **No fixed beta describes oil or interest rates.** The one-year stock–oil correlation was
  positive in 68% of years and the stock–yield correlation in 42%; the stock–yield link was
  negative 99% of the time in 1968–1998, positive 90% of the time in 1999–2021, and mixed since
  2022 (positive 36% of the time).
- **Gold barely moves with stocks, even in crashes** (a "safe haven", Baur & Lucey 2010).
- **Outside markets explain more in wild markets:** 16% of the S&P's daily variance in calm
  markets, 33% in wild ones (thirds by the volatility of the 20 days before). This is the
  opposite of what herding alone would give (in the model, news explains less when chartists
  crowd in), but it is what Longin & Solnik (2001) found across countries: correlations rise in
  falling markets. It cannot say which way the cause runs: in a panic the stock market also
  moves oil and interest rates.

**Fig 17 — take out the outside news.** Each S&P day minus the part that oil, gold, copper
and the 10-year rate explain (on the day before, the day itself and the day after, because
futures settle before the 4 pm stock close; Dimson 1979). 2000–2026, 6,186 days; this removes
too much rather than too little, because some of what moves with those markets is the stock
market's own panic. Percentiles are among 40 held-out S&P-calibrated histories of the same length.

| | S&P | outside news removed | S&P's percentile | after removal |
|---|---|---|---|---|
| share of daily variance explained | | 16% | | |
| kurtosis | 10.8 | 8.8 | 98 | 95 |
| tail α (Hill) | 2.82 | 2.93 | 5 | 18 |
| tail α (paper's way) | 2.24 | 2.37 | 2 | 8 |
| clustering, next day | 0.30 | 0.32 | 88 | 90 |
| clustering, 50 days | 0.14 | 0.13 | 82 | 82 |
| H of \|returns\| | 0.97 | 0.95 | 95 | 92 |
| kurtosis, 25 days | 5.8 | 4.7 | 95 | 95 |

- **The fat tails and the memory survive.** Removing every move the outside markets explain
  changes the tail exponent by about 0.1 and the memory hardly at all. So outside news is not
  what makes them; this is the strongest test so far of the project's answer.
- **What outside news does add is the extreme end of the tail.** With it, the S&P's tails over
  2000–2026 are at the edge of what the model gives (Fig 7 found the same for 1950–2026: the
  paper-style α below every history). Without it, the S&P is typical of the herding model on all
  8 scale-free measures (6 of 8 before). Herding explains the fat tails; outside news makes them
  a little fatter.
- With only oil and interest rates, back to 1987, they explain only 3% of the daily variance,
  and the 25-day kurtosis barely moves (12.1 → 11.1). With all five factors (2006–2026,
  24%) it falls from 7.6 to 1.7: the 2008 crisis moved stocks, commodities, the dollar and rates
  together. This is part of Limitation 6.
- **Fed decisions:** the 9 surprise moves between meetings were 5.3 times a normal day's size
  (2001-09-17, 2008-01-22, 2020-03-16 ...), but scheduled decisions only 1.1 times. Leaving out
  all 271 Fed days changes kurtosis from 10.6 to 9.6 and the Hill α from 2.87 to 2.91. They are
  3% of days and 8% of the biggest 1% of days. News that every trader sees and expects is
  priced calmly; only surprises are big, and there are too few of them to make the tails.

**Fig 18 — the four trading moments, with error bars** (S&P 1950–2026; 95% intervals by block
bootstrap with one-year blocks, so wild spells stay together):

| | S&P (95% interval) | S&P's percentile among 40 model histories |
|---|---|---|
| mean | +3.2 basis points a day (1.7 to 4.5) | 100 (the model's value has no growth, by design) |
| sd | 0.99% a day (0.88 to 1.14) | 90 |
| skewness | −0.95 (−2.27 to −0.16) | 2: unusual |
| kurtosis | 25 (5.5 to 53) | 95 |

- **The S&P falls harder than it rises; the model does not.** Its skewness is clearly below 0
  (the interval excludes 0), while the model's traders behave the same way in both directions.
  This is a real miss of the model (Limitation 10).
- **Kurtosis is a poor yardstick on its own:** its interval runs from 5.5 to 53, because one day
  (Black Monday) dominates it. Tail exponents and the memory measures are much steadier, which
  is why the project relies on them.

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
   percentile, `scripts/analysis/bounce_back_after_big_days.py`). The old model's slow-motion slides
   (+0.31) are gone.
5. **Stability:** the S&P-calibrated settings run away in 2 of 40 held-out histories
   of 100,000 days (about 400 years each; `scripts/analysis/long_run_stability.py`) and 2 of 20 on
   the stability-screen seeds. The paper-matched settings: 0 of 40 and 0 of 20.
   (The rejected stability-first refit: 0 of 40, but it lost the fat tails.)
6. **Long-horizon tails:** the real S&P keeps fat tails at 25 days (kurtosis 3.3);
   both models return to the bell curve faster. Part of the S&P's long-horizon tail is outside
   news moving every market together: over 2006–2026 the 25-day kurtosis falls from 7.6 to 1.7
   once commodities, the dollar and interest rates are taken out (Fig 17).
7. **The S&P score can trade away fat tails** (see "Recalibration" above): a
   stability-first refit fitted better by its own score but lost the tails, so it
   was not used.
8. **The S&P fit rejected bursting bubbles:** it treats any day with the price 2.7×
   away from its value as a failure (the S&P never was), which favours calm settings.
   The paper fit only rejects bubbles that last 250 days, like the finished model,
   which keeps bubbles that burst on their own (SPEC.md, "Runaways").
9. **The paper-matched results depend on the time step** (found by a stress test,
   November 2026; `python3 scripts/analysis/time_step_check.py paper`, 12 held-out histories of
   19,329 days, medians):

   | dt | kurtosis | tail α (paper's fit) | clustering, 1 day | H of \|returns\| |
   |---|---|---|---|---|
   | 0.01 (used everywhere) | 29 | 2.67 | 0.33 | 0.87 |
   | 0.005 | 17 | 3.06 | 0.20 | 0.83 |
   | 0.002 | 3.4 | 3.33 | 0.10 | 0.75 |

   With a finer time step the same settings lose most of their fat tails and
   clustering, so the paper-matched numbers (α 2.62, H|r| 0.81) belong to the model
   *as simulated at dt = 0.01*, not to its equations alone. Lux & Marchesi also
   simulated at dt = 0.01 (refining to 0.002 only in bursts), so this is the same
   kind of model as theirs, but it is a real caveat. **The S&P-calibrated settings,
   which answer this project's question, are not affected:** every measure stays
   within about one seed-to-seed spread at dt = 0.005 and 0.002 (largest change:
   next-day correlation, −0.02 → −0.05). The paper fit was not redone at a smaller dt.
10. **No skew:** the S&P falls harder than it rises (skewness −0.95, 95% interval −2.27 to
    −0.16), while the model's buyers and sellers are mirror images, so its skewness is about 0
    (S&P at the 2nd percentile, Fig 18). A model that reproduces this would need traders who
    react differently to falls (for example, forced selling).
