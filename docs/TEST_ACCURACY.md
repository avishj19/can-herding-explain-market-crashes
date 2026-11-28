# How accurate is each test? (November 2026)

A test passes when the code's answer is close enough to a known answer. "Close enough" is
the test's **tolerance**. Too loose, and a real bug slips through; too tight, and a correct
program fails by bad luck. This page records how every tolerance was chosen and how much
error is left.

Tools (rerun any time):
- `PYTHONPATH=tests python3 -m pytest tests/ -q -p tolerance_recorder`: records all
  34,399 `pytest.approx` comparisons in the suite (expected, actual, allowed) in
  `scratch/tolerances.csv`.
- `python3 scripts/analysis/test_accuracy_audit.py`: runs each noisy check on 200 seeds
  (1000–1199, never the test seeds 0–4) and prints its bias, sd and tolerance.
- **mpmath** (50-digit arithmetic) and Python's **fractions** (exact arithmetic) give
  reference answers with no rounding error at all.

## Three kinds of test, three kinds of error

| Kind | Tests | Where the error comes from | What reduces it |
|---|---|---|---|
| Exact math (a formula, worked by hand) | 90 | rounding in 64-bit numbers (about 16 digits) | nothing more: already at the limit |
| Noisy estimates (random samples) | about 35 | luck of the draw: sd shrinks like 1/√(sample size) | more samples, not more digits |
| Published facts (a price, a date) | about 15 | how many digits the source printed | nothing: can't be more exact than the source |

**More bits only help the first kind, and only for the reference answer.** The project's code
runs in 64-bit numbers; on this Mac (Apple chip) NumPy's "long double" is also 64-bit, so a
longer type does not exist in hardware. What extra digits *can* do is make the expected answer
exact, so the code's true error can be measured:

| Check | Reference | Code's true error |
|---|---|---|
| Kirman's exact distribution, N up to 500 | exact fractions (`fractions.Fraction`) | at most 6e-15 relative |
| the bell curve's tail line (Figs 2, 4) | 50-digit mpmath `erfc` | at most 4e-15 relative |
| P(\|Z\| > 2) in the ccdf test | 50-digit mpmath: 0.0455002638963584… | the old test used the rounded 0.0455 |

## 1. Exact math: 12 digits, not 6

pytest's default tolerance is 1 part in a million (6 digits). The recorder showed that every
exact test's real error is at most 3.4e-11 (sums over 30,000 days) and usually 1e-15. So a bug
that made an answer wrong in the 7th–12th digit would have passed. `tests/conftest.py` now sets
the project default to **12 digits** (relative tolerance 1e-12); all 90 exact tests pass at
that bar. Two new tests check against the exact references above at 13–14 digits.

## 2. Noisy estimates: tolerance = |bias| + 5 sd

For a random estimate, the right question is not "how close was the one test seed?" but "how
much does the answer wander from seed to seed (sd), and is it off on average (bias)?" A
tolerance of |bias| + 5 sd fails a correct program by bad luck less than once in a million
times, and is not tuned to the lucky test seed. Every recipe is in `tests/helpers.py`
(`RECIPES`), and the tests and the audit use the same recipes; the tolerances are in
`TOLERANCE`, written by the audit (`--write`).

| Check | Before: sample, tolerance (in sd) | After: sample, tolerance (in sd) |
|---|---|---|
| kurtosis of a bell curve = 0 | 200k, ±0.05 (4.8 sd) | 800k, ±0.029 (sd 0.0058) |
| kurtosis of uniform = −1.2 | 200k, ±0.03 (14 sd: far too loose) | 800k, ±0.0068 (sd 0.0013) |
| kurtosis of Laplace = 3 | 500k, ±0.2 | 2M, ±0.13 (sd 0.024) |
| skewness of exponential = 2 | 400k, ±0.05 | 1.6M, ±0.034 (sd 0.0067) |
| skewness of a bell curve = 0 | 400k, ±0.02 | 1.6M, ±0.011 (sd 0.0020) |
| Hill α of a Pareto law | 100k, ±8% | 400k, ±3.6% (sd 0.7%) |
| Hill α of Student-t3 = 3 | 200k, top 1%: ±0.36; bias −0.12 | 2M, top 0.25%: ±0.27; bias −0.049, sd 0.042 |
| share of bell-curve days beyond 2 sd | 200k, ±0.003 around a rounded value | 800k, ±0.00085 around the exact value |
| autocorrelation of AR(1), lags 1–3 | 200k, ±0.01 | 800k, ±0.0033 to ±0.0073 |
| autocorrelation of noise = 0 | 100k, ±0.015 | 400k, ±0.0087 (sd 0.0017) |
| H of white noise = 0.5 | 100k, ±0.04 | 400k, ±0.021 (bias +0.0014, sd 0.0037) |
| H of a random walk = 1.5 | 100k, ±0.06 (3.7 sd) | 400k, ±0.037 (sd 0.0073) |
| lag-50 memory of a regime process | 400k, ±0.01 (3.8 sd) | 1.6M, ±0.007 (sd 0.0014) |
| bell-curve fingerprint: H | one series, ±0.03 (**2.2 sd: fragile**) | average of 16 series, ±0.019 (sd 0.0034) |
| bell-curve fingerprint: kurtosis over 25 days | one series, < 0.2 (3.6 sd) | average of 16, ±0.076 (sd 0.015) |
| paper-style α of an exact power law | ±0.15 | ±0.003 (bias −0.002, sd 0.0001) |
| the tail-fit line at the window's start | ±5% | ±0.2% |
| bootstrap interval width = 1.96/√n | 2,000 resamples, ±10% | 8,000 resamples, ±6.5% (sd 1.3%) |

What the audit found besides:
- **Three tests were fragile** (under 4 sd): the fingerprint's H (2.2 sd: a different seed would
  fail about 3% of the time), the random walk's H (3.7 sd) and the lag-50 memory (3.8 sd).
- **Two estimators have a real bias**, not just noise: Hill's α on Student-t3 (−0.12 with the top
  1%; a known property, Hall 1982) and DFA's H on white noise (+0.0014, more than 5 times its own
  uncertainty over 200 seeds). Their tolerances include the bias.
- **H from `fingerprint` hardly improves with longer series** (sd 0.013 at 200,000 days, 0.010 at
  800,000): its longest windows are always a quarter of the data, so there are only a few of
  them. Averaging independent series is what shrinks it.
- The AR(1) series were built with a Python loop (slow); they now use `scipy.signal.lfilter`,
  the same recursion in C, checked against the loop to 12 digits.

## 3. What was not tightened, and why

- **Published facts** are limited by the source: the 20 big-day percentages in
  `data/big_days_events.csv` are typed to 2 decimals, so their tolerance is exactly half a unit
  of the last digit (0.005), and the data use 99% of it, as rounding should.
- **Model simulations** (`test_model.py`, `test_policy.py`, `test_news_and_reactions.py`) were not re-audited: each of their checks needs whole simulated
  histories, so 200 seeds per check would take hours. The recorder shows they use 5–50% of
  their tolerances, so none is close to failing; auditing them is the next step if they matter.

## 4. Testing the tests: the overnight stress test (16–17 November 2026)

Two questions the sections above cannot answer: does a test pass only because of the
particular random numbers it was written with, and would the tests notice a real bug?
`tests/stress/overnight.py` answers both. It runs on 3–5 cores at once (memory, 8 GB, is the
limit, not the 8 cores), each run in its own copy of the project, and can be stopped and resumed.

**A. Seed sweep: the whole suite with other random numbers.** Every random number in the
project comes from `np.random.default_rng(seed)`. `tests/stress/sitecustomize.py` turns each
seed s into a different (still repeatable) one for "shift" k, without editing any test.
The suite (slow tests included) ran 41 times: twice as written (all 887 passed; the 34,399
recorded numbers agreed bit for bit) and 39 times shifted.

| What failed with other seeds | Tests | What was done |
|---|---|---|
| Describes one documented history (the golden output; set I's seed 1 running away on day 28,511; a bubble that bursts at seed 1038; the held-out score quoted in RESULTS.md; Fig 8 on the project's held-out seeds) | 7 | marked `pinned_seed` (conftest.py): skipped in shifted runs |
| Needed a runaway, but set I runs away only on some seeds (26 of 39 lasted 30,000 days) | 4 | use strong trend-following (a2 = 6), which ran away on 300 of 300 seeds within 2,505 days |
| A test of something else stopped by a runaway (set I with tight breakers ran away within 2,000 days on 26 of 300 seeds) | 7 | breaker tests use a market without herding pull, which ran away on 0 of 300 while every breaker level still fired; the news test switches the traders off |
| Tolerance or bound too tight for the luck in it | 5 | measured over 4,000–40,000 seeds and reset (below) |
| Not robust in design | 2 | the smallest-floor test assumed the chartists empty; on 26 of 39 seeds the fundamentalists do. The planted-gap test now checks an exact identity (+0.3% a day adds exactly 75.6% a year) |

Tolerances reset from measurements:
- **Policy surprise** (15 of 39 failed): after 10 days the price had not always settled on its
  new value (up to 7 ticks away over 4,000 seeds). After 30 days it is within 1.73 ticks on all
  4,000, as it must be (the price rests on whole ticks, so before and after it sits up to about
  a tick from its value): the test now waits 30 days and allows 2 ticks, and 4 ticks of
  overshoot (largest seen 2.7).
- **"Strategy switching is a fair game"** (1 of 39): not chance. Over 40,000 seeds the chartist
  share rose from 0.1 to 0.1048 in 3 days (20 standard errors). Each trader moves at most once
  per step and chartists also spend moves on changing mood, so they keep less of their outflow
  than fundamentalists: drift about 0.0017 a day by formula, 0.00161 ± 0.00013 measured,
  halving with the time step. The test now checks the formula. This is a property of the
  dt = 0.01 model, written into SPEC.md sec. 5 and KEY_POINTS limitation 10; the model was not changed.
- **S&P bootstrap tolerance** for α: centre 0.099 (the average over 39 seeds), not one seed's 0.09.
- **Model yardstick and false-positive rates** (politics_heavy, removed with the party study on 28 Nov 2026;
  failed on 10 and 1 of 39): rerun on 24
  seed sets. One measure alone can dip to 15% (the old bound was 22%), so the bound is now on
  the average of the 4 measures (35% and 32%, sd 2.4 and 2.8; must be 18–50%). The shifted
  calendar's false-positive rate averaged 11.7%, not 10%: slightly generous, now stated in
  RESULTS.md; its band is 11.7% ± 5 chance sds (3.5–20%).
- With fewer than 4 sd of room over the 39 seeds, more data instead of a looser bound where
  possible: the news tests use 200,000 days (was 50,000), the spring test 80,000 (was 20,000),
  the mood-flip test 40,000, the group z-score 16,000 rows; R² = 0.5 now allowed 5 sd by formula
  (0.004). (The group z-score and R² tests were removed with the election and outside-news parts on
  28 Nov 2026.)

**B. Mutation testing: would the tests catch a bug?** One small deliberate bug at a time
("+" to "-", "<" to "<=", a number plus 1, "and" to "or", a minus sign dropped) was put into a
copy of the code, and the tests that use that file were run. 2,638 bugs in 19 files (all of
`herding/` and the scripts), in 2 h 20 min with 5 runs at a time (the seed sweep before it took 1 h 20 min, 3 at a time). The unmutated code, rewritten the same way,
passed in every file (the control).

| File | Bugs planted | Caught overnight | Caught now |
|---|---|---|---|
| `calibrate.py` | 132 | 79 (60%) | 92 (70%) |
| `checks.py` | 5 | 5 (100%) | 5 (100%) |
| `experiments.py` | 45 | 36 (80%) | 39 (87%) |
| `factors.py` | 84 | 74 (88%) | 74 (88%) |
| `kirman.py` | 90 | 74 (82%) | 85 (94%) |
| `market_data.py` | 39 | 30 (77%) | 32 (82%) |
| `model.py` | 400 | 322 (80%) | 358 (90%) |
| `plots.py` | 78 | 19 (24%) | 19 (24%) |
| `policy.py` | 89 | 62 (70%) | 73 (82%) |
| `politics.py` | 65 | 64 (98%) | 64 (98%) |
| `stats.py` | 135 | 99 (73%) | 115 (85%) |
| **all of `herding/`** | **1162** | **864 (74%)** | **956 (82%)** |
| the scripts (figures, analysis) | 1476 | 398 (27%) | not rerun |

`factors.py` and `politics.py` (and their tests) were removed on 28 Nov 2026 with the outside-news and
president's-party studies; their rows stay as the record of that run.

"Caught now" reran every bug that survived in `herding/` against the new tests, plus a random
fifth of the caught ones, to check the edited tests still catch them: one did not (a "+" turned
to "-" in the warm-up days, after the runaway test got shorter), and a test now covers it. The
remaining survivors in `herding/` are listed below by kind. `plots.py` is drawing code (colours,
line widths), checked by looking at the figures.

What the surviving bugs showed, and what was done:
- **Tested now:** the runaway rule's numbers (e^1 for 250 days, e^20) and its day counter
  (checked against the rule applied by hand) and the emergency stop for an absurd number of ticks in one step, which no test had reached; every number in `fingerprint()` against the
  function it names (a wrong lag or horizon went unnoticed before); the bootstrap percentiles
  of `moments()` against the same draws by hand (the 6.5% tolerance on the interval's width
  could not see 3.5 instead of 2.5); `typical()` by hand; Kirman's step count, starting count
  and the "chances add to at most 1" limit at its edge; the fit's documented targets and seeds;
  `paper_loss` by hand; the stability screen with several seeds; limits exactly at their edge
  (N = 4 floor, the smallest dt and tick, 1 day, 1 minute); zero prices and repeated dates.
- **A real bug:** `RunawayError` could not be sent back from a worker process (it was rebuilt
  without its `day`), and a process pool then hung for good. Fixed in `model.py`, with a test.
- **Cannot matter** (equivalent): "<" vs "<=" where equality never happens, the order of
  a circular buffer, a constant shift that a straight-line fit removes, settings that are
  never used with breakers off.
- **Cannot be seen in daily data:** a pause one step shorter, or a breaker level firing twice
  in a day after it already fired (the output records each day's highest level only).
- **Left as they are:** most survivors in the figure scripts are drawing details (colours,
  sizes, label positions) or each script's fixed choices (which seeds, how many days). The
  tests run every script at a small size to check it works, not the exact numbers it prints;
  those numbers are checked by hand in RESULTS.md.

**Also measured overnight:** the S&P-calibrated settings ran away in 11 of 400 histories of
100,000 days on new seeds (2.75%; the stability screen saw 0 of 20, which has a 57% chance
at that rate).
