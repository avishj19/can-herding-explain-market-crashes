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
| regression R² = 0.5 (`factors.regress`) | 200k, ±0.01 | 800k, ±0.0036 (sd 0.0007) |
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
  of the last digit (0.005), and the data use 99% of it, as rounding should. The Fed-day check
  (33 basis points, Lucca & Moench 2015) allows ±4: their number is an *excess* return (minus the
  Treasury-bill rate) over a slightly different window, ours a plain close-to-close return.
- **Model simulations** (`test_model.py`, `test_policy.py`, `test_news_and_reactions.py`,
  `test_party_scale.py`) were not re-audited: each of their checks needs whole simulated
  histories, so 200 seeds per check would take hours. The recorder shows they use 5–50% of
  their tolerances, so none is close to failing; auditing them is the next step if they matter.
