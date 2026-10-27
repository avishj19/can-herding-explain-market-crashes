# Analysis scripts (exploration behind the findings)

One-off probes run while testing and calibrating the model. They are not part of
the program; each backs up a finding in the build journal. Run from the project
folder, e.g. `python3 analysis/typical.py`.

| Script | Question it answered |
|---|---|
| `probe.py` | How do kurtosis, tail exponent and autocorrelations change with a given setting? |
| `knobs.py` | Does each setting move the model in the direction the science predicts? |
| `a1.py` | Does stronger opinion herding (a1) fatten the tails? (No: it thins them.) |
| `regime.py` | How far does the price drift from its value in healthy vs runaway settings? (sets the guardrail at 1.0) |
| `audit.py` | Which bad inputs were silently accepted, and how often are probabilities clipped? |
| `fresh.py`, `lag2.py`, `lags.py` | Day-to-day and multi-day return correlations on seeds never used in tests |
| `holdout.py`, `holdout2.py` | Calibrated vs paper settings on held-out seeds |
| `typical.py` | Does the real S&P 500 look like a typical history of the model? |

`history/` keeps earlier versions for comparison: the original single-file
`main.py` and its tests, and the calibrations fitted to 8 targets
(before the reversal and 50-day memory targets were added).

Most probes were run while the code was still one file (`main.py`); they now
reach the same functions in `herding/` through a small adapter at the top.
