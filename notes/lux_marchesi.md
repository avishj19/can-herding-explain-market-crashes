# Lux–Marchesi model: my notes

## In my own words
This model is trying to show, with the Lux–Marchesi model, why the stock market moves
beyond what the news should make it and why the distribution has fat tails.

We look at fundamentalists and chartists and see how many of each there are, then
calculate α (tail exponent) and the kurtosis value. We then compare with the bell curve
and see the more extreme days.

## Fixes from the tutoring session (added)
- *How it works:* fundamentalists pull the price back toward its true (fundamental)
  value; chartists amplify moves by following the trend and herding (copying the
  majority). Traders switch between the two groups, so calm and wild periods alternate
  (volatility clustering).
- β is a model setting (how strongly price reacts to excess demand = buyers − sellers),
  not something we measure. We measure α and kurtosis.

## Study list (from the quiz)
- Kurtosis is a *ruler* for fat tails, not their cause. The cause is chartists.
- Smaller α means a fatter tail, so more crashes. S&P 500 α ≈ 2.9 is fatter than my
  model's α ≈ 3.3, so my model still makes too few extreme days.

## Things I found by testing
- No chartists → returns are a bell curve (kurtosis ≈ 0). This is my base-case check.
- ~~Herding off (a1 = 0) but trend-following on → tails still somewhat fat
  (kurtosis 1.9 vs 3.0 with herding). So herding adds to fat tails but isn't the only cause.~~
  *(Outdated: measured on the old, buggy model. After the fixes, stronger mood-herding
  `a1` actually makes tails thinner. See below.)*

# Session 2: how the calibrated model works

## In my own words
The calibrated model makes a crash by multiple chartists following each other, and ends
it by the fundamentalists switching their ideas.

## Fixes from the tutoring session (added)
- *Which copying:* traders **join the chartist group** because chartists recruit them
  (copying *strategy*, like Kirman's ants). Copying *mood* (`a1`) is weak in the
  calibrated model (0.2).
- *How it ends:* **chartists switch back to fundamentalists**: the price is far from
  its value, so betting on its return looks profitable. With few fundamentalists left
  during the crash, little pulls the price back until they return.
- *Calibration:* score = sum of squared gaps, each gap measured in units of normal
  random wobble. Squaring stops +3 and −3 from cancelling (score 18, not 0).
- *Judging a random model:* run many histories (20) and check the real S&P is typical:
  its percentile should be far from 0% and 100% (e.g. 19 of 20 below it = 95% = edge).

## Study list (from the quiz)
- Strategy-copying (crowding into chartists) vs mood-copying (`a1`): the crash
  mechanism is the first.
- Arithmetic with squares: square each gap first, then add (3² + 3² = 18).
