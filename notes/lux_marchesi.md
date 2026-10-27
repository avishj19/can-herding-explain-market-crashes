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
- Herding off (a1 = 0) but trend-following on → tails still somewhat fat
  (kurtosis 1.9 vs 3.0 with herding). So herding adds to fat tails but isn't the only cause.
