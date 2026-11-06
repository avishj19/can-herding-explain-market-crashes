# Model specification (rebuild, November 2026)

What the program must do, equation by equation, with the source of every piece.
If the code and this file disagree, one of them is a bug.

## Sources, and how much to trust each

| Tag | Source | What it gives |
|---|---|---|
| **[N]** | Lux & Marchesi (1999), *Nature* 397:498 (`paper/`, local only) | The model in words, the results to reproduce (Figs 1-3). **No equations, no parameter values.** |
| **[S]** | Samanidou, Zschischang, Stauffer & **Lux** (2007), *Rep. Prog. Phys.* 70:409, sec. 9 ([arXiv physics/0701140](https://arxiv.org/abs/physics/0701140)) | All the equations, plus simulation details (time step, trend window, group floors). Co-written by Lux, so the most trusted. |
| **[C]** | Cristelli, Pietronero & Zaccaria (2011), sec. 4.5 ([arXiv 1101.1847](https://arxiv.org/abs/1101.1847)) | The same equations; used as a cross-check. |
| **[L]** | Lux & Marchesi (2000), *IJTAF* 3:675 | The parameter tables. **Paywalled and not in hand.** Values below marked [L?] are from secondary memory and are unverified. |

## State
N traders (fixed): n_f fundamentalists, n_+ optimist chartists, n_- pessimist
chartists; n_c = n_+ + n_-. Price p, fundamental value p_f. [N][S]
Derived: opinion x = (n_+ - n_-)/n_c, chartist share z = n_c/N.

## 1. Chartists change mood (optimist <-> pessimist) [S eq 17]; [C eq 16] checks the rate form only
Rate per trader per unit time (first index = group moved TO):

    pi_{+-} = v1 (n_c/N) exp(U1)          (pessimist -> optimist)
    pi_{-+} = v1 (n_c/N) exp(-U1)         (optimist -> pessimist)
    U1 = a1 x + (a2/v1) (dp/dt)/p

a1 = herding (copy the majority mood), a2 = trend following. [N] "a weighted combination of both factors".

## 2. Traders switch strategy (chartist <-> fundamentalist) [S eqs 18-21]; [C eqs 14-15] check the rate forms only
(The formulas for U1, U21 and U22 come from [S] alone: [C] calls them "a sort of utility functions" and gives no formula.)

    pi_{+f} = v2 (n_+/N) exp(U21)   (fundamentalist -> optimist)
    pi_{f+} = v2 (n_f/N) exp(-U21)  (optimist -> fundamentalist)
    pi_{-f} = v2 (n_-/N) exp(U22)   (fundamentalist -> pessimist)
    pi_{f-} = v2 (n_f/N) exp(-U22)  (pessimist -> fundamentalist)
    U21 = a3 [ (r + (dp/dt)/v2)/p - R - s |p_f - p|/p ]
    U22 = a3 [ R - (r + (dp/dt)/v2)/p - s |p_f - p|/p ]

Each rate is proportional to the size of the group being joined: recruitment,
as in Kirman's ants. r = dividend, R = return on other investments, r/p_f = R [S].
s < 1 discounts fundamentalists' future profits [N][S].

## 3. Price [S eqs 22-25][C sec 4.5.3]

    ED_c = (n_+ - n_-) t_c
    ED_f = n_f gamma (p_f - p)/p
    up-tick rate   = max(0,  beta (ED_c + ED_f + mu)) / tick
    down-tick rate = max(0, -beta (ED_c + ED_f + mu)) / tick
    so on average  (dp/dt)/p = beta (ED_c + ED_f)        [S eq 25]

Each tick changes the price by 0.1% of the current price ([S] eq 24 text:
"Delta p = +/-0.001 p"), which is one cent at the starting price p_f = 10 [S][C].
mu is white noise [C], taken here as Gaussian with sd sigma_mu (an assumption:
[L], which would give its form, is not in hand).
**Disagreements:** [C] writes ED_f = n_f gamma (p - p_f) (wrong sign, no 1/p)
and puts mu outside beta. We follow [S]. **Units:** [S] gives the tick rate as
beta (ED + mu) (eq 24) but also says the price moves on average at
(dp/dt)/p = beta ED (eq 25). Both hold only if the tick rate is divided by the
tick size, as written above. With eq 24 taken literally, parameter set I gives a
price that hardly moves (35% of days with no change at all; tested November 2026).
We follow [S] because Lux co-wrote it, and [N] says
fundamentalists react to the *relative* deviation.

## 4. News [N][S]
ln p_f(t) - ln p_f(t-1) = eps_t, eps ~ Normal(0, sigma_eps). Gaussian, with no memory.

## 5. Simulation details [S]
- Time step dt = 0.01 of a unit of time; one unit = one trading day [N: "unit time step ... daily frequencies"].
  Probability of a switch within dt = 1 - exp(-rate*dt) (exact for a constant rate).
  ([S]'s rate*dt is the same to first order; [S] also refined dt to 0.002 in bursts.
  This project keeps dt = 0.01 throughout, so dt = 0.01 is part of the model as fitted:
  the S&P-calibrated results barely change at dt = 0.002, but the paper-matched ones do
  (most of their fat tails and clustering go; `analysis/dt_check.py`, RESULTS.md
  Limitations). The exact switching form does not remove that dependence.)
- Price ticks within dt: Poisson(rate*dt) of them. This is the exact Poisson process
  [S] approximates, so there is no cap of one tick per step.
- Step limit (this project's check): the rate is held fixed for a whole step, so the
  fundamentalists can close at most beta N gamma dt of a mispricing per step
  (`model.step_pull`). At 2 or more each step overshoots the value by more than the
  last and the run blows up even without chartists, so Fig 9 does not run such
  settings. Both fitted settings are far below 1.
- mu is drawn fresh each step with standard deviation sigma_mu * sqrt(0.01/dt): exactly
  [S] at dt = 0.01, and the same noise per day at any other dt.
- dp/dt = the average price change over the last 10 steps [t - 10 dt, t] [S].
  In the code, (dp/dt)/p is (p_t - p_(t-10dt)) / (0.1 p_(t-10dt)): divided by the price
  at the start of the window, not the current one (they differ by the price's move over
  the window, usually well under 1%; [S] does not say which). At the start the window
  is filled with the starting price, so the trend is 0 for the first 0.1 day.
- Order within a step: traders switch (secs. 1-2, all from the same state), then the
  price ticks (sec. 3), then the news moves the value (sec. 4) by Normal(0, sigma_eps sqrt(dt)).
- Floors: n_c and n_f never fall below a minimum, so neither group can die out for
  good (an "absorbing state") [S]. The value, 4 traders, is this project's choice:
  [S] and [C] give no number. When a step's moves would push a group below the floor,
  just enough of the moves out of that group are undone (the larger outflow first).
- Returns are measured once per unit of time: ret_t = ln p_t - ln p_(t-1) [N Fig 1].

## Runaways (not in the sources)
With some settings the price leaves its value for good: almost all traders become
chartists and the bubble never ends (parameter set I does this after about 28,500
days). The paper says this does not happen ("no persistent deviations"), so:
- a **runaway** = the price more than e (2.7x) away from its value for 250 trading
  days in a row, or ever more than e^20 away. The run stops with RunawayError.
- A bubble that goes past 2.7x and **bursts on its own** is not a runaway; it is
  the paper's mechanism, and it is kept.
- The S&P fit is stricter: the S&P 500 never sat 2.7x from any sensible value, so
  any such day disqualifies a setting.

## Fitting (herding/calibrate.py)
Random search over wide ranges, then Nelder-Mead from the best candidates. Every
candidate is scored on the same training seeds (100-107; the paper fit's search and
first refinement use only 100-101, because its runs are long, and `--refine` uses all 8);
every check uses other seeds (1000-1039).
- Paper fit: squared gaps to alpha 2.64 +/- 0.15, H 0.48 +/- 0.03, H|r| 0.85 +/- 0.03,
  plus a penalty if 25-day kurtosis is above 1. Runs of 50,000 days.
  These +/- values are the project's chosen yardsticks, not [N]'s error bars
  (alpha +/- 0.077, H +/- 0.003, H|r| +/- 0.010, Figs 2-3 captions). Those are
  regression errors from one run, smaller than the model's run-to-run spread
  (H|r| varies by about 0.03), so no setting could meet them on every seed.
- S&P fit: squared gaps between the model's median and the S&P 500 on all 11
  fingerprint measures, each gap divided by the S&P's own uncertainty (spread
  over 200 block-bootstrap resamples with one-year blocks).
  Lesson learned: the first version divided by the model's spread instead, and the
  fit "won" by being wildly variable (1,000-day stretches of 10% daily moves).
  A fixed yardstick removes that loophole.
- Stability screen (recalibration of November 2026): a runaway can be too rare to
  show up in the fitting runs (the first S&P fit ran away in 2 of 40 histories of
  100,000 days). So each refined candidate is also run for 100,000 days on seeds
  2000-2019, and the best-scoring candidate with the fewest runaways is kept.
  The November 2026 recalibration scored 3,000 random settings (`--draws 3000`);
  the code's default is 1,200. Neither kept settings file came from that run.
- Overnight recalibration (6 November 2026, outside the repository): tail measures
  weighted 3x, differential evolution (spx and paper), and the S&P from 1928 as the
  target. None passed the judging rule fixed in advance, so both settings files are
  unchanged (figures/RESULTS.md, "Overnight recalibration").

## Policy experiments (an extension, not in the sources; herding/policy.py)
Off by default; with both off the model is exactly sections 1-5.
- **Circuit breakers** (US rules since 2013, SEC-approved NYSE Rule 7.12, formerly
  Rule 80B): when the price falls 7% or 13% below the previous close **before 3:25 pm**
  (minute 355 of a 390-minute day, step 91 of 100), trading pauses for 15 minutes; a
  7% or 13% fall later in the day does nothing. At 20% it closes for the day, at any time. Each level fires at most once
  a day. "Paused" means the price is frozen: no ticks, while traders keep
  switching groups (their moods and strategies do not stop). Levels are scaled by
  (model daily sd / S&P daily sd), so 7% means the same number of typical days.
  Comparison: each seed is run with and without breakers. The two runs use the
  same random numbers until the first halt, so each history's first halt is a
  clean paired comparison (later days differ by chance too).
- **Policy surprise**: at the start of one day the fundamental value jumps by a
  fraction (here -10%), like an unexpected rate hike. The price's fall in units of
  the surprise is 1 - (change in log price - log value since the day before) / |ln(0.9)|:
  1 = the price followed the value exactly. A placebo (the same history without the
  surprise) measures how much of that the ordinary wobble gives by itself.

## Parameters
[L?] Parameter set I (unverified, from memory of [L] Table 1):
N=500, v1=3, v2=2, beta=6, T_c = N t_c = 10, T_f = N gamma = 5, a1=0.6, a2=0.2,
a3=0.5, p_f=10, r=0.004, R=0.0004, s=0.75, sigma_mu=0.05, sigma_eps=0.005.
Because these cannot be checked, the rebuild does not rely on them. Instead:
1. **Paper-matched settings:** fitted so the model reproduces [N]'s published numbers (below).
2. **S&P-calibrated settings:** fitted to the S&P 500 (the project's question).

## Results to reproduce [N]
| Check | Paper's value | Where |
|---|---|---|
| Price tracks its value closely; returns are not Gaussian | visual | Fig 1 |
| Tail exponent (log-log regression, largest 30% of \|ret\|, unit time) | 2.64 +/- 0.08 | Fig 2 |
| Under time aggregation (tau = 5, 15, 25), returns cross over to a bell curve | visual | Fig 2 |
| DFA Hurst exponent of raw returns | 0.48 +/- 0.003 | Fig 3b |
| DFA Hurst exponent of news (input) | 0.49 +/- 0.03 | Fig 3b |
| DFA Hurst exponent of \|returns\| | 0.85 +/- 0.010 | Fig 3a |
| DFA Hurst exponent of \|news\| | 0.51 +/- 0.004 | Fig 3a |
| Volatility is high when the chartist share is high; a critical share exists | text | p. 500 |
| Bursts end on their own (fundamentalists return) | text | p. 500 |
| Results hold for an open set of parameters | text | p. 500 |

## Project checks (from the proposal)
- No chartists: returns follow the news, so they are Gaussian (excess kurtosis about 0).
- Kirman's ant model reproduces his exact solution.
- With herding: tail exponent about 3 (inverse cubic law), like the S&P 500.
- Extreme moves and volatility clustering comparable to the S&P 500.

## Differences from the paper (allowed, documented)
- Exact switching probability 1 - exp(-rate dt) and Poisson ticks (see sec. 5).
- Parameter values are fitted, not copied, since [L] is not in hand.
