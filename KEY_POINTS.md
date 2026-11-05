# Key points: Can herding explain market crashes?

A self-contained guide to this project for someone who has never seen it and
doesn't know finance or agent-based modeling. It explains the ideas behind the
code, what each figure shows and how to read it, and what the model gets right
and wrong. Every number comes from code in this repository (section 10).
Figure-by-figure details: [`figures/RESULTS.md`](figures/RESULTS.md). The
model's equations, each with its source: [`notes/SPEC.md`](notes/SPEC.md).

---

## 1. The question in one paragraph

Stock markets have two famous habits. First, **huge days happen far more often
than a bell curve predicts**. On Black Monday (19 October 1987) the S&P 500 fell
20.5% in one day: a 23-standard-deviation move, which a bell curve says should
essentially never happen. Second, **wild days come in bunches**: a violent week is
usually followed by more violent weeks. A popular view in economics, the *efficient
market hypothesis*, says prices simply react to news, so big swings must come from
big news. **Lux & Marchesi (1999, *Nature*)** argue the opposite: even when the news
is perfectly calm and bell-shaped, traders copying each other can create both
habits on their own. This project rebuilds their model from its equations, checks
it against the numbers in their paper, fits it to 77 years of the S&P 500, and asks
whether **herding alone** produces the real market's fat tails and clustering.

## 2. Words you need

| Word | Meaning |
|---|---|
| **Return** | How much the price changed in a day, as a log change (about the percentage change). |
| **Standard deviation (sd)** | The typical size of a daily move. A "10-sd day" is ten times a typical day. |
| **Bell curve (Gaussian)** | The usual shape of random noise: moves beyond 4–5 sd are almost impossible. |
| **Fat tail** | Big moves are far more common than the bell curve allows. |
| **Tail exponent α** | How fast big moves become rare. Smaller α = fatter tail. Real markets: α ≈ 3 (the "inverse cubic law", Gopikrishnan et al. 1998). |
| **Kurtosis** | One number for tail fatness: 0 for a bell curve, larger = more extreme days. |
| **Volatility clustering / memory** | Big moves come in bunches: today's move size predicts move sizes weeks later. |
| **H (Hurst exponent)** | A memory number from *detrended fluctuation analysis* (DFA). 0.5 = no memory; above 0.5 = spells that persist; below 0.5 = moves that reverse. |
| **Fundamental value** | What the asset is "really worth". In the model only news moves it. |
| **Fundamentalist** | Buys below the value, sells above it: pulls the price back toward value. |
| **Chartist (noise trader)** | Ignores value; follows the trend and the crowd. Either an **optimist** (buys) or a **pessimist** (sells). |
| **Herding** | Traders copying other traders. |
| **Agent-based model** | A simulation of many individual "agents" (here 500 traders) with simple rules; the market's behaviour emerges from their interactions. |
| **Fitting (calibration)** | Searching for the settings whose output best matches target numbers. |
| **Seed, held-out seed** | The model is random; a seed fixes its random numbers. Fits use some seeds, checks use **other** seeds, so the model is never graded on its own homework. |
| **Percentile** | Where a real value falls among many simulated histories: 50% = typical; below 5% or above 95% = unusual. |
| **Runaway** | A bubble that never ends: the price leaves its value for good. |

## 3. The model in plain words

Imagine 500 traders. Each trading day is cut into 100 small steps. In each step:

1. **News** nudges the fundamental value by a small bell-curve amount. This is the
   only outside input; it has no fat tails and no memory.
2. **Chartists change mood** (optimist ↔ pessimist). They are pushed by the
   **majority mood** (herding, setting `a1`) and by the **price trend** (`a2`).
3. **Traders change strategy** (fundamentalist ↔ chartist) by comparing profits.
   Chartists profit from trends; fundamentalists from the gap between price and value,
   discounted (factor `s` = 0.75) because that profit only arrives later.
   Joining a group is **recruitment**: the bigger a group, the faster it recruits
   (Kirman's ant model, Fig 6).
4. **The price moves in small ticks** (0.1% each). More buyers than sellers (excess
   demand) makes up-ticks more likely.

The equations, and which source each comes from, are in
[`notes/SPEC.md`](notes/SPEC.md); the code is [`herding/model.py`](herding/model.py).

**Two fitted settings are compared throughout** (the paper's own parameter values
are in a paper we could not get, so they are fitted):
- **Paper-matched** (orange): fitted to reproduce the Nature paper's published
  numbers.
- **S&P-calibrated** (blue): fitted to the S&P 500's fingerprint, 1950–2026.

## 4. How the model is checked

- **Every function is tested** against answers worked out independently. Examples:
  - the price–value gap behaves like a spring whose stiffness follows from the
    price equation;
  - chartists' moods flip at the rate the two-state formula predicts;
  - Kirman's exact formula;
  - a Pareto law with known α;
  - white noise has H = 0.5.

  All 293 tests pass; run them with `python3 -m pytest tests/ -q` (about 3 minutes).
- **Every result is judged over many histories** on held-out seeds: one history can
  be calm and another wild with identical settings.
- **The real market counts as typical** if it falls inside the middle 90% of the
  simulated histories.

## 5. Key findings, with the evidence

**1. The rebuilt model reproduces the paper** (Figs 1–3). On 8 held-out runs of
100,000 days:

| | Paper | Rebuilt model |
|---|---|---|
| Tail α | 2.64 | 2.62 |
| H of returns | 0.48 | 0.47 |
| H of \|returns\| | 0.85 | 0.81 |
| H of news | 0.49 | 0.50 |

Returns cross over to a bell curve by 25 days. *How to read Fig 2:* each curve shows
the share of days with a bigger move than x, on log-log axes. A bell curve drops
like a cliff; a fat tail comes down slowly, like a straight line.

**2. Without herding, no fat tails** (Fig 4). With no chartists the returns are
just news: kurtosis 0.01, which is a bell curve. This was the proposal's base-case
check.

**3. With herding, the inverse cubic law** (Fig 4). The Hill tail exponent is 3.10
(S&P-calibrated) and 2.90 (paper-matched); the S&P 500's is 2.92.

**4. Long memory of move sizes, created by the traders** (Fig 3). Move sizes have
H = 0.81–0.88 in the models (S&P: 0.90), while the news has 0.49. Calm and wild spells
come from the traders, not from the news.

**5. The mechanism: crowds of chartists** (Fig 5). Move size rises steadily with the
share of chartists: 0.6× average when they are rare, 3.5× when they are a fifth of
the market. Bursts end on their own when the mispricing lures traders back to
fundamentals. *How to read:* the left panels show a burst and the chartist share at
the same time; the middle panel averages over 100,000 days.

**6. Kirman's ants match his math** (Fig 6). Simulated and exact distributions differ
by 0.4–2% (total variation). This is the herding rule inside the model.

**7. The S&P 500 looks like a typical history on 9 of 11 measures** (Fig 7). The two
misses:
- the model's returns reverse slightly (H 0.39 vs the S&P's 0.50);
- its middle-to-large tail is a bit thinner (paper-style α 2.60 vs 2.35).

*How to read:* each panel is one measure; the dots are 40 model histories, the
black line is the S&P.

**8. Markets it never saw** (Fig 8): indexes are mostly typical, but single stocks
have fatter tails than an index-calibrated model. That makes sense: companies also
have their own news.

**9. Robust, within limits** (Fig 9). 44% of random settings run away into endless
bubbles. Of the stable ones, 65% give fat tails with long memory. The paper's "open
set" of good parameters is real, but stability is the main limit.

**10. More herding is not always more crashes** (Fig 10). In the paper-matched
model, tails are fattest at a middling herding strength (a1 ≈ 0.6). Above a1 = 1
the chartists lock into one mood, and returns become a bell curve again. The
extremes come from being *near* a tipping point, not from herding as such. That is
the paper's word "criticality".

## 6. The paper's economic standpoint, and what we found

Lux & Marchesi argue **against the efficient market hypothesis** as the explanation
of fat tails, while agreeing that the market is efficient *on average*. They make no
statements about government or regulation. Measured with
[`analysis/efficiency_claims.py`](analysis/efficiency_claims.py) (4 histories ×
100,000 days):

| Paper's claim | Paper-matched | S&P-calibrated |
|---|---|---|
| The price equals its value on average | ✅ mean gap +0.0000 | ✅ +0.0004 |
| No persistent pathological mispricing | ✅ over 10% off on 0.02% of days, never more than 4 days | ✅ 0.47% of days, at most 29 days |
| Volatility rises with the noise-trader share | ✅ 0.59× → 1.41× | ✅ 0.51× → 1.49× |
| Price direction is unpredictable (H ≈ 0.5) | ✅ 0.47 | ⚠️ 0.39 (mild reversal) |
| Fat tails come from traders, not news | ✅ | ✅ |
| Results hold for many settings | ⚠️ for the 56% of random settings that stay stable | |

## 7. What the model gets wrong (honest limitations)

1. **Settings are fitted, not the paper's own.** The parameter tables are in Lux &
   Marchesi (2000), which we couldn't get. The remembered values run away after
   about 28,500 days in our build.
2. **The sources disagree on two details** (SPEC.md sec. 3); we followed the review
   co-written by Lux.
3. **Mild daily reversal** in the S&P-calibrated model (H 0.39 vs 0.50).
4. **No bounce-back after big days:** after a 3-sd day the S&P tends to reverse
   (−0.19); the model doesn't (+0.02).
5. **Rare runaways:** the S&P-calibrated settings run away in 2 of 40 histories of
   100,000 days (about 400 years each).
6. **Tails at long horizons:** the S&P keeps fat tails at 25 days (kurtosis 3.3);
   the models return to a bell curve faster (1.1–1.6).
7. **One whole market, no company news**, so single stocks look more extreme than
   the model.

## 8. How to read any figure in this project

- **Colours:** black = real data, blue = S&P-calibrated, orange = paper-matched,
  green = news or no chartists, grey dashes = reference lines (bell curve, α = 3).
- **Log-log axes:** a straight line is a power law; steeper = thinner tail.
- **Dots in a strip:** one dot per model history; the black line is the real market.

## 9. Sources

- Lux, T. & Marchesi, M. (1999). Scaling and criticality in a stochastic multi-agent
  model of a financial market. *Nature* 397, 498–500. (Kept locally in `paper/`,
  not in the public repository, because it is copyrighted.)
- Samanidou, E., Zschischang, E., Stauffer, D. & Lux, T. (2007). Agent-based models
  of financial markets. *Rep. Prog. Phys.* 70, 409–450 (arXiv physics/0701140).
  The model's equations.
- Cristelli, M., Pietronero, L. & Zaccaria, A. (2011). Critical overview of agent-based
  models for economics (arXiv 1101.1847). Cross-check of the equations.
- Lux, T. & Marchesi, M. (2000). Volatility clustering in financial markets: a
  microsimulation of interacting agents. *Int. J. Theor. Appl. Finance* 3, 675–702.
  (The parameter tables; not in hand.)
- Gopikrishnan, P. et al. (1998). Inverse cubic law for the distribution of stock
  price variations. *Eur. Phys. J. B* 3, 139–140. (In `paper/`.)
- Kirman, A. (1993). Ants, rationality, and recruitment. *Q. J. Econ.* 108, 137–156.
- Peng, C.-K. et al. (1994). Mosaic organization of DNA nucleotides. *Phys. Rev. E*
  49, 1685–1689. (The DFA method behind H.)
- Data (`data/download.py`, every series ends 2026-10-30): Yahoo Finance (^GSPC,
  ^NDX, ^VIX, 20 stocks) and FRED (NBER recessions).

## 10. Reproduce everything

```bash
python3 -m pip install -r requirements.txt
python3 data/download.py               # data files (already in the repository)
python3 main.py                        # S&P vs 20 held-out histories (about 15 s)
python3 make_figures.py                # figures 1-10 (about 1-2 min on 8 cores)
python3 -m pytest tests/ -q -m "not slow"   # quick tests (about 1 min)
python3 -m pytest tests/ -q            # all 293 tests (about 3 min)
python3 -m herding.calibrate paper     # refit to the paper (about 20 min), then --refine
python3 -m herding.calibrate spx       # refit to the S&P 500 (about 1.5 h), then --refine
```
