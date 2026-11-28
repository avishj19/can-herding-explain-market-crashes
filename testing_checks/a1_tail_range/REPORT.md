# A1: which tail-exponent range means "about 3"? And what is a1 called?

Rerun: `env -u PYTHONDONTWRITEBYTECODE python3 testing_checks/a1_tail_range/measure.py` (writes `results.json`).
Data: saved S&P 500 closes (`ex.saved_prices("S&P 500")`), 19,329 daily log returns, 1950-01-04 to 2026-10-30.
Model: 40 held-out histories from the recipe in `web/numbers.json` (S&P fit, seeds 1000-1039, 19,329 days, burn 1000); all 40 ran fine.
Hill = `stats.hill_alpha` on |returns| (or one signed side). Regression = `stats.alpha_regression` (paper's method, largest `top` share of days).

## MEASUREMENTS

### 1. S&P 500 tail exponent, many ways
| Method | Full 1950-2026 | 1950-1986 (n=9,295) | 1988-2026 (n=9,781) |
|---|---|---|---|
| Hill, \|r\|, top 2.5% | 3.11 | 3.88 | 3.03 |
| Hill, \|r\|, top 5% | 2.92 | 3.65 | 2.92 |
| Hill, \|r\|, top 10% | 2.73 | 3.11 | 2.60 |
| Hill, positive tail, 2.5% / 5% / 10% | 3.14 / 3.03 / 2.83 | 3.83 / 3.64 / 2.95 | 2.86 / 3.01 / 2.63 |
| Hill, negative tail, 2.5% / 5% / 10% | 3.04 / 2.87 / 2.70 | 3.89 / 3.64 / 3.33 | 3.03 / 2.97 / 2.59 |
| Paper regression, top 30% (paper's setting) | 2.35 | 2.64 | 2.30 |
| Regression, top 10% | 2.91 | 3.60 | 2.84 |
| Regression, top 5% | 3.03 | 3.88 | 3.01 |

How much the estimate itself moves (S&P Hill):
| Noise source | Hill 2.5% | Hill 5% |
|---|---|---|
| 95% interval, iid bootstrap (ignores clustering, too optimistic) | 2.86 to 3.39 | 2.76 to 3.08 |
| 95% interval, block bootstrap (50-day blocks, keeps clustering) | 2.60 to 3.78 | 2.57 to 3.34 |
| Ten equal chunks (~7.7 years each): min to max | 2.63 to 4.29 | 2.53 to 4.10 |

Ten-chunk Hill-5% values: 3.21, 2.77, 3.95, 4.10, 3.12, 3.73, 3.65, 2.53, 3.25, 2.74.

### 2. Model (40 histories, Hill on |r|)
| Estimate | Median | Mean | SD | 10th-90th pct | Min-Max |
|---|---|---|---|---|---|
| Hill top 2.5% | 3.48 | 3.48 | 0.38 | 3.00-4.00 | 2.51-4.37 |
| Hill top 5% (the site's) | 3.10 | 3.17 | 0.34 | 2.78-3.63 | 2.64-3.96 |
| Hill top 10% | 2.85 | 2.91 | 0.25 | 2.65-3.20 | 2.49-3.55 |
| Hill 5%, positive / negative tail | 3.23 / 3.09 | | 0.34 / 0.33 | | |
| Paper regression, top 30% | 2.64 | 2.65 | 0.14 | 2.45-2.83 | 2.37-2.91 |

The saved `web/numbers.json` tail values reproduce exactly (median 3.1047).

Share of the 40 histories whose Hill-5% falls inside each rule: 2.5-3.5: **82.5%**; 2.6-3.0: **37.5%**. Inside the S&P's own block-bootstrap 95% interval (2.57-3.34): 67.5%.

Observations (measurement, not opinion):
- One S&P history's Hill-5% ranges 2.5 to 4.1 across 7.7-year chunks; a model history of the same length ranges 2.64 to 3.96 (SD 0.34). So the model's spread across seeds looks like the S&P's spread across eras.
- The estimate depends on the cutoff: the S&P falls 3.11, 2.92, 2.73 as the tail window grows 2.5%, 5%, 10%; the model falls 3.48, 3.10, 2.85. The model is above the S&P at all three cutoffs by 0.12 to 0.37 (a model-vs-S&P gap of 0.18 at 5%, about half of one model-history SD). Its far tail (2.5%) is a bit thinner than the S&P's.
- With the paper's own regression (top 30%), the model gives 2.64, the same as Lux-Marchesi's reported 2.64 (below), while the S&P gives 2.35 (this method also fits the bell-shaped middle, so it reads lower).

### 3. What the sources say (all quotes under 15 words)
| Source | Value | Error bar / note |
|---|---|---|
| Gopikrishnan et al. 1998, `docs/papers/Gopikrishnan1998_inverse_cubic_law.pdf` (1000 US stocks, 1994-95, 5-min data, ~4e7 points) | "α = 3.1 ± 0.03 (positive tail) and α = 2.84 ± 0.12 (negative tail)" (cumulative fit, 2 to 100 SD) | Second method (local slope, moves > 5 SD): "α = 2.84 ± 0.12 for the positive and α = 2.73 ± 0.13" negative. They conclude the "inverse cubic law α ≈ 3". Error bars are fit errors on huge samples (not daily index data). Exponents across their two methods and two tails run 2.73-3.1. |
| Lux & Marchesi 1999, `docs/papers/LuxMarchesi1999_Nature.pdf` | Regression on the tails "yields an estimate of 2.64 for data with a unit time step" | Single run, no error bar. They call it "in good agreement" with daily empirical results. |
| Cont 2001, Quantitative Finance 1, 223-236 (fetched from finance.martinsewell.com/stylized-facts/dependence/Cont2001.pdf) | "a tail index which is finite, higher than two and less than five" | Also: "precise form of the tails is difficult to determine"; for daily returns of stocks, indices and FX, ξ of 0.2-0.4 means "2 < α(T) ⩽ 5"; for most assets the tail index is "typically close to four (sometimes less)". Not verified against a second copy, but the text was read from the PDF itself. |

The proposal's "around 2.6-3" appears to come from Lux-Marchesi's 2.64 and the inverse cubic law's 3. I did not check how the proposal derived it beyond that.

## CONCLUSIONS (judgement, separate from the numbers)

**Recommended range: 2.5 to 3.5 (keep the site's rule).** Justification: the two published anchors the project cites (2.64 and 2.73-3.1) are all inside 2.5-3.5, but the proposal's 2.6-3.0 is only 0.4 wide, narrower than the noise of the measurement itself. A single S&P 500 series gives 2.57-3.34 (95%, block bootstrap) at one cutoff; the cutoff alone moves the S&P from 2.73 to 3.11; eras move it 2.5 to 4.1; and an honest model history of the same length has SD 0.34, so even a perfect model would land outside 2.6-3.0 about 60% of the time (only 37.5% of our histories land inside). Cont's broader 2 to 5 band shows the literature treats "about 3" loosely, so 2.5-3.5 is already a conservative reading, not a lax one. Note the rule is on the median across 40 histories, which has much smaller noise (about 0.34/sqrt(40) = 0.05), so the median test is stricter than the per-history shares above suggest, and 2.6-3.0 would still be defensible for the median alone; but 3.10 sits only 0.10 outside it, and 0.10 is about the cutoff-choice effect, so failing the check on that basis would not mean anything.

**Does 3.10 count as matching?** Yes, as "about 3 and consistent with the S&P 500": it is inside 2.5-3.5, 0.18 above the S&P's 2.92 (about half a model-history SD, and inside the S&P's own block-bootstrap interval 2.57-3.34), and inside Gopikrishnan's quoted range for the positive tail (3.1). Caveats to say plainly on the site: the model's far tail (top 2.5%) is somewhat thinner than the S&P's (3.48 vs 3.11), and the regression method gives 2.64 for the model vs 2.35 for the S&P. I would not claim a match to better than about ±0.3.

If the student prefers the proposal's wording, report "2.6-3" as the target quoted from the proposal and "2.5-3.5" as the tolerance used because of measurement noise, with these numbers as the reason. That keeps the proposal and the check honest and defensible.

## Question 2: the name of a1

Evidence:
- Lux & Marchesi (1999) PDF: no name for the parameter (the Nature paper does not list a1); it describes herding as "herding effects" using the "majority opinion among noise traders", and a "tendency towards herding behaviour".
- `docs/SPEC.md` (line 30): "a1 = herding (copy the majority mood)".
- Samanidou et al. 2007 review (arXiv physics/0701140, read from the PDF): α1 and α2 "are measures of ... the importance of 'flows' (i.e. the observed behaviour of others)" versus charts. So the review calls it the importance of "flows" and does not name it "herding strength" (the same review says "herding tendency" only for the Cont-Bouchaud model).
- arXiv cond-mat/0204574 ("Price Drops, Fluctuations, and Correlation in a Multi-Agent Model of Stock Markets"), parameter table: α1 is "strength of herding effect". This is the only Lux-Marchesi-based source I found that labels it directly.
- Kirman side: arXiv 1506.03708 calls the herding coefficient "herding intensity h" (separate from the idiosyncratic switching coefficient a). I could not verify what Kirman (1993) himself calls it; his paper centres on "recruitment", per its title, and the web summaries describe the term as recruitment.
- Searches found no use of "amount of herding" for this parameter; "herding intensity" and "strength of herding" both occur in the wider literature.

**Recommendation: "herding strength"**, written once with the symbol as "herding strength (a1)". Reasons: it is the plainest beginner wording; it matches the one Lux-Marchesi table that names a1 ("strength of herding effect"); "intensity" is also used (Kirman-type model, 1506.03708) so it is a fair synonym worth a single parenthetical, but a longer word costs a beginner; "amount of herding" is not used in the literature and wrongly suggests a count of herders rather than how strongly a trader weighs the crowd's mood. Also say in one line what it does: how much a trend chaser's mood switch is driven by the majority mood versus the price trend (U1 = a1 x + ...). The site's current wording (`web/page_guide.py` line 385, "herding strength (a1)") already follows this; no change needed.

Limits: the web results come from search summaries plus the PDFs I extracted myself (Cont 2001, Samanidou et al. 2007, cond-mat/0204574, 1506.03708); the Kirman 1993 original and Cont's primary statements beyond the quoted lines were not checked.
