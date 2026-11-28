# Analysis scripts

Each script answers one question about the rebuilt model and prints the numbers
quoted in `figures/RESULTS.md` and `KEY_POINTS.md`. They use the fitted settings
in `settings/` and held-out seeds only. Run from the project folder, e.g.
`python3 scripts/analysis/long_run_stability.py`.

| Script | Question |
|---|---|
| `efficient_market_claims.py` | Lux & Marchesi's efficient-market claims: price = value on average, no persistent mispricing, volatility vs chartist share |
| `bounce_back_after_big_days.py` | After a big move, does the price bounce back (like the S&P) or keep sliding? |
| `news_and_reactions.py` | Do traders react logically to news? The model's moves against its own news (with and without herding), and the documented causes of the Dow's 20 biggest days |
| `time_step_check.py` | Do results change with the time step dt? (`spx` or `paper`; the paper-matched ones do) |
| `test_accuracy_audit.py` | How accurate is each noisy test? Bias and sd of every checked number over 200 seeds, and the tolerance \|bias\| + 5 sd the tests use (`docs/TEST_ACCURACY.md`) |
| `web_stress_test.py` | Is the web app precise, fast and correct on all the data? Every market's numbers recomputed independently (scipy, plain loops), a herding x seed grid, slider corners, live Yahoo vs saved files, every page run headless (report: `scratch/web_stress/report.md`) |
| `data_crosscheck.py` | Are the saved files in `data/` right? Each index against an independent copy (FRED SP500, DJIA, NASDAQ100; Yahoo ^DJI for the MeasuringWorth Dow), the Brent oil file against today's FRED, and the overlapping files against each other: missing days, wrong closes, return differences, and the app's numbers with each copy (downloads in `scratch/crosscheck/`; run with `python3 -I`) |
| `panic_skew.py` | Can "panic" (traders reacting harder to falls, Bouchaud et al. 2001) give the model the S&P 500's skew? Mostly no: the 1%/99% ratio goes 1.00 -> 1.03 against the S&P's 1.09 |
| `long_run_stability.py` | How often do the fitted settings run away over 100,000 days? |

Earlier exploration scripts (for the model before the November 2026 rebuild)
are in git history.

The president's-party and outside-news studies (Figs 12-18) were removed on 28 Nov 2026 as scope
beyond the proposal; their scripts are in git history.
