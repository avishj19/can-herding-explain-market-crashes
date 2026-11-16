# Analysis scripts

Each script answers one question about the rebuilt model and prints the numbers
quoted in `figures/RESULTS.md` and `KEY_POINTS.md`. They use the fitted settings
in `settings/` and held-out seeds only. Run from the project folder, e.g.
`python3 scripts/analysis/long_run_stability.py`.

| Script | Question |
|---|---|
| `efficient_market_claims.py` | Lux & Marchesi's efficient-market claims: price = value on average, no persistent mispricing, volatility vs chartist share |
| `bounce_back_after_big_days.py` | After a big move, does the price bounce back (like the S&P) or keep sliding? |
| `president_terms.py` | Term by term: which presidencies can the politics-free model explain? (Fig 12b) |
| `president_party_gap.py` | Is the S&P different under Democratic and Republican presidents, beyond luck? (Fig 12) |
| `party_scale.py` | The party test at scale: 24 markets (Dow 1896–, S&P, Nasdaq, 20 stocks) against 1,000 model histories, with a bootstrap range and each presidency left out in turn |
| `news_and_reactions.py` | Do traders react logically to news? The model's moves against its own news (with and without herding), the Dow's reaction to 33 presidential elections, and the documented causes of its 20 biggest days |
| `time_step_check.py` | Do results change with the time step dt? (`spx` or `paper`; the paper-matched ones do) |
| `test_accuracy_audit.py` | How accurate is each noisy test? Bias and sd of every checked number over 200 seeds, and the tolerance \|bias\| + 5 sd the tests use (`docs/TEST_ACCURACY.md`) |
| `long_run_stability.py` | How often do the fitted settings run away over 100,000 days? |

Earlier exploration scripts (for the model before the November 2026 rebuild)
are in git history.
