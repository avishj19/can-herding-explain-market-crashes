# A6 Data audit (28 Nov 2026)

Read-only audit of `data/`, written by a subagent (its own write of this file was refused; this copy was saved by
the main session from its report). Helper scripts in this folder: `dow_verify.py`, `ndx_verify.py`,
`spx_files.py`, `stocks_table.py`. Nothing in `data/` was changed.

## Decisions needed before deleting anything
1. **Commodities:** the site's Commodities group (Brent oil, gold, copper) reads `data/factors/oil_brent.csv`,
   `gold.csv`, `copper.csv` through `factors.FACTOR_DIR` (herding/explore.py lines 27, 32, 134). Option A
   (recommended): move those three to `data/commodities/` with their own path table in explore.py, delete the other
   factor files. Option B: drop the Commodities group too (catalog, COMMODITIES, YAHOO, SAVED_SPOT, asset_kind,
   page_compare.py, page_story.py:395 and four test files change).
2. **The `party` argument runs through the whole Test program** (run_mixes, model_market, party_shocks,
   party_calendar; web/common.py cache keys; page_explore.py slider, toggle, panel, share-link keys;
   tests/test_live_and_model_market.py). Removing politics is a code change; with party = 0 the numbers must come out
   identical (party = 0 is the model of SPEC.md), so test that.
3. **scripts/analysis/news_and_reactions.py is half politics:** its `biggest_days` part uses `big_days_events.csv`,
   which the site's `day_story` needs. Split it; don't delete it (same for tests/test_news_and_reactions.py).
4. **Doc wording:** CLAUDE.md and docs/memory.md say `stats.hill`; the function is `stats.hill_alpha`. The proposal
   says Stooq; the project uses Yahoo ^GSPC with FRED as an independent check.

## Inventory (all series end 2026-10-30; made by data/download_market_data.py)
Classes: (a) needed for the proposal's checks; (b) kept extensions or site data; (c) only for removed parts; (d) unused.

| File | Range | Source | Read by | Class |
|---|---|---|---|---|
| sp500_daily.csv | 1950-, 19,329 returns | Yahoo ^GSPC | calibration target, Figs 1-11, site, tests | a |
| sp500_daily_from_1928.csv | 1928-, 24,814 returns | Yahoo ^GSPC | Fig 8 panel, BAD_CLOSES tests, data_crosscheck | b |
| nasdaq100_daily.csv | 1985- | Yahoo ^NDX | INDEXES, Fig 8, All markets | b |
| dow_jones_daily_from_1896.csv | 1896- | MeasuringWorth (minus BAD_CLOSES) | INDEXES, All markets, tests | b |
| dow_jones_daily_1927_1952.csv | 1927-1952 | MeasuringWorth | re-derives the S&P bad closes (test), data_crosscheck | b |
| big_days_events.csv | 20 rows | hand-checked, with sources | explore.day_events -> Test program's big-day reader | b |
| vix_fear_index_daily.csv | 1990- | Yahoo ^VIX | one test only | d |
| us_recessions_monthly.csv | 1854- | FRED USREC | file-listing tests only | d |
| stocks/ (20 CSVs, README, market_beta.csv) | 2000- | Yahoo | stock_tickers, catalog, Fig 8, All markets | b |
| factors/oil_brent, gold, copper | 1987-/2000- | FRED / Yahoo futures | explore.COMMODITIES | b |
| factors/dollar_index, treasury_10y_yield, fed_funds_target; fed_decisions.csv | | FRED / federalreserve.gov | factors.py and its tests only | c |

## Delete lists (with what must change alongside)
- **Politics:** herding/politics.py; tests test_politics*, test_party_figures, test_party_scale; scripts/party_figures.py;
  scripts/analysis/president_party_gap.py, president_terms.py, party_scale.py; figures fig12-fig15. Edit: herding/__init__.py,
  plots.py (party shading), explore.py (party block and the `party=` parameter; keep day_events/day_story),
  make_figures.py (fig12), page_findings.py (party section), page_checks.py, page_explore.py, common.py, page_sources.py,
  the tests that use party, news_and_reactions.py (split), and the docs (RESULTS, GALLERY, KEY_POINTS, SPEC, TEST_ACCURACY,
  analysis README, CLAUDE.md, README, docs/redesign).
- **Outside news:** fed_decisions.csv, factors/dollar_index, treasury_10y_yield, fed_funds_target (and per option A/B the
  three commodity files); herding/factors.py; tests/test_factors.py; scripts/outside_news_figures.py; figures fig16-fig18.
  Edit: herding/__init__.py, explore.py (import, commodity loader), test_plots.py, download_market_data.py (factor and FOMC
  parts), data_crosscheck.py (factor comparisons), MISSING_DATA.md, RESULTS, GALLERY, KEY_POINTS, CLAUDE.md, page_sources.py.
- **Unused data:** vix_fear_index_daily.csv, us_recessions_monthly.csv, their two tests, their downloader entries and text
  mentions (keep ^VIX in explore.NOT_A_PRICE: that is a blocked-ticker list).
- **Do not touch:** both S&P files, Nasdaq-100, both Dow files, big_days_events.csv, kept stocks, market_data.py,
  policy.py (Fig 11 only needs model, checks, experiments), garch.py, kirman.py.

## Condensed stock set
All 20 saved stocks cover the same span (2000-01-03 to 2026-10-30, 6,748 returns), so history length can't choose.
Over the whole file, 7 of the 20 have beta outside 0.9-1.1 (AAPL 1.145, CSCO 1.204, ORCL 1.202, TXN 1.227, CVX 0.868,
CI 0.895, TRV 0.896). Proposed rule: full-file beta 0.9-1.1, excess kurtosis below 25, a household name, different sectors.

| Ticker | Sector | Beta | Kurtosis | Hill alpha | Big days |
|---|---|---|---|---|---|
| MSFT | technology | 1.09 | 9.5 | 2.85 | 43 |
| DIS | media | 1.04 | 9.6 | 2.83 | 46 |
| HD | retail | 1.00 | 23.0 | 3.22 | 37 |
| NKE | consumer goods | 0.90 | 13.4 | 2.71 | 51 |
| HON | industrial | 1.06 | 13.7 | 2.79 | 47 |

Swaps: SBUX for NKE, UNP for HON. AAPL fails the rule (beta 1.145, kurtosis 115 from one -52% day). Changing the set means
editing market_beta.csv (it drives stock_tickers everywhere), deleting 15 CSVs, updating stocks/README.md, the downloader,
explore.COMPANY and every hard-coded "20"; Fig 8 (22 markets -> 7) and its RESULTS.md numbers must be rerun and rewritten.
Typed tickers still work live.

## Missing or unused data against the proposal
- The proposal's checks need real data only for S&P 500 daily closes. The 1950- file is the right one (cleaner, identical to
  the 1928 file from 1950 on); state in the write-up that the S&P check is on 1950-2026.
- Limits to state, not chase before Dec 1: no 5-minute data (the inverse cubic law was measured on 5-minute returns); no Lux &
  Marchesi (2000) parameter tables; no second source for S&P 1950-2016 (FRED covers 2016-2026); proposal said Stooq.
- Unused: VIX and recessions.

## Data quality
- **Nasdaq-100, Yahoo vs FRED:** differences cluster in 2003-2005 (69% of days in 2003 and 2004). On the 117 days where daily
  returns differ by over 1%, Yahoo is closer to the S&P's move on 99. Verdict: Yahoo is more likely right. Effect only on the
  Nasdaq panel (tail 3.149 Yahoo, 3.187 FRED). Keep Yahoo; note it in a caption.
- **Dow BAD_CLOSES (17 dates):** all supported by Yahoo ^DJI and the S&P's move that day; three weak ones differ by under 0.3%.
  No suspect day is missing. 2002-09-25 (7481.82 for 7841.82) strongly supported. Effect: tail 2.788 / 2.809 / 2.786.
- MeasuringWorth lacks 8 trading days in 2026 (the next return spans two days): harmless, worth a footnote.
- S&P Yahoo vs FRED (2016-2026): one day above 0.1%. Harmless.

## Suggested order
1. Decide commodities A or B and the five stocks. 2. Remove outside news, run quick tests. 3. Remove politics (the `party`
parameter last), run the suite and the stress test. 4. Condense the stocks, rerun Fig 8. 5. Delete VIX and recessions.
6. Re-download into scratch and confirm the kept files are byte-identical. 7. Docs last. One commit per step, in Avish's words.
