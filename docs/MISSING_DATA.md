# What data the project has, and what is still missing

A checklist for improving the accuracy of the project's numbers (November 2026). Each line says
what the data would fix, where it comes from, and whether it is in the repository.

## Added in November 2026 (`data/factors/`, `data/fed_decisions.csv`)

| Data | Source | Fixes |
|---|---|---|
| Brent oil, 1987– | FRED `DCOILBRENTEU` (US Energy Information Administration) | outside news that every trader sees (Figs 16–17) |
| Gold and copper futures, 2000– | Yahoo `GC=F`, `HG=F` (front month) | the same; gold as a "safe haven" |
| 10-year Treasury yield, 1962– | FRED `DGS10` (Federal Reserve Board H.15) | interest-rate news |
| Broad US dollar, 2006– | FRED `DTWEXBGS` (Federal Reserve Board H.10) | currency news |
| Fed funds target, 1982– | FRED `DFEDTAR`, `DFEDTARU` | which Fed decisions changed rates, and by how much |
| FOMC decision days, 1994– | federalreserve.gov FOMC calendars | scheduled vs surprise Fed news (checked against Lucca & Moench 2015) |

Every file is rebuilt byte for byte by `python3 data/download_market_data.py` (checked
16 November 2026: all 34 files identical).

## Still missing, in order of how much it would help

1. **Lux & Marchesi (2000) parameter tables** (*Int. J. Theor. Appl. Finance* 3, 675–702). The
   biggest gap: the settings are fitted instead of the paper's own (Limitation 1). Available
   through the CMU library.
2. **Intraday prices (5-minute) for the S&P and large stocks.** The inverse cubic law
   (Gopikrishnan et al. 1998) was measured on 5-minute returns. Daily data give only about 19,000
   points since 1950, so the tail exponent's last digit is uncertain. Source: NYSE TAQ through
   WRDS (CMU has a subscription). Not free, and too large for this repository.
3. **The news behind each big day, classified.** Baker, Bloom, Davis & Sammon (2021, NBER
   w28687) read the next day's newspaper for every daily move of 2.5% or more since 1900. Their
   spreadsheet (stockmarketjumps.com) needs a Google sign-in, and their raw files are scanned Wall
   Street Journal pages (copyrighted, about 130 MB), so neither is in the repository. Asking the
   authors for the classified table would replace the 20 hand-checked days in
   `data/big_days_events.csv`.
4. **Scheduled economic releases** (jobs report, inflation). Savor & Wilson (2013) show they
   are news days like Fed days. Their past dates are in ALFRED (St. Louis Fed), which needs a free
   API key. That key must stay out of this public repository.
5. **Total returns (with dividends).** The index files are price-only. That matters for the
   average return (Fig 12's party gap), not for tails or clustering. Source: CRSP through WRDS.
6. **More countries.** Gopikrishnan et al. and Longin & Solnik used several markets. Yahoo has
   the FTSE 100 (1984–), DAX (1987–) and Nikkei 225 (1965–). These would test whether the model
   fits markets it never saw (like Fig 8), and are free.

## Limits of the data that was added

- **Futures roll over each month.** A continuous futures price switches contracts, which can
  add small jumps. The biggest gold and copper days were checked and all are real events
  (2013-04-15 gold crash, July 2025 copper tariff news), not rolls.
- **Different closing times.** Gold futures settle at 1:30 pm, oil at 2:30 pm and stocks at
  4 pm (New York time). The regressions include the day before and after (Dimson 1979).
- **Fed dates before 1994** are not included: until February 1994 the Fed did not announce its
  decisions, so markets learned of them at different times.
