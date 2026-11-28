# What data the project has, and what is still missing

A checklist for improving the accuracy of the project's numbers (November 2026). Each line says
what the data would fix, where it comes from, and whether it is in the repository.

## Kept, and removed, on 28 November 2026

The data files the project keeps: the S&P 500 (1950- and 1928-, Yahoo ^GSPC), the Nasdaq-100 (^NDX),
the Dow Jones (1896- and 1927-1952, MeasuringWorth), 5 stocks (`data/stocks/README.md` has the rule),
the Dow's 20 biggest days with their causes (`big_days_events.csv`), and three commodities for the
site's Commodities group (`data/commodities/`: Brent oil from FRED `DCOILBRENTEU`, gold and copper
futures from Yahoo `GC=F`, `HG=F`).

Removed as scope beyond the proposal (in git history): the outside-news files (10-year yield, broad
dollar, Fed funds target, FOMC decision days), 15 of the 20 stocks, the VIX and the NBER recession
months. `python3 data/download_market_data.py` rebuilt every file byte for byte on 16 November 2026;
the slimmed-down downloader has not been rerun since.

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
4. **Total returns (with dividends).** The index files are price-only. That matters for the
   average return, not for tails or clustering. Source: CRSP through WRDS.
5. **More countries.** Gopikrishnan et al. used several markets. Yahoo has
   the FTSE 100 (1984–), DAX (1987–) and Nikkei 225 (1965–). These would test whether the model
   fits markets it never saw (like Fig 8), and are free.

## Limits of the kept data

- **Futures roll over each month.** A continuous futures price switches contracts, which can
  add small jumps. The biggest gold and copper days were checked and all are real events
  (2013-04-15 gold crash, July 2025 copper tariff news), not rolls.
- **Brent oil's saved file is a spot price (FRED)**, while its live Yahoo symbol (BZ=F) is a futures
  price; the two usually differ by 1-2% (the site says so).
