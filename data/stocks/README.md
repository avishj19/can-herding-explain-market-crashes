# Stocks matched by market beta

Daily closing prices, 2000 to 2026-10-30, from Yahoo Finance (`yfinance`, `auto_adjust=False`).
Prices are **split-adjusted but not dividend-adjusted**, the same as the S&P 500 and
Nasdaq-100 index files, so all returns are measured the same way.

`market_beta.csv` lists each stock's market beta: the slope of its daily log return
against the S&P 500's, over the whole file (2000-01-04 to 2026-10-30, 6,747 days). A beta
of 1 means the stock moves about 1% when the market moves 1%.

## The rule for choosing the five (28 Nov 2026)

A stock is kept when all three hold:
1. its market beta over the whole file is between 0.9 and 1.1 (it moves about as much as the market);
2. it is a household name (a visitor knows the company);
3. no two kept stocks are in the same sector.

These rules narrow the list but do not pick the five on their own: eight other stocks of the old twenty also had a whole-file beta of 0.9-1.1 (SBUX, FDX, LOW, ALL, CB, ITW, RTX, UNP). The final five are a judgment call: the best-known name in each of five different sectors.

| Ticker | Company | Sector | Beta (2000-2026) |
|---|---|---|---|
| NKE | Nike | consumer goods | 0.903 |
| HD | Home Depot | retail | 1.002 |
| DIS | Disney | media | 1.042 |
| HON | Honeywell | industrial | 1.056 |
| MSFT | Microsoft | technology | 1.093 |

The rule does **not** look at fat tails, kurtosis or any other measure the model is compared on:
choosing stocks by those would pick the answer before the test.

These five come from an earlier list of 20 stocks (chosen from about 80 large companies with full
price history since 2000, by beta over 2005-2025). Over the whole file, 7 of those 20 fall outside
0.9-1.1 (AAPL 1.15, CSCO 1.20, ORCL 1.20, TXN 1.23, CVX 0.87, CI 0.89, TRV 0.90), so the beta is now
measured on the whole file. The other 15 were removed to keep the project to its question; they are
in git history, and any ticker can still be typed on the site's Test program (fetched live from Yahoo).

Caveat (survivorship bias): the candidates are companies that are large today, so
firms that failed or shrank since 2000 are missing.
