# Stocks matched by market beta

Daily closing prices, 2000 to 2026-10-30, from Yahoo Finance (`yfinance`, `auto_adjust=False`).
Prices are **split-adjusted but not dividend-adjusted**, the same as the S&P 500 and
Nasdaq-100 index files, so all returns are measured the same way.

`market_beta.csv` lists each stock's market beta: the slope of its daily log return
against the S&P 500's (2005-2025). A beta of 1 means the stock moves about 1% when the
market moves 1%. The 20 stocks were chosen from about 80 large companies with full
price history since 2000, keeping those with beta between 0.9 and 1.1.

Caveat (beta window): the betas are measured over 2005-2025, a narrower window than
the data (2000-2026). Over the whole file several fall outside 0.9-1.1 (TXN 1.23,
CSCO 1.20, ORCL 1.20, AAPL 1.15, CVX 0.87, CI 0.90, TRV 0.90): the dot-com years
made some technology stocks move more than the market.

Caveat (survivorship bias): the candidates are companies that are large today, so
firms that failed or shrank since 2000 are missing.
