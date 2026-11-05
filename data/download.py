"""Download every data file this project uses, from the original sources.

    python3 data/download.py            # writes into data/
    python3 data/download.py OUTDIR     # writes somewhere else (e.g. to compare)

Sources
    Yahoo Finance (yfinance, auto_adjust=False: split-adjusted, NOT dividend-adjusted)
        ^GSPC  S&P 500 index        -> spx_daily.csv (1950-), spx_daily_1928.csv (1928-)
        ^NDX   Nasdaq-100 index     -> ndx_daily.csv (1985-)
        ^VIX   volatility index     -> vix_daily.csv (1990-)
        20 stocks with market beta 0.9-1.1 (see stocks/README.md) -> stocks/<TICKER>.csv
    FRED (Federal Reserve Bank of St. Louis)
        USREC  NBER recession months -> us_recessions_monthly.csv

Every series stops at the same fixed date (END), so results can be reproduced
exactly even after new trading days are published.
"""
import sys
import urllib.request
from pathlib import Path

import pandas as pd

END = "2026-10-30"                  # last trading day included (a Friday)
INDEXES = {                         # file: (ticker, first day)
    "spx_daily.csv": ("^GSPC", "1950-01-01"),
    "spx_daily_1928.csv": ("^GSPC", "1927-12-01"),
    "ndx_daily.csv": ("^NDX", "1985-10-01"),
    "vix_daily.csv": ("^VIX", "1990-01-01"),
}
STOCK_START = "2000-01-01"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"


def daily_closes(ticker, start, end=END):
    """Daily closes from Yahoo as a DataFrame with a Date index and a Close column."""
    import yfinance as yf
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end_exclusive, auto_adjust=False,
                     progress=False, actions=False)
    if df.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):          # newer yfinance: one column per ticker
        close = close.iloc[:, 0]
    out = close.dropna().rename("Close").to_frame()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out.index.name = "Date"
    return out


def recessions(end=END):
    """NBER recession indicator by month (1 = recession) from FRED, up to END."""
    import ssl
    import certifi                       # trusted certificates (installed with yfinance)
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(FRED_URL, timeout=30, context=ctx) as resp:
        df = pd.read_csv(resp, parse_dates=["observation_date"])
    return df[df["observation_date"] <= pd.Timestamp(end)]


def main(outdir):
    outdir = Path(outdir)
    (outdir / "stocks").mkdir(parents=True, exist_ok=True)
    for name, (ticker, start) in INDEXES.items():
        daily_closes(ticker, start).to_csv(outdir / name)
        print("saved", name)
    here = Path(__file__).resolve().parent
    for ticker in pd.read_csv(here / "stocks" / "market_beta.csv")["ticker"]:
        daily_closes(ticker, STOCK_START).to_csv(outdir / "stocks" / f"{ticker}.csv")
        print("saved stocks/" + ticker)
    recessions().to_csv(outdir / "us_recessions_monthly.csv", index=False)
    print("saved us_recessions_monthly.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent)
