"""Download every data file this project uses, from the original sources.

    python3 data/download_market_data.py            # writes into data/
    python3 data/download_market_data.py OUTDIR     # writes somewhere else (e.g. to compare)

Sources
    Yahoo Finance (yfinance, auto_adjust=False: split-adjusted, NOT dividend-adjusted)
        ^GSPC  S&P 500 index        -> sp500_daily.csv (1950-), sp500_daily_from_1928.csv (1928-)
        ^NDX   Nasdaq-100 index     -> nasdaq100_daily.csv (1985-)
        ^VIX   volatility index     -> vix_fear_index_daily.csv (1990-)
        20 stocks with market beta 0.9-1.1 (see stocks/README.md) -> stocks/<TICKER>.csv
    FRED (Federal Reserve Bank of St. Louis)
        USREC  NBER recession months -> us_recessions_monthly.csv
    MeasuringWorth (S. H. Williamson, "Daily Closing Values of the DJA in the United States,
    1885 to Present", measuringworth.com/datasets/DJA)
        Dow Jones Industrial Average 1927-1952 -> dow_jones_daily_1927_1952.csv
        (used only to find bad closes in the early S&P file: herding/market_data.py)

Every series stops at the same fixed date (END), so results can be reproduced
exactly even after new trading days are published.
"""
import sys
import urllib.request
from pathlib import Path

import pandas as pd

END = "2026-10-30"                  # last trading day included (a Friday)
INDEXES = {                         # file: (ticker, first day)
    "sp500_daily.csv": ("^GSPC", "1950-01-01"),
    "sp500_daily_from_1928.csv": ("^GSPC", "1927-12-01"),
    "nasdaq100_daily.csv": ("^NDX", "1985-10-01"),
    "vix_fear_index_daily.csv": ("^VIX", "1990-01-01"),
}
STOCK_START = "2000-01-01"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"
DOW_URL = "https://www.measuringworth.com/datasets/DJA/result.php"


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


def dow_jones(first=1927, last=1952):
    """Daily Dow closes from MeasuringWorth (its query form returns an HTML table)."""
    import html
    import re
    import ssl
    import certifi
    from urllib.parse import urlencode
    form = urlencode(dict(monthStartD=1, dayStartD=1, yearStartD=first,
                          monthEndD=12, dayEndD=31, yearEndD=last)).encode()
    req = urllib.request.Request(DOW_URL, data=form, headers={"User-Agent": "Mozilla/5.0"})
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        page = resp.read().decode("utf-8", errors="replace")
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page)).replace("\xa0", " "))
    rows = re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4}) ([\d.]+)", text)
    df = pd.DataFrame({"Date": pd.to_datetime([f"{y}-{m}-{d}" for m, d, y, _ in rows]),
                       "Close": [float(c) for *_, c in rows]})
    return df[(df["Date"].dt.year >= first) & (df["Date"].dt.year <= last)].set_index("Date")


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
    dow_jones().to_csv(outdir / "dow_jones_daily_1927_1952.csv")
    print("saved dow_jones_daily_1927_1952.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent)
