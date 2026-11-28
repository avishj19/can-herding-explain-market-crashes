"""Download every data file this project uses, from the original sources.

    python3 data/download_market_data.py            # writes into data/
    python3 data/download_market_data.py OUTDIR     # writes somewhere else (e.g. to compare)

Sources
    Yahoo Finance (yfinance, auto_adjust=False: split-adjusted, NOT dividend-adjusted)
        ^GSPC  S&P 500 index        -> sp500_daily.csv (1950-), sp500_daily_from_1928.csv (1928-)
        ^NDX   Nasdaq-100 index     -> nasdaq100_daily.csv (1985-)
        5 stocks with market beta 0.9-1.1, the tickers in stocks/market_beta.csv (see stocks/README.md) -> stocks/<TICKER>.csv
    FRED, the site's Commodities group (in commodities/)
        DCOILBRENTEU  Brent crude oil spot price, $/barrel (1987-) -> commodities/oil_brent.csv
                      (Brent, not WTI: WTI closed below zero on 2020-04-20, which has no log return)
    Yahoo Finance, front-month futures (continuous contract, 2000-)
        GC=F gold -> commodities/gold.csv, HG=F copper -> commodities/copper.csv
    MeasuringWorth (S. H. Williamson, "Daily Closing Values of the DJA in the United States,
    1885 to Present", measuringworth.com/datasets/DJA)
        Dow Jones Industrial Average 1927-1952 -> dow_jones_daily_1927_1952.csv
        Dow Jones Industrial Average 1896-     -> dow_jones_daily_from_1896.csv (first DJIA day: 26 May 1896),
        a market of its own (the site's Dow Jones); the 1927-1952 slice is used to find bad closes in the
        early S&P file (herding/market_data.py)

Every series stops at the same fixed date (END), so results can be reproduced
exactly even after new trading days are published.
"""
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

END = "2026-10-30"                  # last trading day included (a Friday)
INDEXES = {                         # file: (ticker, first day)
    "sp500_daily.csv": ("^GSPC", "1950-01-01"),
    "sp500_daily_from_1928.csv": ("^GSPC", "1927-12-01"),
    "nasdaq100_daily.csv": ("^NDX", "1985-10-01"),
}
STOCK_START = "2000-01-01"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
COMMODITIES_FRED = {"oil_brent.csv": "DCOILBRENTEU"}
COMMODITIES_YAHOO = {"gold.csv": "GC=F", "copper.csv": "HG=F"}
DOW_URL = "https://www.measuringworth.com/datasets/DJA/result.php"
DOW_START = "1896-05-26"             # the first day of the Dow Jones Industrial Average


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


def _fetch(url, data=None, tries=6):
    """The text of a web page over HTTPS, retried if it times out or comes back empty.
    Uses curl when it is installed: FRED's bot screen often stalls Python's own client
    (November 2026) but lets curl through. A missing page raises at once."""
    import shutil
    import subprocess
    import time
    error = None
    for attempt in range(tries):
        try:
            if shutil.which("curl"):
                # --speed-time: give up only if the transfer stalls for 60 s (a big page, like the
                # Dow since 1896, can take minutes); a cut-off transfer is an error, never a file
                cmd = ["curl", "-sSL", "--connect-timeout", "30", "--speed-time", "60", "--speed-limit", "1",
                       "-w", "\n%{http_code}", url]
                if data is not None:
                    cmd[1:1] = ["--data", data.decode()]
                out = subprocess.run(cmd, capture_output=True, text=True)
                text, _, status = out.stdout.rpartition("\n")
                if out.returncode == 0 and status.isdigit() and int(status) >= 400:   # a 4xx/5xx answer
                    raise urllib.error.HTTPError(url, int(status), "HTTP error", None, None)
                if out.returncode != 0:
                    text = ""                                   # failed or cut off: try again
            else:
                import ssl
                import certifi                        # trusted certificates (installed with yfinance)
                req = urllib.request.Request(url, data=data, headers={"User-Agent": "Mozilla/5.0"})
                ctx = ssl.create_default_context(cafile=certifi.where())
                with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
            if text.strip():
                return text
            error = RuntimeError(f"{url} came back empty")
        except urllib.error.HTTPError:
            raise
        except (TimeoutError, urllib.error.URLError) as e:
            error = e
        time.sleep(5 * (attempt + 1))
    raise error


def fred(series, end=END):
    """One FRED series up to END as a Series indexed by date (days without a value dropped)."""
    import io
    df = pd.read_csv(io.StringIO(_fetch(FRED_URL.format(series))), parse_dates=["observation_date"],
                     index_col="observation_date")
    out = pd.to_numeric(df[series], errors="coerce").dropna()
    out.index.name = "Date"
    return out[out.index <= pd.Timestamp(end)]


def dow_jones(first=1927, last=1952):
    """Daily Dow closes from MeasuringWorth (its query form returns an HTML table)."""
    import html
    import re
    from urllib.parse import urlencode
    form = urlencode(dict(monthStartD=1, dayStartD=1, yearStartD=first,
                          monthEndD=12, dayEndD=31, yearEndD=last)).encode()
    page = _fetch(DOW_URL, data=form)
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page)).replace("\xa0", " "))
    rows = re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4}) ([\d.]+)", text)
    df = pd.DataFrame({"Date": pd.to_datetime([f"{y}-{m}-{d}" for m, d, y, _ in rows]),
                       "Close": [float(c) for *_, c in rows]}).drop_duplicates("Date")
    return df[(df["Date"].dt.year >= first) & (df["Date"].dt.year <= last)].set_index("Date")


def main(outdir):
    outdir = Path(outdir)
    (outdir / "stocks").mkdir(parents=True, exist_ok=True)
    fetched = {}                         # each ticker downloaded once, from its earliest start
    for name, (ticker, start) in sorted(INDEXES.items(), key=lambda kv: kv[1][1]):
        if ticker not in fetched:
            fetched[ticker] = daily_closes(ticker, start)
        df = fetched[ticker]
        df[df.index >= pd.Timestamp(start)].to_csv(outdir / name)
        print("saved", name)
    here = Path(__file__).resolve().parent
    for ticker in pd.read_csv(here / "stocks" / "market_beta.csv")["ticker"]:
        daily_closes(ticker, STOCK_START).to_csv(outdir / "stocks" / f"{ticker}.csv")
        print("saved stocks/" + ticker)
    long = dow_jones(1896, pd.Timestamp(END).year)       # one download; the short file is a slice
    long[(long.index.year >= 1927) & (long.index.year <= 1952)].to_csv(outdir / "dow_jones_daily_1927_1952.csv")
    long[(long.index >= DOW_START) & (long.index <= END)].to_csv(outdir / "dow_jones_daily_from_1896.csv")
    print("saved dow_jones_daily_1927_1952.csv and dow_jones_daily_from_1896.csv")
    save_commodities(outdir)


def save_commodities(outdir):
    """The site's Commodities group: Brent oil (FRED), gold and copper (Yahoo futures)."""
    outdir = Path(outdir)
    (outdir / "commodities").mkdir(parents=True, exist_ok=True)
    for name, series in COMMODITIES_FRED.items():
        fred(series).rename("Close").to_frame().to_csv(outdir / "commodities" / name)
        print("saved commodities/" + name)
    for name, ticker in COMMODITIES_YAHOO.items():
        daily_closes(ticker, STOCK_START).to_csv(outdir / "commodities" / name)
        print("saved commodities/" + name)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--commodities"]:         # only the three commodity files
        save_commodities(args[1] if len(args) > 1 else Path(__file__).resolve().parent)
    else:
        main(args[0] if args else Path(__file__).resolve().parent)
