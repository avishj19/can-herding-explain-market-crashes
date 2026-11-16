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
    FRED, outside factors (all in factors/; FRED series ids in FACTORS_FRED)
        DCOILBRENTEU  Brent crude oil spot price, $/barrel (1987-) -> factors/oil_brent.csv
                      (Brent, not WTI: WTI closed below zero on 2020-04-20, which has no log return)
        DGS10         10-year Treasury yield, percent (1962-)   -> factors/treasury_10y_yield.csv
        DTWEXBGS      broad trade-weighted US dollar (2006-)    -> factors/dollar_index.csv
        DFEDTAR, DFEDTARU  federal funds target (upper end after 2008-12-16) -> factors/fed_funds_target.csv
    Yahoo Finance, front-month futures (continuous contract, 2000-)
        GC=F gold -> factors/gold.csv, HG=F copper -> factors/copper.csv
    Federal Reserve Board (federalreserve.gov/monetarypolicy/fomccalendars.htm and
    fomchistorical<YEAR>.htm), with the FRED target above
        FOMC decision days since February 1994 -> fed_decisions.csv
    MeasuringWorth (S. H. Williamson, "Daily Closing Values of the DJA in the United States,
    1885 to Present", measuringworth.com/datasets/DJA)
        Dow Jones Industrial Average 1927-1952 -> dow_jones_daily_1927_1952.csv
        Dow Jones Industrial Average 1896-     -> dow_jones_daily_from_1896.csv (first DJIA day: 26 May 1896),
        a market of its own (Figs 13-15); the 1927-1952 slice is used to find bad closes in the
        early S&P file (herding/market_data.py)

Every series stops at the same fixed date (END), so results can be reproduced
exactly even after new trading days are published.
"""
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

END = "2026-10-30"                  # last trading day included (a Friday)
INDEXES = {                         # file: (ticker, first day)
    "sp500_daily.csv": ("^GSPC", "1950-01-01"),
    "sp500_daily_from_1928.csv": ("^GSPC", "1927-12-01"),
    "nasdaq100_daily.csv": ("^NDX", "1985-10-01"),
    "vix_fear_index_daily.csv": ("^VIX", "1990-01-01"),
}
STOCK_START = "2000-01-01"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FACTORS_FRED = {"oil_brent.csv": "DCOILBRENTEU", "treasury_10y_yield.csv": "DGS10",
                "dollar_index.csv": "DTWEXBGS"}
FACTORS_YAHOO = {"gold.csv": "GC=F", "copper.csv": "HG=F"}
FOMC_FIRST = 1994                    # the Fed has announced its decisions the same day since Feb 1994
FED = "https://www.federalreserve.gov/monetarypolicy/"
MONTHS = {m: i + 1 for i, m in enumerate(("January", "February", "March", "April", "May", "June", "July",
                                          "August", "September", "October", "November", "December"))}
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


def recessions(end=END):
    """NBER recession indicator by month (1 = recession) from FRED, up to END."""
    return fred("USREC", end).astype(int).rename("USREC").reset_index().rename(
        columns={"Date": "observation_date"})


def fed_funds_target(end=END):
    """The federal funds target in percent: one rate until 2008-12-15, then the top of the range."""
    old, new = fred("DFEDTAR", end), fred("DFEDTARU", end)
    return pd.concat([old[old.index < new.index[0]], new]).rename("Close").to_frame()


def _last_day(month, days, year):
    """The last day of a meeting written like ('March', '17-18') or ('Oct/Nov', '31-1')."""
    month = month.split("/")[-1].strip()
    month = next(m for m in MONTHS if m.startswith(month[:3]))
    return pd.Timestamp(year, MONTHS[month], int(days.split("-")[-1]))


def scheduled_fomc_days(first=FOMC_FIRST, last=None):
    """Last days of the FOMC's scheduled meetings (the days decisions are announced). Unscheduled
    meetings, conference calls, notation votes and cancelled meetings are left out."""
    import re
    last = last or pd.Timestamp(END).year
    days = []
    for year in range(first, last + 1):
        try:
            page = _fetch(f"{FED}fomchistorical{year}.htm")
        except urllib.error.HTTPError:            # recent years are on the calendar page instead
            break
        # e.g. "March 17-18 Meeting - 2015", "January 31-February 1 Meeting - 2006" or
        # "Jan/Feb 31-1 Meeting - 2017"
        for m1, d1, m2, d2 in re.findall(r"<h5[^>]*>\s*([A-Z][a-z]+(?:/[A-Z][a-z]+)?) (\d+)"
                                         r"(?:-(?:([A-Z][a-z]+) )?(\d+))?\s+"
                                         r"Meeting - \d{4}", page):
            days.append(_last_day(m2 or m1, d2 or d1, year))
    else:
        return _one_per_decision(days)
    page = _fetch(f"{FED}fomccalendars.htm")
    for block in re.split(r'<h4><a id="\d+">', page)[1:]:
        y = int(block[:4])
        if y < year or y > last:
            continue
        pairs = re.findall(r'fomc-meeting__month[^>]*><strong>([^<]+)</strong>.*?'
                           r'fomc-meeting__date[^>]*>([^<]+)<', block, flags=re.S)
        for month, d in pairs:
            d = d.strip().rstrip("*").strip()
            if re.fullmatch(r"\d+(-\d+)?", d):     # skips "(unscheduled)", "(notation vote)", "(cancelled)"
                days.append(_last_day(month, d, y))
    return _one_per_decision(days)


def _one_per_decision(days):
    """Sorted meeting days, with back-to-back days (2003-09-15 and -16 are listed as two
    meetings) kept once, as the later day, when the decision was announced."""
    days = pd.DatetimeIndex(sorted(set(days)))
    return days[np.append(np.diff(days.values) > np.timedelta64(1, "D"), True)]


RANGE_START = pd.Timestamp("2008-12-16")   # the Fed's target became a range


def fed_decisions(target, end=END):
    """Fed decision days since February 1994, from the announcement dates (not the dates
    the new target took effect). Columns: date, kind, change_bp.
      scheduled    every scheduled FOMC meeting's last day, with the target change it
                   decided (0 = no change). Since 2008-12-16 a new target range takes
                   effect the day after the announcement, so a change up to 3 days after a
                   meeting belongs to that meeting.
      unscheduled  a change on any other day (a surprise between meetings). Since
                   2008-12-16 it was announced the day before it took effect (2020-03-15, a
                   Sunday, first reached the market on Monday 2020-03-16)."""
    sched = scheduled_fomc_days()
    sched = sched[sched <= pd.Timestamp(end)]
    t = target["Close"]
    t = t[t.index >= pd.Timestamp(FOMC_FIRST, 1, 1)]
    change = (t.diff() * 100).round().fillna(0)
    by_meeting = dict.fromkeys(sched, 0)
    surprise = []
    for day, c in change[change != 0].items():
        before = sched[(sched <= day) & (sched >= day - pd.Timedelta(days=3))]
        if len(before):
            by_meeting[before[-1]] += int(c)
        else:
            surprise.append((day - pd.Timedelta(days=1) if day > RANGE_START else day, int(c)))
    rows = [(d, "scheduled", c) for d, c in by_meeting.items()] + [(d, "unscheduled", c) for d, c in surprise]
    return pd.DataFrame(rows, columns=["date", "kind", "change_bp"]).sort_values("date")


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
    recessions().to_csv(outdir / "us_recessions_monthly.csv", index=False)
    print("saved us_recessions_monthly.csv")
    long = dow_jones(1896, pd.Timestamp(END).year)       # one download; the short file is a slice
    long[(long.index.year >= 1927) & (long.index.year <= 1952)].to_csv(outdir / "dow_jones_daily_1927_1952.csv")
    long[(long.index >= DOW_START) & (long.index <= END)].to_csv(outdir / "dow_jones_daily_from_1896.csv")
    print("saved dow_jones_daily_1927_1952.csv and dow_jones_daily_from_1896.csv")
    save_factors(outdir)


def save_factors(outdir):
    """The outside factors (commodities, rates, the dollar) and the Fed's decision days."""
    outdir = Path(outdir)
    (outdir / "factors").mkdir(parents=True, exist_ok=True)
    for name, series in FACTORS_FRED.items():
        fred(series).rename("Close").to_frame().to_csv(outdir / "factors" / name)
        print("saved factors/" + name)
    for name, ticker in FACTORS_YAHOO.items():
        daily_closes(ticker, STOCK_START).to_csv(outdir / "factors" / name)
        print("saved factors/" + name)
    target = fed_funds_target()
    target.to_csv(outdir / "factors" / "fed_funds_target.csv")
    fed_decisions(target).to_csv(outdir / "fed_decisions.csv", index=False)
    print("saved factors/fed_funds_target.csv and fed_decisions.csv")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--factors"]:             # only the outside factors and Fed days
        save_factors(args[1] if len(args) > 1 else Path(__file__).resolve().parent)
    else:
        main(args[0] if args else Path(__file__).resolve().parent)
