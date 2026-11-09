"""Real-market data, loaded with checks so a bad file fails loudly.

All files come from data/download_market_data.py and end on the same day (2026-10-30).

Markets (daily closes from Yahoo Finance):
    spx       S&P 500, 1950-today   (the calibration target)
    spx_1928  S&P 500, 1928-today   (adds the 1929 crash and the Great Depression;
              Yahoo omits Saturday sessions before 1952, and early prices are coarser)
    ndx       Nasdaq-100, 1985-today
Dow Jones 1927-1952 (MeasuringWorth): only to find wrong closes in spx_1928 (BAD_CLOSES).
Stocks: data/stocks/<TICKER>.csv, large S&P 500 companies whose market beta
(how much they move per 1% market move, 2005-2025) is between 0.9 and 1.1.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent      # the project folder
DATA = ROOT / "data"
SPX = DATA / "sp500_daily.csv"
MARKETS = {"spx": SPX, "spx_1928": DATA / "sp500_daily_from_1928.csv", "ndx": DATA / "nasdaq100_daily.csv"}
STOCKS = DATA / "stocks"
BLACK_MONDAY = pd.Timestamp("1987-10-19")
DOW = DATA / "dow_jones_daily_1927_1952.csv"     # MeasuringWorth; only to check the early S&P

# Single wrong closes in Yahoo's early S&P file, found by find_bad_closes against the Dow
# (November 2026): each "moves" the S&P 3-10% against the Dow and is undone the next day.
# S&P's own record agrees (none is among its 20 largest daily moves; Wikipedia, "List of
# largest daily changes in the S&P 500"). returns() leaves these closes out, so the next
# day's return runs from the last good close. 1934-05-17 and 1938-01-06 are the weakest:
# there the Dow made the larger move, so either series could have the wrong day.
BAD_CLOSES = {"spx_1928": ("1930-03-31", "1931-01-07", "1934-05-17", "1935-04-16", "1935-05-16",
                           "1935-08-16", "1935-12-16", "1936-09-16", "1938-01-06", "1938-10-10",
                           "1946-09-03")}


def load_prices(path):
    """Daily closes from a CSV with Date and Close columns, checked: sorted,
    unique dates, positive prices."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Download every data file with: "
                                f"python3 data/download_market_data.py")
    missing = {"Date", "Close"} - set(pd.read_csv(path, nrows=0).columns)
    if missing:
        raise ValueError(f"{path}: needs a Date and a Close column, missing {sorted(missing)}")
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    close = pd.to_numeric(df["Close"], errors="coerce")       # text -> NaN
    if close.isna().any():
        raise ValueError(f"{path}: every Close price must be a number (found a blank or text)")
    if not np.isfinite(close).all():
        raise ValueError(f"{path}: every Close price must be finite (found inf)")
    if not (close > 0).all():
        raise ValueError(f"{path}: every Close price must be positive (found <= 0)")
    df["Close"] = close
    if not df.index.is_monotonic_increasing or df.index.duplicated().any():
        raise ValueError(f"{path}: dates must be sorted and unique")
    return df


def load_spx(path=SPX):
    """Daily S&P 500 closes since 1950 (the calibration target)."""
    return load_prices(path)


def _log_returns(df):
    return np.log(df["Close"]).diff().dropna()


def spx_returns(spx=None):
    """Daily log returns of the S&P 500, as a Series indexed by the later date."""
    return _log_returns(load_spx() if spx is None else spx)


def returns(market):
    """Daily log returns of a named market: 'spx', 'spx_1928' or 'ndx' (without its
    known bad closes, BAD_CLOSES)."""
    if market not in MARKETS:
        raise ValueError(f"unknown market {market!r}; choose from {sorted(MARKETS)}")
    prices = load_prices(MARKETS[market])
    return _log_returns(prices.drop(pd.DatetimeIndex(BAD_CLOSES.get(market, ()))))


def find_bad_closes(close, reference, gap=0.03, agree=0.015):
    """Days whose close looks wrong when checked against another index of the same market:
    on the day, the two daily log moves differ by more than `gap` (0.03 = 3 points, over
    5x their usual difference of about 0.55 points in 1928-1952); the next day they differ
    by more than `gap` the opposite way; and over the two days together they agree within
    `agree`. A real crash moves both indexes and stays, so it is not flagged."""
    both = close.index.intersection(reference.index)
    d = (np.log(close[both]).diff() - np.log(reference[both]).diff()).values
    bad = [both[i] for i in range(1, len(d) - 1)
           if abs(d[i]) > gap and abs(d[i + 1]) > gap and np.sign(d[i]) != np.sign(d[i + 1])
           and abs(d[i] + d[i + 1]) < agree]
    return pd.DatetimeIndex(bad)


def stock_tickers():
    """Tickers of the saved market-beta-matched stocks."""
    return sorted(pd.read_csv(STOCKS / "market_beta.csv")["ticker"])


def stock_returns(ticker):
    """Daily log returns of one saved stock (split-adjusted, not dividend-adjusted,
    the same as the index files; see data/stocks/README.md)."""
    if ticker not in stock_tickers():
        raise ValueError(f"unknown stock {ticker!r}; saved stocks: {stock_tickers()}")
    return _log_returns(load_prices(STOCKS / f"{ticker}.csv")).rename(ticker)


def black_monday(returns):
    """Log return on 1987-10-19 (the record one-day crash)."""
    if BLACK_MONDAY not in returns.index:
        raise ValueError("the S&P 500 data has no 1987-10-19 (Black Monday); re-download it")
    return float(returns.loc[BLACK_MONDAY])
