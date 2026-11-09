"""Real-market data, loaded with checks so a bad file fails loudly.

All files come from data/download_market_data.py and end on the same day (2026-10-30).

Markets (daily closes from Yahoo Finance):
    spx       S&P 500, 1950-today   (the calibration target)
    spx_1928  S&P 500, 1928-today   (adds the 1929 crash and the Great Depression;
              Yahoo omits Saturday sessions before 1952, and early prices are coarser)
    ndx       Nasdaq-100, 1985-today
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
    """Daily log returns of a named market: 'spx', 'spx_1928' or 'ndx'."""
    if market not in MARKETS:
        raise ValueError(f"unknown market {market!r}; choose from {sorted(MARKETS)}")
    return _log_returns(load_prices(MARKETS[market]))


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
