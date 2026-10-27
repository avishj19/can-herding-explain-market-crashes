"""Real-market data, loaded with checks so a bad file fails loudly."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent      # the project folder
SPX = ROOT / "data" / "spx_daily.csv"
BLACK_MONDAY = pd.Timestamp("1987-10-19")


def load_spx(path=SPX):
    """Daily S&P 500 closes, checked: sorted, unique dates, positive prices."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it with: python3 -c \"import yfinance as yf; "
            f"yf.download('^GSPC', start='1950-01-01')[['Close']].to_csv('{path}')\"")
    spx = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    close = spx["Close"]
    if close.isna().any() or not (close > 0).all():
        raise ValueError(f"{path}: every Close price must be positive (found NaN or <= 0)")
    if not spx.index.is_monotonic_increasing or spx.index.duplicated().any():
        raise ValueError(f"{path}: dates must be sorted and unique")
    return spx


def spx_returns(spx=None):
    """Daily log returns of the S&P 500, as a Series indexed by the later date."""
    spx = load_spx() if spx is None else spx
    return np.log(spx["Close"]).diff().dropna()


def black_monday(returns):
    """Log return on 1987-10-19 (the record one-day crash)."""
    if BLACK_MONDAY not in returns.index:
        raise ValueError("the S&P 500 data has no 1987-10-19 (Black Monday); re-download it")
    return float(returns.loc[BLACK_MONDAY])
