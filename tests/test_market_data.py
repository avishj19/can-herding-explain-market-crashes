"""Tests for data.py and the downloaded data files.  Run: python3 -m pytest tests/ -q

Every test states what "correct" means, and uses inputs whose answer we know
independently (math or published facts), across several random seeds.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, SEEDS
from herding import market_data as data



# --------------------------------------------------------------- real data
# Spec: data files are clean and match known historical values.

@pytest.fixture(scope="module")
def spx():
    return pd.read_csv(ROOT / "data/spx_daily.csv", parse_dates=["Date"], index_col="Date")


def test_spx_clean(spx):
    assert spx.index.is_monotonic_increasing and not spx.index.duplicated().any()
    assert spx["Close"].notna().all() and (spx["Close"] > 0).all()


def test_spx_black_monday_known_value(spx):
    # Published: S&P 500 closed at 282.70 on 1987-10-16 and 224.84 on 1987-10-19 (-20.47%)
    assert spx.loc["1987-10-16", "Close"] == pytest.approx(282.70, abs=0.01)
    assert spx.loc["1987-10-19", "Close"] == pytest.approx(224.84, abs=0.01)


def test_spx_no_large_gaps(spx):
    gaps = spx.index.to_series().diff().dt.days
    assert gaps.max() <= 7, f"largest gap {gaps.max()} days at {gaps.idxmax().date()}"


def test_vix_known_peak():
    vix = pd.read_csv(ROOT / "data/vix_daily.csv", parse_dates=["Date"], index_col="Date")
    # Published record close: 82.69 on 2020-03-16
    assert vix["Close"].idxmax().date().isoformat() == "2020-03-16"
    assert vix["Close"].max() == pytest.approx(82.69, abs=0.01)


def test_recessions_match_nber():
    rec = pd.read_csv(ROOT / "data/us_recessions_monthly.csv", parse_dates=["observation_date"])
    rec = rec.set_index("observation_date")["USREC"]
    for month in ["2008-06-01", "2020-04-01", "1974-06-01"]:
        assert rec[month] == 1
    for month in ["2019-06-01", "1995-06-01"]:
        assert rec[month] == 0


def test_load_spx_missing_file_explains_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="yfinance"):
        data.load_spx(tmp_path / "nope.csv")


def test_load_spx_rejects_bad_prices(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("Date,Close\n2020-01-02,100\n2020-01-03,-5\n2020-01-06,101\n")
    with pytest.raises(ValueError, match="positive"):
        data.load_spx(f)


def test_load_spx_real_file_ok():
    spx = data.load_spx(ROOT / "data/spx_daily.csv")
    assert len(spx) > 19000 and spx.index.is_monotonic_increasing



def test_spx_returns_are_log_returns():
    spx = data.load_spx()
    r = data.spx_returns(spx)
    assert len(r) == len(spx) - 1 and r.notna().all()
    assert r.iloc[0] == pytest.approx(np.log(spx["Close"].iloc[1] / spx["Close"].iloc[0]))


def test_black_monday_value():
    # Published: -20.47% on 1987-10-19 -> log return ln(224.84/282.70) = -0.2290
    assert data.black_monday(data.spx_returns()) == pytest.approx(np.log(224.84 / 282.70), abs=1e-4)


def test_black_monday_missing_date_explains_fix():
    r = data.spx_returns()
    with pytest.raises(ValueError, match="Black Monday"):
        data.black_monday(r.drop(data.BLACK_MONDAY))
