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


# ------------------------------------- more markets: 1928 S&P, Nasdaq-100, stocks
# Spec: every series loads through the same checks and matches published history.

def test_spx_1928_great_crash_and_depression():
    r = data.returns("spx_1928")
    s = data.load_prices(data.MARKETS["spx_1928"])["Close"]
    # Published: Oct 28 1929 -12.94%, Oct 29 1929 -10.16%
    assert np.expm1(r.loc["1929-10-28"]) == pytest.approx(-0.1294, abs=0.0005)
    assert np.expm1(r.loc["1929-10-29"]) == pytest.approx(-0.1016, abs=0.0005)
    # Sept 1929 peak 31.86 -> June 1932 low 4.40: an 86% fall
    assert s.loc["1929"].max() == pytest.approx(31.86, abs=0.01)
    assert s.loc["1932"].min() == pytest.approx(4.40, abs=0.01)


def test_spx_1928_agrees_with_1950_file_where_they_overlap():
    a = data.returns("spx_1928").loc["1950-02-01":"2026-01-01"]
    b = data.returns("spx").loc["1950-02-01":"2026-01-01"]
    assert len(a) == len(b) and np.allclose(a.values, b.values, atol=1e-6)


def test_ndx_dot_com_crash():
    # Published: Nasdaq-100 closing peak 4704.73 on 2000-03-27; the dot-com bust took
    # it down about 83% to a bottom in October 2002 (closing low ~805; ~795 intraday)
    s = data.load_prices(data.MARKETS["ndx"])["Close"]
    assert s.loc["2000"].max() == pytest.approx(4704.73, abs=0.5)
    assert s.loc["2000"].idxmax().date().isoformat() == "2000-03-27"
    low = s.loc["2000-03-27":"2003-12-31"]
    assert low.idxmin().strftime("%Y-%m") == "2002-10"
    assert 1 - low.min() / s.loc["2000"].max() == pytest.approx(0.83, abs=0.01)


def test_known_closures_are_the_only_long_gaps():
    # 1933 Bank Holiday (12 days) and the 9/11 closure (7 days); nothing else > 5 days
    for name, allowed in [("spx_1928", {"1933-03-15", "2001-09-17"}),
                          ("ndx", {"2001-09-17"})]:
        idx = data.load_prices(data.MARKETS[name]).index.to_series()
        gaps = idx.diff().dt.days
        long = {d.date().isoformat() for d in gaps[gaps > 5].index}
        assert long <= allowed, f"{name}: unexpected gaps {long - allowed}"


def test_stocks_have_market_beta_near_one_recomputed_independently():
    table = pd.read_csv(ROOT / "data/stocks/market_beta.csv")
    assert len(table) >= 15 and table.market_beta.between(0.9, 1.1).all()
    m = data.returns("spx").loc["2005-01-01":"2025-12-31"]
    for t, beta in table.itertuples(index=False):
        x = pd.concat([data.stock_returns(t), m], axis=1, join="inner").dropna()
        x = x.loc["2005-01-01":"2025-12-31"]
        slope = np.polyfit(x.iloc[:, 1], x.iloc[:, 0], 1)[0]   # regression slope = beta
        assert slope == pytest.approx(beta, abs=0.02), t


def test_stock_list_matches_files():
    names = set(data.stock_tickers())
    files = {p.stem for p in (ROOT / "data/stocks").glob("*.csv")} - {"market_beta"}
    assert names == files


def test_unknown_market_name_is_rejected():
    with pytest.raises(ValueError, match="nasdaq"):
        data.returns("nasdaq")


def test_stocks_are_price_only_like_the_indexes():
    # Bug fix: indexes are price-only, so stocks must be too (split-adjusted,
    # NOT dividend-adjusted). Published: MSFT closed at 116.5625 on 2000-01-03,
    # i.e. 58.28 after its 2003 2-for-1 split. A dividend-adjusted series would
    # be far lower (~35).
    s = data.load_prices(data.STOCKS / "MSFT.csv")["Close"]
    assert s.loc["2000-01-03"] == pytest.approx(58.28, abs=0.01)
    readme = (data.STOCKS / "README.md").read_text()
    assert "not dividend-adjusted" in readme
