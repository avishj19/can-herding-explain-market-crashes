"""Tests for herding/factors.py, the outside-factor data files, the Fed decision days, and
scripts/outside_news_figures.py (Figs 16-18).

Every test states what "correct" means and checks it against an answer found another way:
made-up data with a known answer (an exact line, a mirror image), a property least squares
must have (residuals uncorrelated with every regressor), or a published historical fact.
"""
import importlib.util

import numpy as np
import pandas as pd
import pytest
from matplotlib.image import imread

from conftest import ROOT, SEEDS
from herding import factors, market_data, stats


def downloader():
    spec = importlib.util.spec_from_file_location("download", ROOT / "data/download_market_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def spx():
    return market_data.spx_returns()


# --------------------------------------------------------------- data files
# Spec: each file is clean, ends on the project's fixed day, and matches known history.

def test_factor_files_are_clean_and_end_on_the_fixed_day():
    end = pd.Timestamp(downloader().END)
    for name, f in factors.FACTORS.items():
        prices = market_data.load_prices(factors.FACTOR_DIR / f)        # checks sorted, unique, > 0
        assert prices.index[-1] <= end, name
        assert (end - prices.index[-1]).days <= 4, name                 # at most a long weekend early


def test_brent_known_prices():
    # EIA: Brent's record close was $143.95 on 2008-07-03; it fell to $9.12 on 2020-04-21
    brent = market_data.load_prices(factors.FACTOR_DIR / "oil_brent.csv")["Close"]
    assert brent["2008-07-03"] == pytest.approx(143.95)
    assert brent.idxmax() == pd.Timestamp("2008-07-03")
    assert brent["2020-04-21"] == pytest.approx(9.12)


def test_ten_year_yield_known_values():
    # The 10-year yield's all-time high was 15.84% (1981-09-30); it fell 0.75 points the
    # day after Black Monday as money fled to bonds
    y = market_data.load_prices(factors.FACTOR_DIR / "treasury_10y_yield.csv")["Close"]
    assert y.idxmax() == pd.Timestamp("1981-09-30") and y.max() == pytest.approx(15.84)
    assert factors.changes("rates")["1987-10-20"] == pytest.approx(-0.75)


def test_fed_target_known_moves():
    # Fed history: cut to a 0-0.25% range on 2008-12-16; to 0-0.25% again on 2020-03-15
    t = market_data.load_prices(factors.FACTOR_DIR / "fed_funds_target.csv")["Close"]
    assert t["2008-12-15"] == pytest.approx(1.0) and t["2008-12-16"] == pytest.approx(0.25)
    assert t["2020-03-16"] == pytest.approx(0.25) and t["2020-03-13"] == pytest.approx(1.25)


# --------------------------------------------------------------- Fed decision days
def test_surprise_moves_are_exactly_the_feds_intermeeting_moves():
    # The Fed's moves between scheduled meetings since 1994 (federalreserve.gov, "Open market
    # operations" history), dated by announcement, typed in here by hand
    known = {"1994-04-18": 25, "1998-10-15": -25, "2001-01-03": -50, "2001-04-18": -50,
             "2001-09-17": -50, "2008-01-22": -75, "2008-10-08": -50, "2020-03-03": -50,
             "2020-03-15": -100}
    d = factors.decisions("unscheduled")
    assert dict(zip(d["date"].dt.strftime("%Y-%m-%d"), d["change_bp"])) == known


def test_scheduled_meetings_known_decisions():
    d = factors.decisions("scheduled").set_index("date")["change_bp"]
    assert d["2008-12-16"] == -75             # 1.00% -> 0-0.25%
    assert d["2022-06-15"] == 75              # the first 0.75-point hike since 1994
    assert d["2023-09-20"] == 0               # a pause
    assert d["2006-01-31"] == 25


def test_scheduled_meetings_are_weekdays_and_eight_a_year():
    # The FOMC holds 8 scheduled meetings a year. Exceptions: March 2020 was cancelled
    # (Covid), and the data end on 2026-10-30, before the December 2026 meeting.
    d = factors.decisions("scheduled")["date"]
    assert (d.dt.dayofweek < 5).all()
    per_year = d.groupby(d.dt.year).size()
    expected = pd.Series(8, index=per_year.index)
    expected[[2020, 2026]] = 7
    pd.testing.assert_series_equal(per_year, expected, check_names=False)
    assert d.is_unique


def test_every_target_change_is_assigned_to_exactly_one_decision():
    # The changes in the target series must add up to the changes in the decision file
    t = market_data.load_prices(factors.FACTOR_DIR / "fed_funds_target.csv")["Close"]
    t = t[t.index >= "1994-01-01"]
    total = round((t.iloc[-1] - t.iloc[0]) * 100)
    assert factors.decisions()["change_bp"].sum() == total


def test_decisions_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        factors.decisions("emergency")


# The rules that build fed_decisions.csv, on made-up targets and meetings
@pytest.fixture
def dm(monkeypatch):
    mod = downloader()
    monkeypatch.setattr(mod, "scheduled_fomc_days",
                        lambda: pd.DatetimeIndex(["2005-03-22", "2005-05-03", "2015-12-16", "2016-03-16"]))
    return mod


def target(changes):
    days = pd.bdate_range("2005-01-03", "2016-06-30")
    rate = pd.Series(1.0, index=days)
    for day, bp in changes:
        rate[rate.index >= day] += bp / 100
    return rate.rename("Close").to_frame()


def test_fed_decisions_same_day_change_before_2008(dm):
    d = dm.fed_decisions(target([("2005-03-22", 25)]))
    assert d.set_index("date").loc["2005-03-22", "change_bp"] == 25
    assert (d["kind"] == "unscheduled").sum() == 0


def test_fed_decisions_next_day_change_after_2008_belongs_to_the_meeting(dm):
    # Since 2008-12-16 a new range takes effect the day after the announcement
    d = dm.fed_decisions(target([("2015-12-17", 25)]))
    assert d.set_index("date").loc["2015-12-16", "change_bp"] == 25
    assert (d["kind"] == "unscheduled").sum() == 0


def test_fed_decisions_surprise_is_dated_by_its_announcement(dm):
    d = dm.fed_decisions(target([("2005-04-12", -50), ("2016-01-20", -25)]))
    s = d[d["kind"] == "unscheduled"].set_index("date")["change_bp"]
    assert s.to_dict() == {pd.Timestamp("2005-04-12"): -50, pd.Timestamp("2016-01-19"): -25}


def test_last_day_of_meeting_parsing():
    mod = downloader()
    assert mod._last_day("March", "17-18", 2026) == pd.Timestamp("2026-03-18")
    assert mod._last_day("Oct/Nov", "31-1", 2023) == pd.Timestamp("2023-11-01")
    assert mod._last_day("September", "20", 2005) == pd.Timestamp("2005-09-20")


# --------------------------------------------------------------- changes / table
def test_changes_are_log_changes_or_yield_differences():
    oil = market_data.load_prices(factors.FACTOR_DIR / "oil_brent.csv")["Close"]
    assert factors.changes("oil")["2020-04-21"] == pytest.approx(np.log(9.12 / 17.36))
    y = market_data.load_prices(factors.FACTOR_DIR / "treasury_10y_yield.csv")["Close"]
    assert factors.changes("rates").values == pytest.approx(np.diff(y.values))
    assert len(factors.changes("oil")) == len(oil) - 1


def test_changes_rejects_unknown_factor():
    with pytest.raises(ValueError, match="unknown factor"):
        factors.changes("silver")


def test_table_keeps_only_days_every_series_has(spx):
    df = factors.table(spx, ["oil", "dollar"])
    assert not df.isna().any().any()
    assert df.index.isin(spx.index).all()
    assert df.index.isin(factors.changes("dollar").index).all()
    assert df.index[0] >= pd.Timestamp("2006-01-03")          # the dollar series starts in 2006


def test_table_needs_factors_and_enough_days(spx):
    with pytest.raises(ValueError, match="at least one"):
        factors.table(spx, [])
    with pytest.raises(ValueError, match="500"):
        factors.table(spx.iloc[-300:], ["oil"])


# --------------------------------------------------------------- with_leads_lags
def test_leads_and_lags_shift_the_right_way():
    df = pd.DataFrame({"x": [1.0, 2, 3, 4, 5]}, index=pd.bdate_range("2020-01-06", periods=5))
    out = factors.with_leads_lags(df, ["x"], 1)
    assert list(out.columns) == ["x[-1]", "x[+0]", "x[+1]"]
    assert out["x[-1]"].tolist() == [1, 2, 3]       # yesterday
    assert out["x[+0]"].tolist() == [2, 3, 4]
    assert out["x[+1]"].tolist() == [3, 4, 5]       # tomorrow
    assert factors.with_leads_lags(df, ["x"], 0)["x[+0]"].tolist() == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_leads_and_lags_rejects_bad_lags(bad):
    df = pd.DataFrame({"x": [1.0, 2, 3]})
    with pytest.raises(ValueError, match="lags"):
        factors.with_leads_lags(df, ["x"], bad)


# --------------------------------------------------------------- regress
@pytest.mark.parametrize("seed", SEEDS)
def test_regress_recovers_an_exact_line(seed):
    g = np.random.default_rng(seed)
    X = g.normal(size=(200, 2))
    beta, icept, r2, resid = factors.regress(2 + 3 * X[:, 0] - X[:, 1], X)
    assert beta == pytest.approx([3, -1]) and icept == pytest.approx(2)
    assert r2 == pytest.approx(1) and np.abs(resid).max() < 1e-10


@pytest.mark.parametrize("seed", SEEDS)
def test_regress_r2_matches_the_signal_share(seed):
    # y = x + noise with equal variances: R^2 should be about 1/2
    g = np.random.default_rng(seed)
    x = g.normal(size=800_000)
    _, _, r2, resid = factors.regress(x + g.normal(size=x.size), x)
    assert r2 == pytest.approx(0.5, abs=0.0036)               # measured: bias 0, sd 0.0007
    assert abs(np.corrcoef(resid, x)[0, 1]) < 1e-10          # least squares: residual _|_ regressor


def test_regress_rejects_bad_input():
    with pytest.raises(ValueError, match="constant"):
        factors.regress(np.ones(10), np.arange(10.0))
    with pytest.raises(ValueError, match="one row"):
        factors.regress(np.arange(10.0), np.arange(9.0))
    with pytest.raises(ValueError, match="more rows"):
        factors.regress(np.arange(3.0), np.ones((3, 2)))
    with pytest.raises(ValueError, match="NaN"):
        factors.regress(np.arange(5.0), np.array([1, 2, np.nan, 4, 5]))


# --------------------------------------------------------------- remove / betas
def test_remove_leaves_nothing_the_factors_can_explain(spx):
    names = ["oil", "rates"]
    resid, r2 = factors.remove(spx, names)
    X = factors.with_leads_lags(factors.table(spx, names), names).loc[resid.index]
    for col in X:
        assert abs(np.corrcoef(resid, X[col])[0, 1]) < 1e-8, col
    assert resid.mean() == pytest.approx(spx.loc[resid.index].mean())
    assert 0 < r2 < 1


def test_more_factors_never_explain_less(spx):
    # Nested least squares: adding regressors can only raise R^2 (on the same days)
    small = factors.regress(*_design(spx, ["oil", "gold", "copper", "dollar", "rates"], ["oil"]))[2]
    big = factors.regress(*_design(spx, ["oil", "gold", "copper", "dollar", "rates"]))[2]
    assert big >= small


def _design(spx, names, use=None):
    df = factors.table(spx, names)
    X = factors.with_leads_lags(df, use or names)
    return df["spx"].loc[X.index].values, X.values


def test_betas_agree_with_numpy_and_with_each_other(spx):
    b = factors.betas(spx, ["copper", "dollar"])
    df = factors.table(spx, ["copper", "dollar"])
    for n in ("copper", "dollar"):
        c = np.corrcoef(df["spx"], df[n])[0, 1]
        assert b[n]["corr"] == pytest.approx(c)
        assert b[n]["spx_beta"] == pytest.approx(np.polyfit(df[n], df["spx"], 1)[0])
        assert b[n]["spx_beta"] * b[n]["factor_beta"] == pytest.approx(c ** 2)   # the two slopes multiply to r^2
    assert b["dollar"]["corr"] < 0 < b["copper"]["corr"]


# --------------------------------------------------------------- rolling / crash corr
def test_rolling_corr_of_a_series_with_its_multiple():
    days = pd.bdate_range("2000-01-03", periods=600)
    a = pd.Series(np.random.default_rng(0).normal(size=600), index=days)
    assert factors.rolling_corr(a, 2 * a, 50).values == pytest.approx(1)
    assert factors.rolling_corr(a, -a, 50).values == pytest.approx(-1)
    assert len(factors.rolling_corr(a, a, 50)) == 600 - 49
    with pytest.raises(ValueError, match="window"):
        factors.rolling_corr(a, a, 2)


def test_crash_corr_sees_a_link_that_exists_only_in_crashes():
    g = np.random.default_rng(1)
    days = pd.bdate_range("2000-01-03", periods=4000)
    r = pd.Series(g.normal(size=4000), index=days)
    f = pd.Series(g.normal(size=4000), index=days)
    worst = r <= r.quantile(0.05)
    f[worst] = r[worst]                       # moves exactly with the S&P, but only on its worst days
    on_worst, other = factors.crash_corr(r, f.rename("f"))
    assert on_worst == pytest.approx(1)
    assert abs(other) < 0.05


# --------------------------------------------------------------- on_days / event_days
def test_on_days_maps_holidays_to_the_next_trading_day():
    idx = pd.DatetimeIndex(["2020-03-12", "2020-03-13", "2020-03-16", "2020-03-17"])
    m = factors.on_days(idx, ["2020-03-15", "2020-03-12", "2020-03-20"])   # Sunday, Thursday, after the end
    assert m.tolist() == [True, False, True, False]
    with pytest.raises(ValueError, match="sorted"):
        factors.on_days(idx[::-1], ["2020-03-12"])


def test_event_days_on_the_biggest_days():
    g = np.random.default_rng(2)
    r = pd.Series(g.standard_t(3, 10_000))
    a = np.abs(r - r.mean())
    mask = (a >= np.quantile(a, 0.99)).values                # exactly the biggest 1%
    e = factors.event_days(r, mask)
    assert e["big_share"] == 1 and e["chance_share"] == pytest.approx(0.01, abs=1e-3)
    assert e["size_ratio"] == pytest.approx(a[mask].mean() / a[~mask].mean())
    assert e["kurtosis_without"] < e["kurtosis_all"]          # the fattest days are gone
    assert e["kurtosis_without"] == pytest.approx(stats.kurtosis(r.values[~mask]))


def test_event_days_rejects_bad_masks():
    r = pd.Series(np.random.default_rng(0).normal(size=100))
    for bad in (np.zeros(100, bool), np.ones(100, bool)):
        with pytest.raises(ValueError, match="some days"):
            factors.event_days(r, bad)
    with pytest.raises(ValueError, match="one True/False"):
        factors.event_days(r, np.zeros(99, bool))


def test_r2_by_regime_is_three_shares(spx):
    out = factors.r2_by_regime(spx, ["oil", "copper"])
    assert list(out) == ["calm", "middle", "wild"]
    assert all(0 <= v <= 1 for v in out.values())


# --------------------------------------------------------------- the figures
def test_figures_draw_and_report_their_own_numbers(spx, tmp_path):
    import outside_news_figures as onf
    from multiprocessing import Pool
    onf.plots.style()
    f16 = onf.fig16_comovement(spx, tmp_path / "16.png")
    assert f16["regimes"] == factors.r2_by_regime(spx, onf.ALL)
    with Pool(3) as pool:
        f17 = onf.fig17_without_outside_news(spx, pool, tmp_path / "17.png", seeds=(1000, 1001, 1002))
        f18 = onf.fig18_moments(spx, pool, tmp_path / "18.png", seeds=(1000, 1001, 1002), draws=50)
    resid, r2 = factors.remove(spx, onf.WINDOW)
    assert f17["r2"] == pytest.approx(r2) and f17["days"] == len(resid)
    assert f17["resid"]["kurtosis"] == pytest.approx(stats.kurtosis(resid.values))
    assert f18["moments"]["skew"][0] == pytest.approx(stats.skewness(spx.values))
    for name in ("16", "17", "18"):
        assert imread(tmp_path / f"{name}.png").shape[0] > 200


def test_back_to_back_meeting_days_are_one_decision():
    mod = downloader()
    days = mod._one_per_decision([pd.Timestamp("2003-09-16"), pd.Timestamp("2003-09-15"),
                                  pd.Timestamp("2003-10-28"), pd.Timestamp("2003-10-28")])
    assert list(days) == [pd.Timestamp("2003-09-16"), pd.Timestamp("2003-10-28")]


# --------------------------------------------------------------- the downloader's fetch
# Spec: a transfer curl reports as failed (e.g. cut off by a time limit) is never returned as
# a page, even if part of the page arrived; a 4xx/5xx answer raises HTTPError at once.
class FakeRun:
    def __init__(self, results):
        self.results, self.calls = list(results), 0

    def __call__(self, cmd, capture_output, text):
        self.calls += 1
        code, out = self.results.pop(0)
        return type("Done", (), dict(returncode=code, stdout=out))()


def test_fetch_retries_a_cut_off_transfer_and_never_returns_it(monkeypatch):
    import urllib.error
    mod = downloader()
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/curl")
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake = FakeRun([(28, "Date,Close\n1896-05-26,40.94\n"), (0, "Date,Close\n1896-05-26,40.94\n2026-10-30,1\n\n200")])
    monkeypatch.setattr("subprocess.run", fake)
    assert mod._fetch("https://example.org/x").endswith("2026-10-30,1\n")
    assert fake.calls == 2
    monkeypatch.setattr("subprocess.run", FakeRun([(28, "partial")] * 3))
    with pytest.raises(RuntimeError, match="empty"):
        mod._fetch("https://example.org/x", tries=3)
    monkeypatch.setattr("subprocess.run", FakeRun([(0, "Not found\n404")]))
    with pytest.raises(urllib.error.HTTPError):
        mod._fetch("https://example.org/missing")


def test_fomc_days_reproduce_the_published_pre_fomc_drift(spx):
    # Lucca & Moench (2015, J. Finance 70, 329-371; NY Fed Liberty Street summary): over
    # 1994-2011 the S&P earned about 33 basis points on scheduled FOMC days and about 1 on
    # other days. An independent check that fed_decisions.csv has the right days.
    x = spx["1994-01-01":"2011-03-31"]
    m = factors.on_days(x.index, factors.decisions("scheduled")["date"])
    assert 1e4 * x[m].mean() == pytest.approx(33, abs=4)
    assert 1e4 * x[~m].mean() == pytest.approx(1, abs=1.5)
