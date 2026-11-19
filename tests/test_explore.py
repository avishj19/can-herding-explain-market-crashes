"""Tests for herding/explore.py (the science behind the web app).  Run: python3 -m pytest tests/test_explore.py -q

Each test says what "correct" means and checks it against an answer worked out separately
(by hand, from a formula, a published price, or a direct call to model.simulate).
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from herding import explore as e
from herding import market_data, model, stats


# ------------------------------------------------------------------ catalog / saved_prices
# Spec: three groups; every listed asset loads as positive, sorted daily closes.

def test_catalog_groups_and_contents():
    c = e.catalog()
    assert list(c) == ["Stocks", "Index funds", "Commodities"]
    assert c["Stocks"] == market_data.stock_tickers() and len(c["Stocks"]) == 20
    assert c["Index funds"] == ["S&P 500", "Nasdaq-100", "Dow Jones"]
    assert c["Commodities"] == ["Brent oil", "Gold", "Copper"]


@pytest.mark.parametrize("name", sum(e.catalog().values(), []))
def test_every_saved_asset_loads_clean(name):
    close = e.saved_prices(name)
    assert close.name == name and len(close) > e.MIN_DAYS
    assert (close > 0).all() and close.index.is_monotonic_increasing and not close.index.duplicated().any()


def test_sp500_black_monday_published_closes():
    # Published: 282.70 on 1987-10-16 and 224.84 on 1987-10-19
    close = e.saved_prices("S&P 500")
    assert close["1987-10-16"] == pytest.approx(282.70, abs=0.01)
    assert close["1987-10-19"] == pytest.approx(224.84, abs=0.01)


def test_saved_returns_match_the_existing_loaders():
    # two separate code paths must give the same returns, to the last digit
    for name, key in e.INDEXES.items():
        assert np.array_equal(e.log_returns(e.saved_prices(name)).values, market_data.returns(key).values)
    assert np.array_equal(e.log_returns(e.saved_prices("AAPL")).values,
                          market_data.stock_returns("AAPL").values)


def test_commodity_first_close_matches_file():
    raw = pd.read_csv(ROOT / "data/factors/gold.csv")
    assert e.saved_prices("Gold").iloc[0] == raw["Close"].iloc[0]


def test_unknown_asset_rejected():
    with pytest.raises(ValueError, match="unknown asset"):
        e.saved_prices("Bitcoin")


# ------------------------------------------------------------------ clean_ticker
# Spec: upper-case and trim; only letters, digits and . - = ^, 1 to 12 characters.

@pytest.mark.parametrize("text, want", [(" nvda ", "NVDA"), ("brk-b", "BRK-B"), ("^gspc", "^GSPC"),
                                        ("GC=F", "GC=F"), ("A", "A"), ("ABCDEFGHIJKL", "ABCDEFGHIJKL")])
def test_clean_ticker_accepts(text, want):
    assert e.clean_ticker(text) == want


@pytest.mark.parametrize("text", ["", "   ", None, 5, "AAPL; rm -rf", "ABCDEFGHIJKLM", ".ABC",
                                  "AA PL", "AAPL/x", "ÄPPLE"])
def test_clean_ticker_rejects(text):
    with pytest.raises(ValueError):
        e.clean_ticker(text)


# ------------------------------------------------------------------ live_prices (no internet: fake Yahoo)
# Spec: cleaned (numbers > 0, sorted, one per day), named by ticker; too short, empty, bad
# ticker -> ValueError; Yahoo failure -> ConnectionError.

def _fake(n, start="2000-01-03"):
    idx = pd.bdate_range(start, periods=n)
    return lambda t, s: pd.Series(np.linspace(10, 20, n), index=idx)


def test_live_prices_cleans_and_names():
    n = e.MIN_DAYS + 10
    idx = pd.bdate_range("2000-01-03", periods=n)
    s = pd.Series(np.arange(1.0, n + 1), index=idx)
    s.iloc[3] = np.nan
    s.iloc[4] = 0.0
    s.iloc[5] = -1.0
    messy = pd.concat([s.iloc[::-1], s.iloc[:2]])          # reversed, with 2 repeated days
    out = e.live_prices("abc", download=lambda t, st: messy)
    assert out.name == "ABC"
    assert len(out) == n - 3                               # NaN, 0 and -1 gone, repeats gone
    assert out.index.is_monotonic_increasing and not out.index.duplicated().any()
    assert (out > 0).all()


def test_live_prices_length_boundary():
    with pytest.raises(ValueError, match="only 1000 days"):
        e.live_prices("ABC", download=_fake(e.MIN_DAYS))
    assert len(e.live_prices("ABC", download=_fake(e.MIN_DAYS + 1))) == e.MIN_DAYS + 1


def test_live_prices_unknown_ticker():
    with pytest.raises(ValueError, match="no prices found"):
        e.live_prices("ZZZZ", download=lambda t, s: pd.Series(dtype=float))


def test_live_prices_network_error_becomes_connection_error():
    def broken(t, s):
        raise OSError("blocked")
    with pytest.raises(ConnectionError, match="Yahoo"):
        e.live_prices("ABC", download=broken)


def test_live_prices_bad_ticker_never_reaches_yahoo():
    calls = []
    with pytest.raises(ValueError):
        e.live_prices("AAPL; rm", download=lambda t, s: calls.append(t))
    assert calls == []


def test_live_prices_passes_clean_ticker_and_start():
    seen = []

    def spy(t, s):
        seen.append((t, s))
        return _fake(e.MIN_DAYS + 1)(t, s)
    e.live_prices(" msft ", start="2010-01-01", download=spy)
    assert seen == [("MSFT", "2010-01-01")]


# ------------------------------------------------------------------ log_returns
# Spec: log(p_t / p_t-1), indexed by the later day.

def test_log_returns_known_values():
    idx = pd.date_range("2020-01-01", periods=3)
    r = e.log_returns(pd.Series([1.0, math.e, math.e ** 3], index=idx))
    assert list(r.index) == list(idx[1:])
    assert r.values == pytest.approx([1.0, 2.0])


@pytest.mark.parametrize("bad", [[5.0], [], [1.0, 0.0], [1.0, -2.0], [1.0, np.nan], [1.0, np.inf]])
def test_log_returns_rejects(bad):
    with pytest.raises(ValueError):
        e.log_returns(pd.Series(bad, dtype=float))


# ------------------------------------------------------------------ fitted_settings / mix_settings
# Spec: settings exactly as saved; each mix changes only what it should.

def test_fitted_settings_match_file():
    d = json.loads((ROOT / "settings/sp500_calibrated.json").read_text())
    p, burn = e.fitted_settings()
    assert p == d["settings"] and burn == d["burn"] == 1000


def test_mix_settings_each_mix():
    p, _ = e.fitted_settings()
    before = dict(p)
    q, c = e.mix_settings(p, "fundamentalists", 0.7)
    assert q == p and c is False and q is not p
    q, c = e.mix_settings(p, "chartists", 0.7)
    assert c is True and q["a1"] == 0.0 and {k: v for k, v in q.items() if k != "a1"} == \
        {k: v for k, v in p.items() if k != "a1"}
    q, c = e.mix_settings(p, "herding", 0.7)
    assert c is True and q["a1"] == 0.7
    q, _ = e.mix_settings(p, "herding", 0)            # a1 = 0 is allowed (the boundary)
    assert q["a1"] == 0.0
    assert p == before                                 # the caller's dict is untouched


@pytest.mark.parametrize("mix, a1", [("ants", 1.0), ("herding", -0.1), ("herding", np.nan),
                                     ("herding", True), ("herding", "1"), ("herding", np.inf)])
def test_mix_settings_rejects(mix, a1):
    p, _ = e.fitted_settings()
    with pytest.raises(ValueError):
        e.mix_settings(p, mix, a1)


# ------------------------------------------------------------------ run_mix
# Spec: the same numbers as calling model.simulate directly, minus the burn-in days.

@pytest.mark.parametrize("mix, chartists, a1", [("fundamentalists", False, None),
                                                ("herding", True, 1.0844126632402482)])
def test_run_mix_equals_direct_simulation(mix, chartists, a1):
    p, burn = e.fitted_settings()
    r, err = e.run_mix(p, mix, p["a1"], 600, 3, burn)
    q = dict(p) if a1 is None else dict(p, a1=a1)
    direct = model.simulate(q, 600 + burn, seed=3, chartists=chartists)["r"][burn:]
    assert err is None and len(r) == 600 and np.array_equal(r, direct)


def test_run_mix_zero_burn():
    p, _ = e.fitted_settings()
    r, _ = e.run_mix(p, "fundamentalists", 1.0, 500, 1, 0)
    assert np.array_equal(r, model.simulate(p, 500, seed=1, chartists=False)["r"])


@pytest.mark.pinned_seed
def test_run_mix_reports_runaway():
    # measured Nov 2026: no herding, seed 0, runs away as a bubble on day 2735
    p, burn = e.fitted_settings()
    r, err = e.run_mix(p, "chartists", 1.0, 10000, 0, burn)
    assert r is None and isinstance(err, model.RunawayError) and err.day == 2735
    assert "bubble" in str(err)


@pytest.mark.parametrize("days", [499, e.MAX_DAYS + 1, 1000.0, True, "1000"])
def test_run_mix_rejects_days(days):
    p, burn = e.fitted_settings()
    with pytest.raises(ValueError, match="days"):
        e.run_mix(p, "fundamentalists", 1.0, days, 0, burn)


@pytest.mark.parametrize("burn", [-1, 1.5, True])
def test_run_mix_rejects_burn(burn):
    p, _ = e.fitted_settings()
    with pytest.raises(ValueError, match="burn"):
        e.run_mix(p, "fundamentalists", 1.0, 500, 0, burn)


# ------------------------------------------------------------------ bell_big_days
# Spec: n * P(|Z| > size) for a standard normal Z.

def test_bell_big_days_known_values():
    assert e.bell_big_days(1, 0) == 1.0                                  # every day beats 0
    assert e.bell_big_days(1, 1.959963984540054) == pytest.approx(0.05)  # the 95% two-sided cutoff
    assert e.bell_big_days(1, 4) == pytest.approx(6.334248366623996e-05) # tables: P(|Z|>4)
    assert e.bell_big_days(10000, 4) == pytest.approx(10000 * 6.334248366623996e-05)
    assert e.bell_big_days(0, 4) == 0


# ------------------------------------------------------------------ summary
# Spec: each number equals its definition, computed here separately.

def _pattern(n=1000, seed=0):
    g = np.random.default_rng(seed)
    return g.standard_normal(n) * np.repeat(g.uniform(0.5, 2, n // 50 + 1), 50)[:n]


def test_summary_rejects_equal_move_sizes():
    # +-1 alternating: every move is the same size, so there is no tail (and no memory) to measure
    with pytest.raises(ValueError, match="constant"):
        e.summary(np.tile([1.0, -1.0], 300))


def test_summary_matches_definitions():
    r = _pattern()
    s = e.summary(r)
    z = (r - r.mean()) / r.std()
    a = np.abs(r)
    assert s["kurtosis"] == pytest.approx(np.mean(z ** 4) - 3)
    assert s["worst"] == pytest.approx(z.min())
    assert s["big_days"] == int(np.sum(np.abs(z) > 4))
    assert s["memory"] == pytest.approx(np.corrcoef(a[:-20], a[20:])[0, 1])
    assert s["bell_big_days"] == pytest.approx(1000 * math.erfc(4 / math.sqrt(2)))
    k = 50                                                   # Hill by hand: top 5% of 1000
    top = np.sort(a)[::-1]
    assert s["tail"] == pytest.approx(k / np.sum(np.log(top[:k] / top[k])))


def test_summary_counts_planted_big_days():
    r = np.tile([1.0, -1.0], 500)
    r[[10, 400, 800]] = [60.0, -60.0, 60.0]                  # 3 huge days, everything else +-1
    assert e.summary(r)["big_days"] == 3


def test_summary_length_boundary():
    assert e.summary(_pattern(500))["days"] == 500
    with pytest.raises(ValueError, match="500"):
        e.summary(_pattern(499))


# ------------------------------------------------------------------ closest_mix
# Spec: smallest relative gap in tail + memory; runaways skipped; none left -> error.

def _s(tail, memory):
    return {"tail": tail, "memory": memory}


def test_closest_mix_picks_by_tail_and_memory():
    target = _s(3.0, 0.2)
    mixes = {"fundamentalists": _s(6.0, 0.0), "chartists": _s(3.5, 0.3), "herding": _s(2.9, 0.22)}
    assert e.closest_mix(target, mixes) == "herding"
    mixes["herding"] = _s(5.0, 0.0)
    assert e.closest_mix(target, mixes) == "chartists"


def test_closest_mix_hand_computed_gap():
    # gaps: a = |3.3-3|/3 + |0.1-0.2|/0.2 = 0.6;  b = |2.4-3|/3 + |0.2-0.2|/0.2 = 0.2
    assert e.closest_mix(_s(3.0, 0.2), {"a": _s(3.3, 0.1), "b": _s(2.4, 0.2)}) == "b"


def test_closest_mix_memory_floor():
    # target memory 0: the 0.05 floor keeps the gap finite; 0.01 away beats 0.04 away
    assert e.closest_mix(_s(3.0, 0.0), {"a": _s(3.0, 0.04), "b": _s(3.0, 0.01)}) == "b"


def test_closest_mix_skips_runaways():
    assert e.closest_mix(_s(3, .2), {"chartists": None, "fundamentalists": _s(6, 0)}) == "fundamentalists"
    with pytest.raises(ValueError):
        e.closest_mix(_s(3, .2), {"chartists": None})


# ------------------------------------------------------------------ insight
# Spec: the sentence states the counts and the closest mix in plain words.

def test_insight_sentence():
    s = {"big_days": 108, "bell_big_days": 1.22}
    text = e.insight("S&P 500", s, "herding")
    assert "S&P 500 had 108 days" in text and "about 1.2" in text
    assert "about 89 times fewer" in text and "herding mix" in text


def test_insight_no_times_when_close_to_bell():
    text = e.insight("X", {"big_days": 1, "bell_big_days": 0.9}, "fundamentalists")
    assert "times fewer" not in text and "fundamentalists-only mix" in text


# ------------------------------------------------------------------ chart data
# Spec: the points are a subset of the exact curves from stats.py / their definitions.

def test_tail_curve_is_a_thinned_ccdf():
    r = _pattern(5000)
    x, cc = e.tail_curve(r, points=100)
    fx, fc = stats.ccdf(r)
    assert 0 < len(x) <= 100
    assert np.all(np.diff(x) >= 0) and np.all(np.diff(cc) <= 0) and np.all(x > 0.1)
    for xi, ci in zip(x, cc):                       # every point lies on the full curve
        i = np.searchsorted(fx, xi)
        assert fx[i] == xi and fc[i] == ci
    assert x[-1] == fx[-1] and cc[-1] == pytest.approx(1 / 5000)   # the biggest day is kept


def test_histogram_known_density():
    # standardized +-1: half the days in the bin around -1, half around +1; width 0.4
    c, d = e.histogram(np.tile([3.0, 5.0], 300))
    assert d[np.argmin(abs(c - 1.0))] == pytest.approx(1.25)
    assert d[np.argmin(abs(c + 1.0))] == pytest.approx(1.25)
    assert np.sum(d * 0.4) == pytest.approx(1.0)


def test_memory_curve_matches_corrcoef():
    r = _pattern(3000)
    lags, c = e.memory_curve(r, (1, 5, 50))
    a = np.abs(r)
    assert list(lags) == [1, 5, 50]
    assert c == pytest.approx([np.corrcoef(a[:-k], a[k:])[0, 1] for k in (1, 5, 50)])


# ------------------------------------------------------------------ bell_check / tail_check
# Spec: pass when every kurtosis is inside +-5 sqrt(24/n); pass when the mean tail is in [lo, hi].

def test_bell_check_band_and_boundary():
    tol = 5 * math.sqrt(24 / 10000)                      # 0.2449...
    assert e.bell_check([0.0], 10000)["tol"] == pytest.approx(tol)
    assert e.bell_check([0.01, -0.05, 0.2], 10000)["passed"]
    assert not e.bell_check([0.01, tol + 1e-9], 10000)["passed"]
    assert not e.bell_check([-tol], 10000)["passed"]     # the band is open: exactly tol fails
    r = e.bell_check([0.1, -0.3, 0.2], 10000)
    assert r["mean"] == pytest.approx(0.0) and r["worst"] == -0.3


@pytest.mark.parametrize("k, n", [([], 100), ([np.nan], 100), ([0.0], 1), ([0.0], 10.0), ([0.0], True)])
def test_bell_check_rejects(k, n):
    with pytest.raises(ValueError):
        e.bell_check(k, n)


def test_tail_check():
    r = e.tail_check([2.8, 3.0, 3.2])
    assert r["passed"] and r["mean"] == pytest.approx(3.0) and (r["low"], r["high"]) == (2.8, 3.2)
    assert e.tail_check([2.5])["passed"] and e.tail_check([3.5])["passed"]   # ends included
    assert not e.tail_check([2.49])["passed"] and not e.tail_check([5.9, 6.0])["passed"]
    with pytest.raises(ValueError):
        e.tail_check([])
    with pytest.raises(ValueError):
        e.tail_check([3.0], lo=3, hi=3)


# ------------------------------------------------------------------ kirman_match
# Spec: simulated shares sum to 1; exact = kirman.stationary; distance 0 when they agree.

def test_kirman_match_hand_counts():
    from herding import kirman
    sim, exact, tv = e.kirman_match(np.array([0, 2, 2, 4]), 4, 0.5, 0.5)
    assert sim == pytest.approx([0.25, 0, 0.5, 0, 0.25])
    assert exact == pytest.approx(kirman.stationary(4, 0.5, 0.5))
    assert tv == pytest.approx(0.5 * np.abs(sim - exact).sum())


def test_kirman_match_long_run_agrees():
    # 2 million steps: simulation and exact answer overlap almost fully (measured Nov 2026:
    # TV 0.004-0.008 over seeds 0-5, so 0.03 is about 4x the worst)
    from herding import kirman
    k = kirman.simulate(20, 0.05, 0.1, 2_000_000, seed=4)
    _, _, tv = e.kirman_match(k, 20, 0.05, 0.1)
    assert tv < 0.03


def test_kirman_match_rejects_out_of_range():
    with pytest.raises(ValueError):
        e.kirman_match(np.array([0, 5]), 4, 0.5, 0.5)
    with pytest.raises(ValueError):
        e.kirman_match(np.array([], dtype=int), 4, 0.5, 0.5)


# ------------------------------------------------------------------ market_table
# Spec: one row per saved asset, numbers equal to summary() of that asset.

def test_market_table_rows_match_summary():
    t = e.market_table()
    assert list(t["name"]) == sum(e.catalog().values(), [])
    row = t[t["name"] == "Gold"].iloc[0]
    s = e.summary(e.log_returns(e.saved_prices("Gold")).values)
    assert row["group"] == "Commodities" and row["tail"] == s["tail"] and row["big_days"] == s["big_days"]
    assert row["times_bell"] == pytest.approx(s["big_days"] / s["bell_big_days"])


# ------------------------------------------------------------------ gallery_entries
# Spec: every image becomes one entry with its section and the caption text after it.

GALLERY = """# Figure gallery
Intro text, not a caption.

## 0. Start
![Quick](fig00.png)

**fig00** first caption
second line

---

## 1. Checks
![A](figA.png)
caption A
![B](figB.png)
caption B
"""


def test_gallery_entries_parse():
    g = e.gallery_entries(GALLERY)
    assert [x["file"] for x in g] == ["fig00.png", "figA.png", "figB.png"]
    assert [x["section"] for x in g] == ["0. Start", "1. Checks", "1. Checks"]
    assert g[0]["title"] == "Quick" and g[0]["caption"] == "**fig00** first caption\nsecond line"
    assert g[1]["caption"] == "caption A" and g[2]["caption"] == "caption B"
    assert e.gallery_entries("") == [] and e.gallery_entries("## Only a heading\ntext") == []


def test_real_gallery_lists_every_figure_file():
    g = e.gallery_entries((ROOT / "figures/GALLERY.md").read_text())
    files = {x["file"] for x in g}
    assert files == {p.name for p in (ROOT / "figures").glob("fig*.png")}
    assert all(x["caption"] for x in g)


# ------------------------------------------------------------------ runaway_path
# Spec: the same path as the guarded run, up to the day it stopped.

def test_runaway_path_equals_guarded_run_while_it_lasts():
    p, _ = e.fitted_settings()
    guarded = model.simulate(p, 800, seed=2)            # does not run away in 800 days
    path = e.runaway_path(p, "herding", p["a1"], 2, 800)
    for k in ("lp", "lf", "z"):
        assert np.array_equal(path[k], guarded[k])


@pytest.mark.pinned_seed
def test_runaway_path_shows_the_bubble():
    # no herding, seed 0 runs away on day 2735 (test_run_mix_reports_runaway):
    # by then the price is far above its value and nearly everyone is a chartist
    p, _ = e.fitted_settings()
    path = e.runaway_path(p, "chartists", 0.0, 0, 2735)
    assert len(path["lp"]) == 2736
    assert path["lp"][-1] - path["lf"][-1] > model.MAX_GAP and path["z"][-1] > 0.95


@pytest.mark.parametrize("day", [0, -3, 2.5, True])
def test_runaway_path_rejects_day(day):
    p, _ = e.fitted_settings()
    with pytest.raises(ValueError):
        e.runaway_path(p, "chartists", 0.0, 0, day)


# ------------------------------------------------------------------ max_drawdown
# Spec: the biggest fall from a running peak of exp(cumsum r), as a fraction.

def test_max_drawdown_hand_paths():
    up, down = math.log(2), math.log(0.5)
    assert e.max_drawdown([up, up]) == 0.0                         # only rises: no drawdown
    assert e.max_drawdown([down]) == pytest.approx(0.5)            # 1 -> 0.5 from the start
    assert e.max_drawdown([up, down, down, up, up]) == pytest.approx(0.75)   # 1,2,1,0.5,1,2: peak 2 -> 0.5
    assert e.max_drawdown([up, down, up, up, down]) == pytest.approx(0.5)    # two falls of 50%: worst 0.5


def test_max_drawdown_matches_brute_force():
    r = _pattern(800) * 0.01
    p = np.exp(np.concatenate([[0], np.cumsum(r)]))
    brute = max(1 - p[j] / p[:j + 1].max() for j in range(len(p)))
    assert e.max_drawdown(r) == pytest.approx(brute)


def test_max_drawdown_sp500_black_monday_era():
    # the S&P 500 fell about 57% from Oct 2007 to Mar 2009 (1565.15 -> 676.53, published closes)
    r = e.log_returns(e.saved_prices("S&P 500")["2007-01-01":"2009-12-31"])
    assert e.max_drawdown(r.values) == pytest.approx(1 - 676.53 / 1565.15, abs=1e-4)


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf]])
def test_max_drawdown_rejects(bad):
    with pytest.raises(ValueError):
        e.max_drawdown(bad)


# ------------------------------------------------------------------ big_day_list
# Spec: exactly the days beyond 4 sd, with their simple return and sd, biggest first.

def test_big_day_list_planted_days():
    idx = pd.bdate_range("2020-01-01", periods=1000)
    r = pd.Series(np.tile([0.01, -0.01], 500), index=idx)
    r.iloc[[100, 500]] = [0.5, -0.8]
    out = e.big_day_list(r)
    assert list(out.index) == [idx[500], idx[100]]                  # -0.8 is the bigger move
    assert out["move"].tolist() == pytest.approx([math.expm1(-0.8), math.expm1(0.5)])
    z = (r - r.mean()) / r.std(ddof=0)
    assert out["sd"].tolist() == pytest.approx([z.iloc[500], z.iloc[100]])


def test_big_day_list_black_monday_tops_sp500():
    out = e.big_day_list(e.log_returns(e.saved_prices("S&P 500")))
    assert out.index[0] == pd.Timestamp("1987-10-19") and out["move"].iloc[0] == pytest.approx(-0.2047, abs=1e-4)
    assert len(out) == e.summary(e.log_returns(e.saved_prices("S&P 500")).values)["big_days"]


# ------------------------------------------------------------------ mix_gap
def test_mix_gap_hand_values():
    assert e.mix_gap(_s(3.0, 0.2), _s(3.0, 0.2)) == 0.0
    assert e.mix_gap(_s(3.0, 0.2), _s(3.3, 0.1)) == pytest.approx(0.6)
    assert e.mix_gap(_s(3.0, 0.0), _s(3.0, 0.01)) == pytest.approx(0.2)    # memory floor 0.05


# ------------------------------------------------------------------ thin_indices
# Spec: at most max_points positions, sorted, unique, first and last kept, and every slice's
# lowest and highest value kept, so no crash or spike disappears from a chart.

def test_thin_indices_short_input_unchanged():
    assert list(e.thin_indices([3.0, 1.0, 2.0], 10)) == [0, 1, 2]
    assert list(e.thin_indices(np.arange(10.0), 10)) == list(range(10))     # exactly max_points


def test_thin_indices_keeps_extremes_and_shape():
    y = np.cumsum(np.random.default_rng(1).standard_normal(50_000))
    y[31_337] = y.max() + 100                                  # a spike
    y[777] = y.min() - 100                                     # a crash
    i = e.thin_indices(y, 1500)
    assert len(i) <= 1500 and np.all(np.diff(i) > 0)
    assert i[0] == 0 and i[-1] == len(y) - 1 and 31_337 in i and 777 in i
    edges = np.linspace(0, len(y), 1500 // 2).astype(int)     # brute force: every slice's min and max
    for lo, hi in zip(edges[:-1], edges[1:]):
        assert lo + y[lo:hi].argmin() in i and lo + y[lo:hi].argmax() in i


@pytest.mark.parametrize("y, n", [([1.0, np.nan], 4), ([[1.0, 2.0]], 4), ([1.0] * 10, 3), ([1.0] * 10, True)])
def test_thin_indices_rejects(y, n):
    with pytest.raises(ValueError):
        e.thin_indices(y, n)


# ------------------------------------------------------------------ sweep
# Spec: one row per (a1, seed), the same numbers as run_mix + summary on that cell.

def test_sweep_matches_run_mix():
    p, burn = e.fitted_settings()
    t = e.sweep(p, [0.0, p["a1"]], [0, 1], 3000, burn)
    assert list(zip(t["a1"], t["seed"])) == [(0.0, 0), (0.0, 1), (p["a1"], 0), (p["a1"], 1)]
    for _, row in t.iterrows():
        r, err = e.run_mix(p, "herding", row["a1"], 3000, int(row["seed"]), burn)
        if r is None:
            assert np.isnan(row["tail"]) and row["runaway_day"] == err.day - burn
        else:
            assert row["tail"] == e.summary(r)["tail"] and np.isnan(row["runaway_day"])


# ------------------------------------------------------------------ intraday_prices / range_stats
def test_intraday_prices_with_fake_yahoo():
    idx = pd.date_range("2026-11-18 09:30", periods=78, freq="5min")
    seen = []

    def fake(t, period, interval):
        seen.append((t, period, interval))
        return pd.Series(np.linspace(100, 101, 78), index=idx)
    out = e.intraday_prices(" aapl ", "1D", download=fake)
    assert seen == [("AAPL", "1d", "5m")] and len(out) == 78 and out.name == "AAPL"
    with pytest.raises(ValueError):
        e.intraday_prices("AAPL", "1Y", download=fake)
    with pytest.raises(ValueError, match="no intraday"):
        e.intraday_prices("AAPL", "7D", download=lambda t, p, i: pd.Series([5.0]))
    with pytest.raises(ConnectionError):
        e.intraday_prices("AAPL", "7D", download=lambda t, p, i: 1 / 0)


def test_range_stats_hand_values():
    idx = pd.bdate_range("2026-01-01", periods=5)
    c = pd.Series([100.0, 120.0, 90.0, 110.0, 99.0], index=idx)
    s = e.range_stats(c, daily=c, year_days=3)
    assert s["last"] == 99.0 and s["first"] == 100.0 and s["change"] == pytest.approx(-0.01)
    assert (s["high"], s["low"]) == (120.0, 90.0) and s["drawdown"] == pytest.approx(0.25)   # 120 -> 90
    assert (s["high_52w"], s["low_52w"]) == (110.0, 90.0)          # the last 3 days only
    r = np.diff(np.log(c.values))
    assert s["volatility"] == pytest.approx(r.std(ddof=1) * math.sqrt(3))


def test_range_stats_sp500_2008():
    # published closes: 1,468.36 on 2007-12-31 and 903.25 on 2008-12-31 (-38.5% for the year)
    c = e.saved_prices("S&P 500")["2007-12-31":"2008-12-31"]
    s = e.range_stats(c)
    assert s["change"] == pytest.approx(903.25 / 1468.36 - 1, abs=1e-5)


# ------------------------------------------------------------------ web/numbers.json matches the backend
# Spec: the saved numbers are exactly what the model gives for those seeds, and their medians are the
# ones quoted in figures/RESULTS.md (Fig 4: herding tail 3.10, no-chartist kurtosis 0.01).

@pytest.fixture(scope="module")
def web():
    return e.web_numbers()


def test_web_numbers_recipe_matches_the_figures(web):
    from herding import calibrate
    assert web["recipe"]["seeds"] == list(calibrate.HELD_OUT_SEEDS)
    assert web["recipe"]["days"] == len(market_data.spx_returns()) and web["recipe"]["burn"] == calibrate.BURN


def test_web_numbers_medians_match_results_md(web):
    tails = [v for v in web["checks"]["herding"]["tail"] if v is not None]
    assert len(tails) == 40 and round(float(np.median(tails)), 2) == 3.10
    assert round(float(np.median(web["checks"]["fundamentalists"]["kurtosis"])), 2) == 0.01
    assert "3.10" in (ROOT / "figures/RESULTS.md").read_text()


@pytest.mark.slow
@pytest.mark.parametrize("i", [0, 17, 39])
def test_web_numbers_rerun_exactly(web, i):
    p, burn = e.fitted_settings()
    seed, days = web["recipe"]["seeds"][i], web["recipe"]["days"]
    r, _ = e.run_mix(p, "herding", p["a1"], days, seed, burn)
    s = e.summary(r)
    for k in ("tail", "kurtosis", "memory", "worst"):
        assert web["checks"]["herding"][k][i] == pytest.approx(s[k])
    f, _ = e.run_mix(p, "fundamentalists", p["a1"], days, seed, burn)
    assert web["checks"]["fundamentalists"]["kurtosis"][i] == pytest.approx(e.summary(f)["kurtosis"])


def test_intraday_prices_drop_time_zone_and_mix_with_daily():
    # Yahoo's bars come time-zoned; comparing them with the daily files' plain dates crashed the 1D view
    idx = pd.date_range("2026-11-18 14:30", periods=6, freq="5min", tz="UTC")
    out = e.intraday_prices("AAPL", "1D", download=lambda t, p, i: pd.Series(np.arange(1.0, 7.0), index=idx))
    assert out.index.tz is None and out.index[0] == pd.Timestamp("2026-11-18 09:30")   # 14:30 UTC = 9:30 New York
    daily = e.saved_prices("AAPL")
    s = e.range_stats(out, daily=daily)                      # must not raise
    assert s["change"] == pytest.approx(5.0) and s["high_52w"] == daily.iloc[-252:].max()
