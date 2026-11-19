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
    assert list(c) == ["Stocks", "Indexes", "Commodities"]
    assert c["Stocks"] == market_data.stock_tickers() and len(c["Stocks"]) == 20
    assert c["Indexes"] == ["S&P 500", "Nasdaq-100", "Dow Jones"]
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
    assert "about 89 times more" in text and "herding mix" in text     # more, not fewer (WS-1)


def test_insight_no_times_when_close_to_bell():
    text = e.insight("X", {"big_days": 1, "bell_big_days": 0.9}, "fundamentalists")
    assert "times" not in text and "fundamentalists-only mix" in text


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


# ================================================================== bug fixes, November 2026 stress test
# Each test below failed before its fix (ids BE-n / WS-n / DUP-n from the stress test) and checks the
# fixed behaviour against an answer worked out separately (by hand or with plain numpy).

def _bdays(end, n, start_value=100.0, seed=0):
    idx = pd.bdate_range(end=end, periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(start_value * np.exp(np.cumsum(rng.standard_normal(n) * 0.01)), index=idx)


# ------------------------------------------------------------------ price_range (BE-1, DUP-1)
# Spec: the closes the chart shows for each range button, always at least 2 prices; YTD starts at the
# last close of the previous year (the usual "year to date": the change since last year's final close).

def test_price_range_ytd_on_the_first_trading_day_of_the_year():
    close = _bdays("2027-01-01", 1500)                    # 2027-01-01 is a Friday: one close this year
    cl, note = e.price_range(close, "YTD")
    assert list(cl.index) == [pd.Timestamp("2026-12-31"), pd.Timestamp("2027-01-01")] and note == ""
    assert e.range_stats(cl)["change"] == pytest.approx(close.iloc[-1] / close.iloc[-2] - 1)   # must not raise


def test_price_range_ytd_2008_matches_the_published_year():
    # published: S&P 500 1,468.36 on 2007-12-31 and 903.25 on 2008-12-31, -38.49% for 2008
    cl, _ = e.price_range(e.saved_prices("S&P 500")[:"2008-12-31"], "YTD")
    assert cl.index[0] == pd.Timestamp("2007-12-31") and cl.index[-1] == pd.Timestamp("2008-12-31")
    assert e.range_stats(cl)["change"] == pytest.approx(903.25 / 1468.36 - 1, abs=1e-5)


def test_price_range_calendar_spans_and_all():
    # 1Y = the last 52 weeks (the same span as the 52-week high/low), 5Y and 20Y = calendar years
    close = _bdays("2026-10-30", 6000)
    last = close.index[-1]
    for frame, start in (("1Y", last - pd.Timedelta(days=364)), ("5Y", pd.Timestamp("2021-10-30")),
                         ("20Y", pd.Timestamp("2006-10-30"))):
        cl, note = e.price_range(close, frame)
        want = [d for d in close.index if d > start]                     # by hand, date by date
        assert list(cl.index) == want and cl.index[-1] == last and note == ""
    assert e.price_range(close, "All")[0].equals(close)
    short = close.iloc[:2]                                 # every range of a 2-price history has 2 prices
    for frame in e.FRAMES:
        assert len(e.price_range(short, frame)[0]) == 2


def test_price_range_intraday_and_fallback():
    close = _bdays("2026-11-18", 1500)
    bars = pd.Series([100.0, 101.0, 102.0], index=pd.date_range("2026-11-18 09:30", periods=3, freq="5min"))
    cl, note = e.price_range(close, "1D", intraday=bars)
    assert cl.equals(bars) and note == ""
    for frame, n in (("1D", 2), ("7D", 6)):                # no intraday bars: the last daily closes
        cl, note = e.price_range(close, frame)
        assert cl.equals(close.iloc[-n:]) and "daily closes" in note
    cl, note = e.price_range(close, "1D", intraday=pd.Series([5.0]))     # too few bars: fall back too
    assert cl.equals(close.iloc[-2:]) and note


@pytest.mark.parametrize("frame", ["2Y", "", None, "ytd"])
def test_price_range_rejects_frame(frame):
    with pytest.raises(ValueError):
        e.price_range(_bdays("2026-11-18", 50), frame)


# ------------------------------------------------------------------ cleaning live prices (BE-2, BE-3, BE-7)

def test_intraday_prices_drop_infinite_bars():
    idx = pd.DatetimeIndex(["2026-11-18 14:30", "2026-11-18 15:00", "2026-11-18 15:30", "2026-11-18 16:00"], tz="UTC")
    out = e.intraday_prices("AAPL", "1D", download=lambda t, p, i: pd.Series([100.0, np.inf, 101.0, 102.0], index=idx))
    assert out.tolist() == [100.0, 101.0, 102.0]
    e.range_stats(out)                                     # must not raise


def test_live_prices_drop_infinite_closes():
    s = _bdays("2026-11-18", 1200)
    s.iloc[500] = np.inf
    s.iloc[-1] = np.inf                                    # the last close too: it used to become "the price"
    out = e.live_prices("ZZZZ", download=lambda t, st: s)
    assert len(out) == 1198 and np.isfinite(out).all() and out.index[-1] == s.index[-2]


def test_live_prices_drop_time_zone():
    # Yahoo can return daily closes labelled with New York time; the saved files have plain dates
    s = _bdays("2026-11-18", 1200)
    tz = s.copy()
    tz.index = s.index.tz_localize("America/New_York")
    out = e.live_prices("ZZZZ", download=lambda t, st: tz)
    assert out.index.tz is None and out.index.equals(s.index)
    utc = s.copy()                                          # 05:00 UTC is midnight in New York (winter)
    utc.index = (s.index + pd.Timedelta(hours=5)).tz_localize("UTC")
    assert e.live_prices("ZZZZ", download=lambda t, st: utc).index.equals(s.index)
    bars = pd.Series([100.0, 101.0], index=pd.date_range("2026-11-18 14:30", periods=2, freq="30min", tz="UTC"))
    cl = e.intraday_prices("ZZZZ", "1D", download=lambda t, p, i: bars)
    daily = out
    assert len(e.daily_within(daily, cl)) == 1             # comparing the two used to raise TypeError
    e.range_stats(cl, daily=daily)


def test_swapping_the_yahoo_functions_reaches_every_caller(monkeypatch):
    # the app's tests fake Yahoo by replacing these module functions; the defaults must look them up
    calls = []
    monkeypatch.setattr(e, "_yahoo", lambda t, s: calls.append(("daily", t)) or _bdays("2026-11-18", 1200))
    monkeypatch.setattr(e, "_yahoo_intraday", lambda t, p, i: calls.append(("intraday", t)) or
                        pd.Series([1.0, 2.0], index=pd.date_range("2026-11-18 09:30", periods=2, freq="5min")))
    e.live_prices("ZZZZ")
    e.intraday_prices("ZZZZ", "1D")
    assert calls == [("daily", "ZZZZ"), ("intraday", "ZZZZ")]


# ------------------------------------------------------------------ range_stats and its daily closes (BE-4, BE-8)

def test_range_stats_volatility_includes_the_first_day_of_an_intraday_range():
    daily = pd.Series(np.linspace(100, 110, 10) * (1 + 0.01 * np.tile([1, -1], 5)),
                      index=pd.bdate_range("2026-11-09", periods=10))
    bars = pd.Series(np.linspace(100, 101, 5), index=pd.DatetimeIndex(
        [f"2026-11-{d} 09:30" for d in (16, 17, 18, 19)] + ["2026-11-20 15:55"]))
    five = daily["2026-11-16":"2026-11-20"].values          # 16th to 20th: five closes, four moves
    want = np.diff(np.log(five)).std(ddof=1) * math.sqrt(252)
    assert len(e.daily_within(daily, bars)) == 5
    assert e.range_stats(bars, daily=daily)["volatility"] == pytest.approx(want)


@pytest.mark.parametrize("kind", ["reversed", "numpy", "zero", "nan", "duplicate", "short"])
def test_range_stats_rejects_bad_daily(kind):
    daily = _bdays("2026-11-18", 400)
    c = daily.iloc[-20:]
    bad = {"reversed": daily.iloc[::-1], "numpy": daily.values,
           "zero": daily.where(daily.index != daily.index[100], 0.0),
           "nan": daily.where(daily.index != daily.index[100], np.nan),
           "duplicate": pd.concat([daily.iloc[:5], daily.iloc[4:]]), "short": daily.iloc[:1]}[kind]
    with pytest.raises(ValueError):
        e.range_stats(c, daily=bad)


@pytest.mark.parametrize("year_days", [0, 1, True, 252.0, -5, "252"])
def test_range_stats_rejects_year_days(year_days):
    daily = _bdays("2026-11-18", 400)
    with pytest.raises(ValueError):
        e.range_stats(daily.iloc[-20:], daily=daily, year_days=year_days)


def test_range_stats_needs_dates_to_use_daily():
    daily = _bdays("2026-11-18", 400)
    with pytest.raises(ValueError):
        e.range_stats(daily.values[-20:], daily=daily)      # no dates: the range can't be found in daily
    assert e.range_stats(daily.values[-20:])["last"] == daily.iloc[-1]   # without daily, a plain list is fine


# ------------------------------------------------------------------ dates must be in order (BE-10)

@pytest.mark.parametrize("index", [["2020-01-03", "2020-01-01", "2020-01-02"],
                                   ["2020-01-01", "2020-01-01", "2020-01-02"]])
def test_log_returns_rejects_unordered_or_repeated_dates(index):
    for idx in (index, pd.DatetimeIndex(index)):
        with pytest.raises(ValueError, match="order"):
            e.log_returns(pd.Series([1.0, 2.0, 3.0], index=idx))


def test_log_returns_plain_list_still_fine():
    assert e.log_returns([1.0, 2.0]).tolist() == pytest.approx([math.log(2)])


# ------------------------------------------------------------------ big_day_list (BE-9, DUP-5)

def test_big_day_list_repeated_dates_give_one_row_per_day():
    idx = pd.DatetimeIndex(["2020-01-01"] * 3).append(pd.bdate_range("2020-01-02", periods=997))
    r = pd.Series(np.tile([0.01, -0.01], 500), index=idx)
    r.iloc[0] = 10.0
    z = (r.values - r.values.mean()) / r.values.std()
    out = e.big_day_list(r)
    assert len(out) == 1 and out.index[0] == pd.Timestamp("2020-01-01")
    assert out["sd"].iloc[0] == pytest.approx(z[0]) and out["move"].iloc[0] == pytest.approx(math.expm1(10.0))


def test_big_day_list_within_a_range():
    idx = pd.bdate_range("2020-01-01", periods=1000)
    r = pd.Series(np.tile([0.01, -0.01], 500), index=idx)
    r.iloc[[100, 500, 900]] = [0.5, -0.8, 0.6]
    every = e.big_day_list(r)
    assert list(e.big_day_list(r, within=r.iloc[400:])) == list(every)
    assert list(e.big_day_list(r, within=r.iloc[400:]).index) == [idx[500], idx[900]]
    assert list(e.big_day_list(r, within=idx[:200]).index) == [idx[100]]
    assert e.big_day_list(r, within=idx[200:300]).empty


# ------------------------------------------------------------------ closest_mix / mix_gap (BE-11)

def test_closest_mix_skips_a_mix_with_nan_numbers_in_any_order():
    a, b = _s(np.nan, 0.2), _s(5.0, 0.9)
    assert e.closest_mix(_s(3.0, 0.2), {"a": a, "b": b}) == "b"
    assert e.closest_mix(_s(3.0, 0.2), {"b": b, "a": a}) == "b"
    with pytest.raises(ValueError):
        e.closest_mix(_s(3.0, 0.2), {"a": a})


@pytest.mark.parametrize("target", [_s(0.0, 0.2), _s(np.nan, 0.2), _s(3.0, np.nan), _s(np.inf, 0.2)])
def test_mix_gap_rejects_target_without_a_shape(target):
    with pytest.raises(ValueError):
        e.mix_gap(target, _s(3.0, 0.2))
    with pytest.raises(ValueError):
        e.closest_mix(target, {"a": _s(3.0, 0.2)})


# ------------------------------------------------------------------ kirman_match / bell_check / tail_check (BE-13, BE-6)

@pytest.mark.parametrize("k", [np.array([0, np.nan]), np.array([0.5, 1.7, 2.2]), np.array([[0, 1], [2, 3]])])
def test_kirman_match_rejects_non_whole_counts(k):
    with pytest.raises(ValueError):
        e.kirman_match(k, 3, 0.01, 0.1)


def test_kirman_match_whole_floats_accepted():
    sim, _, _ = e.kirman_match(np.array([0.0, 2.0, 2.0, 4.0]), 4, 0.5, 0.5)
    assert sim == pytest.approx([0.25, 0, 0.5, 0, 0.25])


def test_bell_check_and_tail_check_reject_2d():
    with pytest.raises(ValueError):
        e.bell_check([[0.1, 0.2]], 1000)
    with pytest.raises(ValueError):
        e.tail_check([[3.0, 3.1]])


def test_tail_check_uses_the_median():
    # the Checks page states "the median tail exponent lies between 2.5 and 3.5"
    r = e.tail_check([1.0, 3.0, 3.2])                    # mean 2.4 (outside), median 3.0 (inside)
    assert r["passed"] and r["median"] == 3.0 and r["n"] == 3 and (r["low"], r["high"]) == (1.0, 3.2)
    assert e.tail_check([3.0, 3.2, 6.0])["passed"]        # mean 4.07, median 3.2
    assert not e.tail_check([2.0, 2.4, 3.4])["passed"]    # mean 2.6, median 2.4
    r = e.tail_check([3.0, None, 3.4, None])              # None = a history that ran away: left out
    assert r["n"] == 2 and r["median"] == pytest.approx(3.2) and r["left_out"] == 2


def test_tail_check_on_the_saved_histories_matches_results_md():
    tails = e.web_numbers()["checks"]["herding"]["tail"]
    r = e.tail_check(tails)
    assert r["passed"] and round(r["median"], 2) == 3.10 and r["n"] == 40


# ------------------------------------------------------------------ run_mix / sweep seeds (BE-14)

@pytest.mark.parametrize("seed", [-1, 1.5, True, "1", None, 1.0])
def test_run_mix_rejects_seed(seed):
    p, burn = e.fitted_settings()
    with pytest.raises(ValueError, match="seed"):
        e.run_mix(p, "fundamentalists", 1.0, 500, seed, burn)


def test_sweep_rejects_fractional_seed_and_keeps_columns_when_empty():
    p, burn = e.fitted_settings()
    with pytest.raises(ValueError, match="seed"):
        e.sweep(p, [1.0], [1.7], 500, burn)
    t = e.sweep(p, [], [0], 500, burn)
    assert t.empty and list(t.columns) == ["a1", "seed", "tail", "memory", "kurtosis", "runaway_day"]


# ------------------------------------------------------------------ insight wording (WS-1, BE-15)

def test_insight_says_more_not_fewer():
    text = e.insight("S&P 500", {"big_days": 108, "bell_big_days": 1.22}, "herding")
    assert "about 89 times more" in text and "fewer" not in text      # 108 / 1.22 = 88.5


def test_insight_one_day_and_no_bell_days():
    assert "had 1 day bigger" in e.insight("X", {"big_days": 1, "bell_big_days": 0.3}, "herding")
    text = e.insight("X", {"big_days": 3, "bell_big_days": 0.0}, "herding")   # used to divide by zero
    assert "had 3 days bigger" in text and "times" not in text
    text = e.insight("X", {"big_days": 0, "bell_big_days": 5.0}, "fundamentalists")
    assert "had no days bigger" in text and "times more" not in text


# ------------------------------------------------------------------ small backend functions for the pages (DUP-2..18)

def test_drawdown_series_hand_values():
    c = pd.Series([100.0, 120.0, 90.0, 110.0, 99.0], index=pd.bdate_range("2026-01-01", periods=5))
    dd = e.drawdown_series(c)
    assert dd.tolist() == pytest.approx([0.0, 0.0, -0.25, 110 / 120 - 1, 99 / 120 - 1])
    assert dd.index.equals(c.index)
    assert e.range_stats(c)["drawdown"] == pytest.approx(0.25)


def test_moves_in_sd_hand_values_and_population_sd():
    z = e.moves_in_sd(np.tile([3.0, 5.0], 10))           # mean 4, population sd 1
    assert z.tolist() == pytest.approx([-1.0, 1.0] * 10)
    assert e.moves_in_sd(np.array([6.0, 4.0]), ref=np.array([3.0, 5.0])).tolist() == pytest.approx([2.0, 0.0])
    r = pd.Series([1.0, 2.0, 4.0], index=pd.bdate_range("2026-01-01", periods=3))
    zs = e.moves_in_sd(r)
    want = (r.values - 7 / 3) / math.sqrt(((1 - 7 / 3) ** 2 + (2 - 7 / 3) ** 2 + (4 - 7 / 3) ** 2) / 3)
    assert isinstance(zs, pd.Series) and zs.index.equals(r.index) and zs.tolist() == pytest.approx(want.tolist())
    with pytest.raises(ValueError):
        e.moves_in_sd([1.0, 2.0], ref=[3.0, 3.0])          # no spread to measure against


def test_day_change():
    assert e.day_change(pd.Series([50.0, 100.0, 110.0])) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        e.day_change(pd.Series([100.0]))


def test_histogram_empty_bins_can_be_nan():
    r = np.tile([3.0, 5.0], 300)
    c0, d0 = e.histogram(r)
    c1, d1 = e.histogram(r, empty=np.nan)
    assert np.array_equal(c0, c1) and np.isnan(d1).sum() == len(d1) - 2
    assert d1[np.argmin(abs(c1 - 1.0))] == pytest.approx(1.25) and np.all(d0[np.isnan(d1)] == 0)


def test_bell_tail_and_slope_guide():
    assert e.bell_tail(4.0) == pytest.approx(6.334248366623996e-05)      # P(|Z| > 4), standard tables
    x = np.array([0.5, 1.0, 2.0])
    assert e.bell_tail(x) == pytest.approx([e.bell_big_days(1, v) for v in x])
    r = _pattern(5000)
    gx, gy = e.slope_guide(r)
    tx, ty = e.tail_curve(r)
    assert list(gx) == [2.0, 30.0]
    assert gy[0] * (2 / 3) ** 3 == pytest.approx(np.interp(3.0, tx, ty))   # the guide passes through the curve at 3 sd
    assert math.log(gy[1] / gy[0]) / math.log(30 / 2) == pytest.approx(-3.0)


def test_sweep_band_hand_rows():
    rows = [{"a1": 0.4, "seed": s, "tail": (2.24 if s == 0 else None)} for s in range(10)] + \
           [{"a1": 1.0, "seed": s, "tail": float(s + 1)} for s in range(10)]
    b = e.sweep_band(rows, min_survivors=5)
    assert list(b["a1"]) == [0.4, 1.0]
    lo, hi = b.iloc[0], b.iloc[1]
    assert (lo["n"], lo["runaways"], bool(lo["shown"])) == (1, 9, False) and np.isnan(lo["median"])
    # 1..10: median 5.5, quartiles at positions 2.25 and 6.75 -> 3.25 and 7.75
    assert (hi["n"], hi["runaways"], bool(hi["shown"])) == (10, 0, True)
    assert (hi["median"], hi["q1"], hi["q3"]) == pytest.approx((5.5, 3.25, 7.75))


def test_sweep_band_hides_the_saved_points_that_mostly_ran_away():
    b = e.sweep_band(e.web_numbers()["sweep"]["rows"]).set_index("a1")
    assert b.loc[0.4, "n"] == 1 and not b.loc[0.4, "shown"] and b.loc[0.4, "runaways"] == 9
    assert b.loc[1.0, "n"] == 10 and b.loc[1.0, "shown"]


def _mixes_by_hand():
    rng = np.random.default_rng(3)
    market = rng.standard_t(3, 2000) * 0.01
    fund = rng.standard_normal(2000) * 0.01
    herd = rng.standard_t(4, 2000) * 0.01
    return market, {"fundamentalists": (fund, None), "chartists": (None, 123), "herding": (herd, None)}


def test_mix_summaries_and_tails():
    _, mixes = _mixes_by_hand()
    sums = e.mix_summaries(mixes)
    assert sums["chartists"] is None and sums["herding"] == e.summary(mixes["herding"][0])
    assert e.mix_tails(mixes) == {"fundamentalists": sums["fundamentalists"]["tail"], "chartists": None,
                                  "herding": sums["herding"]["tail"]}


def test_mix_table_numbers():
    market, mixes = _mixes_by_hand()
    s = e.summary(market)
    t = e.mix_table("ZZZZ", s, market, e.mix_summaries(mixes), mixes)
    assert list(t.index) == ["ZZZZ", "fundamentalists", "chartists", "herding", "bell"]
    price = np.exp(np.concatenate([[0.0], np.cumsum(market)]))         # brute-force drawdown
    worst = max(1 - price[j] / price[:j + 1].max() for j in range(len(price)))
    assert t.loc["ZZZZ", "max_drawdown"] == pytest.approx(worst, rel=1e-9)
    assert t.loc["ZZZZ", "tail"] == s["tail"] and t.loc["ZZZZ", "big_days"] == s["big_days"]
    assert t.loc["chartists", "runaway_day"] == 123 and np.isnan(t.loc["chartists", "tail"])
    assert np.isnan(t.loc["herding", "runaway_day"])
    assert t.loc["bell", "big_days"] == pytest.approx(2000 * math.erfc(4 / math.sqrt(2)))
    assert t.loc["bell", "kurtosis"] == 0 and t.loc["bell", "memory"] == 0 and np.isnan(t.loc["bell", "tail"])


def test_differs_from_fit():
    p = {"a1": 1.0, "a2": 0.5, "v1": 2.0}
    assert not e.differs_from_fit({"a1": 1.7, "a2": 0.5, "v1": 2.0}, p)          # a1 is the dial itself
    assert e.differs_from_fit({"a1": 1.0, "a2": 0.51, "v1": 2.0}, p)
    assert e.differs_from_fit({"a1": 1.0, "a2": 0.5, "v1": 2.0 + 1e-6}, p)
    assert not e.differs_from_fit({"a1": 1.0, "a2": 0.5 + 1e-12, "v1": 2.0}, p)


def test_past_year():
    close = pd.Series(np.arange(1.0, 301.0) / 3)
    assert e.past_year(close) == [round(v / 3, 4) for v in range(49, 301, 5)]   # the last 252 closes, every 5th
    assert e.past_year(close.iloc[:10]) == [round(v / 3, 4) for v in (1, 6)]


def test_run_mixes_equals_direct_simulation():
    p, burn = e.fitted_settings()
    out = e.run_mixes(p, burn, a1=p["a1"], a2=0.3, v1=1.5, days=2000, seed=4)
    q = dict(p, a2=0.3, v1=1.5)
    for m, chartists, a1 in (("fundamentalists", False, p["a1"]), ("herding", True, p["a1"]), ("chartists", True, 0.0)):
        try:
            direct = model.simulate(dict(q, a1=a1) if chartists else q, 2000 + burn, seed=4, chartists=chartists)
            assert np.array_equal(out[m][0], direct["r"][burn:]) and out[m][1] is None
        except model.RunawayError as err:
            assert out[m][0] is None and out[m][1] == err.day - burn


def test_runaway_story_matches_the_path():
    p, burn = e.fitted_settings()
    for seed in range(10):                       # the first no-herding history that runs away
        r, err = e.run_mix(p, "chartists", 0.0, 5000, seed, burn)
        if err is not None:
            break
    else:
        pytest.skip("no runaway in seeds 0-9")
    day = e.runaway_day(err, burn)
    st = e.runaway_story(p, seed, day, burn)
    path = model.simulate(dict(p, a1=0.0), day + burn, seed=seed, chartists=True, max_gap=None)
    assert np.array_equal(st["z"], path["z"])
    assert st["gap"] == pytest.approx(np.expm1(np.clip(path["lp"] - path["lf"], -0.5, 0.5)))
    assert st["share_before"] == path["z"][-6] and st["calm_days"] == day - 3
    assert st["gap_cap"] == pytest.approx(math.expm1(0.5))


def test_spread_hand_values():
    s = e.spread([5.0, 1.0, None, 3.0, np.nan, 2.0, 4.0])
    # 1..5: median 3; 5th percentile at position 0.2 -> 1.2; 95th at 3.8 -> 4.8
    assert s == pytest.approx({"median": 3.0, "low": 1.2, "high": 4.8, "n": 5})
    with pytest.raises(ValueError):
        e.spread([None, np.nan])


def test_runaway_count_and_check_table():
    nums = e.web_numbers()
    gone, total = e.runaway_count(nums, "chartists")
    days = nums["checks"]["chartists"]["runaway_day"]
    assert total == 40 and gone == len([d for d in days if d is not None])
    assert e.runaway_count({"checks": {"x": {"runaway_day": [None, 5, None]}}}, "x") == (1, 3)
    spx = e.summary(e.log_returns(e.saved_prices("S&P 500")).values)
    t = e.check_table(nums, spx)
    assert list(t.index) == ["tail", "kurtosis", "memory", "worst"]
    v = np.array([x for x in nums["checks"]["herding"]["memory"] if x is not None])
    assert t.loc["memory", "median"] == pytest.approx(np.median(v))
    w, pos = np.sort(v), 0.05 * (len(v) - 1)              # 5th percentile by hand: linear between neighbours
    lo = w[int(pos)] + (pos - int(pos)) * (w[int(pos) + 1] - w[int(pos)])
    assert t.loc["memory", "low"] == pytest.approx(lo) and t.loc["memory", "n"] == len(v)
    assert t.loc["tail", "sp500"] == spx["tail"]


def test_kirman_path():
    steps, counts = e.kirman_path(np.arange(10) * 2, upto=7, every=3)
    assert list(steps) == [0, 3, 6] and list(counts) == [0, 6, 12]


def test_daily_within_and_symbols():
    daily = pd.Series(np.arange(1.0, 11.0), index=pd.bdate_range("2026-11-09", periods=10))
    bars = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-11-17 09:30", "2026-11-18 15:55"]))
    assert list(e.daily_within(daily, bars).index) == list(pd.bdate_range("2026-11-17", periods=2))
    assert e.is_futures("GC=F") and e.is_futures("bz=f") and not e.is_futures("AAPL") and not e.is_futures("^GSPC")
    assert e.yahoo_symbol("S&P 500") == "^GSPC" and e.yahoo_symbol("Gold") == "GC=F"
    assert e.yahoo_symbol("AAPL") == "AAPL" and e.yahoo_symbol("NVDA") == "NVDA"
    assert e.asset_kind("Dow Jones") == "Index" and e.asset_kind("Copper") == "Commodity"
    assert e.asset_kind("AAPL") == "Stock" and e.asset_kind("NVDA") == "Stock"


# ================================================================== breaker round 1 (WS-19..22, BE-16, BE-17)

def test_range_stats_drawdown_of_a_rising_range_is_plus_zero():
    # WS-19: a range that only rises fell 0%, and must print "0.0%", not "-0.0%"
    c = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
    d = e.range_stats(c)["drawdown"]
    assert d == 0.0 and math.copysign(1.0, d) == 1.0 and f"{d:.1%}" == "0.0%"
    d = e.range_stats(pd.Series([5.0, 5.0, 6.0, 7.0], index=pd.bdate_range("2026-03-02", periods=4)))["drawdown"]
    assert f"{d:.1%}" == "0.0%"


def test_moves_in_range_hand_values():
    # WS-20: the moves a range shows are those INTO its 2nd..last price: n prices, n - 1 moves
    days = pd.bdate_range("2026-03-02", periods=6)                 # Mon 2 .. Mon 9 March
    r = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=days[1:])     # the move into each day
    prices = pd.Series([1.0, 2.0, 3.0], index=days[2:5])           # Wed, Thu, Fri
    out = e.moves_in_range(r, prices)
    assert list(out.index) == list(days[3:5]) and out.tolist() == [0.3, 0.4]


@pytest.mark.parametrize("frame", ["YTD", "1Y", "5Y", "All"])
def test_moves_in_range_matches_price_range(frame):
    c = e.saved_prices("S&P 500")
    r = e.log_returns(c)
    cl, _ = e.price_range(c, frame)
    rr = e.moves_in_range(r, cl)
    assert len(rr) == len(cl) - 1
    assert rr.index[0] == cl.index[1] and rr.index[-1] == cl.index[-1]
    if frame == "YTD":                    # every move shown happened this year
        assert (rr.index.year == c.index[-1].year).all()
    if frame == "1Y":                     # the 52 weeks before the last close (range_stats' 52 weeks)
        assert (rr.index > c.index[-1] - pd.Timedelta(weeks=52)).all() and 245 <= len(rr) <= 255


def test_moves_in_range_on_2_january():
    # YTD on the first trading day of a year: 2 prices (31 Dec, 2 Jan) and the one move into 2 Jan
    days = pd.bdate_range("2026-12-28", "2027-01-04")
    c = pd.Series(np.linspace(100, 110, len(days)), index=days)[:"2027-01-01"]
    cl, _ = e.price_range(c, "YTD")
    rr = e.moves_in_range(e.log_returns(c), cl)
    assert list(rr.index) == [pd.Timestamp("2027-01-01")]
    assert rr.iloc[0] == pytest.approx(math.log(c.iloc[-1] / c.iloc[-2]))


def test_asset_kind_by_symbol():
    # WS-21: an index (^), futures (=F), a currency (=X) and crypto (-USD) are not stocks
    assert e.asset_kind("^IXIC") == "Index" and e.asset_kind("^GSPC") == "Index"
    assert e.asset_kind("CL=F") == "Futures" and e.asset_kind("EURUSD=X") == "Currency"
    assert e.asset_kind("SHIB-USD") == "Crypto" and e.asset_kind("BTC-USD") == "Crypto"
    assert e.asset_kind("ETH-EUR") == "Crypto"
    assert e.asset_kind("BRK-B") == "Stock" and e.asset_kind("AAPL") == "Stock"   # a share class is a stock
    assert e.asset_kind("S&P 500") == "Index" and e.asset_kind("Gold") == "Commodity"


def test_trading_hours():
    stock = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-11-16 09:30", "2026-11-17 15:55"]))
    weekend = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-11-14 00:00", "2026-11-14 00:30"]))  # Saturday
    assert e.trading_hours("AAPL") == {"nights": False, "weekends": False}
    assert e.trading_hours("AAPL", stock) == {"nights": False, "weekends": False}
    assert e.trading_hours("GC=F") == {"nights": True, "weekends": False}
    assert e.trading_hours("EURUSD=X") == {"nights": True, "weekends": False}
    assert e.trading_hours("SHIB-USD") == {"nights": True, "weekends": True}
    assert e.trading_hours("ZZZ", weekend) == {"nights": True, "weekends": True}   # bars on a Saturday


def test_range_stats_52_weeks_by_date_for_a_market_open_every_day():
    # WS-21: a market open 365 days a year: "52-week" must mean 52 weeks of dates, not 252 rows
    idx = pd.date_range(end="2026-11-18", periods=800, freq="D")
    v = np.full(800, 10.0)
    v[-300] = 50.0                     # 300 days ago: inside 52 weeks, outside the last 252 rows
    v[-400] = 99.0                     # 400 days ago: outside 52 weeks
    v[-10] = 2.0
    d = pd.Series(v, index=idx)
    s = e.range_stats(d.iloc[-30:], daily=d)
    last = idx[-1]                     # by hand: dates after the same weekday 52 weeks ago
    want = [x for t, x in zip(idx, v) if t > last - pd.Timedelta(days=364)]
    assert len(want) == 364 and (s["high_52w"], s["low_52w"]) == (max(want), min(want)) == (50.0, 2.0)
    r = np.diff(np.log(v[-30:]))       # a market open every day: annualized with 365 days, not 252
    assert s["volatility"] == pytest.approx(r.std(ddof=1) * math.sqrt(365))


def test_range_stats_52_weeks_by_date_for_a_stock():
    daily = e.saved_prices("AAPL")
    s = e.range_stats(daily.iloc[-5:], daily=daily)
    cut = daily.index[-1] - pd.Timedelta(weeks=52)
    year = [x for t, x in daily.items() if t > cut]
    assert (s["high_52w"], s["low_52w"]) == (max(year), min(year))
    r = np.diff(np.log(daily.iloc[-5:].values))
    assert s["volatility"] == pytest.approx(r.std(ddof=1) * math.sqrt(252))   # weekdays only: 252


@pytest.mark.parametrize("x, want", [(2.013e-5, "0.00002013"), (1234.5, "1,234.50"), (1.0, "1.00"),
                                     (0.5, "0.5000"), (0.0123456, "0.01235"), (99999.999, "100,000.00")])
def test_price_text(x, want):
    # WS-22: prices below a cent keep 4 significant digits instead of reading 0.00
    assert e.price_text(x) == want


def test_price_decimals_for_hover_templates():
    assert e.price_decimals(150.0) == 2 and e.price_decimals(2.013e-5) == 8 and e.price_decimals(0.5) == 4
    assert e.price_decimals(float("nan")) == 2


def test_runaway_story_day_1_has_no_negative_calm_days():
    # BE-16: a runaway on day 1 gave calm_days = -2
    p, _ = e.fitted_settings()
    for day, want in ((1, 0), (3, 0), (4, 1), (10, 7)):
        assert e.runaway_story(p, 0, day, 0)["calm_days"] == want


def test_tail_check_treats_nan_like_none():
    # BE-17: NaN is how a runaway shows up in pandas or JSON-to-DataFrame
    for tails in (pd.Series([3.0, np.nan, 2.9]), [3.0, float("nan"), 2.9], np.array([3.0, np.nan, 2.9])):
        r = e.tail_check(tails)
        assert r["n"] == 2 and r["left_out"] == 1 and r["median"] == pytest.approx(2.95)
    with pytest.raises(ValueError):
        e.tail_check([np.nan, None])
    with pytest.raises(ValueError):
        e.tail_check([3.0, np.inf])


@pytest.mark.parametrize("N", [2.5, "4", None, True, 1])
def test_kirman_match_rejects_bad_n(N):
    with pytest.raises(ValueError, match="N"):
        e.kirman_match([0, 1, 1], N, 0.01, 0.1)


@pytest.mark.parametrize("k", [5, np.zeros((2, 3))])
def test_kirman_path_needs_a_list(k):
    with pytest.raises(ValueError, match="ant counts"):
        e.kirman_path(k)


@pytest.mark.parametrize("x, want", [(0.99996, "1.00"), (0.0099996, "0.01000"), (0.99994, "0.9999"),
                                     (0.000099996, "0.0001000")])
def test_price_text_rounds_before_counting_decimals(x, want):
    # BE-19: the decimals were worked out before rounding, so 0.99996 read 1.0000 (5 significant
    # digits). Expected strings by hand: 4 significant digits, and 2 decimals from 1 up
    assert e.price_text(x) == want


def test_live_prices_keep_a_utc_midnight_daily_bar_on_its_own_date():
    # BE-18: crypto's daily bars are stamped 00:00 UTC; moved to New York they fell on the day before
    dates = pd.date_range(end="2026-11-18", periods=1500, freq="D")
    d = pd.Series(np.arange(1.0, 1501.0), index=dates.tz_localize("UTC"))
    out = e.live_prices("BTC-USD", download=lambda t, s: d)
    assert out.index[-1] == pd.Timestamp("2026-11-18") and out.index.tz is None
    assert out.index.equals(dates) and (out.values == np.arange(1.0, 1501.0)).all()


def test_live_market_says_plainly_when_a_price_barely_moves():
    # WS-25: a pegged or halted asset showed stats' "need more than 10 nonzero returns, got 1"
    idx = pd.bdate_range(end="2026-11-18", periods=1500)
    step = pd.Series([1.0] * 1400 + [2.0] * 100, index=idx)
    with pytest.raises(ValueError, match=r"^STEP's price barely moves \(only 1 day changed\), so there are "
                                         r"no tails to measure"):
        e.live_market("STEP", download=lambda t, s: step)
    flat = pd.Series(1.0, index=idx)
    with pytest.raises(ValueError, match=r"^FLAT's price barely moves \(only 0 days changed\)"):
        e.live_market("FLAT", download=lambda t, s: flat)
    with pytest.raises(ValueError, match="only 3 days of prices"):     # other errors pass through
        e.live_market("TINY", download=lambda t, s: flat.iloc[:3])


def test_live_market_matches_its_three_parts():
    s = _bdays("2026-11-18", 1500)
    close, r, summ = e.live_market("ZZZZ", download=lambda t, st: s)
    assert close.equals(e.live_prices("ZZZZ", download=lambda t, st: s))
    want = np.log(s.values[1:] / s.values[:-1])                 # by hand, not through log_returns
    # log(a) - log(b) vs log(a / b): both ~1e-16 off on log(100) ~ 4.6, so compare absolutely
    assert np.allclose(r.values, want, rtol=0, atol=1e-14) and summ == e.summary(r.values)
    assert summ["days"] == 1499 and summ["tail"] == pytest.approx(e.stats.hill_alpha(want, 0.05), rel=1e-9)


def test_dial_runs_flag_any_plotted_run_off_the_fit():
    # WS-24: an earlier a2 = 0.2 run stayed on the a1-only dial with no warning once the current
    # run was back at the fit
    p = {"a1": 1.084, "a2": 0.5, "v1": 2.0}
    fit_run = ("t2", {"a1": 1.084, "a2": 0.5, "v1": 2.0}, {"herding": 2.88})
    off_run = ("t1", {"a1": 1.0, "a2": 0.2, "v1": 2.0}, {"herding": 2.71})
    gone = ("t0", {"a1": 2.0, "a2": 0.9, "v1": 2.0}, {"herding": None})     # ran away: not plotted
    assert e.dial_runs([fit_run, off_run], p) == ([(1.084, 2.88), (1.0, 2.71)], True)
    assert e.dial_runs([fit_run, gone], p) == ([(1.084, 2.88)], False)
    assert e.dial_runs([], p) == ([], False)


# ------------------------------------------------------------------ bell_density
# Spec: the standard normal density, checked against scipy; it integrates to 1.

def test_bell_density_matches_scipy_and_integrates_to_one():
    from scipy.stats import norm
    x = np.linspace(-8, 8, 2001)
    assert e.bell_density(x) == pytest.approx(norm.pdf(x))
    assert e.bell_density(0) == pytest.approx(1 / math.sqrt(2 * math.pi))
    assert np.trapezoid(e.bell_density(x), x) == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ValueError):
        e.bell_density([0.0, np.nan])
