"""Tests for scripts/analysis/party_scale.py (the political-party test at scale)."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from helpers import SerialPool

sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import party_scale as ps

from herding import calibrate, experiments, market_data, politics

YEAR = politics.DAYS_PER_YEAR


# ---- the seeds ----

def test_seeds_are_a_fresh_range():
    # 60000-60999, and none used by the fit, the held-out set or the stability screen
    assert ps.SEEDS == tuple(range(60000, 61000))
    used = set(calibrate.TRAIN_SEEDS) | set(calibrate.HELD_OUT_SEEDS) | set(calibrate.STABILITY_SEEDS)
    assert not used & set(ps.SEEDS)


# ---- market_names, load_returns ----

def test_market_names_are_four_indexes_then_twenty_stocks():
    names = ps.market_names()
    assert names[:4] == ["dow", "spx_1928", "spx", "ndx"]
    assert names[4:] == market_data.stock_tickers() and len(names) == 24


def test_load_returns_reads_indexes_and_stocks():
    assert ps.load_returns("spx").equals(market_data.returns("spx"))
    assert ps.load_returns("AAPL").equals(market_data.stock_returns("AAPL"))


def test_load_returns_keeps_the_last_days():
    full, cut = market_data.returns("ndx"), ps.load_returns("ndx", days=7)
    assert len(cut) == 7 and cut.equals(full.iloc[-7:])


@pytest.mark.parametrize("days", [0, -5, 2.5, True])
def test_load_returns_rejects_bad_days(days):
    with pytest.raises(ValueError, match="days must be"):
        ps.load_returns("spx", days=days)


def test_load_returns_rejects_unknown_market():
    with pytest.raises(ValueError, match="unknown market"):
        ps.load_returns("ftse")


# ---- return_gap ----

def test_return_gap_by_hand():
    # D days 0.03 and 0.01 (mean 0.02); R days 0.01, -0.01, 0.03 (mean 0.01): gap 0.01 a day
    r = [0.03, 0.01, 0.01, -0.01, 0.03]
    assert ps.return_gap(r, [True, True, False, False, False]) == pytest.approx(0.01 * YEAR)


def test_return_gap_flips_sign_when_parties_swap():
    rng = np.random.default_rng(1)
    r, dem = rng.normal(size=50), rng.random(50) < 0.4
    assert ps.return_gap(r, ~dem) == pytest.approx(-ps.return_gap(r, dem))


def test_return_gap_one_day_each():
    assert ps.return_gap([0.5, 0.25], [False, True]) == pytest.approx(-0.25 * YEAR)


def test_return_gap_matches_party_gap_mean_times_sd():
    # the project's other route to the same number: party_gap's mean is in daily sds
    rng = np.random.default_rng(2)
    r, dem = rng.normal(0, 0.01, 2000), np.arange(2000) < 900
    assert ps.return_gap(r, dem) == pytest.approx(politics.party_gap(r, dem)["mean"] * r.std())


@pytest.mark.parametrize("dem", [[True, True, True], [False, False, False]])
def test_return_gap_needs_both_parties(dem):
    with pytest.raises(ValueError, match="each party"):
        ps.return_gap([1.0, 2.0, 3.0], dem)


def test_return_gap_rejects_bad_input():
    with pytest.raises(ValueError, match="one party per return"):
        ps.return_gap([1.0, 2.0], [True, False, True])
    with pytest.raises(ValueError, match="NaN"):
        ps.return_gap([1.0, np.nan], [True, False])


# ---- bootstrap_interval ----

def test_bootstrap_of_a_fixed_gap_is_that_gap():
    # every D day returns 0.02 and every R day 0.01: any resample has a gap of exactly 0.01
    dem = np.tile([True] * 30 + [False] * 30, 20)
    b = ps.bootstrap_interval(np.where(dem, 0.02, 0.01), dem, block=50, draws=200)
    assert b["low"] == pytest.approx(0.01 * YEAR) and b["high"] == pytest.approx(0.01 * YEAR)
    assert b == dict(low=b["low"], high=b["high"], block=50, draws=200, used=200)


def test_bootstrap_matches_the_textbook_interval_for_independent_days():
    # independent days: gap = difference of two means, se = sd * sqrt(1/nD + 1/nR)
    rng = np.random.default_rng(3)
    n, dem = 4000, np.arange(4000) % 2 == 0
    r = rng.normal(0, 0.01, n) + np.where(dem, 0.001, 0.0)
    se = 0.01 * np.sqrt(1 / 2000 + 1 / 2000) * YEAR
    b = ps.bootstrap_interval(r, dem, block=1, draws=4000, seed=5)
    assert b["high"] - b["low"] == pytest.approx(2 * 1.96 * se, rel=0.08)
    assert (b["low"] + b["high"]) / 2 == pytest.approx(ps.return_gap(r, dem), abs=0.3 * se)


def test_bootstrap_matches_a_slow_rebuild_of_the_same_draws():
    # brute force: build each resample day by day with plain loops, then the percentiles
    rng = np.random.default_rng(4)
    r, dem = rng.normal(size=23), np.arange(23) % 3 == 0
    gaps, slow = [], np.random.default_rng(9)
    for _ in range(40):
        days = []
        for s in slow.integers(0, 23 - 5 + 1, 5):          # 5 blocks of 5 cover 23 days
            days += [s + j for j in range(5)]
        days = days[:23]
        d = [r[i] for i in days if dem[i]]
        g = [r[i] for i in days if not dem[i]]
        gaps.append((sum(d) / len(d) - sum(g) / len(g)) * YEAR)
    b = ps.bootstrap_interval(r, dem, block=5, draws=40, seed=9)
    assert b["used"] == 40
    assert [b["low"], b["high"]] == pytest.approx(list(np.percentile(gaps, [2.5, 97.5])))


def test_bootstrap_blocks_widen_the_interval_when_returns_come_in_spells():
    # returns that stay high or low for 100 days at a time: single-day resampling hides it
    rng = np.random.default_rng(6)
    r = np.repeat(rng.normal(0, 0.01, 60), 100) + rng.normal(0, 0.001, 6000)
    dem = np.repeat(np.arange(12) % 2 == 0, 500)
    width = lambda b: b["high"] - b["low"]
    assert width(ps.bootstrap_interval(r, dem, block=250, draws=500)) > \
        3 * width(ps.bootstrap_interval(r, dem, block=1, draws=500))


def test_bootstrap_repeats_with_the_same_seed_and_changes_with_another():
    rng = np.random.default_rng(7)
    r, dem = rng.normal(size=600), np.arange(600) < 250
    a = ps.bootstrap_interval(r, dem, block=20, draws=100, seed=1)
    assert a == ps.bootstrap_interval(r, dem, block=20, draws=100, seed=1)
    assert a != ps.bootstrap_interval(r, dem, block=20, draws=100, seed=2)


def test_bootstrap_skips_resamples_with_one_party():
    # one block the length of the data = the data itself, every time
    r, dem = np.arange(10.0), np.arange(10) < 5
    b = ps.bootstrap_interval(r, dem, block=10, draws=30)
    assert b["used"] == 30 and b["low"] == b["high"] == pytest.approx(ps.return_gap(r, dem))
    # 1 Democratic day in 10, blocks of 2: many resamples miss it, and are skipped
    dem = np.arange(10) == 0
    assert 2 <= ps.bootstrap_interval(r, dem, block=2, draws=300)["used"] < 300


def test_bootstrap_rejects_bad_input():
    r, dem = np.arange(10.0), np.arange(10) < 5
    with pytest.raises(ValueError, match="longer than the data"):
        ps.bootstrap_interval(r, dem, block=11)
    with pytest.raises(ValueError, match="block must be"):
        ps.bootstrap_interval(r, dem, block=0)
    with pytest.raises(ValueError, match="draws must be"):
        ps.bootstrap_interval(r, dem, block=2, draws=1.5)
    with pytest.raises(ValueError, match="one party per return"):
        ps.bootstrap_interval(r, dem[:9])
    with pytest.raises(ValueError, match="too short"):
        ps.bootstrap_interval(r, np.zeros(10, dtype=bool), block=2, draws=10)


# ---- leave_one_out ----

def span(name, party, a, b):
    return dict(name=name, party=party, a=a, b=b, start=pd.Timestamp("2001-01-20"),
                end=pd.Timestamp("2005-01-19"))


def test_leave_one_out_by_hand():
    # three presidencies of two days: A (D) 0.04, 0.02; B (R) 0.01, 0.01; C (D) 0.00, -0.02
    r = [0.04, 0.02, 0.01, 0.01, 0.00, -0.02]
    dem = [True, True, False, False, True, True]
    rows = ps.leave_one_out(r, dem, [span("A", "D", 0, 2), span("B", "R", 2, 4), span("C", "D", 4, 6)])
    # all days: D mean 0.01, R mean 0.01 -> gap 0.  Without A: D mean -0.01 -> gap -0.02 a day.
    # Without C: D mean 0.03 -> +0.02 a day. Without B there are no Republican days left.
    assert [t["gap_year"] for t in rows] == pytest.approx([-0.02 * YEAR, None, 0.02 * YEAR])
    assert [t["change"] for t in rows] == pytest.approx([-0.02 * YEAR, None, 0.02 * YEAR])
    assert rows[0] == dict(name="A", party="D", start="2001-01-20", end="2005-01-19", days=2,
                           gap_year=rows[0]["gap_year"], change=rows[0]["change"])


def test_leave_one_out_change_is_measured_from_the_full_gap():
    rng = np.random.default_rng(8)
    r, dem = rng.normal(size=300), np.arange(300) % 100 < 40
    row, = ps.leave_one_out(r, dem, [span("X", "D", 100, 140)])
    keep = np.r_[0:100, 140:300]                               # drop days 100..139 by hand
    assert row["gap_year"] == pytest.approx(ps.return_gap(r[keep], dem[keep]))
    assert row["change"] == pytest.approx(row["gap_year"] - ps.return_gap(r, dem))
    assert row["days"] == 40


def test_leave_one_out_with_no_presidencies_is_empty():
    assert ps.leave_one_out([1.0, 2.0], [True, False], []) == []


@pytest.mark.parametrize("a, b", [(-1, 2), (2, 2), (3, 7)])
def test_leave_one_out_rejects_spans_outside_the_data(a, b):
    with pytest.raises(ValueError, match="outside the data"):
        ps.leave_one_out(np.arange(6.0), [True, False] * 3, [span("A", "D", a, b)])


def test_leave_one_out_on_real_terms_covers_every_day_once():
    # with politics.terms, the presidencies' days add up to the whole series
    r = market_data.returns("ndx")
    rows = ps.leave_one_out(r.values, politics.democrat(r.index), politics.terms(r.index))
    assert sum(t["days"] for t in rows) == len(r)
    assert [t["name"] for t in rows] == ["Reagan", "Bush", "Clinton", "Bush", "Obama", "Trump",
                                         "Biden", "Trump"]


# ---- simulate ----

def test_simulate_uses_the_first_n_fresh_seeds_and_counts_runaways(monkeypatch):
    seen = {}

    def fake(p, seeds, days, pool=None):
        seen.update(p=p, seeds=tuple(seeds), days=days, pool=pool)
        return [None if s == 60001 else {"r": np.full(days, float(s))} for s in seeds]

    monkeypatch.setattr(experiments, "histories", fake)
    runs, bad = ps.simulate("settings", 4, 6, pool="a pool")
    assert seen == dict(p="settings", seeds=(60000, 60001, 60002, 60003), days=6, pool="a pool")
    assert bad == [60001]
    assert [x.tolist() for x in runs] == [[60000.0] * 6, [60002.0] * 6, [60003.0] * 6]


def test_simulate_fails_loudly_when_everything_runs_away(monkeypatch):
    monkeypatch.setattr(experiments, "histories", lambda p, seeds, days, pool=None: [None] * len(seeds))
    with pytest.raises(ValueError, match="every model history ran away"):
        ps.simulate("settings", 3, 10)


@pytest.mark.parametrize("n", [0, 1001, 2.5, True])
def test_simulate_rejects_bad_n(n):
    with pytest.raises(ValueError, match="n must be"):
        ps.simulate("settings", n, 10)


def test_simulate_runs_the_real_model():
    p = calibrate.load("spx")
    runs, bad = ps.simulate(p, 2, 300, SerialPool())
    assert len(runs) + len(bad) == 2 and all(len(x) == 300 for x in runs)
    direct = experiments.histories(p, [60000], 300, pool=SerialPool())[0]
    if 60000 not in bad:
        assert np.array_equal(runs[0], direct["r"])


# ---- yardstick ----

def gaps_of(values):
    return [dict(mean=v, vol=1.0, kurtosis=-v, crashes=10 * v) for v in values]


def test_yardstick_by_hand():
    # the yardstick's "mean" values are 0..100, so percentiles can be read off
    real = dict(mean=25.0, vol=1.0, kurtosis=5.0, crashes=2000.0)
    y = ps.yardstick(real, gaps_of(range(101)))
    assert y["n"] == 101
    assert y["pct"]["mean"] == pytest.approx(100 * 25.5 / 101)      # 25 below, 1 tie counts half
    assert y["pct"]["vol"] == pytest.approx(50.0)                   # all ties
    assert y["pct"]["kurtosis"] == pytest.approx(100.0)             # above every value (all <= 0)
    assert y["pct"]["crashes"] == pytest.approx(100.0)
    assert y["p5"]["mean"] == pytest.approx(5.0) and y["p95"]["mean"] == pytest.approx(95.0)
    assert y["p5"]["kurtosis"] == pytest.approx(-95.0) and y["p95"]["crashes"] == pytest.approx(950.0)
    assert y["values"]["mean"] == list(range(101)) and set(y["values"]) == set(politics.MEASURES)


def test_yardstick_with_one_value():
    y = ps.yardstick(dict(mean=0.0, vol=2.0, kurtosis=3.0, crashes=3.0), gaps_of([3.0]))
    assert y["pct"] == dict(mean=0.0, vol=100.0, kurtosis=100.0, crashes=0.0)
    assert y["p5"]["mean"] == y["p95"]["mean"] == 3.0


def test_yardstick_rejects_an_empty_list():
    with pytest.raises(ValueError, match="empty"):
        ps.yardstick(dict(mean=0.0, vol=1.0, kurtosis=0.0, crashes=0.0), [])


# ---- false_discovery ----

def cell(name, shifted, model):
    return dict(name=name, shifted=dict(pct=dict(zip(politics.MEASURES, shifted))),
                model=dict(pct=dict(zip(politics.MEASURES, model))))


def test_false_discovery_counts_by_hand():
    markets = [cell("a", [50, 96, 4.9, 5], [100, 0, 50, 50]),       # 5 and 95 are inside
               cell("b", [95, 95.1, 50, 50], [50, 50, 50, 50])]
    fd = ps.false_discovery(markets)
    assert fd["expected_share"] == pytest.approx(0.10)
    s, m = fd["shifted"], fd["model"]
    assert s["cells"] == m["cells"] == 8
    assert s["flagged"] == 3 and s["share"] == pytest.approx(3 / 8)
    assert s["flags"] == [["a", "vol", 96], ["a", "kurtosis", 4.9], ["b", "vol", 95.1]]
    assert s["by_measure"] == dict(mean=0, vol=2, kurtosis=1, crashes=0)
    assert m["flagged"] == 2 and m["flags"] == [["a", "mean", 100], ["a", "vol", 0]]


def test_false_discovery_other_limits_and_no_markets():
    fd = ps.false_discovery([cell("a", [50, 96, 4.9, 5], [100, 0, 50, 50])], low=1, high=99)
    assert fd["expected_share"] == pytest.approx(0.02)
    assert fd["shifted"]["flagged"] == 0 and fd["model"]["flagged"] == 2
    empty = ps.false_discovery([])
    assert empty["model"] == dict(cells=0, flagged=0, share=0.0, flags=[],
                                  by_measure=dict(mean=0, vol=0, kurtosis=0, crashes=0))


def test_false_discovery_finds_about_ten_percent_in_pure_luck():
    # percentiles drawn evenly from 0-100: about 10% land outside 5-95
    rng = np.random.default_rng(10)
    markets = [cell(str(i), rng.uniform(0, 100, 4), rng.uniform(0, 100, 4)) for i in range(5000)]
    assert ps.false_discovery(markets)["shifted"]["share"] == pytest.approx(0.10, abs=0.01)


# ---- market_result ----

@pytest.fixture(scope="module")
def ndx_result():
    r = market_data.returns("ndx")
    rng = np.random.default_rng(11)
    runs = [rng.normal(0, 0.01, len(r) + 50) for _ in range(6)]
    return r, runs, ps.market_result("ndx", r, runs)


def test_market_result_describes_the_market(ndx_result):
    r, runs, m = ndx_result
    dem = politics.democrat(r.index)
    assert (m["name"], m["kind"], m["days"]) == ("ndx", "index", len(r))
    assert (m["start"], m["end"]) == ("1985-10-02", "2026-10-30")
    assert m["dem_days"] + m["rep_days"] == len(r) and m["dem_days"] == int(dem.sum())
    assert m["party_changes"] == 6                 # 1993, 2001, 2009, 2017, 2021, 2025
    assert m["sd_daily"] == pytest.approx(np.std(r.values))
    assert m["real"] == politics.party_gap(r.values, dem)
    assert m["gap_year"] == pytest.approx(m["real"]["mean"] * m["sd_daily"])
    assert m["bootstrap"]["low"] < m["gap_year"] < m["bootstrap"]["high"]
    assert len(m["leave_one_out"]) == 8


def test_market_result_cuts_the_last_days_of_each_history(ndx_result):
    r, runs, m = ndx_result
    dem = politics.democrat(r.index)
    by_hand = [politics.party_gap(x[50:], dem)["mean"] for x in runs]      # 50 extra days in front
    assert m["model"]["n"] == 6 and m["model"]["values"]["mean"] == pytest.approx(by_hand)
    assert m["model"]["pct"]["mean"] == experiments.percentile(m["real"]["mean"], by_hand)


def test_market_result_shifted_yardstick_matches_politics(ndx_result):
    r, runs, m = ndx_result
    shifted = politics.shifted_gaps(r.values, politics.democrat(r.index))
    assert m["shifted"]["n"] == len(shifted)
    assert m["shifted"]["values"]["vol"] == pytest.approx([g["vol"] for g in shifted])


def test_market_result_marks_stocks_and_rejects_short_histories():
    r = market_data.stock_returns("AAPL")
    with pytest.raises(ValueError, match="shorter than the market"):
        ps.market_result("AAPL", r, [np.zeros(len(r) - 1)])
    rng = np.random.default_rng(12)
    assert ps.market_result("AAPL", r, [rng.normal(size=len(r))])["kind"] == "stock"


# ---- the printed tables ----

def test_tables_print_the_numbers(ndx_result, capsys):
    m = ndx_result[2]
    ps.print_headlines([m]); ps.print_measures([m]); ps.print_leave_one_out([m])
    ps.print_false_discovery(ps.false_discovery([m]))
    text = capsys.readouterr().out
    assert f"{m['gap_year']:+.1%}" in text and f"{m['bootstrap']['low']:+.1%} to" in text
    assert f"{m['real']['kurtosis']:+.2f}" in text
    biggest = max(m["leave_one_out"], key=lambda t: abs(t["change"]))
    assert f"without {biggest['name']} {biggest['start'][:4]}" in text
    assert "of 4 cells" in text


# ---- main, end to end ----

def test_main_small_run_saves_everything(tmp_path, capsys):
    out = tmp_path / "deep" / "results.json"
    got = ps.main(n=3, out=out, markets=["spx", "ndx", "MSFT"], days=3000, pool=SerialPool())
    saved = json.loads(out.read_text())                     # plain JSON, no NaN
    assert saved == json.loads(json.dumps(got))
    a = saved["about"]
    assert a["n_asked"] == 3 and a["seeds"] == [60000, 60002] and a["days_simulated"] == 3000
    assert a["histories"] + a["runaways"] == 3 and len(a["runaway_seeds"]) == a["runaways"]
    assert (a["block"], a["draws"], a["days"]) == (250, 2000, 3000)
    assert a["measures"] == politics.MEASURES
    assert [m["name"] for m in saved["markets"]] == ["spx", "ndx", "MSFT"]
    for m in saved["markets"]:
        assert m["days"] == 3000 and m["model"]["n"] == a["histories"]
        assert m["shifted"]["n"] == len(range(252, 3000 - 252 + 1, 63))
        for w in ("shifted", "model"):
            assert set(m[w]["pct"]) == set(politics.MEASURES)
            assert all(0 <= v <= 100 for v in m[w]["pct"].values())
    # the real gap does not depend on the model: check it against the data directly
    r = market_data.returns("spx").iloc[-3000:]
    dem = politics.democrat(r.index)
    assert saved["markets"][0]["gap_year"] == pytest.approx(
        (r[dem].mean() - r[~dem].mean()) * 252)
    fd = saved["false_discovery"]
    assert fd["model"]["cells"] == fd["shifted"]["cells"] == 12
    assert fd["shifted"]["flagged"] == sum(
        not 5 <= m["shifted"]["pct"][k] <= 95 for m in saved["markets"] for k in politics.MEASURES)
    assert "saved" in capsys.readouterr().out


def test_main_default_file_depends_on_days():
    # a quick run (days given) must not overwrite the full run's results
    assert ps.OUT.name == "scale_results.json" and ps.QUICK_OUT != ps.OUT
    assert ps.OUT.parent == ROOT / "scratch" / "party" == ps.QUICK_OUT.parent


def test_main_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError, match="at least one market"):
        ps.main(n=2, out=tmp_path / "x.json", markets=[])
    with pytest.raises(ValueError, match="unknown market"):
        ps.main(n=2, out=tmp_path / "x.json", markets=["ftse"])
    with pytest.raises(ValueError, match="n must be"):
        ps.main(n=0, out=tmp_path / "x.json", markets=["ndx"], days=3000, pool=SerialPool())
    assert not (tmp_path / "x.json").exists()
