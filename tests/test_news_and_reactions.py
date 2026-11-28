"""Tests for scripts/analysis/news_and_reactions.py: every function, with answers
worked out independently (by hand, from the public record, or from data built so
the answer is known)."""
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from herding import market_data

spec = importlib.util.spec_from_file_location(
    "news_and_reactions", ROOT / "scripts" / "analysis" / "news_and_reactions.py")
nr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nr)

RNG = np.random.default_rng(7)
# 200,000 days (was 50,000): the overnight stress test (Nov 2026) reran these tests with 39
# other seeds, and three checks had less than 4 sd of room; 4 times the days halves the sd
NEWS = RNG.normal(0, 0.01, 200_000)
NOISE = RNG.normal(0, 0.01, 200_000)         # independent of NEWS, same variance


@pytest.fixture(scope="module")
def dow():
    return market_data.returns("dow")


# ---- variance_explained -------------------------------------------------------

def test_variance_explained_is_1_when_the_return_is_the_news():
    assert nr.variance_explained(NEWS, NEWS) == pytest.approx(1.0, abs=1e-12)
    assert nr.variance_explained(3 * NEWS + 0.002, NEWS) == pytest.approx(1.0, abs=1e-12)  # scale, shift


def test_variance_explained_is_about_0_for_independent_series():
    assert nr.variance_explained(NOISE, NEWS) < 0.001          # chance level is about 1/50,000


def test_variance_explained_is_about_half_with_equal_variance_noise():
    assert nr.variance_explained(NEWS + NOISE, NEWS) == pytest.approx(0.5, abs=0.01)


def test_variance_explained_lags_find_yesterdays_news():
    late = np.concatenate([[0.0], NEWS[:-1]])                  # the price reacts one day late
    assert nr.variance_explained(late, NEWS) < 0.001           # same-day news explains nothing
    assert nr.variance_explained(late, NEWS, lags=1) == pytest.approx(1.0, abs=1e-9)
    half = 0.5 * NEWS + 0.5 * late                             # half today, half tomorrow
    assert nr.variance_explained(half, NEWS) == pytest.approx(0.5, abs=0.01)
    assert nr.variance_explained(half, NEWS, lags=1) == pytest.approx(1.0, abs=1e-9)


def test_variance_explained_by_hand():
    # r = (1, 2, 4), news = (1, 2, 3): best line 1.5 * news - 2/3, leftovers (1, -2, 1)/6,
    # leftover sum of squares 1/6 against a total of 14/3, so R squared = 1 - 1/28
    assert nr.variance_explained([1, 2, 4], [1, 2, 3]) == pytest.approx(1 - 1 / 28)


@pytest.mark.parametrize("r, news, lags", [
    ([1, 2, 3], [1, 2], 0),             # different lengths
    ([1, 2], [1, 2], 0),                # too short
    ([1, 1, 1], [1, 2, 3], 0),          # returns never change
    ([1, 2, np.nan], [1, 2, 3], 0),     # NaN
    ([1, 2, 3, 4], [1, 2, 3, 5], -1),   # negative lags
    ([1, 2, 3, 4], [1, 2, 3, 5], 1.5),  # fractional lags
    ([1, 2, 3, 4], [1, 2, 3, 5], 2),    # too short for 2 lags
])
def test_variance_explained_rejects_bad_input(r, news, lags):
    with pytest.raises(ValueError):
        nr.variance_explained(r, news, lags)


# ---- biggest, reaction_ratio, big_news_share, crowd_on_overreactions -----------

def test_biggest_marks_the_largest_absolute_values():
    x = [1, -9, 3, 0, 5, -2, 4, 8, -7, 6]
    assert list(np.flatnonzero(nr.biggest(x, 0.2))) == [1, 7]       # |-9| and 8
    assert list(np.flatnonzero(nr.biggest(x, 0.01))) == [1]         # never fewer than one day
    assert nr.biggest(x, 1).all()
    assert list(nr.biggest([2, -2, 1], 0.34)) == [True, True, False]   # ties at the cut are kept


@pytest.mark.parametrize("x, share", [([], 0.5), ([1, 2], 0), ([1, 2], 1.5), ([1, np.inf], 0.5)])
def test_biggest_rejects_bad_input(x, share):
    with pytest.raises(ValueError):
        nr.biggest(x, share)


def test_reaction_ratio_by_hand():
    r = np.array([0.1, -0.5, 0.2, 0.4, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    news = np.array([0.1, 0.1, 0.1, -0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert nr.reaction_ratio(r, news, 0.2) == pytest.approx(3.5)    # days 1 and 3: ratios 5 and 2
    assert nr.reaction_ratio(r, news, 0.1) == pytest.approx(5.0)    # day 1 alone
    assert nr.reaction_ratio(NEWS, NEWS) == pytest.approx(1.0)      # logical market: exactly 1
    assert nr.reaction_ratio(4 * NEWS, NEWS) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        nr.reaction_ratio([1.0, 0.1], [0.0, 1.0], 0.5)              # zero news on the big day


def test_big_news_share_by_hand():
    r = np.array([9, 8, 1, 1, 1, 1, 1, 1, 1, 1.0])
    news = np.array([9, 1, 8, 1, 1, 1, 1, 1, 1, 1.0])
    # biggest 20% of moves: days 0, 1. biggest 20% of news: days 0, 2. Overlap: day 0 only.
    assert nr.big_news_share(r, news, 0.2, 0.2) == pytest.approx(0.5)
    assert nr.big_news_share(NEWS, NEWS) == 1.0                     # logical market
    assert nr.big_news_share(NOISE, NEWS) == pytest.approx(0.05, abs=0.03)   # chance = 5%


def test_crowd_on_overreactions_by_hand():
    r = np.array([0.5, -0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    news = np.array([0.1, -0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    z = np.array([0.6, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    out = nr.crowd_on_overreactions(r, news, z, share=0.2, over=2.0)
    # big days 0 and 1; only day 0 moved more than 2x its news (5x against 1.33x)
    assert out["over_share"] == pytest.approx(0.5)
    assert out["z_over"] == pytest.approx(0.6)
    assert out["z_all"] == pytest.approx(0.16)
    assert out["z_times"] == pytest.approx(0.6 / 0.16)


def test_crowd_on_overreactions_edges():
    none = nr.crowd_on_overreactions(NEWS, NEWS, np.zeros(len(NEWS)))   # logical, no chartists
    assert none["over_share"] == 0 and np.isnan(none["z_over"]) and np.isnan(none["z_times"])
    with pytest.raises(ValueError):
        nr.crowd_on_overreactions([1, 2], [1, 2], [0.1])


# ---- news_measures, summarise, model_part --------------------------------------

def test_news_measures_on_a_logical_market():
    lf = np.concatenate([[0.0], np.cumsum(NEWS)])
    m = nr.news_measures(dict(r=NEWS.copy(), lf=lf, z=np.zeros(len(lf))))
    assert set(m) == set(nr.MEASURES)
    assert m["same_day"] == pytest.approx(1.0) and m["with_lags"] == pytest.approx(1.0)
    assert m["vol_ratio"] == pytest.approx(1.0) and m["ratio"] == pytest.approx(1.0)
    assert m["big_news"] == 1.0 and m["over_share"] == 0.0 and m["z_all"] == 0.0


def test_news_measures_on_an_overreacting_market():
    lf = np.concatenate([[0.0], np.cumsum(NEWS)])
    m = nr.news_measures(dict(r=NEWS + 3 * NOISE, lf=lf, z=np.full(len(lf), 0.2)))
    assert m["same_day"] == pytest.approx(0.1, abs=0.01)             # 1 / (1 + 9)
    assert m["vol_ratio"] == pytest.approx(np.sqrt(10), rel=0.02)
    assert m["ratio"] > 3 and m["over_share"] > 0.8
    assert m["z_all"] == pytest.approx(0.2) and m["z_over"] == pytest.approx(0.2)


def test_news_measures_needs_one_more_value_than_returns():
    with pytest.raises(ValueError):
        nr.news_measures(dict(r=NEWS, lf=np.cumsum(NEWS), z=np.zeros(len(NEWS))))


def test_summarise_median_and_range():
    out = nr.summarise([dict(a=float(i), b=np.nan if i else 5.0) for i in range(101)])
    assert out["a"] == dict(median=50.0, low=5.0, high=95.0)
    assert out["b"] == dict(median=5.0, low=5.0, high=5.0)          # NaN histories are skipped
    assert np.isnan(nr.summarise([dict(a=np.nan)])["a"]["median"])
    with pytest.raises(ValueError):
        nr.summarise([])


@pytest.mark.slow
def test_model_part_herding_breaks_the_link_to_the_news():
    from herding import calibrate
    part = nr.model_part(calibrate.load("spx"), nr.SEEDS[:2], 5000)
    w, wo = part["with_chartists"], part["without_chartists"]
    assert w["histories"] + w["runaways"] == 2 and wo["histories"] == 2
    assert wo["same_day"]["median"] > w["same_day"]["median"] + 0.2
    assert w["ratio"]["median"] > 2 * wo["ratio"]["median"]
    assert wo["z_all"]["median"] == 0


# ---- the biggest days and the events file ---------------------------------------

def test_biggest_days_by_hand():
    r = pd.Series([0.01, -0.20, 0.05, np.log(1.10)], index=pd.date_range("2020-01-01", periods=4))
    top = nr.biggest_days(r, 2)
    assert list(top.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-04")]
    assert list(top) == pytest.approx([100 * (np.exp(-0.20) - 1), 10.0])
    for n in (0, 5, 2.5):
        with pytest.raises(ValueError):
            nr.biggest_days(r, n)


def test_biggest_days_of_the_dow_starts_with_black_monday(dow):
    top = nr.biggest_days(dow, 20)
    assert top.index[0] == pd.Timestamp("1987-10-19") and top.iloc[0] == pytest.approx(-22.61, abs=0.005)
    assert np.all(np.diff(np.abs(dow[top.index].values)) <= 0)        # largest first


def test_events_file_is_well_formed():
    ev = nr.load_events()
    assert len(ev) == 20 and list(ev.columns) == nr.COLUMNS
    assert set(ev["kind"]) <= set(nr.KINDS)
    for row in ev.itertuples():
        if row.kind == "unverified":
            assert row.event == ""
        else:
            assert len(row.event) > 20 and row.event.endswith(".") and "\n" not in row.event
            assert all(u.startswith("https://") and " " not in u for u in row.source_url.split(" | "))


def test_events_file_matches_the_data(dow):
    ev = nr.load_events()
    nr.check_events(ev, dow)                                         # the function's own check
    # and independently, straight from the price file
    close = pd.read_csv(market_data.MARKETS["dow"], parse_dates=["Date"], index_col="Date")["Close"]
    pct = 100 * close.pct_change()
    assert set(ev["date"]) == set(pct.abs().sort_values(ascending=False).index[:20]) or \
        set(ev["date"]) == set(np.log(close).diff().abs().sort_values(ascending=False).index[:20])
    for row in ev.itertuples():
        assert row.move_percent == pytest.approx(pct[row.date], abs=0.005)


def write_events(tmp_path, rows):
    path = tmp_path / "events.csv"
    pd.DataFrame(rows, columns=nr.COLUMNS).to_csv(path, index=False)
    return path


GOOD = ["2020-01-02", 1.0, "Something happened that day.", "news", "https://example.org/a"]


@pytest.mark.parametrize("row", [
    GOOD[:3] + ["gossip", GOOD[4]],                      # unknown kind
    GOOD[:2] + ["", "news", GOOD[4]],                    # no event sentence
    GOOD[:3] + ["unverified", GOOD[4]],                  # unverified but claims an event
    GOOD[:4] + ["from memory"],                          # no web source
    GOOD[:4] + [""],                                     # empty source
])
def test_load_events_rejects_bad_rows(tmp_path, row):
    assert len(nr.load_events(write_events(tmp_path, [GOOD]))) == 1   # the good row passes
    with pytest.raises(ValueError):
        nr.load_events(write_events(tmp_path, [row]))


def test_load_events_rejects_bad_tables(tmp_path):
    with pytest.raises(ValueError):
        nr.load_events(write_events(tmp_path, [GOOD, GOOD]))          # a date twice
    path = tmp_path / "cols.csv"
    pd.DataFrame([GOOD[:4]], columns=nr.COLUMNS[:4]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        nr.load_events(path)                                          # a column missing


def test_check_events_catches_a_wrong_day_and_a_typed_number(dow):
    ev = nr.load_events()
    wrong = ev.copy()
    wrong.loc[0, "move_percent"] += 0.02
    with pytest.raises(ValueError):
        nr.check_events(wrong, dow)
    other = ev.copy()
    other.loc[0, "date"] = pd.Timestamp("2019-06-03")                 # an ordinary day
    with pytest.raises(ValueError):
        nr.check_events(other, dow)


def test_kind_counts_by_hand():
    ev = pd.DataFrame(dict(kind=["news", "news", "rebound"]))
    assert nr.kind_counts(ev) == {"news": 2, "policy": 0, "no clear news": 0, "rebound": 1, "unverified": 0}
    assert sum(nr.kind_counts(nr.load_events()).values()) == 20


def test_history_part_reports_every_day(dow):
    part = nr.history_part(dow)
    assert len(part["days"]) == 20 and sum(part["counts"].values()) == 20
    assert part["days"][0]["date"] == "1987-10-19" and part["days"][0]["kind"] == "no clear news"


# ---- printing and main -----------------------------------------------------------

def test_print_functions_show_the_numbers(dow, capsys):
    m = {k: dict(median=0.123, low=0.1, high=0.2) for k in nr.MEASURES}
    nr.print_model(dict(with_chartists=dict(histories=2, runaways=0, **m),
                        without_chartists=dict(histories=2, runaways=0, **m)))
    nr.print_history(nr.history_part(dow))
    out = capsys.readouterr().out
    assert "0.123 (0.100 to 0.200)" in out
    assert "-22.61%" in out and "(cause not confirmed)" in out


@pytest.mark.slow
def test_quick_main_writes_the_quick_file_and_leaves_the_full_results_alone(capsys):
    before = nr.RESULTS.read_bytes() if nr.RESULTS.exists() else None
    out = nr.main(days=3000, n=2)
    assert (nr.RESULTS.read_bytes() if nr.RESULTS.exists() else None) == before
    saved = json.loads(nr.QUICK_RESULTS.read_text())
    assert saved["part1"]["days"] == 3000 and saved["part1"]["seeds"] == [62000, 62001]
    assert set(saved) == {"definition", "part1", "part2", "seconds"}
    assert out["part2"]["counts"] == saved["part2"]["counts"]
    assert "LOGICAL" in capsys.readouterr().out
