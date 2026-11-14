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
NEWS = RNG.normal(0, 0.01, 50_000)
NOISE = RNG.normal(0, 0.01, 50_000)          # independent of NEWS, same variance


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


# ---- elections -----------------------------------------------------------------

@pytest.mark.parametrize("year, date", [
    (1896, "1896-11-03"), (1932, "1932-11-08"), (1960, "1960-11-08"), (2000, "2000-11-07"),
    (2008, "2008-11-04"), (2016, "2016-11-08"), (2020, "2020-11-03"), (2024, "2024-11-05"),
    (1904, "1904-11-08"),      # 1 November a Tuesday: the latest possible date, the 8th
    (1988, "1988-11-08"), (1976, "1976-11-02"),   # 1 November a Monday: the earliest, the 2nd
])
def test_election_day_known_dates(year, date):
    assert nr.election_day(year) == pd.Timestamp(date)


def test_election_day_is_always_a_tuesday_from_2_to_8_november():
    years = list(range(1896, 2025, 4))
    assert len(years) == 33
    for y in years:
        d = nr.election_day(y)
        assert d.day_name() == "Tuesday" and d.month == 11 and 2 <= d.day <= 8


@pytest.mark.parametrize("year", [1897, 2022, 2001.5, True, 1800])
def test_election_day_rejects_other_years(year):
    with pytest.raises(ValueError):
        nr.election_day(year)


# The public record, typed here independently of politics.PRESIDENTS.
WINNERS = {1896: "R", 1900: "R", 1904: "R", 1908: "R", 1912: "D", 1916: "D", 1920: "R", 1924: "R",
           1928: "R", 1932: "D", 1936: "D", 1940: "D", 1944: "D", 1948: "D", 1952: "R", 1956: "R",
           1960: "D", 1964: "D", 1968: "R", 1972: "R", 1976: "D", 1980: "R", 1984: "R", 1988: "R",
           1992: "D", 1996: "D", 2000: "R", 2004: "R", 2008: "D", 2012: "D", 2016: "R", 2020: "D",
           2024: "R"}
NAMES = {1896: "McKinley", 1904: "T. Roosevelt", 1912: "Wilson", 1924: "Coolidge", 1932: "Roosevelt",
         1948: "Truman", 1960: "Kennedy", 1964: "Johnson", 1980: "Reagan", 2000: "Bush",
         2008: "Obama", 2016: "Trump", 2020: "Biden", 2024: "Trump"}
CHANGED = {1896, 1912, 1920, 1932, 1952, 1960, 1968, 1976, 1980, 1992, 2000, 2008, 2016, 2020, 2024}


def test_winners_match_the_public_record():
    assert {y: nr.winner(y)[0] for y in WINNERS} == WINNERS
    assert {y: nr.winner(y)[1] for y in NAMES} == NAMES


def test_changed_party_matches_the_public_record():
    assert {y for y in WINNERS if nr.changed_party(y)} == CHANGED


def test_president_on_boundaries():
    assert nr.president_on("1933-03-03") == ("R", "Hoover")
    assert nr.president_on("1933-03-04") == ("D", "Roosevelt")      # inauguration day itself
    assert nr.president_on("1963-11-22") == ("D", "Johnson")
    with pytest.raises(ValueError):
        nr.president_on("1890-01-01")
    with pytest.raises(ValueError):
        nr.winner(1999)


# A hand-made week: returns 1, 2, 4, 8, 16, 32 (x 0.01) on Mon 2 - Mon 9 Nov 2020,
# so every sum of consecutive days is different and easy to check.
WEEK = pd.Series([0.01, 0.02, 0.04, 0.08, 0.16, 0.32],
                 index=pd.to_datetime(["2020-11-02", "2020-11-03", "2020-11-04", "2020-11-05",
                                       "2020-11-06", "2020-11-09"]))


def test_reaction_when_the_market_is_open_on_the_day():
    # Tuesday 3rd open: Monday's close to Wednesday's close = Tuesday + Wednesday
    assert nr.reaction(WEEK, "2020-11-03") == (pytest.approx(0.06), 2)
    assert nr.reaction(WEEK, "2020-11-03", k=3) == (pytest.approx(0.02 + 0.04 + 0.08 + 0.16), 4)


def test_reaction_when_the_market_is_closed_on_the_day():
    closed = WEEK.drop(pd.Timestamp("2020-11-03"))
    # Monday's close to Wednesday's close is now Wednesday's return alone
    assert nr.reaction(closed, "2020-11-03") == (pytest.approx(0.04), 1)
    assert nr.reaction(closed, "2020-11-03", k=2) == (pytest.approx(0.12), 2)
    assert nr.reaction(WEEK, "2020-11-07") == (pytest.approx(0.32), 1)    # a Saturday


def test_reaction_rejects_windows_past_the_data():
    with pytest.raises(ValueError):
        nr.reaction(WEEK, "2020-11-09")            # nothing after the last day
    with pytest.raises(ValueError):
        nr.reaction(WEEK, "2020-11-03", k=5)
    with pytest.raises(ValueError):
        nr.reaction(WEEK, "2020-11-03", k=0)


def test_reaction_on_the_real_dow_2008(dow):
    # independent: straight from the price file, 3 Nov 2008 close to 5 Nov 2008 close
    close = pd.read_csv(market_data.MARKETS["dow"], parse_dates=["Date"], index_col="Date")["Close"]
    want = np.log(close["2008-11-05"] / close["2008-11-03"])
    assert nr.reaction(dow, nr.election_day(2008)) == (pytest.approx(want), 2)
    assert pd.Timestamp("1960-11-08") not in dow.index             # closed on election day then
    want = np.log(close["1960-11-09"] / close["1960-11-07"])
    assert nr.reaction(dow, nr.election_day(1960)) == (pytest.approx(want), 1)


def test_window_sums_by_hand():
    assert list(nr.window_sums([1, 2, 4, 8], 2)) == [3, 6, 12]
    assert list(nr.window_sums([1, 2, 4, 8], 1)) == [1, 2, 4, 8]
    assert list(nr.window_sums([1, 2, 4, 8], 4)) == [15]
    for n in (0, 5, 1.5):
        with pytest.raises(ValueError):
            nr.window_sums([1, 2, 4, 8], n)


def test_yardstick_uses_only_days_before_the_event():
    assert list(nr.yardstick(WEEK, "2020-11-05", 2)) == pytest.approx([0.03, 0.06])   # Mon-Wed only
    assert list(nr.yardstick(WEEK, "2020-11-05", 1, back=2)) == pytest.approx([0.02, 0.04])
    assert list(nr.yardstick(WEEK, "2020-11-05", 1, back=99)) == pytest.approx([0.01, 0.02, 0.04])
    with pytest.raises(ValueError):
        nr.yardstick(WEEK, "2020-11-03", 2)        # only one day before


def test_election_row_on_the_real_dow(dow):
    row = nr.election_row(dow, 2008)
    assert (row["year"], row["day"], row["winner"], row["party"]) == (2008, "2008-11-04", "Obama", "D")
    assert row["changed"] and row["open_that_day"] and row["trading_days"] == 2 and not row["result_unknown"]
    assert row["percent"] == pytest.approx(100 * (np.exp(row["reaction"]) - 1))
    # the percentile, recomputed here: share of the 749 two-day windows before with a smaller size
    before = dow[dow.index < "2008-11-04"].values[-750:]
    two_day = np.abs(before[1:] + before[:-1])
    assert row["pct_abs_near"] == pytest.approx(100 * np.mean(two_day < abs(row["reaction"])), abs=0.2)
    assert nr.election_row(dow, 2000)["result_unknown"]
    assert not nr.election_row(dow, 1960)["open_that_day"]


def test_group_z_by_hand():
    assert nr.group_z([50.0]) == 0
    assert nr.group_z([40, 60]) == 0
    assert nr.group_z([100.0] * 12) == pytest.approx(50 / (100 / 12))    # se = 28.87 / sqrt(12)
    assert nr.group_z([0.0] * 3) == pytest.approx(-3.0)                  # -50 / (28.87 / 1.732)
    u = np.random.default_rng(1).uniform(0, 100, (4000, 33))             # ordinary days: z ~ N(0, 1)
    z = np.array([nr.group_z(row) for row in u])
    assert abs(z.mean()) < 0.06 and z.std() == pytest.approx(1.0, abs=0.05)
    with pytest.raises(ValueError):
        nr.group_z([])


def test_split_row_by_hand():
    rows = [dict(percent=2.0, pct_abs_near=95.0, pct_signed_near=90.0),
            dict(percent=-1.0, pct_abs_near=50.0, pct_signed_near=20.0),
            dict(percent=0.5, pct_abs_near=20.0, pct_signed_near=70.0)]
    s = nr.split_row(rows)
    assert (s["n"], s["up"], s["above_90"]) == (3, 2, 1)
    assert s["mean"] == pytest.approx(0.5) and s["median"] == 0.5 and s["mean_abs"] == pytest.approx(3.5 / 3)
    assert s["pct_abs"] == 50.0 and s["pct_signed"] == 70.0
    assert s["z_abs"] == pytest.approx((55 - 50) / (100 / np.sqrt(12) / np.sqrt(3)))
    with pytest.raises(ValueError):
        nr.split_row([])


def test_election_part_covers_33_elections_and_the_groups_add_up(dow):
    part = nr.election_part(dow, range(1896, 2025, 4))
    assert len(part["elections"]) == 33 * len(nr.WINDOWS)
    for k in nr.WINDOWS:
        s = part["splits"][str(k)]
        assert s["all elections"]["n"] == 33 and s["all but 2000 (result unknown)"]["n"] == 32
        assert s["Republican won"]["n"] == 18 and s["Democrat won"]["n"] == 15
        assert s["White House changed party"]["n"] == 15 and s["White House kept by party"]["n"] == 18


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


def test_fill_moves_computes_the_percent_from_the_returns():
    r = pd.Series([np.log(1.10), np.log(0.75)], index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    ev = pd.DataFrame(dict(date=["2020-01-03", "2020-01-02"], move_percent=[0.0, 0.0]))
    assert list(nr.fill_moves(ev, r)["move_percent"]) == [-25.0, 10.0]
    with pytest.raises(ValueError):
        nr.fill_moves(pd.DataFrame(dict(date=["2020-01-05"], move_percent=[0.0])), r)


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
    nr.print_elections(nr.election_part(dow, [1956, 1960, 2000], windows=(1,)))
    nr.print_history(nr.history_part(dow))
    out = capsys.readouterr().out
    assert "0.123 (0.100 to 0.200)" in out and "Kennedy" in out and "result unknown" in out
    assert "-22.61%" in out and "(cause not confirmed)" in out


@pytest.mark.slow
def test_quick_main_writes_the_quick_file_and_leaves_the_full_results_alone(capsys):
    before = nr.RESULTS.read_bytes() if nr.RESULTS.exists() else None
    out = nr.main(days=3000, n=2)
    assert (nr.RESULTS.read_bytes() if nr.RESULTS.exists() else None) == before
    saved = json.loads(nr.QUICK_RESULTS.read_text())
    assert saved["part1"]["days"] == 3000 and saved["part1"]["seeds"] == [62000, 62001]
    assert set(saved) == {"definition", "part1", "part2", "part3", "seconds"}
    assert out["part3"]["counts"] == saved["part3"]["counts"]
    assert "LOGICAL" in capsys.readouterr().out
