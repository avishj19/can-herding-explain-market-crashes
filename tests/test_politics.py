"""Tests for politics.py (the political-party extension, Fig 12)."""
import numpy as np
import pandas as pd
import pytest

from herding import politics


def test_party_on_known_dates():
    # inauguration days switch the party; the day before belongs to the old president
    dates = ["1933-03-03", "1933-03-04", "1953-01-19", "1953-01-20", "2009-01-19",
             "2009-01-20", "2025-01-17", "2025-01-21", "2026-10-30"]
    assert politics.democrat(dates).tolist() == [False, True, True, False, False, True, True,
                                                 False, False]


def test_dates_before_the_table_are_rejected():
    with pytest.raises(ValueError, match="not covered"):
        politics.democrat(["1890-01-02"])


def test_party_before_1923_matches_the_record():
    # Cleveland (D) to 3 Mar 1897; McKinley, T. Roosevelt, Taft (R) to 3 Mar 1913;
    # Wilson (D) to 3 Mar 1921; Harding, Coolidge (R) after. McKinley died 14 Sep 1901
    # (T. Roosevelt took over, same party); Harding died 2 Aug 1923.
    dates = ["1896-05-26", "1897-03-03", "1897-03-04", "1901-09-13", "1901-09-16", "1909-03-04",
             "1913-03-03", "1913-03-04", "1921-03-03", "1921-03-04", "1923-08-01", "1923-08-03"]
    assert politics.democrat(dates).tolist() == [True, True, False, False, False, False,
                                                 False, True, True, False, False, False]


def test_the_table_is_in_date_order_with_known_parties():
    starts = [pd.Timestamp(d) for d, _, _ in politics.PRESIDENTS]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    assert {p for _, p, _ in politics.PRESIDENTS} == {"D", "R"}
    # 15 changes of party from 1896 on: 1897, 1913, 1921, then the 12 since 1928
    dow_days = pd.bdate_range("1896-05-26", "2026-10-30")
    assert len(politics.switches(dow_days)[0]) == 16


def test_switches_from_1928_on():
    dates = pd.bdate_range("1928-01-03", "2026-10-30")
    starts, party = politics.switches(dates)
    # 1928 starts Republican, then 12 changes of party (1933, 1953, ... 2025)
    assert party[0] == "R" and len(starts) == 13
    assert all(a != b for a, b in zip(party[:-1], party[1:]))      # parties alternate


def test_party_gap_by_hand():
    # D days: alternating 3 / 1 (mean 2, sd 1); R days: alternating 1 / -1 (mean 0, sd 1)
    dem = np.array([True] * 600 + [False] * 600)
    r = np.concatenate([np.tile([3.0, 1.0], 300), np.tile([1.0, -1.0], 300)])
    g = politics.party_gap(r, dem)
    sd = np.sqrt(2.0)                                   # all days: mean 1, values 3,1,1,-1 -> var 2
    assert r.std() == pytest.approx(sd)
    assert g["mean"] == pytest.approx(2.0 / sd * politics.DAYS_PER_YEAR)
    assert g["vol"] == pytest.approx(1.0)
    assert g["kurtosis"] == pytest.approx(0.0)          # same two-point shape on both sides
    assert g["crashes"] == 0                            # nothing falls 4 sds below the mean


def test_party_gap_ignores_the_scale():
    # a market twice as volatile with the same shape gives the same gaps
    g = np.random.default_rng(0)
    r, dem = g.standard_t(3, 5000) * 0.01, g.random(5000) < 0.5
    a, b = politics.party_gap(r, dem), politics.party_gap(3 * r, dem)
    for k in politics.MEASURES:
        assert a[k] == pytest.approx(b[k])


def test_no_gap_when_both_parties_see_the_same_market():
    # the same random market on both sides: the gaps are near 0 (vol ratio near 1)
    r = np.random.default_rng(1).standard_normal(200_000)
    dem = np.arange(200_000) % 2 == 0
    g = politics.party_gap(r, dem)
    assert abs(g["vol"] - 1) < 0.02 and abs(g["kurtosis"]) < 0.1
    # mean gap: one sd of a 100,000-day average is 1/sqrt(100,000) a day, about 1.1 a year
    assert abs(g["mean"]) < 4 * np.sqrt(2 / 100_000) * politics.DAYS_PER_YEAR


@pytest.mark.parametrize("dem", [np.ones(1000, bool), np.arange(1000) < 300, np.ones(999, bool)])
def test_party_gap_needs_enough_days_of_each(dem):
    with pytest.raises(ValueError):
        politics.party_gap(np.random.default_rng(0).standard_normal(1000), dem)


def test_shifted_calendars():
    g = np.random.default_rng(2)
    r, dem = g.standard_normal(4000), np.arange(4000) // 1000 % 2 == 0
    out = politics.shifted_gaps(r, dem)
    assert len(out) == len(range(252, 4000 - 252 + 1, 63))
    # sliding by a whole cycle (2,000 days) gives back the real calendar exactly
    assert politics.party_gap(r, np.roll(dem, 2000)) == politics.party_gap(r, dem)
    with pytest.raises(ValueError, match="too few shifts"):
        politics.shifted_gaps(r[:1200], dem[:1200])


def test_terms_cut_the_days_at_inaugurations():
    dates = pd.bdate_range("2016-01-04", "2026-10-30")
    t = politics.terms(dates)
    assert [x["name"] for x in t] == ["Obama", "Trump", "Biden", "Trump"]
    assert [x["party"] for x in t] == ["D", "R", "D", "R"]
    # the terms tile the days with no gap or overlap
    assert t[0]["a"] == 0 and t[-1]["b"] == len(dates)
    assert all(x["b"] == y["a"] for x, y in zip(t[:-1], t[1:]))
    assert t[1]["start"] == pd.Timestamp("2017-01-20") and t[1]["end"] == pd.Timestamp("2021-01-19")


def test_permutation_test_by_hand():
    from test_analysis import load_script
    pt = load_script("president_terms.py")
    # D = [2, 0], R = [0, 0]: observed +1; of the 6 ways to pick 2 of 4 as "D", the 3 that
    # include the 2 give +1 and the 3 that do not give -1, so every one is as extreme: p = 1
    assert pt.permutation_p([2, 0, 0, 0], ["D", "D", "R", "R"]) == (1.0, 1.0)
    # D = [3, 3], R = [0, 0]: only the real labelling and its mirror are that extreme: p = 2/6
    diff, p = pt.permutation_p([3, 3, 0, 0], ["D", "D", "R", "R"])
    assert diff == 3.0 and p == pytest.approx(2 / 6)


# ---------------------------------------------------------------- shared lookups (November 2026)
def test_president_on_known_days():
    assert politics.president_on("1933-03-03") == ("R", "Hoover")
    assert politics.president_on("1933-03-04") == ("D", "Roosevelt")
    assert politics.president_on("1905-06-01") == ("R", "T. Roosevelt")
    assert politics.president_on(pd.Timestamp("2026-10-30")) == ("R", "Trump")
    for bad in ("1890-01-01", None):
        with pytest.raises(ValueError, match="not covered"):
            politics.president_on(bad)


def test_president_on_agrees_with_democrat_on_every_dow_day():
    days = pd.bdate_range("1896-05-26", "2026-10-30")[::37]
    assert [politics.president_on(d)[0] == "D" for d in days] == politics.democrat(days).tolist()


def test_return_gap_by_hand_and_against_party_gap():
    # D days average 0.03, R days 0.01: gap 0.02 a day, x 252 a year
    assert politics.return_gap([0.02, 0.04, 0.01, 0.01, 0.01], [True, True, False, False, False]) \
        == pytest.approx(0.02 * politics.DAYS_PER_YEAR)
    g = np.random.default_rng(3)
    r, dem = g.standard_t(4, 4000) * 0.01, g.random(4000) < 0.5
    assert politics.return_gap(r, dem) == pytest.approx(politics.party_gap(r, dem)["mean"] * r.std())
    for bad_r, bad_dem in [([1.0, 2.0], [True, True]), ([1.0, 2.0], [True, False, True]), ([1.0, np.nan], [True, False])]:
        with pytest.raises(ValueError):
            politics.return_gap(bad_r, bad_dem)
