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
        politics.democrat(["1920-01-02"])


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
