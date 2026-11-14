"""Heavy tests for politics.py (the political-party extension, Fig 12).

Part A: is the code right? Thousands of random cases, each checked against an answer
        worked out a different way (a slow loop, scipy, or a hand-built calendar).
Part B: is the METHOD right? Markets with NO party effect are fed through the
        shifted-calendar test to count false alarms, and markets with a PLANTED gap are
        fed through it to see how big a gap it can detect. (Slow: about a minute.)

Run:  python3 -m pytest tests/test_politics_heavy.py -q -s      (-s shows the tables)
Quick part only:  add  -m "not slow"

Words used below:
  false-positive rate  how often a test cries "unusual!" when nothing is going on
  power                how often a test spots an effect that really is there
  GARCH(1,1)           a standard recipe for a market whose wild days come in clusters:
                       today's variance = omega + alpha * yesterday's squared return
                                                + beta * yesterday's variance
"""
import bisect
import datetime as dt
import math
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from herding import calibrate, experiments, market_data, politics
from helpers import SerialPool

# ----------------------------------------------------------------------------------
# The public record, typed out here independently of politics.PRESIDENTS
# ----------------------------------------------------------------------------------
RECORD = [
    ("Cleveland", "D", dt.date(1893, 3, 4)), ("McKinley", "R", dt.date(1897, 3, 4)),
    ("T. Roosevelt", "R", dt.date(1901, 9, 14)), ("Taft", "R", dt.date(1909, 3, 4)),
    ("Wilson", "D", dt.date(1913, 3, 4)), ("Harding", "R", dt.date(1921, 3, 4)),
    ("Coolidge", "R", dt.date(1923, 8, 2)), ("Hoover", "R", dt.date(1929, 3, 4)),
    ("Roosevelt", "D", dt.date(1933, 3, 4)), ("Truman", "D", dt.date(1945, 4, 12)),
    ("Eisenhower", "R", dt.date(1953, 1, 20)), ("Kennedy", "D", dt.date(1961, 1, 20)),
    ("Johnson", "D", dt.date(1963, 11, 22)), ("Nixon", "R", dt.date(1969, 1, 20)),
    ("Ford", "R", dt.date(1974, 8, 9)), ("Carter", "D", dt.date(1977, 1, 20)),
    ("Reagan", "R", dt.date(1981, 1, 20)), ("Bush", "R", dt.date(1989, 1, 20)),
    ("Clinton", "D", dt.date(1993, 1, 20)), ("Bush", "R", dt.date(2001, 1, 20)),
    ("Obama", "D", dt.date(2009, 1, 20)), ("Trump", "R", dt.date(2017, 1, 20)),
    ("Biden", "D", dt.date(2021, 1, 20)), ("Trump", "R", dt.date(2025, 1, 20)),
]
FIRST, LAST = dt.date(1893, 3, 4), dt.date(2030, 12, 31)
RECORD_STARTS = [d for _, _, d in RECORD]


def slow_president(day):
    """Index into RECORD of the president on `day` (None before 1893-03-04): walk the
    whole record and keep the last one who had started."""
    found = None
    for i, (_, _, start) in enumerate(RECORD):
        if start <= day:
            found = i
    return found


def slow_democrat(days):
    return [RECORD[slow_president(d)][1] == "D" for d in days]


def random_days(g, n, first=FIRST, last=LAST):
    lo, hi = first.toordinal(), last.toordinal()
    return [dt.date.fromordinal(int(o)) for o in g.integers(lo, hi + 1, n)]


FORMATS = [
    lambda days: [d.isoformat() for d in days],                    # text
    lambda days: list(days),                                       # datetime.date
    lambda days: [pd.Timestamp(d) for d in days],                  # pandas
    lambda days: np.array(days, dtype="datetime64[D]"),            # numpy
    lambda days: pd.DatetimeIndex([pd.Timestamp(d) for d in days]),
]


def blocks(g, n, low=150, high=900):
    """A random party calendar of n days made of alternating runs (like terms in office)."""
    out, party = [], bool(g.integers(2))
    while len(out) < n:
        out += [party] * int(g.integers(low, high))
        party = not party
    return np.array(out[:n])


def balanced_blocks(g, n):
    while True:
        dem = blocks(g, n)
        if dem.sum() >= 500 and (~dem).sum() >= 500:
            return dem


# ==================================================================================
# PART A: correctness
# ==================================================================================

# ---- the table ----

def test_table_matches_the_public_record():
    # correct = every row (start date, party, name) equals the record typed out above
    assert len(politics.PRESIDENTS) == len(RECORD) == 24
    for (start, party, name), (rname, rparty, rstart) in zip(politics.PRESIDENTS, RECORD):
        assert (pd.Timestamp(start).date(), party, name) == (rstart, rparty, rname)


def test_record_itself_is_sane():
    # correct = my own record is in date order, and every start is a real calendar day
    # on which the record says power changed hands (4 Mar before 1937, 20 Jan after,
    # except the five mid-term successions)
    assert RECORD_STARTS == sorted(set(RECORD_STARTS))
    mid_term = {dt.date(1901, 9, 14), dt.date(1923, 8, 2), dt.date(1945, 4, 12),
                dt.date(1963, 11, 22), dt.date(1974, 8, 9)}
    for d in RECORD_STARTS:
        if d not in mid_term:
            assert (d.month, d.day) == ((3, 4) if d.year < 1937 else (1, 20))
            assert d.year % 4 == 1                        # the year after an election


# ---- democrat ----

N_DATE_SETS = 3000


def test_democrat_on_random_date_sets():
    # correct = same answer as the slow walk through the record, for 3,000 random sets
    # of dates (any order, repeats allowed, five ways of writing a date)
    g = np.random.default_rng(101)
    checked = 0
    for case in range(N_DATE_SETS):
        n = int(g.integers(1, 41))
        pool = random_days(g, max(1, n // 2))                       # forces duplicates
        days = [pool[i] for i in g.integers(0, len(pool), n)] if case % 3 == 0 \
            else random_days(g, n)
        got = politics.democrat(FORMATS[case % len(FORMATS)](days))
        assert got.dtype == bool and got.tolist() == slow_democrat(days), (case, days)
        checked += n
    assert checked > 50_000


def test_democrat_at_every_inauguration():
    # correct = the day before (even one second before midnight) belongs to the old
    # president, the day itself and the day after to the new one, for all 23 handovers
    for i in range(1, len(RECORD)):
        day, old, new = RECORD[i][2], RECORD[i - 1][1] == "D", RECORD[i][1] == "D"
        before, after = day - dt.timedelta(days=1), day + dt.timedelta(days=1)
        assert politics.democrat([before, day, after]).tolist() == [old, new, new], RECORD[i]
        last_second = pd.Timestamp(day) - pd.Timedelta(seconds=1)
        assert politics.democrat([last_second, pd.Timestamp(day)]).tolist() == [old, new]


def test_democrat_first_covered_day_and_the_day_before():
    # correct = 4 March 1893 is accepted (Cleveland, D); 3 March 1893 is rejected,
    # also when it hides in the middle of good dates
    assert politics.democrat(["1893-03-04"]).tolist() == [True]
    for bad in (["1893-03-03"], ["1950-01-03", "1893-03-03", "2001-01-03"], ["1066-10-14"][:0] + ["1700-01-01"]):
        with pytest.raises(ValueError, match="not covered"):
            politics.democrat(bad)


def test_democrat_rejects_random_early_dates():
    # correct = any set holding one date before 4 March 1893 raises (500 random sets)
    g = np.random.default_rng(102)
    for _ in range(500):
        days = random_days(g, int(g.integers(0, 10)))
        days.insert(int(g.integers(0, len(days) + 1)),
                    random_days(g, 1, dt.date(1800, 1, 1), dt.date(1893, 3, 3))[0])
        with pytest.raises(ValueError, match="not covered"):
            politics.democrat(days)


@pytest.mark.parametrize("market,days,dem_days,starts", [("spx_1928", 24814, 12988, 13),
                                                         ("dow", 35247, 16350, 16)])
def test_democrat_on_the_real_trading_days(market, days, dem_days, starts):
    # correct = on every real trading day the party equals the slow walk's answer
    # (pinned counts: a change in the data files or the table shows up here)
    idx = market_data.returns(market).index
    got = politics.democrat(idx)
    assert got.tolist() == slow_democrat([t.date() for t in idx])
    assert (len(idx), int(got.sum())) == (days, dem_days)
    assert len(politics.switches(idx)[0]) == starts


def test_democrat_empty_input_gives_nothing():
    # correct = no dates in, no answers out
    assert len(politics.democrat([])) == 0


def test_democrat_empty_input_is_boolean():
    assert politics.democrat([]).dtype == bool


def test_democrat_rejects_a_missing_date():
    with pytest.raises(ValueError):
        politics.democrat(["2010-01-05", None])



def test_democrat_far_future_is_the_last_president():
    # the table has no end date: any later day belongs to the last row (silent on purpose?)
    assert politics.democrat(["2200-01-01"]).tolist() == [RECORD[-1][1] == "D"]


# ---- switches ----

def slow_switches(days):
    party = [RECORD[slow_president(d)][1] for d in days]
    at = [i for i in range(len(days)) if i == 0 or party[i] != party[i - 1]]
    return at, [party[i] for i in at]


def test_switches_on_random_date_sets():
    # correct = index 0 plus every index whose party differs from the one before it,
    # with the party from there on; checked against a plain loop on 2,000 random sets
    # (sorted and unsorted, with repeats)
    g = np.random.default_rng(103)
    for case in range(2000):
        days = random_days(g, int(g.integers(1, 200)))
        if case % 2 == 0:
            days.sort()
        at, party = politics.switches(FORMATS[case % len(FORMATS)](days))
        want_at, want_party = slow_switches(days)
        assert at.tolist() == want_at and party.tolist() == want_party, case
        if case % 2 == 0:                                 # sorted: parties must alternate
            assert all(a != b for a, b in zip(party[:-1], party[1:]))


def test_switches_counts_every_change_of_party_once():
    # correct = on every calendar day 1893-2030 there are exactly 15 changes of party
    # (1897, 1913, 1921, 1933, 1953, 1961, 1969, 1977, 1981, 1993, 2001, 2009, 2017,
    # 2021, 2025 = 15), each on its inauguration day
    days = pd.date_range("1893-03-04", "2030-12-31")
    at, party = politics.switches(days)
    changes = [RECORD[i][2] for i in range(1, len(RECORD)) if RECORD[i][1] != RECORD[i - 1][1]]
    assert len(changes) == 15
    assert [days[i].date() for i in at[1:]] == changes
    assert party[0] == "D" and len(party) == 16


def test_switches_empty_input():
    at, party = politics.switches([])
    assert len(at) == 0 and len(party) == 0


# ---- terms ----

def slow_terms(days, min_days):
    """Group sorted days by president (bisect on my own record), keep the long-enough ones."""
    who = [bisect.bisect_right(RECORD_STARTS, d) - 1 for d in days]
    out = []
    for i, (name, party, _) in enumerate(RECORD):
        mine = [k for k, w in enumerate(who) if w == i]
        if len(mine) >= min_days:
            out.append(dict(name=name, party=party, a=mine[0], b=mine[-1] + 1,
                            start=pd.Timestamp(days[mine[0]]), end=pd.Timestamp(days[mine[-1]])))
    return out


N_TERM_CASES = 400


def test_terms_on_random_sorted_dates():
    # correct = same terms as grouping the days one by one; no two terms overlap; each
    # term has at least min_days days; with min_days=1 the terms cover every day from
    # 4 March 1893 on with no gap
    g = np.random.default_rng(104)
    for case in range(N_TERM_CASES):
        first = dt.date(int(g.integers(1880, 2021)), 1, 1)
        last = min(first + dt.timedelta(days=int(g.integers(30, 20000))), LAST)
        days = sorted(set(random_days(g, int(g.integers(1, 1200)), first, last)))
        min_days = 1 if case % 4 == 0 else int(g.integers(1, 120))
        got = politics.terms(FORMATS[case % len(FORMATS)](days), min_days=min_days)
        assert got == slow_terms(days, min_days), case
        assert all(x["b"] - x["a"] >= min_days for x in got)
        assert all(x["b"] <= y["a"] for x, y in zip(got[:-1], got[1:]))          # no overlap
        covered = [k for k, d in enumerate(days) if d >= FIRST]
        if min_days == 1 and covered:
            assert got[0]["a"] == covered[0] and got[-1]["b"] == len(days)
            assert all(x["b"] == y["a"] for x, y in zip(got[:-1], got[1:]))      # no gap


def test_terms_min_days_boundary():
    # correct = a term with exactly min_days days is kept, one day fewer is dropped
    days = pd.bdate_range("2017-01-20", "2021-01-19")                # Trump's first term
    n = len(days)
    assert [x["name"] for x in politics.terms(days, min_days=n)] == ["Trump"]
    assert politics.terms(days, min_days=n + 1) == []
    assert politics.terms(days[1:], min_days=n) == []


def test_terms_on_the_real_trading_days():
    # correct = 18 terms on the S&P since 1928 (as RESULTS.md Fig 12b says), tiling all days
    idx = market_data.returns("spx_1928").index
    t = politics.terms(idx)
    assert len(t) == 18 and t[0]["name"] == "Coolidge" and t[0]["b"] - t[0]["a"] == 291
    assert t[0]["a"] == 0 and t[-1]["b"] == len(idx)
    assert all(x["b"] == y["a"] for x, y in zip(t[:-1], t[1:]))
    assert t == slow_terms([d.date() for d in idx], 20)


def test_terms_dates_outside_every_term():
    # correct = days before 4 March 1893 belong to no term (and are not an error here,
    # unlike democrat); no days at all gives no terms
    days = pd.bdate_range("1890-01-01", "1893-03-03")
    assert politics.terms(days, min_days=1) == []
    assert politics.terms([]) == [] and politics.terms([], min_days=1) == []
    mixed = pd.bdate_range("1890-01-01", "1898-01-01")
    t = politics.terms(mixed)
    assert [x["name"] for x in t] == ["Cleveland", "McKinley"]
    assert mixed[t[0]["a"]] == pd.Timestamp("1893-03-06")           # 4 March 1893 was a Saturday
    assert mixed[t[0]["a"] - 1] < pd.Timestamp("1893-03-04")


def test_terms_min_days_zero_returns_only_real_terms():
    # correct = every term returned has its start and end inside that presidency
    days = pd.bdate_range("2016-01-04", "2026-10-30")
    for x in politics.terms(days, min_days=0):
        assert x["a"] < x["b"] and x["start"] <= x["end"]


def test_terms_min_days_zero_on_dates_ending_early():
    # fixed November 2026: min_days=0 used to invent a 0-day Cleveland term, and to crash
    # when the dates stopped before the last inauguration
    days = pd.bdate_range("1994-01-04", "1996-10-30")
    assert [x["name"] for x in politics.terms(days, min_days=0)] == ["Clinton"]
    assert politics.terms(pd.bdate_range("2016-01-04", "2026-10-30"), min_days=0)[0]["name"] == "Obama"


def test_terms_rejects_unsorted_dates():
    # fixed November 2026: unsorted days used to file 10 years under one president
    days = pd.bdate_range("2016-01-04", "2026-10-30")[::-1]
    with pytest.raises(ValueError, match="sorted"):
        politics.terms(days)


# ---- party_gap ----

def direct_gap(r, dem):
    """The four gaps computed with scipy and plain Python instead of the module's code."""
    r = [float(x) for x in r]
    d = [x for x, flag in zip(r, dem) if flag]
    g = [x for x, flag in zip(r, dem) if not flag]
    mean_all = math.fsum(r) / len(r)
    sd = math.sqrt(math.fsum((x - mean_all) ** 2 for x in r) / len(r))
    floor = mean_all - 4.0 * sd
    crash = lambda xs: 1000 * sum(x < floor for x in xs) / len(xs)
    return dict(mean=(math.fsum(d) / len(d) - math.fsum(g) / len(g)) / sd * 252,
                vol=math.sqrt(scipy.stats.moment(d, 2) / scipy.stats.moment(g, 2)),
                kurtosis=scipy.stats.kurtosis(d, fisher=True, bias=True)
                - scipy.stats.kurtosis(g, fisher=True, bias=True),
                crashes=crash(d) - crash(g))


def fat_tailed(g, n):
    """Random returns with fat tails (Student-t, 2.5 to 8 degrees of freedom), a random
    scale and a random drift."""
    return g.standard_t(g.uniform(2.5, 8), n) * 10 ** g.uniform(-4, 0) + g.normal(0, 1e-3)


N_GAP_CASES = 1500


def test_party_gap_against_a_direct_computation():
    # correct = all four gaps equal the scipy / plain-Python answer on 1,500 random
    # fat-tailed series with random party calendars
    g = np.random.default_rng(105)
    crash_cases = 0
    for case in range(N_GAP_CASES):
        n = int(g.integers(1000, 4000))
        dem = balanced_blocks(g, n) if case % 2 else g.permutation(np.arange(n) < g.integers(500, n - 499))
        r = fat_tailed(g, n)
        got, want = politics.party_gap(r, dem), direct_gap(r, dem)
        assert set(got) == set(politics.MEASURES) == set(want)
        for k in want:
            assert got[k] == pytest.approx(want[k], rel=1e-8, abs=1e-8), (case, k)
        crash_cases += want["crashes"] != 0
    assert crash_cases > 300          # the crash measure was really exercised, not always 0


def test_party_gap_exactly_500_days_is_enough():
    # correct = 500 days under a party is accepted, 499 is refused (both sides)
    r = np.random.default_rng(106).standard_normal(1200)
    politics.party_gap(r, np.arange(1200) < 500)
    politics.party_gap(r, np.arange(1200) >= 700)
    for dem in (np.arange(1200) < 499, np.arange(1200) >= 701):
        with pytest.raises(ValueError, match="at least 500"):
            politics.party_gap(r, dem)


N_INVARIANCE_CASES = 500


def test_party_gap_invariances():
    # correct = (1) multiplying all returns by a positive number changes nothing;
    # (2) adding the same number to every return changes nothing; (3) swapping the
    # party labels flips the sign of mean, kurtosis and crashes and turns vol into 1/vol
    g = np.random.default_rng(107)
    for case in range(N_INVARIANCE_CASES):
        n = int(g.integers(1000, 5000))
        dem, r = balanced_blocks(g, n), fat_tailed(g, n)
        base = politics.party_gap(r, dem)
        scaled = politics.party_gap(r * 10 ** g.uniform(-3, 3), dem)
        moved = politics.party_gap(r + g.normal(0, 3 * r.std()), dem)
        swapped = politics.party_gap(r, ~dem)
        for k in politics.MEASURES:
            assert scaled[k] == pytest.approx(base[k], rel=1e-7, abs=1e-7), (case, k)
            assert moved[k] == pytest.approx(base[k], rel=1e-6, abs=1e-6), (case, k)
        assert swapped["mean"] == pytest.approx(-base["mean"], rel=1e-9, abs=1e-9)
        assert swapped["kurtosis"] == pytest.approx(-base["kurtosis"], rel=1e-9, abs=1e-9)
        assert swapped["crashes"] == pytest.approx(-base["crashes"], rel=1e-9, abs=1e-9)
        assert swapped["vol"] == pytest.approx(1 / base["vol"], rel=1e-9)


def test_party_gap_accepts_lists_series_and_0_1():
    # correct = the answer does not depend on how the same numbers are handed over
    g = np.random.default_rng(108)
    r, dem = g.standard_t(4, 2000), balanced_blocks(g, 2000)
    want = politics.party_gap(r, dem)
    assert politics.party_gap(list(r), list(dem)) == want
    assert politics.party_gap(pd.Series(r), pd.Series(dem)) == want
    assert politics.party_gap(r, dem.astype(int)) == want


BAD = np.random.default_rng(109).standard_normal(1200)
HALF = np.arange(1200) < 600


@pytest.mark.parametrize("r,dem", [
    (np.where(np.arange(1200) == 7, np.nan, BAD), HALF),            # a NaN return
    (np.where(np.arange(1200) == 7, np.inf, BAD), HALF),            # an infinite return
    (np.where(np.arange(1200) == 7, -np.inf, BAD), HALF),
    (BAD.reshape(2, 600), HALF.reshape(2, 600)),                    # a table, not a list
    (BAD.reshape(1200, 1), HALF),
    (BAD, HALF[:-1]),                                               # one party too few
    (BAD[:-1], HALF),                                               # one return too few
    (BAD, HALF.reshape(2, 600)),
    (np.array([]), np.array([], dtype=bool)),                       # nothing at all
    (BAD, np.ones(1200, bool)), (BAD, np.zeros(1200, bool)),        # one party only
    (BAD, np.arange(1200) < 499), (BAD, np.arange(1200) < 701),     # 499 days for a party
    (np.where(HALF, 1.0, BAD), HALF),                               # D days all the same
    (np.where(HALF, BAD, 1.0), HALF),                               # R days all the same
    (np.ones(1200), HALF),                                          # every day the same
    (BAD, np.array(["D", "R"] * 600)),                              # letters, not True/False
])
def test_party_gap_rejects_bad_input(r, dem):
    # correct = input that cannot give an honest answer raises ValueError
    with pytest.raises(ValueError):
        with np.errstate(all="ignore"):
            politics.party_gap(r, dem)


def test_party_gap_letters_are_rejected_only_by_accident():
    # documents a trap: "D"/"R" letters all count as True (non-empty text), so the call
    # fails with the "500 days" message, not with "need True/False"; and a NaN in dem
    # silently counts as a Democratic day
    with pytest.raises(ValueError, match="at least 500"):
        politics.party_gap(BAD, np.array(["D", "R"] * 600))
    nan_dem = np.where(HALF, np.nan, 0.0)
    assert politics.party_gap(BAD, nan_dem) == politics.party_gap(BAD, HALF)


# ---- shifted_gaps ----

def my_roll(dem, k):
    """dem slid k days later, wrapping round (written without np.roll)."""
    k %= len(dem)
    return np.concatenate([dem[len(dem) - k:], dem[:len(dem) - k]])


N_SHIFT_CASES = 300


def test_shifted_gaps_use_exactly_the_rolled_calendars(monkeypatch):
    # correct = one fake calendar per shift margin, margin+step, ... up to n-margin; each
    # is the real calendar slid by that many days (so it has the same number of
    # Democratic days) and none is the real calendar itself. Checked on 300 random cases
    # by catching every calendar handed to party_gap.
    seen = []
    monkeypatch.setattr(politics, "party_gap", lambda r, dem: seen.append(dem.copy()) or len(seen))
    g = np.random.default_rng(110)
    calendars = 0
    for case in range(N_SHIFT_CASES):
        step, margin = int(g.integers(5, 130)), int(g.integers(0 if case % 10 == 0 else 1, 400))
        n = 2 * margin + step * int(g.integers(20, 60)) + int(g.integers(0, step))
        dem = blocks(g, n, 20, 300)
        while dem.all() or not dem.any():
            dem = blocks(g, n, 20, 300)
        seen.clear()
        out = politics.shifted_gaps(np.zeros(n), dem, step=step, margin=margin)
        shifts = [k for k in range(n + 1) if k >= margin and k <= n - margin and (k - margin) % step == 0]
        assert len(out) == len(seen) == len(shifts) >= 20, case
        for cal, k in zip(seen, shifts):
            assert np.array_equal(cal, my_roll(dem, k)) and cal.sum() == dem.sum()
            if margin > 0:
                assert not np.array_equal(cal, dem), (case, k)
        calendars += len(shifts)
    assert calendars > 8000


def test_shifted_gaps_values_are_party_gaps_of_the_rolled_calendar():
    # correct = the k-th result equals the direct (scipy) gap for the k-th rolled calendar
    g = np.random.default_rng(111)
    for case in range(15):
        n = int(g.integers(2200, 3000))
        dem, r = balanced_blocks(g, n), fat_tailed(g, n)
        out = politics.shifted_gaps(r, dem)
        shifts = list(range(252, n - 252 + 1, 63))
        assert len(out) == len(shifts)
        for i in g.integers(0, len(shifts), 6):
            want = direct_gap(r, my_roll(dem, shifts[i]))
            for k in want:
                assert out[i][k] == pytest.approx(want[k], rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("market,count", [("spx_1928", 386), ("dow", 552)])
def test_shifted_real_calendars_never_repeat_the_real_one(market, count):
    # correct = on the real trading days there are 386 (S&P, as RESULTS.md says) / 552 (Dow)
    # fake calendars, each with the real number of Democratic days, each differing from the
    # real calendar on at least 400 days (a calendar slid a year differs on about 2 x 252
    # days per change of party)
    dem = politics.democrat(market_data.returns(market).index)
    shifts = range(252, len(dem) - 252 + 1, 63)
    assert len(shifts) == count
    differ = [int((my_roll(dem, k) != dem).sum()) for k in shifts]
    assert min(differ) >= 400
    assert all(my_roll(dem, k).sum() == dem.sum() for k in shifts)


def test_shifted_gaps_shortest_series_allowed():
    # correct = 20 fake calendars is the minimum: 2*252 + 19*63 = 1,701 days pass,
    # 1,700 days raise; a step of 0 and mismatched lengths raise too
    g = np.random.default_rng(112)
    r, dem = g.standard_normal(1701), np.arange(1701) < 850
    assert len(politics.shifted_gaps(r, dem)) == 20
    with pytest.raises(ValueError, match="too few shifts"):
        politics.shifted_gaps(r[:1700], dem[:1700])
    with pytest.raises(ValueError):
        politics.shifted_gaps(r, dem, step=0)
    with pytest.raises(ValueError, match="one party per return"):
        politics.shifted_gaps(r[:-1], dem)
    for n in g.integers(1000, 1700, 50):                   # 50 random too-short series
        with pytest.raises(ValueError, match="too few shifts"):
            politics.shifted_gaps(r[:n], dem[:n])


def test_shifted_gaps_traps_current_behaviour():
    # documents two traps: (1) margin=0 puts the REAL calendar among the "fake" ones
    # (shift 0); (2) a calendar that repeats itself comes back unshifted when the shift
    # equals its period (here 2,016 = 252 + 28*63 days). Real calendars do not repeat,
    # so (2) cannot bite on real dates (see the test above).
    g = np.random.default_rng(113)
    r = g.standard_normal(4032)
    dem = np.arange(4032) // 1008 % 2 == 0
    assert politics.shifted_gaps(r, dem, margin=0)[0] == politics.party_gap(r, dem)
    assert politics.shifted_gaps(r, dem)[28] == politics.party_gap(r, dem)


# ---- percentile (the ruler Part B and compare() rely on) ----

def test_percentile_against_scipy():
    # correct = share of simulations below the value, ties counted as half
    # (scipy's percentileofscore, kind="mean"), on 2,000 random cases with ties
    g = np.random.default_rng(114)
    for _ in range(2000):
        sims = g.integers(-5, 6, int(g.integers(1, 60))).astype(float)
        value = float(g.integers(-6, 7))
        assert experiments.percentile(value, sims) == pytest.approx(
            scipy.stats.percentileofscore(sims, value, kind="mean"))


# ---- compare (the function Fig 12 calls) ----

def test_compare_puts_the_pieces_together():
    # correct = on the S&P's last 3,000 days: the real gap is party_gap on those days and
    # the real calendar; one model gap per history; the shifted gaps are shifted_gaps; and
    # every percentile equals scipy's on those lists
    p, seeds = calibrate.load("spx"), [61990, 61991, 61992]
    r, dem, real, nulls, pct = politics.compare("spx_1928", p, seeds, pool=SerialPool(), days=3000)
    full = market_data.returns("spx_1928")
    assert r.equals(full.iloc[-3000:])
    assert dem.tolist() == slow_democrat([t.date() for t in r.index])
    assert real == pytest.approx(direct_gap(r.values, dem), rel=1e-8, abs=1e-8)
    assert len(nulls["model"]) == 3 and len(nulls["shifted"]) == len(range(252, 3000 - 252 + 1, 63))
    assert nulls["shifted"] == politics.shifted_gaps(r.values, dem)
    for name in ("model", "shifted"):
        for k in politics.MEASURES:
            want = scipy.stats.percentileofscore([s[k] for s in nulls[name]], real[k], kind="mean")
            assert pct[name][k] == pytest.approx(want)


# ==================================================================================
# PART B: can the method be trusted?
# ==================================================================================
# Everything here is fixed by seeds, so the measured rates below are the same on every run.
# The asserted bands are deliberately wide: about +-3.5 standard errors around the target
# (for a rate near 10% out of 400 markets, one standard error is 1.5 points).

SP_DAILY_SD = 0.0118            # the S&P 500's daily standard deviation since 1928 (1.18%)
N_MARKETS = 400                 # synthetic no-effect markets per generator
GAPS = [0, 2, 5, 7, 10, 15]     # planted gap, % a year added on Democratic days (0 = none)
MODEL_SEEDS = range(61000, 61200)

# Generator 1: GARCH(1,1) with fat-tailed (Student-t, 5 degrees of freedom) shocks.
#   variance today = OMEGA + 0.08 * (yesterday's return)^2 + 0.91 * yesterday's variance
#   Wild spells fade with a half-life of ln(0.5)/ln(0.99) = 69 trading days.
GARCH_ALPHA, GARCH_BETA, GARCH_DF = 0.08, 0.91, 5
# Generator 2: stochastic volatility with LONG memory: log(volatility) is the sum of a fast
#   wave (half-life 40 days, sd 0.35) and a slow wave (half-life 750 days = 3 years, sd 0.45),
#   times Student-t(5) shocks. The slow wave gives calm and wild ERAS that last years, as in
#   the real market (the 1930s, the 1950s-60s), which is the hard case for this test.
SV_FAST, SV_SLOW, SV_DF = (40, 0.35), (750, 0.45), 5


def garch_markets(seed, markets, days):
    """`markets` no-effect histories of `days` returns (one per row)."""
    g = np.random.default_rng(seed)
    burn = 1000
    z = g.standard_t(GARCH_DF, (days + burn, markets)) * math.sqrt((GARCH_DF - 2) / GARCH_DF)
    omega = SP_DAILY_SD ** 2 * (1 - GARCH_ALPHA - GARCH_BETA)
    var = np.full(markets, SP_DAILY_SD ** 2)
    out = np.empty_like(z)
    for t in range(days + burn):
        out[t] = np.sqrt(var) * z[t]
        var = omega + GARCH_ALPHA * out[t] ** 2 + GARCH_BETA * var
    return np.ascontiguousarray(out[burn:].T)


def sv_markets(seed, markets, days):
    g = np.random.default_rng(seed)
    z = g.standard_t(SV_DF, (days, markets)) * math.sqrt((SV_DF - 2) / SV_DF)
    logvol = np.zeros((days, markets))
    for half_life, sd in (SV_FAST, SV_SLOW):
        phi = 0.5 ** (1 / half_life)
        wave = g.normal(0, sd, markets)                             # start in the long-run state
        kick = g.normal(0, sd * math.sqrt(1 - phi ** 2), (days, markets))
        for t in range(days):
            wave = phi * wave + kick[t]
            logvol[t] += wave
    total_var = SV_FAST[1] ** 2 + SV_SLOW[1] ** 2
    return np.ascontiguousarray((SP_DAILY_SD * np.exp(logvol - total_var) * z).T)


def _shifted_job(args):
    """For one market: its 'real' gaps at the true calendar and their percentiles among
    the module's shifted calendars (this is exactly what politics.compare does)."""
    r, dem = args
    real = politics.party_gap(r, dem)
    fake = politics.shifted_gaps(r, dem)
    return real, {k: experiments.percentile(real[k], [s[k] for s in fake]) for k in politics.MEASURES}


def shifted_percentiles(pool, markets, dem):
    out = pool.map(_shifted_job, [(r, dem) for r in markets], chunksize=4)
    reals = {k: np.array([real[k] for real, _ in out]) for k in politics.MEASURES}
    pcts = {k: np.array([pct[k] for _, pct in out]) for k in politics.MEASURES}
    return reals, pcts


def fast_mean_percentiles(markets, dem, gaps=GAPS, step=63, margin=252):
    """Percentile of the mean gap among shifted calendars after adding X% a year on
    Democratic days, for every market and every X, WITHOUT calling politics: the D-day sum
    under each rolled calendar is one matrix product, and planting x a day adds x times the
    overlap between the rolled and the real calendar. (Dividing by the series' sd, as
    party_gap does, does not change the order, so it is left out.)"""
    n, n_d = len(dem), dem.sum()
    rolled = np.stack([my_roll(dem, k) for k in range(margin, n - margin + 1, step)]).astype(float)
    sums = rolled @ markets.T                         # shifts x markets: sum of returns on "D" days
    overlap = rolled @ dem.astype(float)              # shifts: days that are D in both calendars
    real_sums, totals = markets @ dem.astype(float), markets.sum(axis=1)
    gap = lambda s, total: s / n_d - (total - s) / (n - n_d)
    out = {}
    for x_year in gaps:
        x = x_year / 100 / politics.DAYS_PER_YEAR
        total = totals + x * n_d
        fake = gap(sums + x * overlap[:, None], total)
        real = gap(real_sums + x * n_d, total)
        out[x_year] = 100 * ((fake < real).mean(axis=0) + 0.5 * (fake == real).mean(axis=0))
    return out


def outside(p):
    """Share of percentiles outside 5-95 (a fair test gives 10%)."""
    return float(np.mean((p < 5) | (p > 95)))


def show(title, rows):
    print("\n" + title)
    for row in rows:
        print("   " + row)


@pytest.fixture(scope="module")
def sp():
    idx = market_data.returns("spx_1928").index
    return idx, politics.democrat(idx)


@pytest.fixture(scope="module")
def workers():
    with Pool(min(8, os.cpu_count() or 1)) as pool:
        yield pool


@pytest.fixture(scope="module")
def synthetic(sp, workers):
    """400 GARCH and 400 long-memory markets on the S&P's 24,814 trading days, each with
    its percentiles among the shifted calendars (4 measures) and its planted-gap percentiles."""
    idx, dem = sp
    out = {}
    for name, make, seed in (("garch", garch_markets, 20261114), ("slow-vol", sv_markets, 20261115)):
        markets = make(seed, N_MARKETS, len(idx))
        reals, pcts = shifted_percentiles(workers, markets, dem)
        out[name] = dict(markets=markets, real=reals, pct=pcts,
                         planted=fast_mean_percentiles(markets, dem))
    return out


@pytest.fixture(scope="module")
def model_runs(sp, workers):
    """Model histories (S&P-calibrated, seeds 61000-61199, used nowhere else) on the same
    days, rescaled to the S&P's daily sd so that 'X% a year' means the same as for the S&P."""
    idx, dem = sp
    runs = [h["r"] for h in experiments.histories(calibrate.load("spx"), MODEL_SEEDS, len(idx), pool=workers)
            if h is not None]
    markets = np.array([r / r.std() * SP_DAILY_SD for r in runs])
    reals, pcts = shifted_percentiles(workers, markets, dem)
    return dict(markets=markets, real=reals, pct=pcts, planted=fast_mean_percentiles(markets, dem))


@pytest.mark.slow
def test_generators_look_like_a_market(synthetic):
    # correct = the no-effect markets have what the test must cope with: daily sd near
    # 1.18%, fat tails (excess kurtosis > 3 in the median market), volatility clustering
    # (|returns| 1 day apart correlate > 0.1), and no memory in the returns themselves
    rows = []
    for name, s in synthetic.items():
        m = s["markets"][:100]
        sd = np.median(m.std(axis=1))
        kurt = np.median(scipy.stats.kurtosis(m, axis=1))
        ac = lambda x, lag: np.median([np.corrcoef(row[:-lag], row[lag:])[0, 1] for row in x])
        a1, a250, r1 = ac(np.abs(m), 1), ac(np.abs(m), 250), ac(m, 1)
        rows.append(f"{name:9s} daily sd {100 * sd:.2f}%  excess kurtosis {kurt:.1f}  "
                    f"|r| autocorrelation lag 1: {a1:.2f}, lag 250: {a250:.2f}  r lag 1: {r1:+.3f}")
        assert 0.009 < sd < 0.016 and kurt > 3 and a1 > 0.1 and abs(r1) < 0.03
    show("Synthetic no-effect markets (median of 100)", rows)
    # the slow-vol market still remembers its volatility a year later; GARCH has forgotten
    slow, fast = synthetic["slow-vol"]["markets"][:100], synthetic["garch"]["markets"][:100]
    year = lambda x: np.median([np.corrcoef(abs(row[:-250]), abs(row[250:]))[0, 1] for row in x])
    assert year(slow) > 0.05 > year(fast)


@pytest.mark.slow
def test_fast_formula_agrees_with_the_module(synthetic, sp):
    # correct = my matrix shortcut for the planted-gap percentile gives the same number as
    # really adding the gap to the returns and calling politics.shifted_gaps (so the power
    # table below is a statement about the module's test, not about my shortcut)
    _, dem = sp
    for name, s in synthetic.items():
        for j in (0, 137, 399):
            assert s["planted"][0][j] == pytest.approx(s["pct"]["mean"][j], abs=1e-9)
            for x_year in (5, 10):
                r = s["markets"][j] + x_year / 100 / 252 * dem
                _, pct = _shifted_job((r, dem))
                assert s["planted"][x_year][j] == pytest.approx(pct["mean"], abs=1e-9), (name, j, x_year)


# MEASURED (seeds fixed; 400 markets each; a fair test gives 10% outside 5-95):
#              mean    vol     kurtosis  crashes
#   garch      11.0%   12.2%   10.8%     10.5%
#   slow-vol    8.8%    9.0%   11.5%     10.0%
# All within 1.5 standard errors (1.5 points each) of 10%: the shifted-calendar yardstick
# is neither too narrow nor too wide on these markets.
FALSE_POSITIVE_BAND = (0.05, 0.16)


@pytest.mark.slow
def test_false_positive_rate_of_the_shifted_calendar_test(synthetic):
    # correct = with NO party effect, about 10% of markets land outside the 5th-95th
    # percentile of their own shifted calendars, for each of the 4 measures and both kinds
    # of volatility clustering
    rows, rates = [], {}
    for name, s in synthetic.items():
        for k in politics.MEASURES:
            p = s["pct"][k]
            rates[name, k] = outside(p)
            rows.append(f"{name:9s} {k:9s} outside 5-95: {100 * outside(p):5.1f}%   "
                        f"(below 5: {100 * np.mean(p < 5):4.1f}%, above 95: {100 * np.mean(p > 95):4.1f}%)   "
                        f"outside 2.5-97.5: {100 * np.mean((p < 2.5) | (p > 97.5)):4.1f}%")
    show(f"FALSE-POSITIVE RATE, shifted calendars, {N_MARKETS} no-effect markets each (fair = 10%)", rows)
    for key, rate in rates.items():
        assert FALSE_POSITIVE_BAND[0] <= rate <= FALSE_POSITIVE_BAND[1], (key, rate)


@pytest.mark.slow
def test_percentiles_are_spread_evenly_when_nothing_is_going_on(synthetic):
    # correct = with no effect, a market's percentile among its shifted calendars is equally
    # likely to be anywhere from 0 to 100 (Kolmogorov-Smirnov test against "uniform":
    # p-value not tiny). This checks the whole ruler, not only its two ends.
    rows = []
    for name, s in synthetic.items():
        for k in politics.MEASURES:
            p = scipy.stats.kstest(s["pct"][k] / 100, "uniform").pvalue
            rows.append(f"{name:9s} {k:9s} KS p-value {p:.3f}")
            assert p > 0.001, (name, k, p)
    show("Are no-effect percentiles uniform? (p near 0 = no)", rows)


def textbook_power(x_year, years_d, years_r, sd_year):
    """Chance that a one-sided 5% test sees a gap of x_year, if days were independent:
    the gap's standard error is sd * sqrt(1/years_D + 1/years_R)."""
    se = sd_year * math.sqrt(1 / years_d + 1 / years_r)
    return scipy.stats.norm.cdf(x_year / se - scipy.stats.norm.ppf(0.95)), se


@pytest.mark.slow
def test_power_of_the_shifted_calendar_test(synthetic, sp):
    # correct = a planted gap of X% a year on Democratic days lands above the 95th
    # percentile more often the bigger X is, about as often as the textbook formula for
    # independent days says (within 12 points), and X = 0 lands there about 5% of the time
    _, dem = sp
    years_d, years_r = dem.sum() / 252, (~dem).sum() / 252
    rows = []
    for name, s in synthetic.items():
        sd_year = 100 * np.median(s["markets"].std(axis=1)) * math.sqrt(252)
        rate = {x: float(np.mean(s["planted"][x] > 95)) for x in GAPS}
        for x in GAPS:
            want, se = textbook_power(x, years_d, years_r, sd_year)
            rows.append(f"{name:9s} planted +{x:2d}%/yr: above 95th in {100 * rate[x]:5.1f}% of markets "
                        f"(textbook {100 * want:5.1f}%; the gap's standard error is {se:.1f}%/yr)")
            assert abs(rate[x] - want) < 0.12, (name, x, rate[x], want)
        assert [rate[x] for x in GAPS] == sorted(rate[x] for x in GAPS)       # bigger gap, seen more often
        assert 0.02 <= rate[0] <= 0.09
        # the finding in numbers: a true gap of 5% a year is missed more often than found,
        # and even the S&P's own measured gap size (7%) is missed in over a third of markets
        assert rate[5] < 0.5 and rate[7] < 0.67 and rate[15] > 0.9
    show(f"POWER, shifted calendars, S&P days 1928-2026 ({years_d:.1f} D years, {years_r:.1f} R years, "
         f"12 changes of party), {N_MARKETS} markets each", rows)


@pytest.mark.slow
def test_power_on_the_dow_calendar():
    # correct = same check on the longer Dow record (1896-2026, 15 changes of party):
    # more years, so a little more power than on the S&P days, but the same picture
    idx = market_data.returns("dow").index
    dem = politics.democrat(idx)
    markets = garch_markets(20261116, 300, len(idx))
    planted = fast_mean_percentiles(markets, dem)
    years_d, years_r = dem.sum() / 252, (~dem).sum() / 252
    sd_year = 100 * np.median(markets.std(axis=1)) * math.sqrt(252)
    rows = []
    for x in GAPS:
        rate, (want, se) = float(np.mean(planted[x] > 95)), textbook_power(x, years_d, years_r, sd_year)
        rows.append(f"garch     planted +{x:2d}%/yr: above 95th in {100 * rate:5.1f}% of markets "
                    f"(textbook {100 * want:5.1f}%; standard error {se:.1f}%/yr)")
        assert abs(rate - want) < 0.12, (x, rate, want)
    rows.append(f"outside 5-95 with no gap: {100 * outside(planted[0]):.1f}% (fair = 10%)")
    assert 0.04 <= outside(planted[0]) <= 0.17
    show(f"POWER, shifted calendars, Dow days 1896-2026 ({years_d:.1f} D years, {years_r:.1f} R years, "
         f"15 changes of party), 300 GARCH markets", rows)


# ---- the model as the yardstick ----

@pytest.mark.slow
def test_shifted_calendar_test_on_model_histories(model_runs):
    # correct = model histories have no politics, so the shifted-calendar test should flag
    # about 10% of them per measure; and since the model's long-run averages wobble less
    # than a real market's (RESULTS.md: H 0.39), a planted gap is easier to see in them
    n = len(model_runs["markets"])
    assert n >= 190                                        # nearly all 200 histories survived
    rows = [f"{k:9s} outside 5-95: {100 * outside(model_runs['pct'][k]):5.1f}%" for k in politics.MEASURES]
    rate = {x: float(np.mean(model_runs["planted"][x] > 95)) for x in GAPS}
    rows += [f"planted +{x:2d}%/yr: above 95th in {100 * rate[x]:5.1f}% of histories" for x in GAPS]
    show(f"MODEL HISTORIES judged by their own shifted calendars ({n} histories)", rows)
    for k in politics.MEASURES:
        assert 0.03 <= outside(model_runs["pct"][k]) <= 0.19, k          # 200 runs: wider band
    assert [rate[x] for x in GAPS] == sorted(rate[x] for x in GAPS)


@pytest.mark.slow
def test_model_yardstick_false_positives_and_power(model_runs, synthetic, sp):
    # The model yardstick = "where does a market's gap sit among the model histories' gaps?"
    # correct (if the model were a fair yardstick for ANY no-effect market) = about 10% of
    # no-effect GARCH / slow-vol markets outside the model's 5-95 range, per measure.
    _, dem = sp
    model_gaps = model_runs["real"]
    pct = lambda value, k, sims=None: experiments.percentile(value, model_gaps[k] if sims is None else sims)
    rows, rates = [], {}
    for name, s in synthetic.items():
        for k in politics.MEASURES:
            p = np.array([pct(v, k) for v in s["real"][k]])
            rates[name, k] = outside(p)
            rows.append(f"{name:9s} {k:9s} outside the model's 5-95: {100 * outside(p):5.1f}%   "
                        f"(spread of the gap: these markets {s['real'][k].std():.2f}, model {model_gaps[k].std():.2f})")
    # model judged by model (each history against all the others): fair by construction
    n = len(model_runs["markets"])
    others = lambda i, k: np.delete(model_gaps[k], i)
    for k in politics.MEASURES:
        p = np.array([pct(model_gaps[k][i], k, others(i, k)) for i in range(n)])
        rows.append(f"{'model':9s} {k:9s} outside the other histories' 5-95: {100 * outside(p):5.1f}%")
        assert 0.04 <= outside(p) <= 0.17
    # power of the model yardstick on model histories
    for x in GAPS:
        gaps = np.array([politics.party_gap(r + x / 100 / 252 * dem, dem)["mean"] for r in model_runs["markets"]])
        hit = np.mean([pct(gaps[i], "mean", others(i, "mean")) > 95 for i in range(n)])
        rates["power", x] = float(hit)
        rows.append(f"model     planted +{x:2d}%/yr: above the other histories' 95th in {100 * hit:5.1f}%")
    show(f"MODEL YARDSTICK ({n} model histories as the ruler)", rows)
    assert rates["power", 15] > rates["power", 5] > rates["power", 0]
    # FINDING (measured, not hoped for): against the model's ruler, 29-50% of no-effect
    # GARCH / slow-vol markets look "unusual" on EVERY measure, not 10%:
    #              mean    vol     kurtosis  crashes
    #   garch      29.0%   31.0%   49.5%     33.2%
    #   slow-vol   29.2%   36.5%   40.5%     33.8%
    # The model's gaps spread less than these markets' (mean 1.86 vs 3.1 daily sds a year,
    # vol ratio 0.09 vs 0.2, kurtosis 8 vs 62). So "outside the model's range" does not by
    # itself mean "not luck" unless the real market behaves like the model. If the model is
    # ever changed so that it becomes a fair ruler for such markets, this assert will fail:
    # good news, update it.
    for key in [(name, k) for name in synthetic for k in politics.MEASURES]:
        assert 0.22 <= rates[key] <= 0.60, (key, rates[key])
