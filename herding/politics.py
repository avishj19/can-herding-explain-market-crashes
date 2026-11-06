"""Political parties (an extension, not part of the paper): is the market different
under Democratic and Republican presidents, and could a model with no politics in it
produce a gap that big by luck?

The "presidential puzzle" (Santa-Clara & Valkanov 2003, J. Finance 58: 1841-1872):
1927-1998, US stocks returned about 9 points a year more under Democratic presidents,
with no matching difference in risk. Pastor & Veronesi (2020, J. Political Economy
128: 4011-4045) found it continued, and argue the party is not the cause: when
investors are fearful, voters pick Democrats, and fearful markets later pay more.

Here the model is the yardstick. It has no politics (its news is the same bell curve
every day), so any "party gap" in a model history is pure chance. Cutting many model
histories at the real inauguration dates shows how big a gap chance alone gives.
"""
import numpy as np
import pandas as pd

from . import experiments, market_data, stats

# The party of the president from each date on (inauguration days; public record).
PRESIDENTS = [
    ("1923-08-02", "R", "Coolidge"), ("1929-03-04", "R", "Hoover"),
    ("1933-03-04", "D", "Roosevelt"), ("1945-04-12", "D", "Truman"),
    ("1953-01-20", "R", "Eisenhower"), ("1961-01-20", "D", "Kennedy"),
    ("1963-11-22", "D", "Johnson"), ("1969-01-20", "R", "Nixon"),
    ("1974-08-09", "R", "Ford"), ("1977-01-20", "D", "Carter"),
    ("1981-01-20", "R", "Reagan"), ("1989-01-20", "R", "Bush"),
    ("1993-01-20", "D", "Clinton"), ("2001-01-20", "R", "Bush"),
    ("2009-01-20", "D", "Obama"), ("2017-01-20", "R", "Trump"),
    ("2021-01-20", "D", "Biden"), ("2025-01-20", "R", "Trump"),
]
DAYS_PER_YEAR = 252
CRASH_SD = 4.0          # a "crash day": a fall of more than 4 standard deviations


def democrat(dates):
    """True on each date with a Democratic president (dates: anything pandas reads)."""
    dates = pd.DatetimeIndex(dates)
    first = pd.Timestamp(PRESIDENTS[0][0])
    if len(dates) and dates.min() < first:
        raise ValueError(f"dates before {first.date()} are not covered")
    starts = pd.DatetimeIndex([d for d, _, _ in PRESIDENTS])
    i = starts.searchsorted(dates, side="right") - 1   # the latest inauguration on or before
    return np.array([PRESIDENTS[j][1] == "D" for j in i])


def switches(dates):
    """Day indexes where the party changes, and the party from there on."""
    dem = democrat(dates)
    at = np.flatnonzero(np.diff(dem.astype(int))) + 1
    return np.concatenate([[0], at]), np.where(dem[np.concatenate([[0], at])], "D", "R")


def party_gap(r, dem):
    """How returns differ between Democratic (dem True) and Republican days, in units
    of the whole series' daily standard deviation so that a real market and a model of
    different volatility can be compared:
        mean      (mean D - mean R) per year, in daily sds (x DAYS_PER_YEAR)
        vol       sd on D days / sd on R days (1 = the same)
        kurtosis  excess kurtosis on D days minus on R days
        crashes   crash days (below -CRASH_SD sds) per 1,000 days, D minus R"""
    r = stats._returns(r)
    dem = np.asarray(dem, dtype=bool)
    if dem.shape != r.shape:
        raise ValueError(f"need one party per return: {dem.shape} vs {r.shape}")
    if dem.sum() < 500 or (~dem).sum() < 500:
        raise ValueError("need at least 500 days under each party")
    sd = r.std()
    d, g = r[dem], r[~dem]
    crash = lambda x: 1000 * np.mean(x < r.mean() - CRASH_SD * sd)
    return dict(mean=float((d.mean() - g.mean()) / sd * DAYS_PER_YEAR),
                vol=float(d.std() / g.std()),
                kurtosis=float(stats.kurtosis(d) - stats.kurtosis(g)),
                crashes=float(crash(d) - crash(g)))


MEASURES = {
    "mean": "return gap per year, D minus R (daily sds)",
    "vol": "volatility, D / R",
    "kurtosis": "fat tails (excess kurtosis), D minus R",
    "crashes": "crash days per 1,000, D minus R",
}


def shifted_gaps(r, dem, step=63, margin=DAYS_PER_YEAR):
    """party_gap of the SAME returns with the party calendar slid along by every multiple
    of `step` days (63 = a quarter), wrapping around, at least `margin` days from the
    real calendar. A fake calendar keeps the real terms' lengths and the market's own
    calm and wild spells, so the gaps it gives are what luck alone does with this market."""
    dem = np.asarray(dem, dtype=bool)
    shifts = range(margin, len(dem) - margin + 1, step)
    if len(shifts) < 20:
        raise ValueError(f"too few shifts ({len(shifts)}): need a longer series or a smaller step")
    return [party_gap(r, np.roll(dem, k)) for k in shifts]


def compare(market, p, seeds, pool=None, days=None):
    """The real gap, and two yardsticks for luck, each with the real market's percentile:
    the model's gaps at the same dates (one per history that did not run away), and the
    market's own returns with the party calendar shifted (shifted_gaps).
    days: use only the market's last `days` days (None = all; small for quick tests)."""
    r = market_data.returns(market)
    r = r if days is None else r.iloc[-int(days):]
    dem = democrat(r.index)
    real = party_gap(r.values, dem)
    runs = [h["r"] for h in experiments.histories(p, seeds, len(r), pool=pool) if h is not None]
    nulls = {"model": [party_gap(x, dem) for x in runs],
             "shifted": shifted_gaps(r.values, dem)}
    pct = {name: {k: experiments.percentile(real[k], [s[k] for s in null]) for k in MEASURES}
           for name, null in nulls.items()}
    return r, dem, real, nulls, pct


def terms(dates, min_days=20):
    """Each presidency's stretch of `dates` (sorted trading days): a list of dicts with
    name, party and the day indexes a:b (b not included). Terms with fewer than
    min_days trading days in `dates` (e.g. before the data start) are left out."""
    dates = pd.DatetimeIndex(dates)
    starts = [pd.Timestamp(d) for d, _, _ in PRESIDENTS] + [pd.Timestamp("2100-01-01")]
    out = []
    for i, (_, party, name) in enumerate(PRESIDENTS):
        a, b = int(dates.searchsorted(starts[i])), int(dates.searchsorted(starts[i + 1]))
        if b - a >= min_days:
            out.append(dict(name=name, party=party, a=a, b=b, start=dates[a], end=dates[b - 1]))
    return out
