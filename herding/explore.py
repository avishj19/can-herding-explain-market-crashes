"""The science behind the web app (app.py): load any market, run the three trader mixes,
and describe returns with a few numbers. app.py only draws what these functions return.

Assets come in three groups:
    stocks       the 20 saved stocks (data/stocks/), or any ticker fetched live from Yahoo
    indexes      S&P 500, Nasdaq-100, Dow Jones (data/)
    commodities  Brent oil, gold, copper (data/factors/)

Trader mixes are three versions of the Lux-Marchesi model (docs/SPEC.md):
    fundamentalists  everybody trades on value (chartists=False): the bell-curve base case
    chartists        chartists too, but they do not copy each other's mood (a1 = 0);
                     they still follow the price trend (a2)
    herding          the full model, with herding strength a1

Every mix starts from the S&P 500 fit (settings/sp500_calibrated.json). The comparison with
another asset is made with shape measures only (kurtosis, tail exponent, volatility memory),
which are measured in standard deviations and so do not depend on how volatile the asset is.
"""
import json
import math
import re

import numpy as np
import pandas as pd

from . import factors, market_data, model, stats
from .checks import is_number

FIT = market_data.ROOT / "settings" / "sp500_calibrated.json"
INDEXES = {"S&P 500": "spx", "Nasdaq-100": "ndx", "Dow Jones": "dow"}
COMMODITIES = {"Brent oil": "oil", "Gold": "gold", "Copper": "copper"}
MIXES = ("fundamentalists", "chartists", "herding")
MIN_DAYS = 1000          # about 4 years: fewer big days than this make the tail exponent noise
MAX_DAYS = 20000         # longest model run the web app allows (about 80 years)
BIG = 4.0                # a "big day": a move of more than 4 standard deviations
MEMORY_LAG = 20          # volatility memory = correlation of move sizes 20 days (a month) apart
LAGS = (1, 2, 5, 10, 20, 50, 100, 200)
TICKER = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,11}$")    # ^ starts index tickers (^GSPC)
YEAR_DAYS = 252          # trading days in a year: the 52-week window, the sidebar sparkline, annualizing
# Yahoo symbols of the saved markets that are not stocks (a stock's symbol is its own ticker)
YAHOO = {"S&P 500": "^GSPC", "Nasdaq-100": "^NDX", "Dow Jones": "^DJI", "Brent oil": "BZ=F",
         "Gold": "GC=F", "Copper": "HG=F"}


def _whole(x, name, lo):
    """x as an int if it is a whole number >= lo (not True/False, not 2.0), else a ValueError.
    One rule for days, seeds, window lengths, ..."""
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < lo:
        raise ValueError(f"{name} must be a whole number >= {lo}, got {x!r}")
    return int(x)


def _prices(close, name="prices"):
    """close as a float Series of at least 2 positive, finite prices whose index is in order with no
    repeats. Out-of-order or repeated dates would make every return compare the wrong two days."""
    c = pd.Series(close, dtype=float)
    if len(c) < 2 or not (c > 0).all() or not np.all(np.isfinite(c)):
        raise ValueError(f"need at least 2 positive, finite {name}")
    if not (c.index.is_monotonic_increasing and c.index.is_unique):
        raise ValueError(f"{name} must be in date order with no repeated dates")
    return c


def _dated(c, name):
    """Checks that a price Series is indexed by dates (needed to line two series up in time)."""
    if not isinstance(c.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must be a Series indexed by dates")
    return c


# ------------------------------------------------------------------ assets

def catalog():
    """{group: [asset names]} for the market list, in display order."""
    return {"Stocks": market_data.stock_tickers(), "Indexes": list(INDEXES),
            "Commodities": list(COMMODITIES)}


def saved_prices(name):
    """Daily closes (a Series indexed by date) of one saved asset, by its display name.
    Index files drop their known bad closes (market_data.BAD_CLOSES)."""
    if name in INDEXES:
        key = INDEXES[name]
        df = market_data.load_prices(market_data.MARKETS[key])
        close = df["Close"].drop(pd.DatetimeIndex(market_data.BAD_CLOSES.get(key, ())))
    elif name in COMMODITIES:
        close = market_data.load_prices(factors.FACTOR_DIR / factors.FACTORS[COMMODITIES[name]])["Close"]
    elif name in market_data.stock_tickers():
        close = market_data.load_prices(market_data.STOCKS / f"{name}.csv")["Close"]
    else:
        raise ValueError(f"unknown asset {name!r}; choose from {sum(catalog().values(), [])}")
    return close.rename(name)


def clean_ticker(text):
    """A ticker typed by a visitor, upper-cased and checked (letters, digits and . - = ^ only,
    at most 12 characters), so nothing else is ever sent to Yahoo."""
    if not isinstance(text, str):
        raise ValueError("type a ticker, like NVDA")
    t = text.strip().upper()
    if not t:
        raise ValueError("type a ticker first, like NVDA")
    if not TICKER.match(t):
        raise ValueError(f"{t!r} is not a ticker: use letters, digits and . - = ^ (at most 12)")
    return t


CRYPTO = re.compile(r"-(USD|USDT|USDC|EUR|GBP|JPY|BTC|ETH)$")   # BTC-USD; BRK-B (a share class) is not


def asset_kind(name):
    """'Index' or 'Commodity' for the saved markets; for a typed ticker, read from Yahoo's symbol
    pattern: ^... an index, ...=F futures, ...=X a currency, ...-USD crypto, anything else a stock."""
    if name in INDEXES:
        return "Index"
    if name in COMMODITIES:
        return "Commodity"
    t = str(name).strip().upper()
    if t.startswith("^"):
        return "Index"
    if t.endswith("=F"):
        return "Futures"
    if t.endswith("=X"):
        return "Currency"
    if CRYPTO.search(t):
        return "Crypto"
    return "Stock"


def trading_hours(symbol, bars=None):
    """{nights, weekends}: whether a market trades at night and on weekends, so an intraday chart
    knows which gaps it may hide. Futures and currencies trade almost 24 hours on weekdays; crypto,
    or any market with bars on a Saturday or Sunday, trades around the clock every day."""
    kind = asset_kind(symbol) if isinstance(symbol, str) else "Stock"
    weekends = kind == "Crypto"
    if bars is not None and isinstance(getattr(bars, "index", None), pd.DatetimeIndex) and len(bars):
        weekends = weekends or bool((bars.index.dayofweek >= 5).any())
    return {"nights": weekends or kind in ("Futures", "Currency") or is_futures(symbol),
            "weekends": weekends}


def price_decimals(x):
    """How many decimals a price needs: 2 from 1 up (cents), else enough for 4 significant digits,
    so a coin worth 0.00002013 doesn't read 0.00. Worked out on the price already rounded to 4
    significant digits: 0.99996 rounds to 1.000, a price from 1 up, so it reads 1.00, not 1.0000
    (5 significant digits)."""
    x = float(x)
    if not np.isfinite(x) or x <= 0:
        return 2
    x = float(f"{x:.4g}")
    if x >= 1:
        return 2
    return min(12, 3 - math.floor(math.log10(x)))


def price_text(x):
    """A price for reading: 1,234.50 or 0.00002013 (see price_decimals)."""
    return f"{float(x):,.{price_decimals(x)}f}"


def yahoo_symbol(name):
    """The Yahoo Finance symbol for a market: the saved indexes and commodities have their own
    (S&P 500 -> ^GSPC, Gold -> GC=F); a stock or a typed ticker is its own symbol."""
    return YAHOO.get(name, name)


def is_futures(symbol):
    """True for a futures symbol (ends in =F, like GC=F for gold): futures trade almost 24 hours,
    so an intraday chart must not hide the nights."""
    return isinstance(symbol, str) and symbol.strip().upper().endswith("=F")


def _clean_download(close):
    """What every Yahoo download goes through: numbers only, positive and finite (an inf close used
    to survive and become "the price"), plain New York times without a time-zone label (like the
    saved files), no repeated times, in order."""
    close = pd.to_numeric(pd.Series(close, dtype=float), errors="coerce")
    close = close[np.isfinite(close) & (close > 0)]
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_convert("America/New_York").tz_localize(None)
    return close[~close.index.duplicated()].sort_index()


def _yahoo(ticker, start):
    import yfinance as yf                 # only needed for live tickers
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False, threads=False)
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):   # newer yfinance: one column per ticker
        close = close.iloc[:, 0]
    return close


def live_prices(text, start="2000-01-01", download=None):
    """Daily closes of any ticker from Yahoo Finance (split-adjusted, like the saved files), indexed
    by plain dates. Raises ValueError for a bad or unknown ticker or too short a history, and
    ConnectionError if Yahoo cannot be reached. `download(ticker, start)` is swappable so
    the tests need no internet; left out, it is this module's _yahoo, looked up at call time so a
    test can replace herding.explore._yahoo for the whole app."""
    t = clean_ticker(text)
    download = download or _yahoo
    try:
        close = download(t, start)
    except Exception as e:                # network trouble, Yahoo blocking the server, ...
        raise ConnectionError(f"couldn't reach Yahoo Finance for {t} ({type(e).__name__})") from e
    idx = getattr(close, "index", None)
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        # a daily bar keeps the date in its own time zone: crypto's 18 Nov 00:00 UTC is 18 Nov, but
        # moved to New York (17 Nov 19:00) it became 17 Nov, every date a day early (BE-18).
        # Only intraday bars are moved to New York time (_clean_download)
        close = pd.Series(close).copy()
        close.index = idx.tz_localize(None)
    close = _clean_download(close)
    if isinstance(close.index, pd.DatetimeIndex):   # a bar stamped within the day becomes its date
        close.index = close.index.normalize()
        close = close[~close.index.duplicated()]
    if len(close) == 0:
        raise ValueError(f"no prices found for {t}: check the ticker")
    if len(close) <= MIN_DAYS:
        raise ValueError(f"{t} has only {len(close)} days of prices; need more than {MIN_DAYS} "
                         f"(about 4 years) to measure its tails")
    return close.rename(t)


def live_market(text, start="2000-01-01", download=None):
    """(closes, daily log returns, summary) of any ticker from Yahoo, as the Workspace shows it.
    Raises what live_prices raises, and a ValueError worded for visitors when the price barely
    moves (a pegged or halted asset): the summary's own message ("need more than 13 nonzero
    returns, got 1") means nothing to someone who only typed a ticker (WS-25)."""
    close = live_prices(text, start, download)
    r = log_returns(close)
    try:
        s = summary(r.values)
    except ValueError as e:
        if "nonzero returns" not in str(e) and "constant" not in str(e):
            raise
        moved = int(np.count_nonzero(r.values))
        raise ValueError(f"{close.name}'s price barely moves (only {moved:,} "
                         f"day{'' if moved == 1 else 's'} changed), so there are no tails to measure. "
                         f"Try a stock or an index.") from e
    return close, r, s


def log_returns(close):
    """Daily log returns of a Series of closes (one fewer value, indexed by the later day).
    The closes must be in date order with no repeated dates."""
    close = _prices(close)
    return np.log(close).diff().dropna()


# ------------------------------------------------------------------ trader mixes

def fitted_settings(path=FIT):
    """The S&P 500 fit: (settings dict, burn-in days)."""
    d = json.loads(open(path).read())
    return dict(d["settings"]), int(d["burn"])


def mix_settings(p, mix, a1):
    """(settings, chartists flag) for one trader mix; p itself is left unchanged."""
    if mix not in MIXES:
        raise ValueError(f"unknown trader mix {mix!r}; choose from {MIXES}")
    if not is_number(a1) or a1 < 0:
        raise ValueError(f"herding strength a1 must be a number >= 0, got {a1!r}")
    q = dict(p)
    if mix == "fundamentalists":
        return q, False
    q["a1"] = 0.0 if mix == "chartists" else float(a1)
    return q, True


def run_mix(p, mix, a1, days, seed, burn):
    """One model history for a trader mix: (returns, None), with the first `burn` days
    dropped so the starting state does not matter, or (None, the RunawayError) if the
    price left its value for good. (With the S&P 500 fit and a1 = 0, measured Nov 2026:
    9 of seeds 0-9 run away as a bubble with 99% chartists.)"""
    if isinstance(days, bool) or not isinstance(days, (int, np.integer)) or not 500 <= days <= MAX_DAYS:
        raise ValueError(f"days must be a whole number from 500 to {MAX_DAYS}, got {days!r}")
    if isinstance(burn, bool) or not isinstance(burn, (int, np.integer)) or burn < 0:
        raise ValueError(f"burn must be a whole number >= 0, got {burn!r}")
    seed = _whole(seed, "seed", 0)        # numpy would turn -1, 1.5 or "1" into a confusing error
    q, chartists = mix_settings(p, mix, a1)
    try:
        out = model.simulate(q, int(days) + int(burn), seed=seed, chartists=chartists)
    except model.RunawayError as e:
        return None, e
    return out["r"][int(burn):], None


def runaway_day(err, burn):
    """The day a history ran away, counted after the burn-in (the days the site shows)."""
    return int(err.day - burn)


def run_mixes(p, burn, a1, a2, v1, days, seed):
    """The three trader mixes on one seed, with a2 and v1 replaced: {mix: (returns or None,
    the runaway day after burn-in or None)}."""
    q = dict(p, a2=a2, v1=v1)
    out = {}
    for m in MIXES:
        r, err = run_mix(q, m, a1, days, seed, burn)
        out[m] = (r, None if err is None else runaway_day(err, burn))
    return out


def mix_summaries(mixes):
    """{mix: summary of its returns, or None if it ran away} for run_mixes' output."""
    return {m: (summary(r) if r is not None else None) for m, (r, _) in mixes.items()}


def mix_tails(mixes):
    """{mix: tail exponent, or None if it ran away}: what the Workspace's run log keeps."""
    return {m: (s["tail"] if s is not None else None) for m, s in mix_summaries(mixes).items()}


def differs_from_fit(run, p, keys=("a2", "v1"), tol=1e-9):
    """True if a run's settings other than the herding strength differ from the fit p, so the run
    can't be read against the saved sweep (which changed only a1)."""
    return any(abs(float(run[k]) - float(p[k])) > tol for k in keys)


def dial_runs(log, p):
    """What the Workspace's dial chart plots from the run log [(time, settings, tails), ...]:
    ([(a1, herding tail), ...] for the runs whose herding history didn't run away, and whether any
    of those used an a2 or v1 different from the fit p). The pink sweep changed only a1, so such a
    run sits on the dial without being comparable, and the page must say so even after the run
    shown now is back at the fit (WS-24)."""
    kept = [(prm, tl["herding"]) for _, prm, tl in log if tl.get("herding") is not None]
    return [(prm["a1"], t) for prm, t in kept], any(differs_from_fit(prm, p) for prm, _ in kept)


# ------------------------------------------------------------------ describing returns

def bell_big_days(n, size=BIG):
    """How many of n days a bell curve expects to move more than `size` standard deviations
    either way: n * P(|Z| > size) = n * erfc(size / sqrt 2)."""
    return n * math.erfc(size / math.sqrt(2))


def bell_tail(x):
    """Share of days a bell curve expects to move more than x standard deviations either way:
    P(|Z| > x) = erfc(x / sqrt 2). x may be one number or an array."""
    v = np.asarray(x, dtype=float)
    out = np.array([math.erfc(xi / math.sqrt(2)) for xi in v.ravel()]).reshape(v.shape)
    return float(out) if out.ndim == 0 else out


def bell_density(x):
    """The bell curve's height at x standard deviations (the standard normal density
    exp(-x^2 / 2) / sqrt(2 pi)), for drawing it next to a histogram. x may be a number or an array."""
    v = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(v)):
        raise ValueError("x must be finite")
    out = np.exp(-v ** 2 / 2) / math.sqrt(2 * math.pi)
    return float(out) if out.ndim == 0 else out


def summary(r):
    """The numbers the app shows for one series of daily returns."""
    r = np.asarray(r, dtype=float)
    if len(r) <= max(LAGS) + 2 or len(r) < 500:
        raise ValueError(f"need at least 500 days of returns, got {len(r)}")
    z = stats.standardize(r)
    return {
        "days": len(r),
        "kurtosis": float(stats.kurtosis(r)),
        "tail": float(stats.hill_alpha(r, 0.05)),
        "memory": float(stats.acf(np.abs(r), MEMORY_LAG)[0]),
        "worst": float(z.min()),
        "big_days": int(np.sum(np.abs(z) > BIG)),
        "bell_big_days": bell_big_days(len(r)),
    }


def closest_mix(target, mixes):
    """Name of the mix whose summary is closest to the target's, on the two shape measures
    that are steady in long samples: the tail exponent and volatility memory, each as a
    share of the target's own value. Mixes that are None (ran away) are skipped, and so are mixes
    whose numbers are not finite (NaN would otherwise make the pick depend on the order)."""
    gaps = {k: mix_gap(target, s) for k, s in mixes.items() if s is not None}
    ok = {k: g for k, g in gaps.items() if np.isfinite(g)}
    if not ok:
        raise ValueError("no trader mix to compare with (every one ran away or has no numbers)")
    return min(ok, key=ok.get)


def insight(name, s, closest):
    """One plain sentence about an asset, built only from its numbers. "N times more" only when the
    market had at least twice the bell curve's count (a real market: about 90 times more)."""
    n, bell = s["big_days"], s["bell_big_days"]
    label = {"fundamentalists": "fundamentalists-only", "chartists": "chartists without herding",
             "herding": "herding"}[closest]
    count = "no days" if n == 0 else ("1 day" if n == 1 else f"{n} days")
    times = n / bell if bell > 0 else None        # no bell-curve days expected: no ratio to give
    return (f"{name} had {count} bigger than {BIG:g} standard deviations; "
            f"a bell curve expects about {bell:.1f}"
            + (f", so about {times:.0f} times more" if times is not None and times >= 2 else "")
            + f". Its tails and volatility memory are closest to the {label} mix.")


# ------------------------------------------------------------------ chart data

def tail_curve(r, points=150):
    """(x, share): moves in standard deviations and the share of days at least that big,
    thinned to about `points` points spread evenly on a log scale."""
    x, cc = stats.ccdf(r)
    keep = x > 0.1
    x, cc = x[keep], cc[keep]
    idx = np.unique(np.round(np.geomspace(1, len(x), points)).astype(int) - 1)
    return x[idx], cc[idx]


def histogram(r, edges=np.arange(-8, 8.01, 0.4), empty=0.0):
    """(centers, density) of standardized returns: the density integrates to the share of
    days inside the edges (1 when no day falls outside). Bins with no day get `empty`: pass
    np.nan for a log-scale chart, where 0 cannot be drawn."""
    z = stats.standardize(r)
    counts, edges = np.histogram(z, bins=edges)
    d = counts / (len(z) * np.diff(edges))
    return (edges[:-1] + edges[1:]) / 2, np.where(counts > 0, d, empty)


def slope_guide(r, at=3.0, slope=3.0, xs=(2.0, 30.0)):
    """(x, y): a straight guide line on the log-log tail chart with the given slope (3, the inverse
    cubic law), passing through the tail curve of r at `at` standard deviations, drawn from xs[0] to
    xs[1]: y = y0 * (x / at) ** -slope."""
    for v, nm in ((at, "at"), (slope, "slope")):
        if not is_number(v) or v <= 0:
            raise ValueError(f"{nm} must be a number > 0, got {v!r}")
    x, y = tail_curve(r)
    y0 = float(np.interp(at, x, y))
    xs = np.asarray(xs, dtype=float)
    return xs, y0 * (xs / at) ** -slope


def moves_in_sd(r, ref=None):
    """Each move of r in standard deviations of `ref` (default: r itself): (r - mean) / sd, with
    the population sd (ddof=0) like stats.standardize, so a "4 sd" day means the same on every
    chart. A Series keeps its dates."""
    x = stats._returns(r)
    base = x if ref is None else stats._returns(ref)
    if base.max() == base.min():
        raise ValueError("reference returns are constant: no spread to measure")
    sd = base.std()
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("reference returns are too large or too small to measure their spread")
    z = (x - base.mean()) / sd
    return pd.Series(z, index=r.index, name=r.name) if isinstance(r, pd.Series) else z


def memory_curve(r, lags=LAGS):
    """(lags, correlation of move sizes that many days apart)."""
    return np.asarray(lags), stats.acf(np.abs(np.asarray(r, dtype=float)), lags)


# ------------------------------------------------------------------ checks from the proposal

def bell_check(kurtoses, n):
    """The no-chartists check: every history's excess kurtosis must be within 5 standard
    errors of 0. For n bell-curve days the standard error of the sample excess kurtosis is
    about sqrt(24 / n), so the allowed band is +-5 sqrt(24 / n) (0.24 for 10,000 days)."""
    k = np.asarray(kurtoses, dtype=float)
    if k.ndim != 1 or k.size == 0 or not np.all(np.isfinite(k)):
        raise ValueError("need at least one finite kurtosis")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError(f"n must be a whole number of days >= 2, got {n!r}")
    tol = 5 * math.sqrt(24 / n)
    return {"mean": float(k.mean()), "worst": float(k[np.argmax(np.abs(k))]), "tol": tol,
            "passed": bool(np.all(np.abs(k) < tol))}


def tail_check(tails, lo=2.5, hi=3.5):
    """The inverse-cubic-law check: the median tail exponent must lie in [lo, hi]
    ("about 3"; the proposal expected 2.6 to 3). The median, not the mean, so one odd history
    can't decide it (the Checks page states this rule). None marks a history that ran away: it
    is left out and counted in left_out. low/high are the smallest and largest tail."""
    # NaN is a runaway too (how None shows up after pandas or a JSON-to-DataFrame round trip)
    vals = np.asarray([np.nan if v is None else v for v in tails] if not isinstance(tails, np.ndarray)
                      else tails, dtype=float)
    if vals.ndim != 1:
        raise ValueError("need one list of finite tail exponents (at least one)")
    t = vals[~np.isnan(vals)]
    if t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("need one list of finite tail exponents (at least one)")
    if not lo < hi:
        raise ValueError(f"lo must be below hi, got {lo}, {hi}")
    m = float(np.median(t))
    return {"median": m, "mean": float(t.mean()), "low": float(t.min()), "high": float(t.max()),
            "n": int(t.size), "left_out": int(vals.size - t.size), "passed": lo <= m <= hi}


def kirman_match(k, N, eps, delta):
    """Kirman's ants: (share of steps with each count 0..N in the simulation k, the exact
    long-run share from kirman.stationary, their total variation distance
    0.5 * sum |simulated - exact|: 0 = identical, 1 = no overlap)."""
    from . import kirman
    N = _whole(N, "N (the number of ants)", 2)     # 2.5 used to reach np.bincount as a TypeError
    k = np.asarray(k)
    if k.ndim != 1 or k.size == 0:
        raise ValueError("ant counts must be one non-empty list")
    if not np.all(np.isfinite(k)) or not np.all(k == np.round(k)):
        raise ValueError("ant counts must be whole numbers")      # NaN would be counted as 0 ants
    if k.min() < 0 or k.max() > N:
        raise ValueError(f"ant counts must be between 0 and N={N}")
    sim = np.bincount(k.astype(int), minlength=N + 1) / k.size
    exact = kirman.stationary(N, eps, delta)
    return sim, exact, float(0.5 * np.abs(sim - exact).sum())


def kirman_path(k, upto=5_000_000, every=2_500):
    """(steps, counts): every `every`-th count of the first `upto` steps of an ant run, few enough
    points to draw."""
    upto, every = _whole(upto, "upto", 1), _whole(every, "every", 1)
    k = np.asarray(k)
    if k.ndim != 1:
        raise ValueError("ant counts must be one list (one count per step)")
    steps = np.arange(0, min(upto, len(k)), every)
    return steps, k[steps]


def spread(values, lo=5, hi=95):
    """{median, low, high, n} of the values that are not None or NaN (histories that ran away):
    low and high are the lo-th and hi-th percentiles (5 and 95: the middle 90%)."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[~np.isnan(v)] if v.ndim == 1 else v
    if v.ndim != 1 or v.size == 0:
        raise ValueError("need at least one number")
    if not 0 <= lo <= hi <= 100:
        raise ValueError(f"need 0 <= lo <= hi <= 100, got {lo}, {hi}")
    a, b = np.percentile(v, [lo, hi])
    return {"median": float(np.median(v)), "low": float(a), "high": float(b), "n": int(v.size)}


def runaway_count(nums, mix):
    """(how many saved histories of a mix ran away, how many there are) from web_numbers()."""
    days = nums["checks"][mix]["runaway_day"]
    return sum(d is not None for d in days), len(days)


CHECK_MEASURES = ("tail", "kurtosis", "memory", "worst")


def check_table(nums, spx, mix="herding"):
    """Check 4 of the proposal: for each measure, the median and middle 90% (5th-95th percentile)
    over the saved histories of a mix, next to the S&P 500's own number (spx: its summary).
    A DataFrame indexed by measure with columns median, low, high, n, sp500."""
    rows = {m: {**spread(nums["checks"][mix][m]), "sp500": float(spx[m])} for m in CHECK_MEASURES}
    return pd.DataFrame.from_dict(rows, orient="index")


# ------------------------------------------------------------------ overview of every market

def market_table():
    """One row per saved asset: group, name and its summary numbers, plus how many times more
    big days (> 4 sd) it had than a bell curve expects."""
    rows = []
    for group, names in catalog().items():
        for name in names:
            s = summary(log_returns(saved_prices(name)).values)
            rows.append({"group": group, "name": name, **s,
                         "times_bell": s["big_days"] / s["bell_big_days"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figure gallery

def gallery_entries(text):
    """Figures from figures/GALLERY.md: a list of {section, file, title, caption}. A section is a
    '## ' heading; each '![title](file)' starts a figure whose caption is the text after it,
    up to the next figure, heading or '---' line."""
    out, section, cur = [], "", None
    for line in text.splitlines():
        m = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", line.strip())
        if line.startswith("## ") or line.strip() == "---" or m:
            if cur:
                cur["caption"] = "\n".join(cur["caption"]).strip()
                out.append(cur)
                cur = None
            if line.startswith("## "):
                section = line[3:].strip()
            if m:
                cur = {"section": section, "title": m.group(1), "file": m.group(2), "caption": []}
        elif cur is not None:
            cur["caption"].append(line)
    if cur:
        cur["caption"] = "\n".join(cur["caption"]).strip()
        out.append(cur)
    return out


def runaway_path(p, mix, a1, seed, day):
    """The first `day` + 1 daily closes (lp, lf, z) of a history that ran away on `day`,
    rebuilt by rerunning the same seed with the runaway guard off (max_gap=None). The
    guard only stops a run, it never changes its random draws, so the path is the same."""
    if isinstance(day, bool) or not isinstance(day, (int, np.integer)) or day < 1:
        raise ValueError(f"day must be a whole number >= 1, got {day!r}")
    q, chartists = mix_settings(p, mix, a1)
    out = model.simulate(q, int(day), seed=seed, chartists=chartists, max_gap=None)
    return {k: out[k] for k in ("lp", "lf", "z")}


GAP_CAP = 0.5            # the Story chart shows mispricing up to e^0.5 - 1 = +65%; beyond is "off the chart"


def runaway_story(p, seed, day, burn, mix="chartists", a1=0.0):
    """What the Story page tells about a history that ran away on `day` (counted after burn-in):
    z, the share of chartists each day (burn-in included); gap, how far the price sits above (+) or
    below (-) its value, as a fraction, capped at +-(e^GAP_CAP - 1); gap_cap, that cap;
    share_before, the share of chartists 5 days before the end; calm_days, about how long the
    price stayed near its value (all but the last 3 days)."""
    day = _whole(day, "day", 1)
    path = runaway_path(p, mix, a1, seed, day + _whole(burn, "burn", 0))
    z = path["z"]
    return {"z": z, "gap": np.expm1(np.clip(path["lp"] - path["lf"], -GAP_CAP, GAP_CAP)),
            "gap_cap": float(np.expm1(GAP_CAP)), "share_before": float(z[max(len(z) - 6, 0)]),
            "calm_days": max(day - 3, 0)}       # a runaway on day 1-3 had no calm stretch (not -2 days)


# ------------------------------------------------------------------ workspace numbers

def max_drawdown(r):
    """Largest fall from a running peak, as a fraction of the peak (0.2 = a 20% fall), for
    daily log returns r. The price path is exp(cumsum r), starting at 1 before the first day."""
    r = np.asarray(r, dtype=float)
    if r.size == 0 or not np.all(np.isfinite(r)):
        raise ValueError("need at least one finite return")
    lp = np.concatenate([[0.0], np.cumsum(r)])
    return float(1 - np.exp(np.min(lp - np.maximum.accumulate(lp))))


def big_day_list(r, size=BIG, within=None):
    """Days that moved more than `size` standard deviations: a DataFrame indexed like r with
    columns move (the simple return, e.g. -0.2 = -20%) and sd (the move in standard deviations),
    biggest first. r is a Series of daily log returns. sd is always measured against all of r;
    `within` (dates, or a Series whose dates are used) keeps only the big days among them, e.g.
    the range a chart shows. Rows are found by position, so a repeated date can't mix rows up."""
    r = pd.Series(r, dtype=float)
    z = moves_in_sd(r.values)
    pos = np.flatnonzero(np.abs(z) > size)
    pos = pos[np.argsort(-np.abs(z[pos]), kind="stable")]
    out = pd.DataFrame({"move": np.expm1(r.values[pos]), "sd": z[pos]}, index=r.index[pos])
    if within is not None:
        dates = within.index if isinstance(within, (pd.Series, pd.DataFrame)) else pd.Index(within)
        out = out[out.index.isin(dates)]
    return out


TABLE_MEASURES = ["days", "tail", "kurtosis", "memory", "worst", "big_days", "max_drawdown", "runaway_day"]


def mix_table(name, s, r, sums, mixes):
    """The Workspace's "Model vs market" numbers: one row for the market (`name`, its summary s and
    log returns r), one per trader mix (sums: mix_summaries, mixes: run_mixes output) and one for a
    bell curve as long as the market, with columns TABLE_MEASURES. NaN where a number does not apply:
    a mix that ran away has only its runaway_day (after burn-in); a bell curve has no tail exponent
    (its tail falls faster than any power) and no single worst day or drawdown."""
    def row(sm, rr, gone):
        if sm is None:
            return {**{k: np.nan for k in TABLE_MEASURES}, "runaway_day": gone}
        return {**{k: sm[k] for k in TABLE_MEASURES[:6]}, "max_drawdown": max_drawdown(rr),
                "runaway_day": np.nan}
    rows = {name: row(s, r, None)}
    for m in MIXES:
        rows[m] = row(sums[m], mixes[m][0], mixes[m][1])
    rows["bell"] = {"days": s["days"], "tail": np.nan, "kurtosis": 0.0, "memory": 0.0, "worst": np.nan,
                    "big_days": s["bell_big_days"], "max_drawdown": np.nan, "runaway_day": np.nan}
    return pd.DataFrame.from_dict(rows, orient="index", columns=TABLE_MEASURES)


def mix_gap(target, s):
    """How far a mix's summary is from the target's (what closest_mix minimizes): the
    tail-exponent gap as a share of the target's, plus the volatility-memory gap as a share of
    the target's (at least 0.05). 0 = the same on both measures. The target must have a finite,
    non-zero tail and a finite memory (a gap "as a share of 0" means nothing)."""
    if not (np.isfinite(target["tail"]) and target["tail"] != 0 and np.isfinite(target["memory"])):
        raise ValueError(f"the target needs a finite, non-zero tail and a finite memory, got "
                         f"{target['tail']!r}, {target['memory']!r}")
    return (abs(s["tail"] - target["tail"]) / abs(target["tail"])
            + abs(s["memory"] - target["memory"]) / max(abs(target["memory"]), 0.05))


# ------------------------------------------------------------------ smooth charts, graphing changes

def thin_indices(y, max_points=1500):
    """Positions of at most `max_points` values of y that keep its shape for a chart: y is cut
    into equal slices and each slice keeps its lowest and highest value (plus the first and last
    value overall), so no crash or spike disappears. Shorter y is returned whole."""
    y = np.asarray(y, dtype=float)
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)) or max_points < 4:
        raise ValueError(f"max_points must be a whole number >= 4, got {max_points!r}")
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError("y must be one finite list of numbers")
    n = len(y)
    if n <= max_points:
        return np.arange(n)
    edges = np.linspace(0, n, max_points // 2).astype(int)        # max_points//2 - 1 slices
    keep = [0, n - 1]
    for lo, hi in zip(edges[:-1], edges[1:]):
        seg = y[lo:hi]
        keep += [lo + int(seg.argmin()), lo + int(seg.argmax())]
    return np.unique(keep)


def sweep(p, a1_values, seeds, days, burn, pool=None):
    """The herding mix at each herding strength a1 and seed: a DataFrame with columns
    a1, seed, tail, memory, kurtosis (NaN if the history ran away) and runaway_day (NaN if not)."""
    seeds = [_whole(s, "seed", 0) for s in seeds]      # checked first: int(1.7) used to run seed 1
    jobs = [(p, "herding", float(a1), days, s, burn) for a1 in a1_values for s in seeds]
    out = pool.starmap(run_mix, jobs) if pool is not None else [run_mix(*j) for j in jobs]
    rows = []
    for (_, _, a1, _, s, _), (r, err) in zip(jobs, out):
        sm = summary(r) if r is not None else None
        rows.append({"a1": a1, "seed": s, "tail": sm["tail"] if sm else np.nan,
                     "memory": sm["memory"] if sm else np.nan, "kurtosis": sm["kurtosis"] if sm else np.nan,
                     "runaway_day": float(runaway_day(err, burn)) if err is not None else np.nan})
    return pd.DataFrame(rows, columns=SWEEP_COLUMNS)


SWEEP_COLUMNS = ["a1", "seed", "tail", "memory", "kurtosis", "runaway_day"]


def sweep_band(rows, min_survivors=5):
    """The dial chart's line: for each herding strength a1 in the sweep (rows: sweep() output or its
    saved records), the median and quartiles (q1, q3) of the tail exponent over the histories that
    did not run away, n of them, and how many ran away. A median of 1 or 2 survivors is mostly luck
    (the calm histories are the ones that survive), so median/q1/q3 are NaN and shown is False
    when n < min_survivors."""
    min_survivors = _whole(min_survivors, "min_survivors", 1)
    t = pd.DataFrame(rows)
    if not {"a1", "tail"} <= set(t.columns):
        raise ValueError("rows need a1 and tail")
    out = []
    for a1, g in t.groupby("a1", sort=True):
        ok = pd.to_numeric(g["tail"], errors="coerce").dropna()
        show = len(ok) >= min_survivors
        out.append({"a1": float(a1), "median": float(ok.median()) if show else np.nan,
                    "q1": float(ok.quantile(0.25)) if show else np.nan,
                    "q3": float(ok.quantile(0.75)) if show else np.nan,
                    "n": int(len(ok)), "runaways": int(len(g) - len(ok)), "shown": bool(show)})
    return pd.DataFrame(out, columns=["a1", "median", "q1", "q3", "n", "runaways", "shown"])


# ------------------------------------------------------------------ standard stock numbers

INTRADAY = {"1D": ("1d", "5m"), "7D": ("7d", "30m")}      # Yahoo period and bar size


def _yahoo_intraday(ticker, period, interval):
    import yfinance as yf
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False,
                     threads=False)
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    close = df["Close"]
    return close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close


def intraday_prices(text, frame, download=None):
    """Prices within the day for the 1D (5-minute bars) or 7D (30-minute bars) view, from Yahoo,
    in New York exchange time without a time-zone label (like the daily files' dates, so the two can
    be compared). Raises ValueError for a bad ticker or no prices, ConnectionError if Yahoo can't be
    reached. `download` defaults to this module's _yahoo_intraday, looked up at call time."""
    if frame not in INTRADAY:
        raise ValueError(f"frame must be one of {list(INTRADAY)}, got {frame!r}")
    t = clean_ticker(text)
    download = download or _yahoo_intraday
    period, interval = INTRADAY[frame]
    try:
        close = download(t, period, interval)
    except Exception as e:
        raise ConnectionError(f"couldn't reach Yahoo Finance for {t} ({type(e).__name__})") from e
    close = _clean_download(close)          # drops NaN, inf and <= 0 bars; drops the time zone
    if len(close) < 2:
        raise ValueError(f"no intraday prices for {t} right now")
    return close.rename(t)


FRAMES = ("1D", "7D", "YTD", "1Y", "5Y", "20Y", "All")     # the range buttons, in order
FRAME_YEARS = {"1Y": 1, "5Y": 5, "20Y": 20}     # calendar years (1Y = 52 weeks) in a range
DAILY_FALLBACK = {"1D": 2, "7D": 6}       # without intraday bars: the last 1 or 5 daily moves
FALLBACK_NOTE = "Intraday prices aren't available right now, so this shows daily closes."


def price_range(close, frame, intraday=None):
    """(prices, note): the prices a chart shows for one range button. close is the full daily
    history; intraday, for 1D and 7D, is intraday_prices' bars (None if they couldn't be fetched:
    then the last 2 or 6 daily closes are shown and note says so). YTD starts at the last close of
    the previous year, the usual year-to-date base, so it has 2 prices even on 2 January. Every
    range has at least 2 prices."""
    if frame not in FRAMES:
        raise ValueError(f"frame must be one of {list(FRAMES)}, got {frame!r}")
    c = _prices(close, "daily closes")
    if frame in DAILY_FALLBACK:
        if intraday is not None:
            try:
                return _prices(intraday, "intraday prices"), ""
            except ValueError:
                pass                       # too few or bad bars: daily closes are better than nothing
        return c.iloc[-DAILY_FALLBACK[frame]:], FALLBACK_NOTE
    if frame == "All":
        return c, ""
    if frame == "YTD":
        _dated(c, "daily closes")
        before = c.index[c.index < pd.Timestamp(c.index[-1].year, 1, 1)]
        start = before[-1] if len(before) else c.index[0]
        return c[c.index >= start], ""
    # calendar spans, as on stock sites: 1Y is the same 52 weeks as range_stats' 52-week high and low
    # (a fixed 252 closes could differ from it by a day, so "High" and "52-week high" disagreed: WS-6)
    _dated(c, "daily closes")
    start = c.index[-1] - (pd.Timedelta(weeks=52) if frame == "1Y" else pd.DateOffset(years=FRAME_YEARS[frame]))
    out = c[c.index > start]
    return (out if len(out) >= 2 else c.iloc[-2:]), ""


def daily_within(daily, prices):
    """The daily closes from the first day of `prices` to its last moment. prices may be intraday
    bars (9:30 on the first day), so the first day is compared as a date: comparing midnight with
    9:30 left the first day out."""
    d, c = _dated(_prices(daily, "daily closes"), "daily closes"), _dated(_prices(prices), "prices")
    return d[(d.index >= c.index[0].normalize()) & (d.index <= c.index[-1])]


def drawdown_series(close):
    """How far each price sits below the highest price so far, as a fraction (0 at a new high,
    -0.25 = 25% below it). Its minimum, with the sign flipped, is the max drawdown."""
    c = _prices(close)
    return c / c.cummax() - 1


def day_change(close):
    """The last day's change: last close / the close before it - 1 (0.01 = up 1%)."""
    c = _prices(close)
    return float(c.iloc[-1] / c.iloc[-2] - 1)


def past_year(close, year_days=YEAR_DAYS, step=5):
    """The past year's closes (the last year_days), every step-th one, rounded to 4 decimals: a
    small sparkline for the market list."""
    year_days, step = _whole(year_days, "year_days", 1), _whole(step, "step", 1)
    c = _prices(close)
    return c.iloc[-year_days:].iloc[::step].round(4).tolist()


def moves_in_range(r, prices):
    """The daily moves a price range shows: those into its 2nd..last price (n prices have n - 1
    moves). r is indexed by the later day of each move, so the move into the range's first price
    happened before the range (YTD's would be 31 December of last year) and is left out."""
    r = pd.Series(r, dtype=float)
    p = _prices(prices)
    return r[(r.index > p.index[0]) & (r.index <= p.index[-1])]


def days_per_year(daily):
    """Trading days in a year for annualizing: 365 for a market open every day (crypto: more than
    a tenth of its past year's dates fall on a weekend), else YEAR_DAYS (252)."""
    d = _dated(_prices(daily, "daily closes"), "daily closes")
    year = d.index[d.index > d.index[-1] - pd.Timedelta(weeks=52)]
    return 365 if np.mean(year.dayofweek >= 5) > 0.1 else YEAR_DAYS


def range_stats(close, daily=None, year_days=None):
    """The usual numbers for a price chart's range: last, first, change, high, low, max drawdown
    within the range, and (from `daily`, the full daily closes, when given) the 52-week high and
    low and the annualized volatility of the range's daily returns. The 52 weeks are counted by
    date (a market open every day has 364 closes in them, a stock about 251) and the volatility is
    annualized with days_per_year(daily). Given year_days instead, both use the last year_days
    closes and sqrt(year_days). Both series must be in date order with no repeated dates; with
    daily, both must be indexed by dates."""
    if year_days is not None:
        year_days = _whole(year_days, "year_days", 2)
    c = _prices(close)
    out = {"last": float(c.iloc[-1]), "first": float(c.iloc[0]), "change": float(c.iloc[-1] / c.iloc[0] - 1),
           "high": float(c.max()), "low": float(c.min()),
           # 0.0 - x, not -x: a range that never fell has min 0.0, and -0.0 printed as "-0.0%"
           "drawdown": float(0.0 - drawdown_series(c).min())}
    if daily is not None:
        d = _dated(_prices(daily, "daily closes"), "daily closes")
        _dated(c, "prices")
        if year_days is None:
            year, per_year = d[d.index > d.index[-1] - pd.Timedelta(weeks=52)], days_per_year(d)
        else:
            year, per_year = d.iloc[-year_days:], year_days
        out["high_52w"], out["low_52w"] = float(year.max()), float(year.min())
        r = np.diff(np.log(daily_within(d, c).values))
        out["volatility"] = float(r.std(ddof=1) * math.sqrt(per_year)) if len(r) >= 2 else float("nan")
    return out


# ------------------------------------------------------------------ the saved numbers behind the site

WEB_NUMBERS = market_data.ROOT / "web" / "numbers.json"


def web_numbers(path=WEB_NUMBERS):
    """The per-seed results made by scripts/web_numbers.py with the same recipe as the figures
    (held-out seeds, S&P-length histories), so the site and figures/RESULTS.md agree."""
    d = json.loads(open(path).read())
    need = {"checks", "sweep", "recipe"}
    if not need <= set(d):
        raise ValueError(f"{path} is missing {sorted(need - set(d))}; rerun scripts/web_numbers.py")
    return d
