"""The science behind the web app (app.py): load any market, run the three trader mixes,
and describe returns with a few numbers. app.py only draws what these functions return.

Assets come in three groups:
    stocks       the 20 saved stocks (data/stocks/), or any ticker fetched live from Yahoo
    index funds  S&P 500, Nasdaq-100, Dow Jones (data/)
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


# ------------------------------------------------------------------ assets

def catalog():
    """{group: [asset names]} for the market list, in display order."""
    return {"Stocks": market_data.stock_tickers(), "Index funds": list(INDEXES),
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


def _yahoo(ticker, start):
    import yfinance as yf                 # only needed for live tickers
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False, threads=False)
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):   # newer yfinance: one column per ticker
        close = close.iloc[:, 0]
    return close


def live_prices(text, start="2000-01-01", download=_yahoo):
    """Daily closes of any ticker from Yahoo Finance (split-adjusted, like the saved files).
    Raises ValueError for a bad or unknown ticker or too short a history, and
    ConnectionError if Yahoo cannot be reached. `download(ticker, start)` is swappable so
    the tests need no internet."""
    t = clean_ticker(text)
    try:
        close = download(t, start)
    except Exception as e:                # network trouble, Yahoo blocking the server, ...
        raise ConnectionError(f"couldn't reach Yahoo Finance for {t} ({type(e).__name__})") from e
    close = pd.to_numeric(pd.Series(close, dtype=float), errors="coerce").dropna()
    close = close[close > 0]
    close = close[~close.index.duplicated()].sort_index()
    if len(close) == 0:
        raise ValueError(f"no prices found for {t}: check the ticker")
    if len(close) <= MIN_DAYS:
        raise ValueError(f"{t} has only {len(close)} days of prices; need more than {MIN_DAYS} "
                         f"(about 4 years) to measure its tails")
    return close.rename(t)


def log_returns(close):
    """Daily log returns of a Series of closes (one fewer value, indexed by the later day)."""
    close = pd.Series(close, dtype=float)
    if len(close) < 2 or not (close > 0).all() or not np.isfinite(close).all():
        raise ValueError("need at least 2 positive, finite prices")
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
    q, chartists = mix_settings(p, mix, a1)
    try:
        out = model.simulate(q, int(days) + int(burn), seed=seed, chartists=chartists)
    except model.RunawayError as e:
        return None, e
    return out["r"][int(burn):], None


# ------------------------------------------------------------------ describing returns

def bell_big_days(n, size=BIG):
    """How many of n days a bell curve expects to move more than `size` standard deviations
    either way: n * P(|Z| > size) = n * erfc(size / sqrt 2)."""
    return n * math.erfc(size / math.sqrt(2))


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
    share of the target's own value. Mixes that are None (ran away) are skipped."""
    ok = {k: s for k, s in mixes.items() if s is not None}
    if not ok:
        raise ValueError("no trader mix to compare with (every one ran away)")
    return min(ok, key=lambda k: mix_gap(target, ok[k]))


def insight(name, s, closest):
    """One plain sentence about an asset, built only from its numbers."""
    times = s["big_days"] / s["bell_big_days"]
    label = {"fundamentalists": "fundamentalists-only", "chartists": "chartists without herding",
             "herding": "herding"}[closest]
    return (f"{name} had {s['big_days']} days bigger than {BIG:g} standard deviations; "
            f"a bell curve expects about {s['bell_big_days']:.1f}"
            + (f", so about {times:.0f} times fewer" if times >= 2 else "")
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


def histogram(r, edges=np.arange(-8, 8.01, 0.4)):
    """(centers, density) of standardized returns: the density integrates to the share of
    days inside the edges (1 when no day falls outside)."""
    z = stats.standardize(r)
    counts, edges = np.histogram(z, bins=edges)
    return (edges[:-1] + edges[1:]) / 2, counts / (len(z) * np.diff(edges))


def memory_curve(r, lags=LAGS):
    """(lags, correlation of move sizes that many days apart)."""
    return np.asarray(lags), stats.acf(np.abs(np.asarray(r, dtype=float)), lags)


# ------------------------------------------------------------------ checks from the proposal

def bell_check(kurtoses, n):
    """The no-chartists check: every history's excess kurtosis must be within 5 standard
    errors of 0. For n bell-curve days the standard error of the sample excess kurtosis is
    about sqrt(24 / n), so the allowed band is +-5 sqrt(24 / n) (0.24 for 10,000 days)."""
    k = np.asarray(kurtoses, dtype=float)
    if k.size == 0 or not np.all(np.isfinite(k)):
        raise ValueError("need at least one finite kurtosis")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError(f"n must be a whole number of days >= 2, got {n!r}")
    tol = 5 * math.sqrt(24 / n)
    return {"mean": float(k.mean()), "worst": float(k[np.argmax(np.abs(k))]), "tol": tol,
            "passed": bool(np.all(np.abs(k) < tol))}


def tail_check(tails, lo=2.5, hi=3.5):
    """The inverse-cubic-law check: the average tail exponent must lie in [lo, hi]
    ("about 3"; the proposal expected 2.6 to 3)."""
    t = np.asarray(tails, dtype=float)
    if t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("need at least one finite tail exponent")
    if not lo < hi:
        raise ValueError(f"lo must be below hi, got {lo}, {hi}")
    m = float(t.mean())
    return {"mean": m, "low": float(t.min()), "high": float(t.max()), "passed": lo <= m <= hi}


def kirman_match(k, N, eps, delta):
    """Kirman's ants: (share of steps with each count 0..N in the simulation k, the exact
    long-run share from kirman.stationary, their total variation distance
    0.5 * sum |simulated - exact|: 0 = identical, 1 = no overlap)."""
    from . import kirman
    k = np.asarray(k)
    if k.size == 0 or k.min() < 0 or k.max() > N:
        raise ValueError(f"ant counts must be between 0 and N={N}")
    sim = np.bincount(k.astype(int), minlength=N + 1) / k.size
    exact = kirman.stationary(N, eps, delta)
    return sim, exact, float(0.5 * np.abs(sim - exact).sum())


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


# ------------------------------------------------------------------ workspace numbers

def max_drawdown(r):
    """Largest fall from a running peak, as a fraction of the peak (0.2 = a 20% fall), for
    daily log returns r. The price path is exp(cumsum r), starting at 1 before the first day."""
    r = np.asarray(r, dtype=float)
    if r.size == 0 or not np.all(np.isfinite(r)):
        raise ValueError("need at least one finite return")
    lp = np.concatenate([[0.0], np.cumsum(r)])
    return float(1 - np.exp(np.min(lp - np.maximum.accumulate(lp))))


def big_day_list(r, size=BIG):
    """Days that moved more than `size` standard deviations: a DataFrame indexed like r with
    columns move (the simple return, e.g. -0.2 = -20%) and sd (the move in standard deviations),
    biggest first. r is a Series of daily log returns."""
    r = pd.Series(r, dtype=float)
    z = pd.Series(stats.standardize(r.values), index=r.index)
    big = z[np.abs(z) > size]
    out = pd.DataFrame({"move": np.expm1(r[big.index]), "sd": big})
    return out.reindex(out["sd"].abs().sort_values(ascending=False).index)


def mix_gap(target, s):
    """How far a mix's summary is from the target's (what closest_mix minimizes): the
    tail-exponent gap as a share of the target's, plus the volatility-memory gap as a share of
    the target's (at least 0.05). 0 = the same on both measures."""
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
    jobs = [(p, "herding", float(a1), days, int(s), burn) for a1 in a1_values for s in seeds]
    out = pool.starmap(run_mix, jobs) if pool is not None else [run_mix(*j) for j in jobs]
    rows = []
    for (_, _, a1, _, s, _), (r, err) in zip(jobs, out):
        sm = summary(r) if r is not None else None
        rows.append({"a1": a1, "seed": s, "tail": sm["tail"] if sm else np.nan,
                     "memory": sm["memory"] if sm else np.nan, "kurtosis": sm["kurtosis"] if sm else np.nan,
                     "runaway_day": float(err.day - burn) if err is not None else np.nan})
    return pd.DataFrame(rows)


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


def intraday_prices(text, frame, download=_yahoo_intraday):
    """Prices within the day for the 1D (5-minute bars) or 7D (30-minute bars) view, from Yahoo,
    in New York exchange time without a time-zone label (like the daily files' dates, so the two can
    be compared). Raises ValueError for a bad ticker or no prices, ConnectionError if Yahoo can't be
    reached."""
    if frame not in INTRADAY:
        raise ValueError(f"frame must be one of {list(INTRADAY)}, got {frame!r}")
    t = clean_ticker(text)
    period, interval = INTRADAY[frame]
    try:
        close = download(t, period, interval)
    except Exception as e:
        raise ConnectionError(f"couldn't reach Yahoo Finance for {t} ({type(e).__name__})") from e
    close = pd.to_numeric(pd.Series(close, dtype=float), errors="coerce").dropna()
    close = close[close > 0]
    if getattr(close.index, "tz", None) is not None:      # Yahoo's bars carry a time zone; the daily files don't
        close.index = close.index.tz_convert("America/New_York").tz_localize(None)
    close = close[~close.index.duplicated()].sort_index()
    if len(close) < 2:
        raise ValueError(f"no intraday prices for {t} right now")
    return close.rename(t)


def range_stats(close, daily=None, year_days=252):
    """The usual numbers for a price chart's range: last, first, change, high, low, max drawdown
    within the range, and (from `daily`, the full daily closes, when given) the 52-week high and
    low and the annualized volatility of the range's daily returns."""
    c = pd.Series(close, dtype=float)
    if len(c) < 2 or not (c > 0).all() or not np.all(np.isfinite(c)):
        raise ValueError("need at least 2 positive, finite prices")
    out = {"last": float(c.iloc[-1]), "first": float(c.iloc[0]), "change": float(c.iloc[-1] / c.iloc[0] - 1),
           "high": float(c.max()), "low": float(c.min()),
           "drawdown": float(1 - (c / c.cummax()).min())}
    if daily is not None:
        d = pd.Series(daily, dtype=float)
        year = d.iloc[-year_days:]
        out["high_52w"], out["low_52w"] = float(year.max()), float(year.min())
        inside = d[(d.index >= c.index[0]) & (d.index <= c.index[-1])]
        r = np.log(inside).diff().dropna()
        out["volatility"] = float(r.std(ddof=1) * math.sqrt(year_days)) if len(r) >= 2 else float("nan")
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
