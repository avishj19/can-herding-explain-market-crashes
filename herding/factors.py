"""Outside factors (extension): news that every trader sees at the same moment.

Lux & Marchesi model news as small bell-curve steps in the value. Some real news is
different: an oil shock, a jump in interest rates or a Fed decision reaches every trader
at once, and a fully logical trader acts on it with no herding at all. If such news
carried the S&P 500's fat tails and volatility clustering, herding would not be needed to
explain them. This module measures how much it carries.

Data (data/download_market_data.py; every series ends 2026-10-30):
    oil     Brent crude oil, $/barrel (FRED DCOILBRENTEU, 1987-)
    gold    gold futures, front month (Yahoo GC=F, 2000-)
    copper  copper futures, front month (Yahoo HG=F, 2000-)
    dollar  broad trade-weighted US dollar (FRED DTWEXBGS, 2006-)
    rates   10-year Treasury yield (FRED DGS10, 1962-); its daily change in percentage
            points, since a yield is already a rate (a log change of a rate means little)
    data/fed_decisions.csv: FOMC decision days since February 1994 (scheduled meetings and
            the 9 surprise moves between meetings)

Timing: commodity futures settle in the early afternoon (gold 1:30 pm, oil 2:30 pm New York
time), the S&P closes at 4 pm. A late S&P move therefore shows up in the next day's commodity
change. Following Dimson (1979, J. Financial Economics 7, 197-226) the regressions include
each factor's change on the day before and the day after as well as the same day.
"""
import numpy as np
import pandas as pd

from . import market_data, stats
from .checks import finite_1d

FACTOR_DIR = market_data.DATA / "factors"
FACTORS = {"oil": "oil_brent.csv", "gold": "gold.csv", "copper": "copper.csv",
           "dollar": "dollar_index.csv", "rates": "treasury_10y_yield.csv"}
LEVEL_CHANGE = {"rates"}                 # measured as a difference, not a log change
FED = market_data.DATA / "fed_decisions.csv"


def changes(name):
    """Daily changes of one outside factor, as a Series indexed by the later date: log
    changes of prices, and percentage-point changes of the 10-year yield."""
    if name not in FACTORS:
        raise ValueError(f"unknown factor {name!r}; choose from {sorted(FACTORS)}")
    level = market_data.load_prices(FACTOR_DIR / FACTORS[name])["Close"]
    out = level.diff() if name in LEVEL_CHANGE else np.log(level).diff()
    return out.dropna().rename(name)


def table(r, names):
    """The S&P's returns (column 'spx') next to the factors' changes, on the days when
    every one of them has a value. A factor that did not trade on an S&P day (a commodity
    holiday) drops that day; its change on the next day spans both days."""
    if len(names) == 0:
        raise ValueError("need at least one factor")
    df = pd.concat([pd.Series(r, name="spx")] + [changes(n) for n in names], axis=1, join="inner")
    if len(df) < 500:
        raise ValueError(f"only {len(df)} days in common; need at least 500")
    return df


def with_leads_lags(df, names, lags=1):
    """The columns `names` shifted by -lags..+lags days (Dimson 1979), as a new frame
    named like 'oil[+1]' (tomorrow's oil change). Rows with a missing shift are dropped."""
    if isinstance(lags, bool) or int(lags) != lags or lags < 0:
        raise ValueError(f"lags must be a whole number >= 0, got {lags}")
    cols = {f"{n}[{k:+d}]": df[n].shift(-k) for n in names for k in range(-int(lags), int(lags) + 1)}
    return pd.DataFrame(cols, index=df.index).dropna()


def regress(y, X):
    """Least squares of y on the columns of X with an intercept: (betas, intercept, R^2,
    residuals). R^2 = share of y's variance the columns explain."""
    y = finite_1d(y, "y")
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError(f"X must have one row per value of y: {X.shape} vs {y.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinity")
    if len(y) <= X.shape[1] + 1:
        raise ValueError(f"need more rows ({len(y)}) than coefficients ({X.shape[1] + 1})")
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    tss = np.sum((y - y.mean()) ** 2)
    if tss == 0:
        raise ValueError("y is constant: nothing to explain")
    return coef[1:], float(coef[0]), float(1 - np.sum(resid ** 2) / tss), resid


def remove(r, names, lags=1):
    """The S&P's returns with the part explained by the factors (same day, `lags` days
    before and after) taken out: (residual Series, R^2). The residual keeps the S&P's own
    average, so only the factor-driven moves are removed."""
    df = table(r, names)
    X = with_leads_lags(df, names, lags)
    y = df["spx"].loc[X.index]
    _, _, r2, resid = regress(y.values, X.values)
    return pd.Series(resid + y.mean(), index=X.index, name="spx_residual"), r2


def betas(r, names):
    """For each factor alone, on the same day: the S&P's beta to it (S&P move per unit of
    factor move), the factor's beta to the S&P, and their correlation."""
    df = table(r, names)
    out = {}
    for n in names:
        c = float(np.corrcoef(df["spx"], df[n])[0, 1])
        sd_s, sd_f = df["spx"].std(), df[n].std()
        out[n] = dict(corr=c, spx_beta=c * sd_s / sd_f, factor_beta=c * sd_f / sd_s, days=len(df))
    return out


def rolling_corr(a, b, window=250):
    """Correlation of two aligned Series over a trailing window of `window` days."""
    if isinstance(window, bool) or int(window) != window or window < 3:
        raise ValueError(f"window must be a whole number of at least 3 days, got {window}")
    df = pd.concat([a, b], axis=1, join="inner")
    return df.iloc[:, 0].rolling(int(window)).corr(df.iloc[:, 1]).dropna()


def crash_corr(r, f, share=0.05):
    """Correlation of the S&P with a factor on the S&P's worst `share` of days and on all
    other days (Baur & Lucey 2010: a safe haven moves the other way, or not at all, in a
    crash). Returns (on worst days, on other days)."""
    df = pd.concat([pd.Series(r, name="spx"), f], axis=1, join="inner")
    worst = df["spx"] <= df["spx"].quantile(share)
    if worst.sum() < 10:
        raise ValueError("need at least 10 worst days")
    corr = [float(np.corrcoef(d["spx"], d.iloc[:, 1])[0, 1]) for d in (df[worst], df[~worst])]
    return corr[0], corr[1]


def decisions(kind=None):
    """Fed decision days (data/fed_decisions.csv): columns date, kind, change_bp.
    kind: None (all), 'scheduled' or 'unscheduled'."""
    df = pd.read_csv(FED, parse_dates=["date"])
    if kind is not None:
        if kind not in ("scheduled", "unscheduled"):
            raise ValueError(f"kind must be None, 'scheduled' or 'unscheduled', got {kind!r}")
        df = df[df["kind"] == kind]
    return df


def on_days(index, days):
    """True on each trading day in `index` that first saw one of `days`: a day that is a
    trading day maps to itself, a weekend or holiday (2020-03-15, a Sunday) to the next
    trading day. Days after the last trading day are ignored."""
    index = pd.DatetimeIndex(index)
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted")
    pos = index.searchsorted(pd.DatetimeIndex(days))
    mask = np.zeros(len(index), dtype=bool)
    mask[pos[pos < len(index)]] = True
    return mask


def event_days(r, mask, top=0.01):
    """How a set of days (the mask) differs from the rest: count, typical move size on
    those days relative to others, the share of the biggest `top` days that fall on them
    (against the share expected by chance), and kurtosis / Hill exponent with them left out."""
    r = pd.Series(r)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(r),):
        raise ValueError(f"need one True/False per day: {mask.shape} vs {len(r)}")
    if not mask.any() or mask.all():
        raise ValueError("the mask must mark some days but not all")
    a = np.abs(r.values - r.values.mean())
    big = a >= np.quantile(a, 1 - top)
    rest = r.values[~mask]
    return dict(days=int(mask.sum()), size_ratio=float(a[mask].mean() / a[~mask].mean()),
                big_share=float(mask[big].mean()), chance_share=float(mask.mean()),
                kurtosis_all=float(stats.kurtosis(r.values)), kurtosis_without=float(stats.kurtosis(rest)),
                alpha_all=float(stats.hill_alpha(r.values)), alpha_without=float(stats.hill_alpha(rest)))


def r2_by_regime(r, names, window=20, lags=1):
    """R^2 of the factor regression in calm, middle and wild markets: days are split into
    thirds by the S&P's volatility over the `window` days before (known in advance, so no
    look-ahead). Herding predicts a lower R^2 in wild markets, where traders' own dynamics
    add moves that no outside news explains. Returns {'calm': R^2, 'middle': ..., 'wild': ...}."""
    df = table(r, names)
    X = with_leads_lags(df, names, lags)
    vol = df["spx"].rolling(int(window)).std().shift(1).loc[X.index]
    keep = vol.notna()
    X, vol, y = X[keep], vol[keep], df["spx"].loc[X.index[keep]]
    third = pd.qcut(vol, 3, labels=["calm", "middle", "wild"])
    return {k: regress(y[third == k].values, X[third == k].values)[2] for k in ("calm", "middle", "wild")}
