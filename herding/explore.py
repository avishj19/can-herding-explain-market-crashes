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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import experiments, factors, market_data, model, politics, stats
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
# Symbols whose "price" is a measurement, not something traders buy and sell, so a model of traders
# can't be compared with them: they are turned away before anything is sent to Yahoo.
NOT_A_PRICE = {
    **dict.fromkeys(["^VIX", "^VIX9D", "^VIX3M", "^VIX6M", "^VVIX", "^VXN", "^VXD", "^RVX", "^OVX", "^GVZ",
                     "^EVZ", "^VXEEM", "^VXSLV", "^VXAPL", "^VXAZN", "^VXGOG", "^VXIBM"],
                    "a volatility index (a measure of how nervous traders are, not a price)"),
    **dict.fromkeys(["^TNX", "^IRX", "^FVX", "^TYX"], "an interest rate (a yield in percent, not a price)"),
}
YEAR_DAYS = 252          # trading days in a year: the 52-week window, the sidebar sparkline, annualizing
# Yahoo symbols of the saved markets that are not stocks (a stock's symbol is its own ticker)
YAHOO = {"S&P 500": "^GSPC", "Nasdaq-100": "^NDX", "Dow Jones": "^DJI", "Brent oil": "BZ=F",
         "Gold": "GC=F", "Copper": "HG=F"}
# saved markets whose file is a spot price (FRED) while Yahoo's symbol is a futures price: the two can
# differ (usually by 1-2%), so typing the futures symbol says the saved spot price is shown
SAVED_SPOT = {"Brent oil"}
# the company behind each saved stock ticker, so a visitor who doesn't know DIS reads "Disney"
COMPANY = {"AAPL": "Apple", "ALL": "Allstate", "CB": "Chubb", "CI": "Cigna", "CSCO": "Cisco",
           "CVX": "Chevron", "DIS": "Disney", "FDX": "FedEx", "HD": "Home Depot", "HON": "Honeywell",
           "ITW": "Illinois Tool Works", "LOW": "Lowe's", "MSFT": "Microsoft", "NKE": "Nike",
           "ORCL": "Oracle", "RTX": "RTX (Raytheon)", "SBUX": "Starbucks", "TRV": "Travelers",
           "TXN": "Texas Instruments", "UNP": "Union Pacific"}
# what the ticker messages suggest when a typed ticker can't be used
PICK_INSTEAD = "or pick a market above"


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


def _no_tz(c, name):
    """Checks that dates (if c is indexed by dates) carry no time zone. Every date here is plain
    (live_prices and the downloads strip Yahoo's time zones); comparing UTC dates with plain ones
    would end in a raw pandas TypeError, and only on some ranges (BR3-3)."""
    if isinstance(c.index, pd.DatetimeIndex) and c.index.tz is not None:
        raise ValueError(f"{name}: dates must not carry a time zone, got {c.index.tz}")
    return c


def _dated(c, name):
    """Checks that a price Series is indexed by dates without a time zone (needed to line two
    series up in time)."""
    if not isinstance(c.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must be a Series indexed by dates")
    return _no_tz(c, name)


# ------------------------------------------------------------------ assets

def catalog():
    """{group: [asset names]} for the market list, in display order."""
    return {"Stocks": market_data.stock_tickers(), "Indexes": list(INDEXES),
            "Commodities": list(COMMODITIES)}


def market_group(name):
    """The catalog group ('Indexes', 'Stocks' or 'Commodities') a saved market belongs to, or None
    for anything else (a ticker typed and fetched live)."""
    for group, names in catalog().items():
        if name in names:
            return group
    return None


def display_name(name):
    """How a market is named on screen: a saved stock with its company ("DIS · Disney", or the company
    alone when it already starts with the ticker: "RTX (Raytheon)"); anything else (an index, a commodity, a typed ticker) as it is."""
    if name not in COMPANY:
        return name
    return COMPANY[name] if COMPANY[name].startswith(name) else f"{name} · {COMPANY[name]}"   # not "RTX · RTX ..."


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
    at most 12 characters, and not a measurement like ^VIX: NOT_A_PRICE), so nothing else is ever
    sent to Yahoo."""
    if not isinstance(text, str):
        raise ValueError("type a ticker, like NVDA")
    t = text.strip().upper()
    if not t:
        raise ValueError("type a ticker first, like NVDA")
    if not TICKER.match(t):
        raise ValueError(f"\u201c{t}\u201d is not a ticker. Use only letters, digits and . - = ^ (up to 12), "
                         f"like NVDA or BTC-USD.")
    if t in NOT_A_PRICE:
        raise ValueError(f"{t} is {NOT_A_PRICE[t]}, so the model cannot be compared with it. "
                         f"Try a stock, a fund or an index of prices like ^GSPC (the S&P 500), {PICK_INSTEAD}.")
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


YAHOO_DOWN = f"Could not reach Yahoo Finance. Try again in a minute, {PICK_INSTEAD}."


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
        raise ConnectionError(YAHOO_DOWN) from e
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
        raise ValueError(f"Yahoo Finance has no prices for {t}. Check the spelling (e.g. BRK-B), "
                         f"{PICK_INSTEAD}.")
    if len(close) <= MIN_DAYS:
        raise ValueError(f"{t} has only {len(close):,} days of prices. The site needs more than "
                         f"{MIN_DAYS:,} days (about 4 years for a stock) to count rare big days reliably. "
                         f"Try an older company, {PICK_INSTEAD}.")
    return close.rename(t)


def live_market(text, start="2000-01-01", download=None):
    """(closes, daily log returns, summary) of any ticker from Yahoo, as the Workspace shows it.
    Raises what live_prices raises, and a ValueError worded for visitors when the price barely
    moves (a pegged or halted asset): the summary's own message ("need more than 13 nonzero
    returns, got 1") means nothing to someone who only typed a ticker (WS-25)."""
    return market_from_close(live_prices(text, start, download))


RECENT_DAYS = 10         # calendar days fetched for the fresh end of a live history (a long weekend fits)


def recent_prices(text, download=None, today=None):
    """The last few daily closes of a ticker from Yahoo (RECENT_DAYS calendar days), including today's
    price so far while the market is open. Asked for every minute by the site, so a live chart's end
    is never more than a minute behind Yahoo, while the long history (live_prices) is fetched once a
    day. Raises what live_prices raises, except that a short history is fine here."""
    t = clean_ticker(text)
    start = (pd.Timestamp(today) if today is not None else pd.Timestamp.now()).normalize() \
        - pd.Timedelta(days=RECENT_DAYS)
    download = download or _yahoo
    try:
        close = download(t, start.strftime("%Y-%m-%d"))
    except Exception as e:
        raise ConnectionError(YAHOO_DOWN) from e
    idx = getattr(close, "index", None)
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:    # as in live_prices (BE-18)
        close = pd.Series(close).copy()
        close.index = idx.tz_localize(None)
    close = _clean_download(close)
    if isinstance(close.index, pd.DatetimeIndex):
        close.index = close.index.normalize()
        close = close[~close.index.duplicated(keep="last")]
    if len(close) == 0:
        raise ValueError(f"Yahoo Finance has no recent prices for {t}")
    return close[close.index >= start].rename(t)


# FRED (the St. Louis Fed's public data, no key needed) as a backup source of daily closes when Yahoo
# doesn't answer. It has only these three indexes, and only for the last 10 years (S&P's licence).
FRED_SERIES = {"^GSPC": "SP500", "^DJI": "DJIA", "^NDX": "NASDAQ100"}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def _fred_csv(series):
    import requests                  # plain python User-Agent: FRED blocks browser-like ones (Nov 2026)
    resp = requests.get(FRED_URL.format(series), timeout=10)
    resp.raise_for_status()
    return resp.text


def fred_closes(symbol, download=None):
    """Daily closes of an index from FRED, for the Yahoo symbols in FRED_SERIES: a Series indexed by
    date (FRED's empty days, holidays, are left out). Raises ValueError for any other symbol or a reply
    that isn't FRED's CSV, ConnectionError if FRED can't be reached. download(series) -> CSV text is
    swappable so the tests need no internet."""
    if symbol not in FRED_SERIES:
        raise ValueError(f"FRED has no daily closes for {symbol!r}")
    series = FRED_SERIES[symbol]
    try:
        text = (download or _fred_csv)(series)
    except Exception as e:
        raise ConnectionError("Could not reach FRED") from e
    import io
    try:
        df = pd.read_csv(io.StringIO(text), na_values=["."])
        dates = pd.to_datetime(df.iloc[:, 0], format="%Y-%m-%d")
        close = pd.Series(pd.to_numeric(df[series], errors="coerce").to_numpy(), index=pd.DatetimeIndex(dates))
    except Exception as e:                  # a block page, an error page, a changed format
        raise ValueError(f"FRED's reply for {series} is not its usual table") from e
    close = _clean_download(close)
    if len(close) < 2:
        raise ValueError(f"FRED has no recent closes for {series}")
    return close.rename(symbol)


def _yahoo_quote(ticker):
    """(last price, previous close) from Yahoo's quote (yfinance fast_info): one small request,
    about 0.3 s, against 1-2 s for a download of daily closes."""
    import yfinance as yf
    fi = yf.Ticker(ticker).fast_info
    return fi.last_price, fi.previous_close


def live_quote(text, quote=None):
    """The latest price of a ticker from Yahoo's quote, for the live header: {"price", "prev_close",
    "change"} (change = price / previous close - 1). Raises ConnectionError if Yahoo can't be
    reached and ValueError if it gives no sensible price (missing, zero, negative or not finite).
    `quote(ticker)` is swappable so the tests need no internet (default: _yahoo_quote)."""
    t = clean_ticker(text)
    quote = quote or _yahoo_quote
    try:
        price, prev = quote(t)
    except Exception as e:
        raise ConnectionError(YAHOO_DOWN) from e
    try:
        price, prev = float(price), float(prev)
    except (TypeError, ValueError):
        raise ValueError(f"Yahoo Finance gave no quote for {t}") from None
    if not (np.isfinite(price) and np.isfinite(prev) and price > 0 and prev > 0):
        raise ValueError(f"Yahoo Finance gave no sensible quote for {t}: {price}, {prev}")
    return {"price": price, "prev_close": prev, "change": price / prev - 1}


def splice_recent(close, recent):
    """A long daily history with its end brought up to date: closes in `recent` replace the ones on
    the same dates (today's price so far moves during the day) and later dates are added. Dates in
    recent before the history's last RECENT_DAYS days are ignored, so a short fetch can't rewrite old
    data; and recent prices that don't reach back to the history's last close are not added at all
    (the days between would be missing, and their whole move counted as one day)."""
    close = _dated(close, "close")
    if recent is None or len(recent) == 0:
        return close
    recent = _dated(recent, "recent")
    if recent.index[0] > close.index[-1]:
        return close
    recent = recent[recent.index > close.index[-1] - pd.Timedelta(days=RECENT_DAYS)]
    out = pd.concat([close[~close.index.isin(recent.index)], recent]).sort_index()
    return out[~out.index.duplicated(keep="last")].rename(close.name)


def market_from_close(close):
    """(closes, daily log returns, summary) from a cleaned history of daily closes, with a ValueError
    worded for visitors when the price barely moves (see live_market)."""
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


def run_mix(p, mix, a1, days, seed, burn, party=0.0, paths=None):
    """One model history for a trader mix: (returns, None), with the first `burn` days
    dropped so the starting state does not matter, or (None, the RunawayError) if the
    price left its value for good. (With the S&P 500 fit and a1 = 0, measured Nov 2026:
    9 of seeds 0-9 run away as a bubble with 99% chartists.)
    party: an optional party drift (party_shocks), 0 = the model of SPEC.md, unchanged.
    paths: an optional dict, filled with this history's day-by-day market (market_paths) when it did
    not run away, so the Simulated market chart needn't run the same history a second time."""
    if isinstance(days, bool) or not isinstance(days, (int, np.integer)) or not 500 <= days <= MAX_DAYS:
        raise ValueError(f"days must be a whole number from 500 to {MAX_DAYS}, got {days!r}")
    if isinstance(burn, bool) or not isinstance(burn, (int, np.integer)) or burn < 0:
        raise ValueError(f"burn must be a whole number >= 0, got {burn!r}")
    seed = _whole(seed, "seed", 0)        # numpy would turn -1, 1.5 or "1" into a confusing error
    q, chartists = mix_settings(p, mix, a1)
    shocks = party_shocks(int(days), int(burn), party)
    try:
        out = model.simulate(q, int(days) + int(burn), seed=seed, chartists=chartists, shocks=shocks)
    except model.RunawayError as e:
        return None, e
    if paths is not None:
        paths.update(market_paths(out, int(burn)))
    return out["r"][int(burn):], None


def market_paths(out, burn, ran=None):
    """model.simulate's output as model_market's dict: price and value as an index starting at 100 on
    day 0 (after the burn-in), the chartist share z and mood x, the daily returns r, and runaway_day."""
    lp, lf = out["lp"][burn:], out["lf"][burn:]
    return {"price": 100 * np.exp(lp - lp[0]), "value": 100 * np.exp(lf - lp[0]),
            "z": out["z"][burn:], "x": out["x"][burn:], "r": np.diff(lp), "runaway_day": ran}


def runaway_day(err, burn):
    """The day a history ran away, counted after the burn-in (the days the site shows)."""
    return int(err.day - burn)


def mix_key(mix, a1, a2, v1):
    """The slider settings one trader mix's history actually depends on: none for fundamentalists only
    (no chartists, so a1, a2 and v1 are never used), (a2, v1) for chartists without herding (their a1
    is 0 whatever the slider says), (a1, a2, v1) for chartists who herd. Two runs with the same key,
    seed, days and party drift give the same history for that mix, so it can be reused."""
    if mix not in MIXES:
        raise ValueError(f"unknown trader mix {mix!r}; choose from {MIXES}")
    return {"fundamentalists": (), "chartists": (float(a2), float(v1)),
            "herding": (float(a1), float(a2), float(v1))}[mix]


def run_mixes(p, burn, a1, a2, v1, days, seed, party=0.0, known=None, paths=None):
    """The three trader mixes on one seed, with a2 and v1 replaced: {mix: (returns or None,
    the runaway day after burn-in or None)}. party: the party drift (party_shocks), 0 = none.
    known: {mix: result} already computed for these settings (see mix_key), which is reused instead
    of run again: moving only the herding slider then runs one history, not three.
    paths: an optional dict, filled with {mix: market_paths} for the mixes run here that didn't run away."""
    known = known or {}
    q = dict(p, a2=a2, v1=v1)
    # the three mixes are independent histories, each with its own random generator: run them at
    # the same time (the model's loop releases Python's lock), which gives the same numbers as one
    # after another in about a third of the wait (0.8 s -> 0.3 s for an S&P-length run, Nov 2026)
    with ThreadPoolExecutor(max_workers=len(MIXES)) as pool:
        got = {m: {} for m in MIXES if m not in known}
        jobs = {m: pool.submit(run_mix, q, m, a1, days, seed, burn, party, got[m]) for m in got}
    out = {}
    for m in MIXES:
        if m in known:
            out[m] = known[m]
            continue
        r, err = jobs[m].result()          # a bad input raises here, as before
        out[m] = (r, None if err is None else runaway_day(err, burn))
        if paths is not None and got[m]:
            paths[m] = got[m]
    return out


# How close a model number must be to the market's to count as a match / close (scorecard):
# the tail exponent within 10% / 25% of the market's; big days (beyond 4 sd) per 1,000 days within a
# factor 1.5 / 2.5; the volatility memory within 0.05 / 0.12. Chosen to be about the spread between
# seeds of the S&P fit (figures/RESULTS.md, Fig 4), so "match" means "as close as luck allows".
SCORE_RULES = {"tail": (0.10, 0.25), "big_days": (1.5, 2.5), "memory": (0.05, 0.12)}


def _verdict(gap, rule):
    match, close = rule
    return "match" if gap <= match else "close" if gap <= close else "off"


def scorecard(market, model):
    """Three plain checks of one model history against a market, each a dict with what (the name),
    market and model (the numbers compared) and verdict ("match", "close" or "off"):
    fat tails (tail exponent), big days (days beyond 4 sd per 1,000 days) and bursts of wild days
    (volatility memory). market and model are summary() dicts; model None (it ran away) gives []."""
    if model is None:
        return []
    out = []
    t = abs(model["tail"] - market["tail"]) / abs(market["tail"])
    out.append(dict(what="Fat tails", market=market["tail"], model=model["tail"],
                    verdict=_verdict(t, SCORE_RULES["tail"])))
    per = [1000 * x["big_days"] / x["days"] for x in (market, model)]
    if per[0] == 0 or per[1] == 0:
        ratio = 1.0 if per[0] == per[1] else np.inf
    else:
        ratio = max(per) / min(per)
    out.append(dict(what="Big days (per 1,000 days)", market=per[0], model=per[1],
                    verdict=_verdict(ratio, SCORE_RULES["big_days"])))
    out.append(dict(what="Bursts of wild days", market=market["memory"], model=model["memory"],
                    verdict=_verdict(abs(model["memory"] - market["memory"]), SCORE_RULES["memory"])))
    return out


def mix_summaries(mixes):
    """{mix: summary of its returns, or None if it ran away} for run_mixes' output."""
    return {m: (summary(r) if r is not None else None) for m, (r, _) in mixes.items()}


def mix_tails(mixes):
    """{mix: tail exponent, or None if it ran away}: what the Workspace's run log keeps."""
    return {m: (s["tail"] if s is not None else None) for m, s in mix_summaries(mixes).items()}


def differs_from_fit(run, p, keys=("a2", "v1"), tol=1e-9):
    """True if a run's settings other than the herding strength differ from the fit p, so the run
    can't be read against the saved sweep (which changed only a1). A party drift counts too: the
    saved sweep has none."""
    return any(abs(float(run[k]) - float(p[k])) > tol for k in keys) or abs(float(run.get("party", 0.0))) > tol


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
    label = {"fundamentalists": "value-traders-only", "chartists": "trend-chasers-without-herding",
             "herding": "herding"}[closest]
    count = "no days" if n == 0 else ("1 day" if n == 1 else f"{n} days")
    times = n / bell if bell > 0 else None        # no bell-curve days expected: no ratio to give
    return (f"{name} had {count} bigger than {BIG:g} sd; "
            f"a bell curve expects about {bell:.1f}"
            + (f", so about {times:.0f} times more" if times is not None and times >= 2 else "")
            + f". Its tails and bursts of wild days are closest to the {label} mix.")


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


TAIL_RANGE = (2.5, 3.5)    # check 3's pass range for the median tail exponent ("about 3")


def tail_check(tails, lo=TAIL_RANGE[0], hi=TAIL_RANGE[1]):
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


# ------------------------------------------------------------------ how long between crashes
# The proposal's question: a bell curve says a Black Monday (about 20 standard deviations) should
# come once in 10^86 years; real markets have one within decades. A "crash" here is a daily FALL of at
# least `size` standard deviations (each history in its own sd, like summary's "worst").
CRASH_SIZES = (3, 4, 5, 6, 7, 8, 10, 12, 15, 20)
DECADES = 50             # "a few decades": the longest wait (years) the crash check accepts
CRASH_CHECK_SIZE = 8     # the crash check's size: a bell curve puts these 6 trillion years apart
BIG_FALL = 8             # numbers.json keeps every model fall at least this big (in sd), so any bigger
                         # size (like Black Monday's) can be counted exactly


def big_falls(r, size=BIG_FALL):
    """The sizes (in r's own standard deviations, as positive numbers) of every daily fall of at least `size`."""
    z = moves_in_sd(np.asarray(r, dtype=float))
    return sorted((float(-v) for v in z[z <= -size]), reverse=True)


def bell_crash_years(size, year_days=YEAR_DAYS):
    """Years a bell curve expects between daily falls of at least `size` standard deviations:
    1 / (P(Z < -size) * days per year), with P(Z < -size) = erfc(size / sqrt 2) / 2.
    20 sd gives about 1.4e86 years, the proposal's 10^86."""
    if not is_number(size) or size <= 0:
        raise ValueError(f"size must be a positive number of standard deviations, got {size!r}")
    p = math.erfc(size / math.sqrt(2)) / 2
    return math.inf if p == 0 else 1 / (p * year_days)


def crash_counts(r, sizes=CRASH_SIZES):
    """How many daily falls of r reach each size (in r's own standard deviations): a list of ints."""
    z = moves_in_sd(np.asarray(r, dtype=float))
    return [int(np.sum(z <= -s)) for s in sizes]


def crash_years(counts, days, year_days=YEAR_DAYS):
    """Years between crashes when `counts` crashes (one number, or one per history) happened in `days`
    trading days (the total over all histories): days / year_days / total count; inf when there were none."""
    total = int(np.sum(counts))
    if isinstance(days, bool) or not isinstance(days, (int, np.integer)) or days < 1:
        raise ValueError(f"days must be a whole number >= 1, got {days!r}")
    if total < 0:
        raise ValueError("counts can't be negative")
    return math.inf if total == 0 else days / year_days / total


def crash_table(nums, spx_r):
    """One row per crash size: years between crashes for a bell curve, the S&P 500 (spx_r, its daily
    returns) and the herding model (all saved held-out histories pooled), and how many of those histories
    had at least one. Rows: size, bell, sp500, model, sp500_count, model_count, histories, of."""
    cr = nums["crashes"]
    sizes, per = cr["sizes"], np.asarray(cr["falls"], dtype=int)      # per: histories x sizes
    spx = crash_counts(spx_r, sizes)
    rows = []
    for i, s in enumerate(sizes):
        rows.append({"size": s, "bell": bell_crash_years(s), "sp500": crash_years(spx[i], len(spx_r)),
                     "model": crash_years(per[:, i], int(sum(cr["days"]))),
                     "sp500_count": spx[i], "model_count": int(per[:, i].sum()),
                     "histories": int(np.sum(per[:, i] > 0)), "of": int(per.shape[0])})
    return rows


def crash_check(nums, spx_r, size=CRASH_CHECK_SIZE, limit=DECADES):
    """The proposal's crash check, in two parts:
    decades:      falls of `size` sd, which a bell curve puts trillions of years apart, come in the
                  herding model at least once every `limit` years ("within a few decades").
    black_monday: a fall as big as the S&P 500's worst day (Black Monday) also comes within `limit`
                  years. The result says which parts passed; passed is True only if both did."""
    rows = {r["size"]: r for r in crash_table(nums, spx_r)}
    if size not in rows:
        raise ValueError(f"size must be one of the saved sizes {sorted(rows)}, got {size!r}")
    z = moves_in_sd(np.asarray(spx_r, dtype=float))
    worst = float(-z.min())
    cr = nums["crashes"]
    pooled = int(sum(cr["days"]))
    bm_count = sum(int(np.sum(np.asarray(f, dtype=float) >= worst)) for f in cr["big_falls"])
    bm_model = math.inf if bm_count == 0 else pooled / YEAR_DAYS / bm_count
    out = {"size": size, "limit": limit, "row": rows[size], "worst_sd": worst,
           "bm_bell": bell_crash_years(worst), "bm_model": bm_model, "bm_count": bm_count,
           "decades": rows[size]["model"] <= limit, "black_monday": bm_model <= limit}
    out["passed"] = out["decades"] and out["black_monday"]
    return out


# ------------------------------------------------------------------ the textbook checklist
# Cont (2001), "Empirical properties of asset returns: stylized facts and statistical issues",
# Quantitative Finance 1:223-236: the patterns found in nearly every real market. Each rule's line is set
# where a plain bell curve can't reach it (a bell curve passes only "no trend" and "tails thin out").
# key: (name, what it means, rule in words, test)
FACTS = {
    "no_trend": ("No predictable trend", "Yesterday's move does not tell you today's direction.",
                 "|correlation of a day's move with the next day's| < 0.05", lambda v: abs(v) < 0.05),
    "fat_tails": ("Fat tails", "Huge days happen far more often than a bell curve allows.",
                  "excess kurtosis > 1 (bell curve: 0)", lambda v: v > 1),
    "skew": ("Falls bigger than rises", "The biggest falls are bigger than the biggest rises.",
             "1% worst day / 1% best day > 1.05 (mirror images: 1.00)", lambda v: v > 1.05),
    "bunching": ("Bursts of wild days", "A wild day is usually followed by another wild day.",
                 "correlation of move sizes 1 day apart > 0.1", lambda v: v > 0.1),
    "long_memory": ("Long memory of wild spells", "Wild spells are still felt two months later.",
                    "correlation of move sizes 50 days apart > 0.05", lambda v: v > 0.05),
    "thin_out": ("Tails thin out over a month", "Monthly moves look more like a bell curve than daily ones.",
                 "kurtosis of 20-day moves < kurtosis of daily moves", lambda v: v < 0),
    "leverage": ("Wilder after falls", "Falls are followed by bigger moves than rises are.",
                 "correlation of a day's move with the next 10 days' sizes < -0.02", lambda v: v < -0.02),
    "fat_in_calm": ("Fat tails even in calm spells", "Surprises stay big even after allowing for how wild "
                    "the last month was.", "kurtosis of moves in units of the last 20 days' sd > 1 "
                    "(bell curve: about 0.4 from noise)", lambda v: v > 1),
}
FACT_MONTH = 20          # trading days in a month


def fact_values(r):
    """Each FACTS pattern measured on daily returns r: {key: number} (the test in FACTS turns it into
    yes or no). "thin_out" is the 20-day kurtosis minus the daily one (below 0: tails thin out)."""
    r = np.asarray(r, dtype=float)
    if r.ndim != 1 or r.size < 500 or not np.all(np.isfinite(r)):
        raise ValueError("need a 1-D list of at least 500 finite returns")
    e = r - r.mean()
    size = np.abs(e)
    z = stats.standardize(r)
    lo, hi = np.quantile(z, [0.01, 0.99])
    month = r[: r.size // FACT_MONTH * FACT_MONTH].reshape(-1, FACT_MONTH).sum(axis=1)
    sd_before = np.sqrt(np.convolve(e ** 2, np.ones(FACT_MONTH) / FACT_MONTH, "valid"))[:-1]
    return {
        "no_trend": float(stats.acf(r, 1)[0]),
        "fat_tails": float(stats.kurtosis(r)),
        "skew": float(-lo / hi),
        "bunching": float(stats.acf(size, 1)[0]),
        "long_memory": float(stats.acf(size, 50)[-1]),
        "thin_out": float(stats.kurtosis(month) - stats.kurtosis(r)),
        "leverage": float(np.mean([np.corrcoef(r[:-k], size[k:])[0, 1] for k in range(1, 11)])),
        "fat_in_calm": float(stats.kurtosis(e[FACT_MONTH:] / sd_before)),
    }


def fact_passes(values):
    """{key: True/False} for fact_values' numbers (FACTS' tests)."""
    return {k: bool(FACTS[k][3](v)) for k, v in values.items()}


def fact_table(nums, spx_r):
    """One row per FACTS pattern: the S&P 500's value and whether it passes; for the herding model and
    the GARCH formula (numbers.json "facts"), the median and how many histories pass."""
    spx = fact_values(spx_r)
    rows = []
    for k, (name, meaning, rule, test) in FACTS.items():
        row = {"key": k, "name": name, "meaning": meaning, "rule": rule, "sp500": spx[k], "sp500_pass": test(spx[k])}
        for who in ("herding", "garch"):
            vals = nums["facts"][who][k]
            row[who] = float(np.median(vals))
            row[who + "_pass"] = int(sum(bool(test(v)) for v in vals))
            row["of"] = len(vals)
        rows.append(row)
    return rows


# The Checks page's rule for Kirman's ants: with 20 million steps the simulated and exact long-run
# shares differ by a total variation distance of 0.0024-0.0234 (measured Nov 2026, every slider value
# of eps, seeds 0-3), so 0.05 is about twice the worst seen and a real bug would still be caught.
KIRMAN_PASS = 0.05


def kirman_passes(tv, limit=KIRMAN_PASS):
    """Kirman check: True when the simulation's total variation distance from the exact answer
    (kirman_match's third value) is below `limit`."""
    if not is_number(tv) or not 0 <= tv <= 1:
        raise ValueError(f"tv must be a distance between 0 and 1, got {tv!r}")
    if not is_number(limit) or not 0 < limit <= 1:
        raise ValueError(f"limit must be between 0 and 1, got {limit!r}")
    return bool(tv < limit)


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
    if not np.all(np.isfinite(v)):            # infinity would give NaN percentiles (BR3-5)
        raise ValueError("values must be finite (no infinity)")
    if not 0 <= lo <= hi <= 100:
        raise ValueError(f"need 0 <= lo <= hi <= 100, got {lo}, {hi}")
    a, b = np.percentile(v, [lo, hi])
    return {"median": float(np.median(v)), "low": float(a), "high": float(b), "n": int(v.size)}


# Lux & Marchesi (1999), Nature 397, 498-500: the numbers their paper reports for its model (Figs 2-3),
# with the paper's own error bars. (calibrate.PAPER_TARGETS has the same values with this project's
# wider tolerances, which decide "close".)
PAPER_PUBLISHED = {"alpha_reg": (2.64, 0.077), "H": (0.48, 0.003), "H_abs": (0.85, 0.010)}
PAPER_MEASURES = {"alpha_reg": "Tail exponent (paper's method: largest 30% of days)",
                  "H": "Hurst H of direction (0.5 = no memory)",
                  "H_abs": "Hurst H of move size (above 0.5 = bursts of wild days)"}


def paper_comparison(nums):
    """Our paper-matched model next to the numbers Lux & Marchesi published, as rows
    {measure, paper, paper_err, ours, ours_sd, n, tolerance, close, inside_err}: ours is the mean over the held-out
    histories saved in nums["paper"] (scripts/web_numbers.py), ours_sd their spread, and close
    whether the gap is within this project's tolerance (calibrate.PAPER_TARGETS), inside_err whether
    it is within the paper's own error bar."""
    from .calibrate import PAPER_TARGETS
    rows = []
    for key, label in PAPER_MEASURES.items():
        v = np.asarray([x for x in nums["paper"][key] if x is not None], dtype=float)
        if v.size == 0:
            raise ValueError(f"no saved paper-matched values for {key}")
        paper, err = PAPER_PUBLISHED[key]
        ours = float(v.mean())
        rows.append({"measure": label, "key": key, "paper": paper, "paper_err": err, "ours": ours,
                     "ours_sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size),
                     "tolerance": float(PAPER_TARGETS[key][1]),
                     "close": bool(abs(ours - paper) <= PAPER_TARGETS[key][1]),
                     "inside_err": bool(abs(ours - paper) <= err)})
    return rows


def paper_tail(r):
    """A market's tail exponent by the paper's method: the log-log regression over its largest 30% of
    daily moves (stats.alpha_regression), as Lux & Marchesi measured it. summary()["tail"] is a
    different estimate (Hill, largest 5%), so the two are not interchangeable."""
    return float(stats.alpha_regression(np.asarray(r, dtype=float)))


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
    seed = _whole(seed, "seed", 0)        # numpy would turn -1, 1.5 or "1" into a confusing error
    q, chartists = mix_settings(p, mix, a1)
    out = model.simulate(q, int(day), seed=seed, chartists=chartists, max_gap=None)
    return {k: out[k] for k in ("lp", "lf", "z")}


def model_market(p, burn, mix, a1, a2, v1, days, seed, party=0.0):
    """The simulated market behind one trader mix, day by day after the burn-in, for the Test
    program's "Model market" chart: the same history as run_mixes gives for that mix (same seed,
    settings and party drift). Returns a dict of numpy arrays, one value per day's close:
        price   the price, as an index that starts at 100 on day 0
        value   the fundamental value on the same scale (the price it is pulled back to)
        z       share of traders who are chartists (0-1)
        x       chartists' mood, (optimists - pessimists) / chartists (-1 to 1)
        r       the daily log returns (one fewer value), exactly run_mixes' returns for this mix
    and runaway_day: None, or the day (after burn-in) the history ran away. A history that ran away
    is shown up to that day, rebuilt with the runaway guard off (it never changes the random draws,
    so those days are the same); one that ran away in the warm-up has nothing to show (ValueError)."""
    q, chartists = mix_settings(dict(p, a2=a2, v1=v1), mix, a1)
    days, burn = _whole(days, "days", 1), _whole(burn, "burn", 0)
    seed = _whole(seed, "seed", 0)
    shocks = party_shocks(days, burn, party)
    ran = None
    try:
        out = model.simulate(q, days + burn, seed=seed, chartists=chartists, shocks=shocks)
    except model.RunawayError as e:
        ran = runaway_day(e, burn)
        if ran < 1:
            raise ValueError("this history ran away in the warm-up, before day 0") from e
        out = model.simulate(q, int(e.day), seed=seed, chartists=chartists, max_gap=None,
                             shocks={d: f for d, f in (shocks or {}).items() if d < e.day})
    return market_paths(out, burn, ran)


ZOOM_DAYS = 250          # a year of trading days: the window the Simulated market chart zooms to


def crowd_counts(z, x, n=500):
    """How many of n traders are value traders, optimists and pessimists on each day, for the crowd
    picture on the Start page. z: share of traders who are trend chasers (0-1); x: their mood,
    (optimists - pessimists) / trend chasers (-1 to 1), as model_market returns them. Optimists are a
    share z (1 + x) / 2 of all traders and pessimists z (1 - x) / 2. Returns an int array, one row
    (value traders, optimists, pessimists) per day, each row adding up to exactly n."""
    z, x = np.asarray(z, dtype=float), np.asarray(x, dtype=float)
    if z.ndim != 1 or z.shape != x.shape or z.size == 0:
        raise ValueError("z and x must be 1-D lists of the same, non-zero length")
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(x))):
        raise ValueError("z and x must be finite")
    if np.any((z < 0) | (z > 1)) or np.any((x < -1) | (x > 1)):
        raise ValueError("z must be within 0-1 and x within -1 to 1")
    n = _whole(n, "n", 1)
    opt = np.rint(z * (1 + x) / 2 * n).astype(int)
    pes = np.minimum(np.rint(z * (1 - x) / 2 * n).astype(int), n - opt)     # rounding never makes more than n
    return np.column_stack([n - opt - pes, opt, pes])


def zoom_window(r, kind, width=ZOOM_DAYS):
    """The days (first, last) of the Simulated market chart around something worth watching in the daily
    returns r (move i ends on day i + 1):
        "crash"  the biggest one-day fall, in the middle of the window
        "burst"  the window of `width` moves with the largest total move size (wild days bunched up)
    The window is cut to the history (0 .. len(r)). r shorter than width: the whole history."""
    r = np.asarray(r, dtype=float)
    if r.ndim != 1 or r.size == 0 or not np.all(np.isfinite(r)):
        raise ValueError("need a 1-D list of finite returns")
    width = _whole(width, "width", 2)
    if kind not in ("crash", "burst"):
        raise ValueError(f"kind must be 'crash' or 'burst', got {kind!r}")
    n = r.size
    if n <= width:
        return 0, n
    if kind == "crash":
        end_day = int(np.argmin(r)) + 1
        first = min(max(end_day - width // 2, 0), n - width)
    else:                                     # "burst"
        sums = np.convolve(np.abs(r - r.mean()), np.ones(width), mode="valid")   # move sizes in each window
        first = int(np.argmax(sums))
    return first, first + width


def news_and_moves(mm):
    """The proposal's question in two numbers for one simulated market (model_market's dict): the excess
    kurtosis of the news (daily changes of the true value: bell-curve noise, so about 0) and of the price's
    daily moves; plus the average and peak share of chartists."""
    news = np.diff(np.log(mm["value"]))
    return {"news_kurtosis": float(stats.kurtosis(news)), "price_kurtosis": float(stats.kurtosis(mm["r"])),
            "chartists_mean": float(np.mean(mm["z"])), "chartists_peak": float(np.max(mm["z"]))}


GAP_CAP = 0.5            # the Story chart shows mispricing up to e^0.5 - 1 = +65%; beyond is "off the chart"


def runaway_story(p, seed, day, burn, mix="chartists", a1=0.0):
    """What the Story page tells about a history that ran away on `day` (counted after burn-in):
    z, the share of chartists each day; gap, how far the price sits above (+) or below (-) its
    value, as a fraction, capped at +-(e^GAP_CAP - 1); both from day 0 to `day` with the burn-in
    (warm-up) days left out, so day 0 here is day 0 in the text and the Workspace; gap_cap, that cap;
    share_before, the share of chartists 5 days before the end; calm_days, about how long the
    price stayed near its value (all but the last 3 days)."""
    day = _whole(day, "day", 1)
    burn = _whole(burn, "burn", 0)
    path = runaway_path(p, mix, a1, seed, day + burn)
    z = path["z"][burn:]
    gap = np.expm1(np.clip(path["lp"] - path["lf"], -GAP_CAP, GAP_CAP))[burn:]
    return {"z": z, "gap": gap, "gap_cap": float(np.expm1(GAP_CAP)),
            "share_before": float(z[max(len(z) - 6, 0)]),
            "calm_days": max(day - 3, 0)}       # a runaway on day 1-3 had no calm stretch (not -2 days)


def example_returns(n=20000, seed=3):
    """Made-up daily moves for the Start here page's 'how to read the tail chart' picture:
    (bell, fat), n draws from a bell curve and from Student's t with 3 degrees of freedom, whose
    tail exponent is 3 like real markets. The same seed always gives the same picture."""
    n, seed = _whole(n, "n", 2), _whole(seed, "seed", 0)
    g = np.random.default_rng(seed)
    return g.standard_normal(n), g.standard_t(3, n)


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


def sweep_at(nums, a1, tol=1e-9):
    """What the saved herding-strength sweep (numbers.json "sweep") says at one herding strength a1:
    {"a1", "n": histories, "ran_away": how many ran away, "tail": median tail exponent of the others
    (None if all ran away)}. ValueError if a1 is not one of the sweep's strengths."""
    rows = [r for r in nums["sweep"]["rows"] if abs(r["a1"] - a1) < tol]
    if not rows:
        raise ValueError(f"a1 = {a1!r} is not in the saved sweep")
    tails = [r["tail"] for r in rows if r["tail"] is not None]
    return {"a1": float(a1), "n": len(rows), "ran_away": len(rows) - len(tails),
            "tail": float(np.median(tails)) if tails else None}


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
        raise ConnectionError(YAHOO_DOWN) from e
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
    c = _no_tz(_prices(close, "daily closes"), "daily closes")
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



def moves_in_range(r, prices):
    """The daily moves a price range shows: those into its 2nd..last price (n prices have n - 1
    moves). r is indexed by the later day of each move, so the move into the range's first price
    happened before the range (YTD's would be 31 December of last year) and is left out."""
    r = _no_tz(pd.Series(r, dtype=float), "moves")
    p = _no_tz(_prices(prices), "prices")
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
    date (a market open every day has 364 closes in them, a stock about 251), and the range's own
    prices inside them count too (an intraday high above every close), and the volatility is
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
            start = d.index[-1] - pd.Timedelta(weeks=52)
            # the range's own prices inside those 52 weeks count too: an intraday high above every
            # daily close must not read "High 104" next to "52-week high 100" (BR3-1)
            year, per_year = pd.concat([d[d.index > start], c[c.index > start]]), days_per_year(d)
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
    need = {"checks", "sweep", "recipe", "paper"}
    if not need <= set(d):
        raise ValueError(f"{path} is missing {sorted(need - set(d))}; rerun scripts/web_numbers.py")
    return d


# ------------------------------------------------------------------ the president's party (extension)
# politics.py and Figs 12-15 found: US stocks returned more under Democratic presidents, but the gap
# is about what luck gives (the real returns with the party calendar slid along), and crashes,
# volatility and fat tails show no party difference. The site shows that test for any market, and
# can add an optional party drift to the model's value (off by default; the paper has none).
PARTY_CALENDAR = "Dow Jones"     # the longest party calendar we have (trading days since 1896)
PARTY_MAX = 0.20                 # largest party drift the site allows: 20 points a year


def party_view(r):
    """The president's-party test for one market's daily log returns r (a Series indexed by date):
    {gap_year (D minus R, log return a year: 0.05 = 5 points), dem_year, rep_year, vol_ratio (D / R),
    kurtosis (D minus R), crashes (crash days per 1,000, D minus R), luck {measure: percentile among
    the shifted calendars, 0-100}, calendars (how many), changes (party changes), dem_days,
    rep_days, start, end}. Days before the first president in politics.PRESIDENTS are left out.
    Raises ValueError when there are fewer than 500 days under either party."""
    if not isinstance(r, pd.Series) or not isinstance(r.index, pd.DatetimeIndex):
        raise ValueError("need daily returns as a Series indexed by dates")
    r = r[r.index >= pd.Timestamp(politics.PRESIDENTS[0][0])].dropna()
    dem = politics.democrat(r.index)
    if dem.sum() < 500 or (~dem).sum() < 500:
        raise ValueError(f"needs at least 500 days under each party (has {int(dem.sum()):,} under "
                         f"Democrats and {int((~dem).sum()):,} under Republicans)")
    v = r.values
    real = politics.party_gap(v, dem)
    shifted = politics.shifted_gaps(v, dem)
    luck = {k: experiments.percentile(real[k], [g[k] for g in shifted]) for k in politics.MEASURES}
    return dict(gap_year=politics.return_gap(v, dem),
                dem_year=float(v[dem].mean() * politics.DAYS_PER_YEAR),
                rep_year=float(v[~dem].mean() * politics.DAYS_PER_YEAR),
                vol_ratio=real["vol"], kurtosis=real["kurtosis"], crashes=real["crashes"],
                luck=luck, calendars=len(shifted), changes=len(politics.switches(r.index)[0]) - 1,
                dem_days=int(dem.sum()), rep_days=int((~dem).sum()), start=r.index[0], end=r.index[-1])


def party_terms(dates):
    """Each presidency within these dates (sorted): [{name, party, start, end}], for shading a chart.
    Dates before the first president in the table are skipped."""
    dates = pd.DatetimeIndex(dates)
    dates = dates[dates >= pd.Timestamp(politics.PRESIDENTS[0][0])]
    if len(dates) == 0:
        return []
    return [dict(name=t["name"], party=t["party"], start=t["start"], end=t["end"])
            for t in politics.terms(dates, min_days=1)]


def party_calendar(days):
    """Which of `days` model days had a Democratic president: the model's days are laid on the last
    `days` real trading days of the Dow Jones (PARTY_CALENDAR), as politics.compare does with the
    model. Returns (bool array, those dates)."""
    days = _whole(days, "days", 1)
    dates = saved_prices(PARTY_CALENDAR).index[1:]       # the dates of its daily returns
    if days > len(dates):
        raise ValueError(f"the party calendar has {len(dates):,} trading days; asked for {days:,}")
    dates = dates[-days:]
    return politics.democrat(dates), dates


def party_shocks(days, burn, party):
    """The party drift as model.simulate shocks: on each model day after the burn-in, the true value
    moves by +party/2 a year (spread over the trading days) under a Democratic president and by
    -party/2 under a Republican one, so the value grows `party` a year faster under Democrats (0.05 =
    5 points). The burn-in days get none. party = 0 gives None: the model exactly as in SPEC.md."""
    if not is_number(party) or abs(party) > PARTY_MAX:
        raise ValueError(f"party drift must be a number from -{PARTY_MAX} to {PARTY_MAX} "
                         f"(a fraction a year), got {party!r}")
    if party == 0:
        return None
    dem, _ = party_calendar(days)
    step = party / 2 / politics.DAYS_PER_YEAR
    return {int(burn) + i: math.expm1(step if d else -step) for i, d in enumerate(dem)}


def model_party_gap(r):
    """The yearly D-minus-R return gap of one model history r (after burn-in), cut at the same
    calendar as party_shocks: what a party drift (or luck alone, without one) gives in the model."""
    r = np.asarray(r, dtype=float)
    dem, _ = party_calendar(len(r))
    return politics.return_gap(r, dem)


def party_gap_year(r):
    """Just the yearly D-minus-R return gap of returns r (a Series indexed by date), as party_view's
    gap_year, without the luck test (fast enough for a help text). Days before the first president in
    politics.PRESIDENTS are left out."""
    if not isinstance(r, pd.Series) or not isinstance(r.index, pd.DatetimeIndex):
        raise ValueError("need daily returns as a Series indexed by dates")
    r = r[r.index >= pd.Timestamp(politics.PRESIDENTS[0][0])].dropna()
    return politics.return_gap(r.values, politics.democrat(r.index))


def party_gap_without(r, name):
    """The yearly D-minus-R return gap of returns r (a Series indexed by date) with one presidency's
    days left out (name as in politics.PRESIDENTS, e.g. "Hoover"; every term with that name)."""
    if not isinstance(r, pd.Series) or not isinstance(r.index, pd.DatetimeIndex):
        raise ValueError("need daily returns as a Series indexed by dates")
    r = r[r.index >= pd.Timestamp(politics.PRESIDENTS[0][0])].dropna()
    spans = [t for t in politics.terms(r.index, min_days=1) if t["name"] == name]
    if not spans:
        raise ValueError(f"no presidency named {name!r} in these dates")
    keep = np.ones(len(r), dtype=bool)
    for t in spans:
        keep[t["a"]:t["b"]] = False
    return politics.return_gap(r.values[keep], politics.democrat(r.index[keep]))


# ------------------------------------------------------------------ big days with a known cause

DAY_EVENTS = market_data.DATA / "big_days_events.csv"


def day_events(path=DAY_EVENTS):
    """The Dow's 20 biggest days with their documented causes (data/big_days_events.csv): a DataFrame
    indexed by date with columns move_percent, event, kind and sources (a list of URLs)."""
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    # an event without a source has an empty cell (read as NaN): no sources, not the text "nan"
    df["sources"] = [[u.strip() for u in str(s).split("|") if u.strip()]
                     for s in df.pop("source_url").fillna("")]
    return df


def day_story(day, move, sd, events=None):
    """What the page says about one big day a visitor clicked: the move, and the documented cause
    when that day is one of the Dow's 20 biggest (events: day_events()). Returns (text, sources)."""
    day = pd.Timestamp(day).normalize()
    text = (f"{day:%d %B %Y}: {'rose' if move > 0 else 'fell'} {abs(move):.1%}, "
            f"{abs(sd):.1f} standard deviations.")
    events = day_events() if events is None else events
    if day in events.index:
        e = events.loc[day]
        return text + f" What happened: {e['event']}", list(e["sources"])
    return text + " This day is not among the Dow's 20 biggest days, whose causes the project documents.", []


# ------------------------------------------------------------------ shareable links

LINK_LIMITS = {"a1": (0.0, 2.0), "a2": (0.0, 1.0), "v1": (0.5, 5.0), "party": (-10.0, 10.0)}
LINK_SEEDS = (0, 10_000)
LINK_KEYS = ("market", "a1", "a2", "v1", "party", "days", "seed")


def read_link(params, day_choices, markets):
    """The settings a shared link asks for, checked: {name: value} with only the keys that are present
    and valid, the rest left out (the page keeps its own value). Never raises, whatever a link holds:
    numbers must be finite and inside the sliders' ranges (LINK_LIMITS; kept exactly as written, so a
    link to the fitted settings, like a2 = 0.12993..., gives the fit itself), days one of day_choices, seed a whole number in LINK_SEEDS, market one
    of `markets` (saved markets only, so a link can't make the server fetch anything)."""
    out = {}
    get = (lambda k: params.get(k)) if hasattr(params, "get") else (lambda k: None)
    for k, (lo, hi) in LINK_LIMITS.items():
        v = get(k)
        try:
            x = float(str(v)[:32])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and lo <= x <= hi:
            out[k] = x
    for k, choices in (("days", day_choices), ("seed", None)):
        v = str(get(k))[:12]
        if not re.fullmatch(r"\d{1,6}", v):
            continue
        n = int(v)
        if (choices is not None and n in choices) or (choices is None and LINK_SEEDS[0] <= n <= LINK_SEEDS[1]):
            out[k] = n
    m = get("market")
    if isinstance(m, str) and m in markets:
        out["market"] = m
    return out


def write_link(market, run, default_market=None, default_run=None):
    """The query parameters for a link to this market and model run (read back by read_link). Settings
    equal to the defaults are left out, so the page's own address stays short until something changes."""
    # repr: the shortest text that gives back exactly the same number
    full = {"market": str(market), **{k: repr(float(run[k])) for k in LINK_LIMITS},
            "days": str(int(run["days"])), "seed": str(int(run["seed"]))}
    base = {} if default_run is None else write_link(default_market or market, default_run)
    if default_market is not None:
        base["market"] = str(default_market)
    return {k: v for k, v in full.items() if base.get(k) != v}
