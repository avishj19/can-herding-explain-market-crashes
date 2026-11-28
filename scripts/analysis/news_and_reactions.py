"""Do the day's news and history move the market, and is the reaction "logical"?

THE DEFINITION USED EVERYWHERE BELOW
    news       new information about what shares are worth. In the model it is known
               exactly: the day's change in the log fundamental value (lf), which is
               plain Gaussian noise (a bell curve).
    return     the day's change in the log price (0.01 is about 1%).
    LOGICAL    a reaction is logical when the price moves about as much as the news
               does on that day.
    NOT LOGICAL (an over-reaction) when the move is much bigger than the news (here:
               more than OVER times bigger). This is what herding should produce.

Two tests:
  PART 1  the model, where the news is known. S&P-calibrated histories (seeds
          62000-62039, used nowhere else) with and without chartists (the herding
          traders). How much of the returns does the news explain, and on the biggest
          days was the news big too?
  PART 2  history: the 20 largest daily moves of the Dow and what happened that day,
          read from data/big_days_events.csv. Every row there has a source that was
          opened in November 2026; a cause that could not be confirmed is "unverified".

The classic study behind Part 2 is Cutler, Poterba & Summers (1989), "What moves
stock prices?", Journal of Portfolio Management 15(3): 4-12. It asks the same question
of the S&P's largest moves. No number from that paper is used here: it was not
re-read for this script.

Run from the project folder:  python3 scripts/analysis/news_and_reactions.py
It prints two tables and writes scratch/news/news_results.json (a smaller, quick
run, e.g. main(days=3000, n=2), writes news_results_quick.json instead):
    definition   the sentence above, so the file explains itself
    part1        days, seeds, and for "with_chartists" / "without_chartists":
                 histories, runaways, and per measure {median, low, high} across
                 histories (low/high = 5th/95th percentile). Measures: see MEASURES.
    part2        "days": the rows of big_days_events.csv; "counts": rows per kind
    seconds      how long the run took
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd

from herding import calibrate, experiments, market_data

DEFINITION = ("A reaction is LOGICAL when the price moves about as much as the news does that "
              "day; it is NOT logical (an over-reaction) when the move is much bigger than the news.")
SEEDS = tuple(range(62000, 62040))      # used by no other script or test
DAYS = 20_000                           # simulated days per history (about 80 years)
BIG = 0.01                              # "biggest-move days": the 1% largest |returns|
BIG_NEWS = 0.05                         # "big news": the 5% largest |news|
OVER = 2.0                              # over-reaction: the move is more than 2x the news
LAGS = 5                                # also let the news of the previous 5 days count
RESULTS = market_data.ROOT / "scratch" / "news" / "news_results.json"
QUICK_RESULTS = RESULTS.with_name("news_results_quick.json")   # smaller runs (tests) go here
EVENTS = market_data.DATA / "big_days_events.csv"
KINDS = ("news", "policy", "no clear news", "rebound", "unverified")
COLUMNS = ["date", "move_percent", "event", "kind", "source_url"]


# ---------------------------------------------------------------- PART 1: the model

def variance_explained(r, news, lags=0):
    """Share of the variance of the returns r that the news explains, from 0 to 1
    (the "R squared" of the best straight-line prediction of r from the news).
    1 = the news predicts every move exactly; 0 = the news says nothing about the move.
    lags=k also lets the news of each of the previous k days help the prediction."""
    r, news = experiments.finite_1d(r, "returns"), experiments.finite_1d(news, "news")
    if r.shape != news.shape:
        raise ValueError(f"need one news value per return: {news.shape} vs {r.shape}")
    if isinstance(lags, bool) or int(lags) != lags or lags < 0:
        raise ValueError(f"lags must be a whole number >= 0, got {lags!r}")
    lags = int(lags)
    if len(r) < lags + 3:
        raise ValueError(f"need at least {lags + 3} days, got {len(r)}")
    y = r[lags:]
    if y.var() == 0:
        raise ValueError("the returns never change, so there is no variance to explain")
    cols = [news[lags - k:len(news) - k] for k in range(lags + 1)]
    X = np.column_stack([np.ones(len(y))] + cols)
    left = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - left.var() / y.var())


def biggest(x, share):
    """True on the `share` (e.g. 0.01 = 1%) of days with the largest |x|, at least one day.
    Ties at the cut-off are all kept."""
    x = np.abs(experiments.finite_1d(x, "values"))
    if not 0 < share <= 1:
        raise ValueError(f"share must be above 0 and at most 1, got {share}")
    if x.size == 0:
        raise ValueError("need at least one day")
    k = max(1, int(round(share * x.size)))
    return x >= np.sort(x)[-k]


def reaction_ratio(r, news, share=BIG):
    """On the biggest-move days: the median of |return| / |news|. About 1 = logical
    (the price moved as much as the news); far above 1 = an over-reaction."""
    r, news = np.asarray(r, float), np.asarray(news, float)
    big = biggest(r, share)
    if np.any(news[big] == 0):
        raise ValueError("a big-move day has news of exactly zero, so the ratio is undefined")
    return float(np.median(np.abs(r[big]) / np.abs(news[big])))


def big_news_share(r, news, share=BIG, news_share=BIG_NEWS):
    """Share of the biggest-move days on which the news was itself big (in the top
    `news_share` of all days). 1 = every big move had big news behind it; by chance
    alone it would be about news_share."""
    return float(np.mean(biggest(news, news_share)[biggest(r, share)]))


def crowd_on_overreactions(r, news, z, share=BIG, over=OVER):
    """Do big over-reactions happen when chartists are many? An over-reaction day is a
    biggest-move day whose |return| is more than `over` times the |news|. z is the
    chartist share at the START of each day (so it cannot be a result of the move).
    Returns over_share (share of the biggest days that are over-reactions), z_over
    (mean chartist share on those days; NaN if there are none), z_all (on all days)
    and z_times (z_over / z_all; NaN if there are no chartists at all)."""
    r, news, z = (np.asarray(v, float) for v in (r, news, z))
    if not r.shape == news.shape == z.shape:
        raise ValueError(f"r, news and z need one value per day: {r.shape}, {news.shape}, {z.shape}")
    over_day = biggest(r, share) & (np.abs(r) > over * np.abs(news))
    z_over = float(z[over_day].mean()) if over_day.any() else float("nan")
    z_all = float(z.mean())
    return dict(over_share=float(over_day.sum() / biggest(r, share).sum()), z_over=z_over,
                z_all=z_all, z_times=z_over / z_all if z_all > 0 else float("nan"))


MEASURES = {
    "same_day": "share of return variance explained by the same day's news",
    "with_lags": f"... also allowing the previous {LAGS} days' news",
    "vol_ratio": "size of a typical move / size of typical news (sd / sd)",
    "ratio": "biggest 1% of days: median |move| / |news|",
    "big_news": "biggest 1% of days: share with big news (top 5%) that day",
    "over_share": f"biggest 1% of days: share that are over-reactions (move > {OVER:g}x news)",
    "z_all": "chartist share, average day",
    "z_over": "chartist share at the start of over-reaction days",
}


def news_measures(h):
    """Every Part 1 measure for one history h (a dict with r = daily returns, lf = log
    fundamental value with one more entry than r, z = chartist share, also one more)."""
    r, news, z = np.asarray(h["r"]), np.diff(h["lf"]), np.asarray(h["z"])[:-1]
    if not len(r) == len(news) == len(z):
        raise ValueError("lf and z must each have exactly one more entry than r")
    crowd = crowd_on_overreactions(r, news, z)
    return dict(same_day=variance_explained(r, news), with_lags=variance_explained(r, news, LAGS),
                vol_ratio=float(r.std() / news.std()), ratio=reaction_ratio(r, news),
                big_news=big_news_share(r, news), over_share=crowd["over_share"],
                z_all=crowd["z_all"], z_over=crowd["z_over"])


def summarise(rows):
    """Across histories: for each measure the median and the 5th and 95th percentile
    (ignoring NaN, e.g. a history with no over-reaction days)."""
    if len(rows) == 0:
        raise ValueError("need at least one history")
    out = {}
    for k in rows[0]:
        v = np.array([row[k] for row in rows], float)
        v = v[~np.isnan(v)]
        lo, mid, hi = np.percentile(v, [5, 50, 95]) if v.size else [float("nan")] * 3
        out[k] = dict(median=float(mid), low=float(lo), high=float(hi))
    return out


def model_part(p, seeds, days, pool=None):
    """Part 1: the measures with and without chartists, over one history per seed."""
    out = {}
    for name, chartists in (("with_chartists", True), ("without_chartists", False)):
        runs = experiments.histories(p, seeds, days, chartists, ("r", "lf", "z"), pool)
        kept = [h for h in runs if h is not None]
        out[name] = dict(histories=len(kept), runaways=len(runs) - len(kept),
                         **summarise([news_measures(h) for h in kept]))
    return out


def print_model(part):
    """The Part 1 table: one row per measure, median (5th to 95th percentile)."""
    cell = lambda m: f"{m['median']:7.3f} ({m['low']:.3f} to {m['high']:.3f})"
    print(f"\n{'':<68}{'WITH chartists (herding)':<30}WITHOUT chartists (benchmark)")
    for k, label in MEASURES.items():
        print(f"  {label:<66}{cell(part['with_chartists'][k]):<30}{cell(part['without_chartists'][k])}")
    for name in ("with_chartists", "without_chartists"):
        print(f"  {name}: {part[name]['histories']} histories, {part[name]['runaways']} ran away")


# ---------------------------------------------------------------- PART 2: history

def biggest_days(r, n=20):
    """The n largest daily moves (by absolute log return), largest first, as percent
    moves: 100 * (exp(log return) - 1), so -22.61 means the index lost 22.61%."""
    if isinstance(n, bool) or int(n) != n or not 1 <= n <= len(r):
        raise ValueError(f"n must be a whole number between 1 and {len(r)}, got {n!r}")
    top = r.abs().sort_values(ascending=False, kind="stable").index[:int(n)]
    return (100 * np.expm1(r[top])).rename("move_percent")


def load_events(path=EVENTS):
    """data/big_days_events.csv, checked: the right columns, a known kind and a web
    source on every row, an event sentence on every row except "unverified" ones
    (which must have none), and no date twice."""
    ev = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(ev.columns) != COLUMNS:
        raise ValueError(f"{path}: columns must be {COLUMNS}, got {list(ev.columns)}")
    ev["date"] = pd.to_datetime(ev["date"], format="%Y-%m-%d")
    ev["move_percent"] = ev["move_percent"].astype(float)
    if ev["date"].duplicated().any():
        raise ValueError(f"{path}: a date appears twice")
    for row in ev.itertuples():
        if row.kind not in KINDS:
            raise ValueError(f"{path}: {row.date.date()} has unknown kind {row.kind!r}; use {KINDS}")
        if (row.kind == "unverified") != (row.event.strip() == ""):
            raise ValueError(f"{path}: {row.date.date()}: only 'unverified' rows have an empty event")
        if row.kind != "unverified" and not all(u.startswith("https://") for u in row.source_url.split(" | ")):
            raise ValueError(f"{path}: {row.date.date()} needs a source_url starting with https://")
    return ev


def check_events(ev, r, tol=0.005):
    """Raise unless the events are exactly the len(ev) biggest days of r and each
    move_percent equals the data's (to 2 decimals)."""
    top = biggest_days(r, len(ev))
    if set(ev["date"]) != set(top.index):
        raise ValueError("the events file does not list exactly the biggest days of the data")
    gap = np.abs(ev["move_percent"].values - top[pd.DatetimeIndex(ev["date"])].values)
    if gap.max() > tol:
        raise ValueError(f"move_percent differs from the data on {ev['date'][gap.argmax()].date()}")


def kind_counts(ev):
    """How many days of each kind, in the order of KINDS (0 for a kind with none)."""
    return {k: int((ev["kind"] == k).sum()) for k in KINDS}


def history_part(r):
    """Part 2: the checked events and the count of each kind."""
    ev = load_events()
    check_events(ev, r)
    days = [dict(date=str(row.date.date()), move_percent=row.move_percent, event=row.event,
                 kind=row.kind, source_url=row.source_url) for row in ev.itertuples()]
    return dict(days=days, counts=kind_counts(ev))


def print_history(part):
    """The Part 2 table and the counts."""
    print()
    for d in part["days"]:
        print(f"  {d['date']}  {d['move_percent']:+7.2f}%  {d['kind']:<14} {d['event'] or '(cause not confirmed)'}")
    c = part["counts"]
    print(f"\n  counts: {c}")
    print(f"  a cause that day (news or policy): {c['news'] + c['policy']} of {len(part['days'])};  "
          f"no clear news or a rebound: {c['no clear news'] + c['rebound']};  unverified: {c['unverified']}")


def main(days=None, n=40):
    """Run both parts. days = simulated days per history (None = DAYS); n = number
    of histories. A full-size run (the defaults) writes RESULTS; any smaller, quick run
    writes QUICK_RESULTS instead, so a test can never overwrite the full results."""
    t0, full = time.time(), days is None and n == len(SEEDS)
    days = DAYS if days is None else int(days)
    print(DEFINITION)
    print(f"\nPART 1. The model, where the news is known ({n} histories of {days:,} days; "
          f"median, 5th to 95th percentile across histories)")
    part1 = dict(days=days, seeds=list(SEEDS[:n]), **model_part(calibrate.load("spx"), SEEDS[:n], days))
    print_model(part1)
    dow = market_data.returns("dow")
    print("\nPART 2. The 20 largest daily moves of the Dow, 1896-2026, and what happened that day")
    part2 = history_part(dow)
    print_history(part2)
    out = dict(definition=DEFINITION, part1=part1, part2=part2,
               seconds=round(time.time() - t0, 1))
    path = RESULTS if full else QUICK_RESULTS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"\nsaved {path.relative_to(market_data.ROOT)} ({out['seconds']} s)")
    return out


if __name__ == "__main__":
    main()
