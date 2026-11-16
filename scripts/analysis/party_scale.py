"""The political-party test at scale: every market, 1,000 model histories.

Question: US stocks have returned more under Democratic presidents (Fig 12). Is that gap
bigger than luck, in every market we have, when luck is measured with many more histories?

Run from the project folder:  python3 scripts/analysis/party_scale.py     (about 3 minutes)

For each market (Dow 1896-, S&P 1928-, S&P 1950-, Nasdaq-100 1985-, 20 stocks 2000-):
  1. the real gap, Democrats minus Republicans (politics.party_gap: 4 measures);
  2. two yardsticks for luck, and the real market's percentile in each
     (0-100; 50 = typical; outside 5-95 = a "flag"):
       shifted  the real returns with the party calendar slid along (politics.shifted_gaps)
       model    n model histories cut at the real inauguration dates (the model has no politics)
  3. a 95% interval for the yearly return gap from a block bootstrap: the series is rebuilt
     2,000 times from randomly chosen 250-day blocks (about a year each; a day's return and
     its party stay together), so calm and wild spells shorter than a year are kept;
  4. leave one presidency out: the return gap again without each presidency's days.
Then the false-discovery view: with many (market, measure) cells, about 10% fall outside
5-95 by luck alone, so we count how many do.

Model histories: seeds 60000-60999 (SEEDS), a range used nowhere else in the project (not for
fitting, the stability screen, or any other figure). They are simulated ONCE at the length of
the longest market; a shorter market uses the last N days of each. That is fair because a
history is stationary after its burn-in (any stretch of it is as good as any other), but it
means the markets share the same histories: their model yardsticks are not independent.
Histories that run away are counted and skipped.

Layout of the saved JSON (plain lists, numbers and text only):
  about            n_asked, seeds [first, last], histories (number that did not run away),
                   runaways, runaway_seeds, days_simulated, days (null = all), block,
                   draws, seconds, measures {measure: plain description}
  markets          a list, one dict per market:
    name, kind ("index" or "stock"), start, end (dates), days, dem_days, rep_days,
    party_changes, sd_daily
    gap_year       return gap D - R in log return per year (0.05 = 5 points a year)
    real           {mean, vol, kurtosis, crashes}: politics.party_gap (mean is in daily sds)
    bootstrap      {low, high, block, draws, used}: the 95% interval for gap_year
    shifted, model each {n, pct {measure: percentile of the real value},
                   p5 {measure}, p95 {measure}, values {measure: [one per calendar/history]}}
    leave_one_out  a list, one per presidency in the data: name, party, start, end, days,
                   gap_year (without it; null if a party has no days left),
                   change (gap_year without it minus gap_year with it)
  false_discovery  expected_share (0.10), and for "shifted" and "model": cells, flagged,
                   share, by_measure {measure: flagged}, flags [[market, measure, pct], ...]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from herding import calibrate, experiments, market_data, politics, stats

SEEDS = tuple(range(60000, 61000))     # fresh: used by no fit, screen or other figure
BLOCK = 250                            # bootstrap block, in trading days (about a year)
DRAWS = 2000                           # bootstrap resamples
INDEXES = ("dow", "spx_1928", "spx", "ndx")
OUT = market_data.ROOT / "scratch" / "party" / "scale_results.json"
QUICK_OUT = OUT.with_name("scale_results_quick.json")    # runs on only the last `days` days
YARDSTICKS = ("shifted", "model")


def market_names():
    """Every market we have: the four indexes (longest first), then the 20 stocks."""
    return list(INDEXES) + market_data.stock_tickers()


def load_returns(name, days=None):
    """Daily log returns of an index or a stock by name (only the last `days` if given)."""
    if name in market_data.MARKETS:
        r = market_data.returns(name)
    elif name in market_data.stock_tickers():
        r = market_data.stock_returns(name)
    else:
        raise ValueError(f"unknown market {name!r}; choose from {market_names()}")
    if days is None:
        return r
    if isinstance(days, bool) or int(days) != days or days < 1:
        raise ValueError(f"days must be a positive whole number or None, got {days}")
    return r.iloc[-int(days):]


return_gap = politics.return_gap        # the yearly D-minus-R return gap (one shared definition)


def bootstrap_interval(r, dem, block=BLOCK, draws=DRAWS, seed=0):
    """A 95% interval for return_gap by block bootstrap. Each resample glues together
    randomly chosen stretches of `block` days in a row (starting on any day, repeats
    allowed) until it is as long as the data; each day keeps its own party. The interval
    runs from the 2.5th to the 97.5th percentile of the resamples' gaps. A resample that
    happens to have no day of one party is skipped (`used` counts the rest)."""
    r = experiments.finite_1d(r, "returns")
    dem = np.asarray(dem, dtype=bool)
    if dem.shape != r.shape:
        raise ValueError(f"need one party per return: {dem.shape} vs {r.shape}")
    for name, v in (("block", block), ("draws", draws)):
        if isinstance(v, bool) or int(v) != v or v < 1:
            raise ValueError(f"{name} must be a positive whole number, got {v}")
    if block > len(r):
        raise ValueError(f"block ({block} days) is longer than the data ({len(r)} days)")
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(draws):
        i = stats.block_indices(len(r), int(block), rng, wrap=False)
        d = dem[i]
        if d.any() and not d.all():
            gaps.append(return_gap(r[i], d))
    if len(gaps) < 2:
        raise ValueError("almost every resample had only one party: the data is too short")
    low, high = np.percentile(gaps, [2.5, 97.5])
    return dict(low=float(low), high=float(high), block=int(block), draws=int(draws), used=len(gaps))


def leave_one_out(r, dem, spans):
    """The yearly return gap again with each presidency's days dropped. spans: dicts with
    name, party, a, b (day indexes a:b, b not included), start, end, as politics.terms
    gives. Returns one dict per presidency with the gap without it (None if that leaves a
    party with no days) and the change from the gap with every day in."""
    r = experiments.finite_1d(r, "returns")
    dem = np.asarray(dem, dtype=bool)
    full = return_gap(r, dem)
    out = []
    for s in spans:
        if not 0 <= s["a"] < s["b"] <= len(r):
            raise ValueError(f"{s['name']}: days {s['a']}:{s['b']} are outside the data (0:{len(r)})")
        keep = np.ones(len(r), dtype=bool)
        keep[s["a"]:s["b"]] = False
        d = dem[keep]
        gap = return_gap(r[keep], d) if d.any() and not d.all() else None
        out.append(dict(name=s["name"], party=s["party"], start=str(s["start"])[:10],
                        end=str(s["end"])[:10], days=int(s["b"] - s["a"]), gap_year=gap,
                        change=None if gap is None else gap - full))
    return out


def simulate(p, n, days, pool=None):
    """n model histories of `days` days from the first n SEEDS. Returns the returns of
    the ones that did not run away, and the seeds of the ones that did."""
    if isinstance(n, bool) or int(n) != n or not 1 <= n <= len(SEEDS):
        raise ValueError(f"n must be a whole number from 1 to {len(SEEDS)}, got {n}")
    seeds = SEEDS[:int(n)]
    out = experiments.histories(p, seeds, days, pool=pool)
    runs = [h["r"] for h in out if h is not None]
    if not runs:
        raise ValueError("every model history ran away: nothing to compare with")
    return runs, [s for s, h in zip(seeds, out) if h is None]


def yardstick(real, gaps):
    """Where the real gap sits among a yardstick's gaps (a list of party_gap dicts): per
    measure the real value's percentile, the yardstick's 5th and 95th percentiles, and
    every value (for drawing)."""
    if len(gaps) == 0:
        raise ValueError("the yardstick is empty")
    values = {k: [float(g[k]) for g in gaps] for k in politics.MEASURES}
    return dict(n=len(gaps),
                pct={k: experiments.percentile(real[k], v) for k, v in values.items()},
                p5={k: float(np.percentile(v, 5)) for k, v in values.items()},
                p95={k: float(np.percentile(v, 95)) for k, v in values.items()},
                values=values)


def market_result(name, r, runs):
    """Everything about one market: r = its returns (a Series with dates), runs = model
    histories at least as long (their last len(r) days are used)."""
    if min(len(x) for x in runs) < len(r):
        raise ValueError(f"{name}: the model histories are shorter than the market ({len(r)} days)")
    dem = politics.democrat(r.index)
    v = r.values
    real = politics.party_gap(v, dem)
    return dict(
        name=name, kind="index" if name in INDEXES else "stock",
        start=str(r.index[0].date()), end=str(r.index[-1].date()), days=len(r),
        dem_days=int(dem.sum()), rep_days=int((~dem).sum()),
        party_changes=len(politics.switches(r.index)[0]) - 1, sd_daily=float(v.std()),
        gap_year=return_gap(v, dem), real=real, bootstrap=bootstrap_interval(v, dem),
        shifted=yardstick(real, politics.shifted_gaps(v, dem)),
        model=yardstick(real, [politics.party_gap(x[len(x) - len(r):], dem) for x in runs]),
        leave_one_out=leave_one_out(v, dem, politics.terms(r.index)))


def false_discovery(markets, low=5.0, high=95.0):
    """Count the flags: (market, measure) cells whose percentile is below `low` or above
    `high`, for each yardstick. By luck alone about (low + 100 - high)% of cells are flags."""
    out = {"expected_share": (low + 100 - high) / 100}
    for w in YARDSTICKS:
        flags = [[m["name"], k, m[w]["pct"][k]] for m in markets for k in politics.MEASURES
                 if not low <= m[w]["pct"][k] <= high]
        cells = len(markets) * len(politics.MEASURES)
        out[w] = dict(cells=cells, flagged=len(flags), share=len(flags) / cells if cells else 0.0,
                      by_measure={k: sum(f[1] == k for f in flags) for k in politics.MEASURES},
                      flags=flags)
    return out


def print_headlines(markets):
    """One line per market: the return gap, its interval and its two percentiles."""
    print(f"\nReturn gap, Democrats minus Republicans (log return per year; 95% interval from "
          f"{BLOCK}-day blocks)\n{'market':<9} {'years':<9} {'days':>6} {'changes':>7} "
          f"{'gap/yr':>7}  {'95% interval':<17} {'pct shifted':>11} {'pct model':>9}")
    for m in markets:
        b = m["bootstrap"]
        interval = f"{b['low']:+.1%} to {b['high']:+.1%}"
        print(f"{m['name']:<9} {m['start'][:4]}-{m['end'][:4]} {m['days']:>6,} {m['party_changes']:>7} "
              f"{m['gap_year']:>+7.1%}  {interval:<17} "
              f"{m['shifted']['pct']['mean']:>11.1f} {m['model']['pct']['mean']:>9.1f}")


def print_measures(markets):
    """Per market and measure: the real value and its percentile in each yardstick."""
    print("\nEach measure: real value (percentile among shifted calendars / among model histories)")
    print(f"{'market':<9}" + "".join(f"{k:>26}" for k in politics.MEASURES))
    for m in markets:
        print(f"{m['name']:<9}" + "".join(
            f"{m['real'][k]:>+11.2f} ({m['shifted']['pct'][k]:>5.1f}/{m['model']['pct'][k]:>5.1f})"
            for k in politics.MEASURES))


def print_leave_one_out(markets, top=3):
    """Per market: the presidencies whose removal moves the return gap most."""
    print(f"\nLeave one presidency out: the {top} that move the yearly return gap most")
    for m in markets:
        rows = sorted((t for t in m["leave_one_out"] if t["change"] is not None),
                      key=lambda t: -abs(t["change"]))[:top]
        cells = "; ".join(f"{t['name']} {t['start'][:4]} ({t['party']}) -> {t['gap_year']:+.1%} "
                          f"({t['change']:+.1%})" for t in rows)
        print(f"{m['name']:<9} all days {m['gap_year']:+.1%}; without {cells}")


def print_false_discovery(fd):
    """How many cells are flags, against the share luck alone gives."""
    print(f"\nFalse-discovery view: cells outside 5-95 (luck alone gives about "
          f"{fd['expected_share']:.0%} of cells)")
    for w in YARDSTICKS:
        f = fd[w]
        by = ", ".join(f"{k} {v}" for k, v in f["by_measure"].items())
        print(f"  {w:<8} {f['flagged']} of {f['cells']} cells ({f['share']:.0%}; luck: about "
              f"{fd['expected_share'] * f['cells']:.1f})   by measure: {by}")
        for name, k, pct in f["flags"]:
            print(f"      {name:<9} {k:<9} at {pct:.1f}")
    print("  The stocks share one calendar and move together, so their cells are not "
          "independent:\n  flags come in clusters, and the count can stray from 10% by luck "
          "more than coin flips would.")


def main(n=1000, out=None, markets=None, days=None, pool=None):
    """Run every market (or the named ones) against n model histories, print the tables
    and save everything as JSON. days: only each market's last `days` days (quick tests;
    saved to QUICK_OUT unless `out` is given). pool: a ready pool (tests pass a serial one)."""
    t0 = time.time()
    names = market_names() if markets is None else list(markets)
    if not names:
        raise ValueError("need at least one market")
    out = Path(out) if out is not None else (OUT if days is None else QUICK_OUT)
    series = {name: load_returns(name, days) for name in names}
    longest = max(len(r) for r in series.values())
    runs, runaway_seeds = simulate(calibrate.load("spx"), n, longest, pool)
    print(f"{len(runs)} model histories of {longest:,} days (seeds {SEEDS[0]}-{SEEDS[n - 1]}; "
          f"{len(runaway_seeds)} ran away and were skipped)")
    results = [market_result(name, series[name], runs) for name in names]
    fd = false_discovery(results)
    print_headlines(results)
    print_measures(results)
    print_leave_one_out(results)
    print_false_discovery(fd)
    about = dict(n_asked=int(n), seeds=[SEEDS[0], SEEDS[n - 1]], histories=len(runs),
                 runaways=len(runaway_seeds), runaway_seeds=runaway_seeds, days_simulated=longest,
                 days=days, block=BLOCK, draws=DRAWS, seconds=round(time.time() - t0, 1),
                 measures=dict(politics.MEASURES))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(about=about, markets=results, false_discovery=fd), allow_nan=False))
    print(f"\nsaved {out}  ({about['seconds']} s)")
    return dict(about=about, markets=results, false_discovery=fd)


if __name__ == "__main__":
    main()
