"""Term by term: which presidencies can the politics-free model explain?

Every presidential term of the S&P 500 since 1928 against 40 S&P-calibrated model
histories cut at the same days (each history rescaled to the S&P's daily sd and mean,
because the model has no drift). For seven measures, the S&P's percentile among the
histories; outside 5-95 is a "flag" (the model does not explain it). Then: how many flags
luck alone gives (each history in turn plays the S&P), whether the parties differ (exact
permutation test), and what the flagged terms have in common. RESULTS.md, "Fig 12b".
Merged from three agents' scripts (November 2026; drafts in scratch/review/party_*).

Run from the project folder:  python3 scripts/analysis/party_terms.py
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd

from herding import calibrate, experiments, market_data, politics, stats

Y = politics.DAYS_PER_YEAR
MIN_HILL = 1000                  # the tail exponent only for terms this long
KEYS = ("ret", "vol", "kurt", "worst1", "worst20", "crash", "alpha")
WILD_HIGH = dict(vol=True, kurt=True, worst1=True, worst20=True, crash=True, alpha=False)
# falls in the 1930s undone the next day: suspected bad closes, not checked against another source
SUSPECT = ("1935-04-16", "1935-05-16", "1935-08-16", "1936-09-16", "1938-10-10")


def measures(seg, mu, sd):
    """The seven measures of one stretch of daily log returns (falls as positive numbers;
    crash days are below mu - CRASH_SD sd of the whole series)."""
    return dict(ret=seg.mean() * Y, vol=seg.std() * np.sqrt(Y), kurt=stats.kurtosis(seg),
                worst1=experiments.worst_fall(seg, 1), worst20=experiments.worst_fall(seg, 20),
                crash=int(np.sum(seg < mu - politics.CRASH_SD * sd)),
                alpha=stats.hill_alpha(seg) if len(seg) >= MIN_HILL else np.nan)


def flags(real, sims):
    """Percentiles of `real` among `sims`, and the measures outside 5-95."""
    pct = {k: np.nan if np.isnan(real[k]) else experiments.percentile(real[k], [s[k] for s in sims])
           for k in KEYS}
    return pct, [k for k in KEYS if not np.isnan(pct[k]) and (pct[k] < 5 or pct[k] > 95)]


def kind(k, pct):
    """'wild', 'calm', or for the return 'ret-high' / 'ret-low'."""
    if k == "ret":
        return "ret-high" if pct > 95 else "ret-low"
    return "wild" if (pct > 95) == WILD_HIGH[k] else "calm"


def permutation_p(counts, party):
    """Exact two-sided permutation test of mean(counts) for "D" minus "R":
    (difference, share of all relabellings with a difference at least as big)."""
    counts, party = np.asarray(counts, float), np.asarray(party)
    nd, n = int(np.sum(party == "D")), len(counts)
    obs = counts[party == "D"].mean() - counts[party == "R"].mean()
    diffs = [counts[list(c)].sum() / nd - (counts.sum() - counts[list(c)].sum()) / (n - nd)
             for c in itertools.combinations(range(n), nd)]
    return float(obs), float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12))


def main(days=None, n=40):
    """All terms (or only those inside the last `days` days) against n model histories."""
    s = market_data.returns("spx_1928")
    s = s if days is None else s.iloc[-int(days):]
    x, mu, sd = s.values, s.values.mean(), s.values.std()
    with experiments.Pool() as pool:
        hs = experiments.histories(calibrate.load("spx"), calibrate.HELD_OUT_SEEDS[:n], len(x), pool=pool)
    runs = [(h["r"] - h["r"].mean()) / h["r"].std() * sd + mu for h in hs if h is not None]
    print(f"S&P 500 {s.index[0].year}-{s.index[-1].year}; {len(runs)} of {len(hs)} model histories did not run away")
    print("value (percentile among the model histories); falls are log returns\n")
    rows, sims_of = [], []
    for t in politics.terms(s.index):
        a, b = t["a"], t["b"]
        real, sims = measures(x[a:b], mu, sd), [measures(h[a:b], mu, sd) for h in runs]
        pct, fl = flags(real, sims)
        rows.append(dict(t, flags=fl, kinds=[kind(k, pct[k]) for k in fl]))
        sims_of.append(sims)
        cells = "  ".join(f"{k} {real[k]:.3g} ({pct[k]:.0f})" for k in KEYS if not np.isnan(real[k]))
        print(f"{t['name']:<10} {t['party']} {t['start'].year}-{t['end'].year} {b - a:>5,} days: {cells}"
              f"  -> {len(fl)} flags {', '.join(f'{k}:{w}' for k, w in zip(fl, rows[-1]['kinds']))}")
    # luck alone: each model history in turn plays the S&P against the others
    luck = np.array([[len(flags(sims[j], sims[:j] + sims[j + 1:])[1]) for sims in sims_of]
                     for j in range(len(runs))])
    total = sum(len(r["flags"]) for r in rows)
    print(f"\nluck alone: {luck.mean():.2f} flags per term; over {len(rows)} terms median "
          f"{np.median(luck.sum(1)):.0f}, 95th percentile {np.percentile(luck.sum(1), 95):.0f}. "
          f"S&P: {total} (reached by {np.mean(luck.sum(1) >= total):.0%} of luck runs)")
    counts, party = [len(r["flags"]) for r in rows], [r["party"] for r in rows]
    if len(set(party)) == 2:
        for P in "DR":
            c = [k for k, q in zip(counts, party) if q == P]
            print(f"{P}: {len(c)} terms, {sum(c)} flags = {np.mean(c):.2f} per term")
        diff, p = permutation_p(counts, party)
        print(f"D - R: {diff:+.2f} flags per term, permutation p = {p:.2f}")
    kinds = [w for r in rows for w in r["kinds"]]
    print(f"kinds of flag: wild {kinds.count('wild')}, calm {kinds.count('calm')}, "
          f"return {kinds.count('ret-high') + kinds.count('ret-low')}")
    for r, sims in zip(rows, sims_of):
        if r["name"] == "Roosevelt":           # how much the suspect 1930s closes matter
            i = s.index.get_indexer(pd.DatetimeIndex(SUSPECT))
            y = x.copy()
            y[np.concatenate([i, i + 1])] = 0.0
            print(f"Roosevelt without the {len(SUSPECT)} suspect days: flags "
                  f"{flags(measures(y[r['a']:r['b']], mu, sd), sims)[1]} (with them: {r['flags']})")


if __name__ == "__main__":
    main()
