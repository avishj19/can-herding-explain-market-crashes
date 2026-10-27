"""Stress tests for model.py across seeds and parameter combinations.

Run:  python3 tests/experiments.py
Saves figures/test_*.png and prints a results table.
"""
import sys
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from herding import model  # noqa: E402
from herding import stats  # noqa: E402
from herding import calibrate  # noqa: E402

FIG = ROOT / "figures"


def acf(x, lags):
    x = x - x.mean()
    return np.array([np.corrcoef(x[:-k], x[k:])[0, 1] for k in lags])


def job(args):
    """One simulation -> summary statistics. args = (seed, days, herding, overrides)."""
    seed, days, herding, over = args
    over = dict(over)
    keep = over.pop("keep", False)               # a flag for this script, not a model setting
    base = over.pop("base", model.P)             # start from these settings
    # research script: explore unstable settings too (guard off)
    r, f = model.simulate(days, herding=herding, p=dict(base, **over), seed=seed,
                         max_mispricing=None)
    burn = min(200, days // 10)                  # drop the start-up transient
    r, f = r[burn:], f[burn:]
    return dict(seed=seed, herding=herding, **over,
                kurt=stats.kurtosis(r), alpha=stats.hill_alpha(r), std=r.std(),
                ac1=acf(r, [1])[0], ac1_abs=acf(np.abs(r), [1])[0],
                frac=f.mean(), r=r if keep else None)


def run_all(jobs):
    with Pool() as pool:
        return pd.DataFrame(pool.map(job, jobs))


def main_experiments():
    seeds = range(8)
    spx = pd.read_csv(ROOT / "data/spx_daily.csv", parse_dates=["Date"], index_col="Date")
    real = np.diff(np.log(spx["Close"].values))

    # E1: seed-to-seed variability, herding on vs off
    e1 = run_all([(s, 5000, h, {}) for s in seeds for h in (True, False)])
    # E2: herding strength a1 x chartist trade size tc
    e2 = run_all([(s, 3000, True, dict(a1=a1, tc=tc)) for s in range(3)
                  for a1, tc in product([0.0, 0.3, 0.6, 0.9], [0.02, 0.05, 0.1, 0.2])])
    # E3: time step dt (results should NOT depend on it if the numerics are sound)
    e3 = run_all([(s, 3000, True, dict(dt=dt)) for s in range(4)
                  for dt in [0.01, 0.02, 0.05, 0.1]])
    # E4: number of traders N
    e4 = run_all([(s, 3000, True, dict(N=n)) for s in range(4)
                  for n in [100, 250, 500, 1000, 2000]])
    # E5: price units pf0 (should NOT matter)
    e5 = run_all([(s, 2000, h, dict(pf0=pf)) for s in range(4) for h in (True, False)
                  for pf in [1.0, 10.0, 100.0]])
    # E6: one long run kept for autocorrelation plots
    long_run = job((11, 19600, True, dict(keep=True)))["r"]
    cal_run = job((11, 19600, True, dict(keep=True, base=calibrate.load())))["r"]

    pd.set_option("display.width", 140, "display.precision", 3)
    cols = ["kurt", "alpha", "std", "ac1", "ac1_abs", "frac"]
    print("E1 seeds (mean, std over 8 seeds)\n",
          e1.groupby("herding")[cols].agg(["mean", "std"]).round(3))
    print("\nE2 herding a1 x tc (mean of 3 seeds)\n",
          e2.groupby(["a1", "tc"])[cols].mean().round(3))
    print("\nE3 time step dt\n", e3.groupby("dt")[cols].mean().round(4))
    print("\nE4 traders N\n", e4.groupby("N")[cols].mean().round(3))
    print("\nE5 price units pf0\n", e5.groupby(["herding", "pf0"])[cols].mean().round(4))
    print(f"\nS&P 500: kurt={stats.kurtosis(real):.2f} alpha={stats.hill_alpha(real):.2f} "
          f"ac1={acf(real, [1])[0]:.3f} ac1_abs={acf(np.abs(real), [1])[0]:.3f}")

    # ---- Figures
    FIG.mkdir(exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, (col, ref, lab) in enumerate([("kurt", stats.kurtosis(real), "S&P 500"),
                                          ("alpha", stats.hill_alpha(real), "S&P 500")]):
        for h, c in [(True, "#d94f30"), (False, "#2a6fdb")]:
            v = e1[e1.herding == h][col]
            ax[i].scatter(np.full(len(v), int(h)) + np.random.default_rng(0).uniform(-.1, .1, len(v)),
                          v, color=c, label="herding" if h else "no chartists")
        ax[i].axhline(ref, color="black", ls="--", label=lab)
        ax[i].set_xticks([0, 1], ["no chartists", "herding"])
        ax[i].set_title(["Excess kurtosis (0 = bell curve)", "Tail exponent α (S&P ≈ 3)"][i])
    ax[0].set_yscale("symlog"); ax[1].legend(fontsize=8)
    fig.suptitle("E1: 8 random seeds, 5000 days each")
    fig.tight_layout(); fig.savefig(FIG / "test_seeds.png", dpi=110)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, col, title in zip(ax, ["alpha", "kurt", "frac"],
                             ["Tail exponent α", "Excess kurtosis", "Mean chartist fraction"]):
        g = e2.groupby(["a1", "tc"])[col].mean().unstack()
        im = a.imshow(g.values, origin="lower", cmap="viridis", aspect="auto")
        a.set_xticks(range(len(g.columns)), g.columns); a.set_yticks(range(len(g.index)), g.index)
        a.set_xlabel("tc (chartist trade size)"); a.set_ylabel("a1 (herding strength)")
        for (i, j), v in np.ndenumerate(g.values):
            a.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=8)
        a.set_title(title); fig.colorbar(im, ax=a)
    fig.suptitle("E2: parameter grid (3 seeds each)")
    fig.tight_layout(); fig.savefig(FIG / "test_param_grid.png", dpi=110)

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for a, (df, key, title) in zip(ax, [(e3, "dt", "E3: time step dt"), (e4, "N", "E4: traders N"),
                                        (e5[e5.herding], "pf0", "E5: price units pf0 (herding)"),
                                        (e5[~e5.herding], "pf0", "E5: price units pf0 (no chartists)")]):
        m = df.groupby(key)[["alpha", "kurt"]].mean()
        sd = df.groupby(key)["std"].mean() / df["std"].mean()
        a.plot(m.index, m["alpha"], "o-", label="α")
        a.plot(m.index, m["kurt"], "s-", label="kurtosis")
        a.plot(m.index, sd, "^-", label="return std (relative)")
        a.set_xscale("log"); a.set_xlabel(key); a.set_title(title); a.legend(fontsize=7)
    fig.suptitle("Should these be flat? dt and pf0: yes. N: the paper says effects fade as N grows.")
    fig.tight_layout(); fig.savefig(FIG / "test_sensitivity.png", dpi=110)

    lags = np.arange(1, 101)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r, lab, c in [(real, "S&P 500", "black"), (long_run, "model (paper set)", "#d94f30"),
                      (cal_run, "model (calibrated)", "#2a9d4b")]:
        ax[0].plot(lags[:20], acf(r, lags[:20]), "o-", ms=3, label=lab, color=c)
        ax[1].plot(lags, acf(np.abs(r), lags), lw=1.5, label=lab, color=c)
    for a in ax: a.axhline(0, color="gray", lw=0.5); a.legend()
    ax[0].set_title("Autocorrelation of returns (real: ≈ 0)")
    ax[1].set_title("Autocorrelation of |returns| (volatility clustering)")
    ax[1].set_xlabel("lag (days)"); ax[0].set_xlabel("lag (days)")
    fig.tight_layout(); fig.savefig(FIG / "test_autocorrelation.png", dpi=110)

    tails = [0.005, 0.01, 0.02, 0.05, 0.1]
    g = np.random.default_rng(0).standard_normal(len(real))
    fig, ax = plt.subplots(figsize=(6, 4))
    for r, lab, c in [(real, "S&P 500", "black"), (long_run, "model (paper set)", "#d94f30"),
                      (cal_run, "model (calibrated)", "#2a9d4b"),
                      (g, "Gaussian", "gray")]:
        ax.plot(tails, [stats.hill_alpha(r, t) for t in tails], "o-", label=lab, color=c)
    ax.axhline(3, ls=":", color="green", label="inverse cubic (3)")
    ax.set_xscale("log"); ax.set_xlabel("fraction of data used as 'tail'"); ax.set_ylabel("Hill α")
    ax.set_title("How much does α depend on the tail cut?"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "test_hill_tail_choice.png", dpi=110)
    print("\nSaved figures/test_*.png")


if __name__ == "__main__":
    main_experiments()
