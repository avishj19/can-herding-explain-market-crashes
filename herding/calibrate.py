"""Fit the model's settings to target numbers (notes/SPEC.md, "Parameters").

Two fits, by one method:
    paper   reproduce Lux & Marchesi (1999)'s published numbers (alpha 2.64,
            H 0.48, H|r| 0.85, back to a bell curve at 25 days, no runaways)
    spx     make the model's history look like the S&P 500's (its fingerprint)

Method: (1) score random settings drawn from RANGES; (2) refine the best few with
Nelder-Mead. Every score uses the same TRAIN_SEEDS (so two settings are compared on
the same luck); checks of a fitted model must use other seeds (HELD_OUT_SEEDS).

Run:  python3 -m herding.calibrate paper      -> settings/paper.json
      python3 -m herding.calibrate spx        -> settings/spx.json
      add --refine to polish the saved settings on all 8 training seeds
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

from . import market_data, model, stats

SETTINGS_DIR = market_data.ROOT / "settings"
BURN = 1000                       # days dropped at the start, so the start does not matter
TRAIN_SEEDS = tuple(range(100, 108))
HELD_OUT_SEEDS = tuple(range(1000, 1040))

# Settings that are fitted, with search ranges (low, high, "log" or "lin").
RANGES = {
    "v1": (0.2, 5.0, "log"), "v2": (0.2, 5.0, "log"),
    "a1": (0.0, 1.2, "lin"), "a2": (0.0, 2.0, "lin"), "a3": (0.1, 3.0, "log"),
    "tc": (5e-4, 0.05, "log"), "gamma": (2e-3, 0.05, "log"), "beta": (1.0, 20.0, "log"),
    "sigma_mu": (0.0, 0.1, "lin"),
}
SPX_RANGES = dict(RANGES, sigma_eps=(0.002, 0.02, "log"))   # the S&P also sets the scale
# Settings that stay at parameter set I: N=500, s=0.75, R=0.0004, tick=0.001,
# dt=0.01, floor=4 (and sigma_eps=0.005 in the paper fit).

# Paper fit: (value, tolerance) for each published number, and the longest
# aggregated kurtosis allowed (the paper: a cross-over to the normal distribution).
PAPER_TARGETS = {"alpha_reg": (2.64, 0.15), "H": (0.48, 0.03), "H_abs": (0.85, 0.03)}
PAPER_KURT_25_MAX = 1.0
PAPER_DAYS = 50_000
# S&P fit: every fingerprint measure. The tolerance of each is how uncertain the
# S&P's own value is: its spread when the S&P's history is resampled in one-year
# blocks (a block bootstrap). It does not depend on the model, so a model cannot
# look "typical" just by being wildly variable (the first version of this fit,
# which divided by the model's own spread, did exactly that).
SPX_TARGETS = tuple(stats.FINGERPRINT)
BOOTSTRAP_BLOCK, BOOTSTRAP_DRAWS = 250, 200
SPX_MAX_DAYS = 1
RUNAWAY_LOSS = 1e4


def base():
    return dict(model.SET_I)


# ------------------------------------------------------------------ settings <-> numbers

def to_vector(p, ranges):
    """Settings as optimizer coordinates: log for "log" settings, as-is for "lin"."""
    return np.array([np.log(p[k]) if kind == "log" else p[k]
                     for k, (_, _, kind) in ranges.items()])


def from_vector(v, ranges, start=None):
    """Inverse of to_vector; values outside the ranges are clipped back in."""
    p = dict(base() if start is None else start)
    for x, (k, (lo, hi, kind)) in zip(v, ranges.items()):
        x = float(np.exp(x)) if kind == "log" else float(x)
        p[k] = min(hi, max(lo, x))
    return p


def draw(rng, ranges, start=None):
    """Random settings, uniform within each range (on a log scale for "log")."""
    p = dict(base() if start is None else start)
    for k, (lo, hi, kind) in ranges.items():
        p[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if kind == "log" \
            else float(rng.uniform(lo, hi))
    return p


# ------------------------------------------------------------------ measuring

def histories(p, seeds, days, max_days=model.MAX_DAYS):
    """Daily returns of one history per seed (after BURN days), or None if any runs
    away (max_days=1: any day more than 2.7x from the value counts as a runaway)."""
    out = []
    for s in seeds:
        try:
            out.append(model.simulate(p, days + BURN, seed=s, max_days=max_days)["r"][BURN:])
        except model.RunawayError:
            return None
    return out


def ensemble(p, seeds, days, max_days=model.MAX_DAYS):
    """Fingerprints of one history per seed, as {measure: array}, or None on a runaway."""
    runs = histories(p, seeds, days, max_days)
    if runs is None:
        return None
    rows = [stats.fingerprint(r) for r in runs]
    return {k: np.array([f[k] for f in rows]) for k in stats.FINGERPRINT}


def paper_loss(fp):
    """How far a model is from the paper's numbers, in units of their tolerances."""
    if fp is None:
        return RUNAWAY_LOSS
    loss = sum(((np.mean(fp[k]) - v) / tol) ** 2 for k, (v, tol) in PAPER_TARGETS.items())
    return float(loss + max(0.0, np.mean(fp["kurt_25"]) - PAPER_KURT_25_MAX) ** 2)


def block_bootstrap(r, block=BOOTSTRAP_BLOCK, draws=BOOTSTRAP_DRAWS, seed=0):
    """Fingerprints of `draws` resampled histories, each built from random blocks of
    `block` consecutive days (wrapping around), so day-to-day clustering survives."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    if not 1 <= block <= n:
        raise ValueError(f"block must be between 1 and {n}, got {block}")
    g = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        starts = g.integers(0, n, -(-n // block))
        idx = ((starts[:, None] + np.arange(block)[None, :]).ravel()[:n]) % n
        out.append(stats.fingerprint(r[idx]))
    return out


def spx_tolerance(r):
    """How uncertain each S&P fingerprint measure is (block-bootstrap spread)."""
    rows = block_bootstrap(r)
    return {k: float(np.std([f[k] for f in rows])) for k in SPX_TARGETS}


def spx_loss(fp, target, tol):
    """Mean squared gap between the model's median and the S&P, each gap in units of
    the S&P's own uncertainty (tol). About 1 or less = within the S&P's own noise."""
    if fp is None:
        return RUNAWAY_LOSS
    gaps = [(np.median(fp[k]) - target[k]) / tol[k] for k in SPX_TARGETS]
    return float(np.mean(np.square(gaps)))


def spx_target():
    return stats.fingerprint(market_data.spx_returns().values)


# ------------------------------------------------------------------ one fit

def fit_seeds(kind):
    """The seeds a fit scores every candidate on (the paper fit uses long runs, so 2)."""
    return TRAIN_SEEDS[:2] if kind == "paper" else TRAIN_SEEDS


class Fit:
    """Everything a fit needs, in one picklable object (for multiprocessing)."""

    def __init__(self, kind):
        if kind not in ("paper", "spx"):
            raise ValueError(f"kind must be 'paper' or 'spx', got {kind!r}")
        self.kind = kind
        self.ranges = RANGES if kind == "paper" else SPX_RANGES
        # The S&P never sat 2.7x away from its value, even for a day, so the S&P fit
        # rejects any such bubble; the paper fit only rejects persistent ones, as
        # the paper says ("no persistent deviations").
        self.max_days = SPX_MAX_DAYS if kind == "spx" else model.MAX_DAYS
        if kind == "paper":
            self.days, self.seeds, self.target = PAPER_DAYS, fit_seeds(kind), None
        else:
            self.target = spx_target()
            self.tol = spx_tolerance(market_data.spx_returns().values)
            self.days, self.seeds = len(market_data.spx_returns()), fit_seeds(kind)

    def loss(self, p):
        fp = ensemble(p, self.seeds, self.days, self.max_days)
        return paper_loss(fp) if self.kind == "paper" else spx_loss(fp, self.target, self.tol)

    def score_draw(self, i):
        p = draw(np.random.default_rng(10_000 + i), self.ranges)
        return self.loss(p), p

    def loss_of_vector(self, v):
        return self.loss(from_vector(v, self.ranges))


def _refine(args):
    fit, p, iters = args
    res = minimize(fit.loss_of_vector, to_vector(p, fit.ranges), method="Nelder-Mead",
                   options=dict(maxfev=iters, xatol=1e-3, fatol=1e-3))
    return float(res.fun), from_vector(res.x, fit.ranges)


def run(kind, draws=1200, keep=8, iters=150, workers=8):
    """Random search, then Nelder-Mead from the `keep` best; returns (loss, settings)."""
    fit = Fit(kind)
    t = time.time()
    with Pool(workers) as pool:
        scored = sorted(pool.map(fit.score_draw, range(draws), chunksize=4), key=lambda d: d[0])
        print(f"{kind}: {draws} random settings in {time.time() - t:.0f}s; "
              f"best loss {scored[0][0]:.2f}; runaways {sum(s >= RUNAWAY_LOSS for s, _ in scored)}")
        refined = pool.map(_refine, [(fit, p, iters) for _, p in scored[:keep]])
    refined.sort(key=lambda d: d[0])
    print(f"{kind}: refined {keep} in {time.time() - t:.0f}s total; best loss {refined[0][0]:.3f}")
    return refined[0]


class _PoolFit(Fit):
    """A Fit that runs its seeds in parallel (for refining one candidate at a time)."""

    def __init__(self, kind, pool, seeds):
        super().__init__(kind)
        self.pool, self.seeds = pool, tuple(seeds)

    def loss(self, p):
        rows = self.pool.map(_fingerprint_one, [(p, s, self.days, self.max_days) for s in self.seeds])
        fp = None if any(r is None for r in rows) else \
            {k: np.array([r[k] for r in rows]) for k in stats.FINGERPRINT}
        return paper_loss(fp) if self.kind == "paper" else spx_loss(fp, self.target, self.tol)


def _fingerprint_one(args):
    p, seed, days, max_days = args
    runs = histories(p, (seed,), days, max_days)
    return None if runs is None else stats.fingerprint(runs[0])


def refine_saved(kind, iters=200, workers=8):
    """Nelder-Mead from the saved settings, scoring every candidate on all
    TRAIN_SEEDS (the first fit may use fewer, which can overfit to lucky seeds)."""
    start = load(kind)
    with Pool(workers) as pool:
        fit = _PoolFit(kind, pool, TRAIN_SEEDS)
        first = fit.loss(start)
        res = minimize(fit.loss_of_vector, to_vector(start, fit.ranges), method="Nelder-Mead",
                       options=dict(maxfev=iters, xatol=1e-3, fatol=1e-3))
    print(f"{kind}: loss on all {len(TRAIN_SEEDS)} training seeds {first:.3f} -> {res.fun:.3f}")
    return float(res.fun), from_vector(res.x, fit.ranges)


def save(kind, loss, p, seeds=None):
    SETTINGS_DIR.mkdir(exist_ok=True)
    seeds = fit_seeds(kind) if seeds is None else seeds
    out = dict(kind=kind, loss=loss, train_seeds=list(seeds), burn=BURN,
               fitted=list(SPX_RANGES if kind == "spx" else RANGES), settings=p)
    path = SETTINGS_DIR / f"{kind}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    return path


def load(kind):
    """Fitted settings saved by `python3 -m herding.calibrate <kind>`."""
    path = SETTINGS_DIR / f"{kind}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run: python3 -m herding.calibrate {kind}")
    p = json.loads(path.read_text())["settings"]
    model.check(p)
    return p


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which not in ("paper", "spx"):
        sys.exit("usage: python3 -m herding.calibrate paper|spx [--refine]")
    if "--refine" in sys.argv:
        loss, p = refine_saved(which)
        print("saved", save(which, loss, p, TRAIN_SEEDS))
    else:
        loss, p = run(which)
        print("saved", save(which, loss, p))
