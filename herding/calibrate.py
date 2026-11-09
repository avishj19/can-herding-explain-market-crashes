"""Fit the model's settings to target numbers (docs/SPEC.md, "Parameters").

Two fits, by one method:
    paper   reproduce Lux & Marchesi (1999)'s published numbers (alpha 2.64,
            H 0.48, H|r| 0.85, back to a bell curve at 25 days, no runaways)
    spx     make the model's history look like the S&P 500's (its fingerprint)

Method: (1) score random settings drawn from RANGES; (2) refine the best few with
Nelder-Mead; (3) stability screen: run each refined candidate for STABILITY_DAYS on
STABILITY_SEEDS and keep the best-scoring one among those with the fewest runaways
(a runaway can be too rare to show up in the shorter fitting runs). Every score
uses the same TRAIN_SEEDS (so two settings are compared on the same luck); checks
of a fitted model must use other seeds (HELD_OUT_SEEDS).

Run:  python3 -m herding.calibrate paper      -> settings/paper_matched.json
      python3 -m herding.calibrate spx        -> settings/sp500_calibrated.json
      add --refine to polish the saved settings on all 8 training seeds
      add --draws N to score N random settings in step 1 (default 1200)
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import market_data, model, stats

SETTINGS_DIR = market_data.ROOT / "settings"
SETTINGS_FILE = {"paper": "paper_matched.json", "spx": "sp500_calibrated.json"}


def settings_path(kind):
    """Where the fitted settings of one kind ("paper" or "spx") are saved."""
    if kind not in SETTINGS_FILE:
        raise ValueError(f"unknown settings {kind!r}; choose 'paper' or 'spx'")
    return SETTINGS_DIR / SETTINGS_FILE[kind]
BURN = 1000                       # days dropped at the start, so the start does not matter
TRAIN_SEEDS = tuple(range(100, 108))
HELD_OUT_SEEDS = tuple(range(1000, 1040))
STABILITY_SEEDS = tuple(range(2000, 2020))   # the stability screen: neither training nor held out
STABILITY_DAYS = 100_000                     # about 400 years of trading days per history

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
# The tolerances are this project's own yardsticks, NOT the paper's error bars.
# The paper's bars (from its Fig 2 and 3 captions: alpha 0.077, H 0.003, H|r| 0.010)
# are regression errors from one long run; they are smaller than how much these numbers change from
# one run of the model to the next (H of |returns| varies by about 0.03), so no
# setting could match them on every seed. The wider tolerances allow for that.
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


class SavedLoss:
    """A loss function whose every score is saved as one JSON line in `path`. A
    score already in the file is read back instead of recomputed. Nelder-Mead is
    deterministic, so a refinement restarted after a stop asks for the same points
    in the same order and replays the saved ones instantly."""

    def __init__(self, fn, path):
        self.fn, self.path, self.cache = fn, Path(path), {}
        # a line cut off mid-write would glue onto the next one: end it first
        if self.path.exists() and self.path.stat().st_size \
                and not self.path.read_bytes().endswith(b"\n"):
            with self.path.open("a") as f:
                f.write("\n")
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:          # a line cut off mid-write
                    continue
                self.cache[tuple(rec["v"])] = rec["loss"]

    def __call__(self, v):
        key = tuple(float(x) for x in v)
        if key not in self.cache:
            self.cache[key] = float(self.fn(np.array(key)))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"v": list(key), "loss": self.cache[key]}) + "\n")
        return self.cache[key]


def _refine(args):
    from scipy.optimize import minimize   # here, not on top: only fits need slow-to-load scipy
    fit, p, iters = args[:3]
    loss = fit.loss_of_vector if len(args) < 4 else SavedLoss(fit.loss_of_vector, args[3])
    res = minimize(loss, to_vector(p, fit.ranges), method="Nelder-Mead",
                   options=dict(maxfev=iters, xatol=1e-3, fatol=1e-3))
    return float(res.fun), from_vector(res.x, fit.ranges)


def _runs_away(args):
    p, seed, days = args
    try:
        model.simulate(p, days, seed=seed)
    except model.RunawayError:
        return True
    return False


def runaways(pool, p, seeds=STABILITY_SEEDS, days=STABILITY_DAYS):
    """How many of the long histories (one per seed) run away, by the model's own
    rule (more than e away from the value for MAX_DAYS days in a row)."""
    return int(sum(pool.map(_runs_away, [(p, int(s), int(days)) for s in seeds])))


def screen(pool, candidates, seeds=STABILITY_SEEDS, days=STABILITY_DAYS):
    """Stability screen: sort (loss, settings) candidates by (runaways, loss), so the
    best fit among the never-runaway ones comes first. Returns (runaways, loss, p)."""
    # one pool.map over every (candidate, seed), so no worker waits between candidates
    jobs = [(p, int(s), int(days)) for _, p in candidates for s in seeds]
    ran_away = pool.map(_runs_away, jobs, chunksize=1)
    counts = [int(sum(ran_away[i * len(seeds):(i + 1) * len(seeds)])) for i in range(len(candidates))]
    out = sorted(((bad, loss, p) for bad, (loss, p) in zip(counts, candidates)),
                 key=lambda c: (c[0], c[1]))
    for bad, loss, _ in out:
        print(f"  candidate loss {loss:.3f}: {bad} of {len(seeds)} long histories ran away")
    return out


CHECKPOINT_DIR = market_data.ROOT / "scratch" / "calibrate"   # git-ignored


def checkpointed(pool, fn, jobs, path):
    """pool.map(fn, jobs), but each result is saved as one JSON line in `path` as
    soon as it is done, and results already in the file are not recomputed. So a
    fit that is stopped (a flat battery) loses at most the jobs that were running.
    Every result must be JSON-friendly; lists come back as tuples."""
    path = Path(path)
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:              # a line cut off mid-write
                continue
            done[rec["i"]] = tuple(rec["out"]) if isinstance(rec["out"], list) else rec["out"]
    todo = [i for i in range(len(jobs)) if i not in done]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
            f.write("\n")                            # end a line cut off mid-write
        for i, out in zip(todo, pool.imap(fn, [jobs[i] for i in todo])):
            f.write(json.dumps({"i": i, "out": out}) + "\n")
            f.flush()
            done[i] = tuple(out) if isinstance(out, list) else out
    return [done[i] for i in range(len(jobs))]


def run(kind, draws=1200, keep=8, iters=150, workers=8, checkpoint_dir=CHECKPOINT_DIR):
    """Random search, Nelder-Mead from the `keep` best, then the stability screen;
    returns (loss, settings, runaways). Progress is saved in checkpoint_dir, so
    running the same command again after a stop continues where it was; the
    checkpoint files are removed once the fit has finished."""
    fit = Fit(kind)
    t = time.time()
    tag = Path(checkpoint_dir) / f"{kind}_draws{draws}_keep{keep}_iters{iters}"
    with Pool(workers) as pool:
        scored = checkpointed(pool, fit.score_draw, list(range(draws)), f"{tag}_search.jsonl")
        scored.sort(key=lambda d: d[0])
        print(f"{kind}: {draws} random settings in {time.time() - t:.0f}s; "
              f"best loss {scored[0][0]:.2f}; runaways {sum(s >= RUNAWAY_LOSS for s, _ in scored)}",
              flush=True)
        refined = checkpointed(pool, _refine, [(fit, p, iters, f"{tag}_refine{j}_scores.jsonl")
                                               for j, (_, p) in enumerate(scored[:keep])],
                               f"{tag}_refine.jsonl")
        print(f"{kind}: refined {keep} in {time.time() - t:.0f}s total; stability screen:", flush=True)
        bad, loss, p = screen(pool, refined)[0]
    print(f"{kind}: chose loss {loss:.3f} ({bad} runaways) after {time.time() - t:.0f}s")
    for part in ["search", "refine"] + [f"refine{j}_scores" for j in range(keep)]:
        Path(f"{tag}_{part}.jsonl").unlink(missing_ok=True)
    return loss, p, bad


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
    TRAIN_SEEDS (the first fit may use fewer, which can overfit to lucky seeds).
    The refined settings replace the saved ones only if they pass the stability
    screen at least as well. Returns (loss, settings, runaways)."""
    from scipy.optimize import minimize
    start = load(kind)
    with Pool(workers) as pool:
        fit = _PoolFit(kind, pool, TRAIN_SEEDS)
        first = fit.loss(start)
        res = minimize(fit.loss_of_vector, to_vector(start, fit.ranges), method="Nelder-Mead",
                       options=dict(maxfev=iters, xatol=1e-3, fatol=1e-3))
        print(f"{kind}: loss on all {len(TRAIN_SEEDS)} training seeds {first:.3f} -> {res.fun:.3f}; "
              "stability screen:", flush=True)
        bad, loss, p = screen(pool, [(float(res.fun), from_vector(res.x, fit.ranges)),
                                     (float(first), start)])[0]
    return loss, p, bad


def save(kind, loss, p, seeds=None, runaways=None, draws=None):
    SETTINGS_DIR.mkdir(exist_ok=True)
    seeds = fit_seeds(kind) if seeds is None else seeds
    out = dict(kind=kind, loss=loss, train_seeds=list(seeds), burn=BURN,
               fitted=list(SPX_RANGES if kind == "spx" else RANGES), settings=p)
    if draws is not None:
        out["draws"] = int(draws)
    if runaways is not None:
        out["stability"] = dict(seeds=[STABILITY_SEEDS[0], STABILITY_SEEDS[-1]],
                                days=STABILITY_DAYS, runaways=int(runaways))
    path = settings_path(kind)
    path.write_text(json.dumps(out, indent=1) + "\n")
    return path


def load(kind):
    """Fitted settings saved by `python3 -m herding.calibrate <kind>`."""
    path = settings_path(kind)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run: python3 -m herding.calibrate {kind}")
    p = json.loads(path.read_text())["settings"]
    model.check(p)
    return p


USAGE = "usage: python3 -m herding.calibrate paper|spx [--refine] [--draws N]"


def parse_args(argv):
    """The command line (argv without the program name) as (kind, refine, draws);
    exits with the usage line if it makes no sense. Kept apart from running the fit,
    so a test can check the arguments without starting hours of work."""
    which = argv[0] if argv else ""
    if which not in ("paper", "spx"):
        sys.exit(USAGE)
    refine, draws = "--refine" in argv, 1200
    if "--draws" in argv:
        if refine:
            sys.exit(USAGE + "  (--draws is for a new fit; --refine starts from the saved settings)")
        i = argv.index("--draws") + 1
        if i >= len(argv) or not argv[i].isdigit() or int(argv[i]) < 1:
            sys.exit(USAGE + "  (--draws needs a whole number of at least 1)")
        draws = int(argv[i])
    return which, refine, draws


if __name__ == "__main__":
    which, refine, n = parse_args(sys.argv[1:])
    if refine:
        loss, p, bad = refine_saved(which)
        print("saved", save(which, loss, p, TRAIN_SEEDS, bad))
    else:
        loss, p, bad = run(which, draws=n)
        print("saved", save(which, loss, p, runaways=bad, draws=n))
