"""Small helpers shared by several test files (conftest.py puts tests/ on the path)."""


class SerialPool:
    """Stands in for multiprocessing.Pool: runs every job in this process, in order,
    so tests stay small and their results are the same as with a real pool."""

    def map(self, fn, jobs, chunksize=None):
        return [fn(j) for j in jobs]

    def imap(self, fn, jobs, chunksize=1):
        return (fn(j) for j in jobs)


# ------------------------------------------------- noisy checks with measured tolerances
# Each recipe makes one random sample (from a seed) and returns the numbers a test checks.
# The tests and scripts/analysis/test_accuracy_audit.py use these same recipes: the audit runs
# each on 200 other seeds, measures how far the numbers fall from the exact answer (bias) and
# how much they vary (sd), and sets the tolerance to |bias| + 5 sd (TOLERANCE below). A test
# then fails by bad luck less than once in a million runs, but a real error of that size fails.
import numpy as np
from scipy.signal import lfilter

from herding import stats


def ar1(phi, e):
    """x_t = phi * x_(t-1) + e_t with x_0 = e_0 (lfilter: the same recursion, in C)."""
    return lfilter([1.0], [1.0, -phi], e)


def regime_vol(n_blocks, g):
    """|x| memory process: volatility 1 or 3, redrawn every 100 days, times a bell curve."""
    vol = np.repeat(g.choice([1.0, 3.0], n_blocks), 100)
    return vol * g.standard_normal(len(vol))


def exact_tail_beyond_2():
    """P(|Z| > 2) for a standard bell curve, from a 50-digit erfc."""
    import mpmath
    mpmath.mp.dps = 50
    return float(mpmath.erfc(2 / mpmath.sqrt(2)))


# lag-50 autocorrelation of |regime_vol|, exactly (see test_stats.py)
REGIME_AC50 = (2 / np.pi) * 0.5 / (5.0 - (2.0 * np.sqrt(2 / np.pi)) ** 2)
FP_SERIES = 16                    # bell-curve fingerprint: average of 16 series of 200,000 days


def _bell_fingerprint(seed):
    g = np.random.default_rng(seed)
    fps = [stats.fingerprint(0.01 * g.standard_normal(200_000)) for _ in range(FP_SERIES)]
    return {k: float(np.mean([f[k] for f in fps])) for k in fps[0]}


RECIPES = {   # name: (seed -> {check: value}, {check: exact answer})
    "kurtosis_normal": (lambda s: {"k": stats.kurtosis(np.random.default_rng(s).standard_normal(800_000))}, {"k": 0.0}),
    "kurtosis_uniform": (lambda s: {"k": stats.kurtosis(np.random.default_rng(s).uniform(-1, 1, 800_000))}, {"k": -1.2}),
    "kurtosis_laplace": (lambda s: {"k": stats.kurtosis(np.random.default_rng(s).laplace(0, 1, 2_000_000))}, {"k": 3.0}),
    "skew_exponential": (lambda s: {"s": stats.skewness(np.random.default_rng(s).exponential(size=1_600_000))}, {"s": 2.0}),
    "skew_normal": (lambda s: {"s": stats.skewness(np.random.default_rng(s).standard_normal(1_600_000))}, {"s": 0.0}),
    **{f"hill_pareto_{a}": ((lambda a: lambda s: (lambda g: {"a": stats.hill_alpha(
        (g.pareto(a, 400_000) + 1) * g.choice([-1, 1], 400_000))})(np.random.default_rng(s)))(a), {"a": a})
       for a in (1.5, 3.0, 5.0)},
    "hill_t3": (lambda s: {"a": stats.hill_alpha(np.random.default_rng(s).standard_t(3, 2_000_000), tail=0.0025)},
                {"a": 3.0}),
    "ccdf_normal_2": (lambda s: {"p": float((lambda z, p: np.interp(2.0, z, p))(
        *stats.ccdf(np.random.default_rng(s).standard_normal(800_000))))}, {"p": exact_tail_beyond_2()}),
    **{f"acf_ar1_{phi}": ((lambda phi: lambda s: dict(zip(("l1", "l2", "l3"), stats.acf(
        ar1(phi, np.random.default_rng(s).standard_normal(800_000)), [1, 2, 3]))))(phi),
        {"l1": phi, "l2": phi ** 2, "l3": phi ** 3}) for phi in (0.5, -0.3, 0.8)},
    "acf_noise": (lambda s: dict(zip(("l1", "l2", "l10", "l100"),
                                     stats.acf(np.random.default_rng(s).standard_normal(400_000), [1, 2, 10, 100]))),
                  {"l1": 0.0, "l2": 0.0, "l10": 0.0, "l100": 0.0}),
    "hurst_white": (lambda s: {"H": stats.hurst(np.random.default_rng(s).standard_normal(400_000))}, {"H": 0.5}),
    "hurst_walk": (lambda s: {"H": stats.hurst(np.cumsum(np.random.default_rng(s).standard_normal(400_000)))}, {"H": 1.5}),
    "regime_ac50": (lambda s: (lambda x, g: {"ac50": stats.acf(np.abs(x), [50])[0],
                                             "shuffled_ac50": stats.acf(np.abs(g.permutation(x)), [50])[0]})(
        *(lambda g: (regime_vol(16_000, g), g))(np.random.default_rng(s))), {"ac50": REGIME_AC50, "shuffled_ac50": 0.0}),
    "bell_fingerprint": (_bell_fingerprint, {"sd": 0.01, "kurtosis": 0.0, "kurt_5": 0.0, "kurt_25": 0.0,
                                             "ac_r_1": 0.0, "ac_abs_1": 0.0, "ac_abs_50": 0.0, "H": 0.5, "H_abs": 0.5}),
}

def run_recipe(job):
    """(name, seed) -> (name, seed, the recipe's numbers); a top-level function, so worker
    processes can run it."""
    name, seed = job
    return name, seed, RECIPES[name][0](seed)


# |bias| + 5 sd over 200 seeds (1000-1199), rounded up; written by test_accuracy_audit.py
TOLERANCE = {
    'kurtosis_normal.k': 0.029,
    'kurtosis_uniform.k': 0.0068000000000000005,
    'kurtosis_laplace.k': 0.13,
    'skew_exponential.s': 0.034,
    'skew_normal.s': 0.011,
    'hill_pareto_1.5.a': 0.054,
    'hill_pareto_3.0.a': 0.11,
    'hill_pareto_5.0.a': 0.18,
    'hill_t3.a': 0.27,
    'ccdf_normal_2.p': 0.0008500000000000001,
    'acf_ar1_0.5.l1': 0.005,
    'acf_ar1_0.5.l2': 0.0066,
    'acf_ar1_0.5.l3': 0.007,
    'acf_ar1_-0.3.l1': 0.0055000000000000005,
    'acf_ar1_-0.3.l2': 0.006200000000000001,
    'acf_ar1_-0.3.l3': 0.0057,
    'acf_ar1_0.8.l1': 0.0033,
    'acf_ar1_0.8.l2': 0.0056,
    'acf_ar1_0.8.l3': 0.0073,
    'acf_noise.l1': 0.008400000000000001,
    'acf_noise.l2': 0.0086,
    'acf_noise.l10': 0.0082,
    'acf_noise.l100': 0.008700000000000001,
    'hurst_white.H': 0.021,
    'hurst_walk.H': 0.037,
    'regime_ac50.ac50': 0.007,
    'regime_ac50.shuffled_ac50': 0.004,
    'bell_fingerprint.sd': 2.1e-05,
    'bell_fingerprint.kurtosis': 0.015,
    'bell_fingerprint.ac_r_1': 0.0028,
    'bell_fingerprint.ac_abs_1': 0.0028,
    'bell_fingerprint.ac_abs_50': 0.0029000000000000002,
    'bell_fingerprint.H': 0.019,
    'bell_fingerprint.H_abs': 0.018000000000000002,
    'bell_fingerprint.kurt_5': 0.033,
    'bell_fingerprint.kurt_25': 0.076,
}
