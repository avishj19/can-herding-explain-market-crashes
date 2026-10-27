"""Kirman's (1993) ant model: the herding mechanism behind Lux-Marchesi.

N ants feed at two identical food sources, A and B. Each step one random ant
either switches source on its own (probability eps), or meets another random
ant and is recruited to that ant's source (probability 1 - delta).

Because only one ant moves per step, the number k at source A is a birth-death
chain, so its long-run distribution is known exactly (detailed balance):
    pi(k+1) / pi(k) = up(k) / down(k+1)
Kirman's result: if eps < (1 - delta) / (N - 1) the ants herd (most of them at
one source, with sudden mass switches); otherwise they split about 50/50.
"""
import numpy as np

try:
    import numba
except ImportError:
    numba = None


def check(N, eps, delta):
    if int(N) != N or N < 2:
        raise ValueError(f"N must be a whole number >= 2, got {N}")
    if not 0 < eps <= 1:
        raise ValueError(f"eps must be in (0, 1], got {eps}")
    if not 0 <= delta <= 1:
        raise ValueError(f"delta must be in [0, 1], got {delta}")


def up_prob(k, N, eps, delta):
    """P(k -> k+1): a B ant is picked and joins A (alone or recruited by an A ant)."""
    return (N - k) / N * (eps + (1 - delta) * k / (N - 1))


def down_prob(k, N, eps, delta):
    """P(k -> k-1): an A ant is picked and joins B."""
    return k / N * (eps + (1 - delta) * (N - k) / (N - 1))


def stationary(N, eps, delta):
    """Exact long-run probability of k = 0..N ants at source A."""
    check(N, eps, delta)
    k = np.arange(N)
    ratio = up_prob(k, N, eps, delta) / down_prob(k + 1, N, eps, delta)
    logpi = np.concatenate([[0.0], np.cumsum(np.log(ratio))])
    pi = np.exp(logpi - logpi.max())
    return pi / pi.sum()


def herds(N, eps, delta):
    """Kirman's condition: True if the ants crowd at one source (bimodal)."""
    check(N, eps, delta)
    return eps < (1 - delta) / (N - 1)


def _run_impl(g, N, eps, delta, steps, k):
    out = np.empty(steps + 1, dtype=np.int64)
    out[0] = k
    for t in range(steps):
        u = g.random()
        if u < up_prob(k, N, eps, delta):
            k += 1
        elif u < up_prob(k, N, eps, delta) + down_prob(k, N, eps, delta):
            k -= 1
        out[t + 1] = k
    return out


if numba is not None:
    up_prob = numba.njit(cache=True)(up_prob)
    down_prob = numba.njit(cache=True)(down_prob)
    _run = numba.njit(cache=True)(_run_impl)
else:
    _run = _run_impl


def simulate(N, eps, delta, steps, seed=None, k0=None):
    """Number of ants at source A after each step (length steps + 1)."""
    check(N, eps, delta)
    if int(steps) != steps or steps < 0:
        raise ValueError(f"steps must be a whole number >= 0, got {steps}")
    k0 = N // 2 if k0 is None else k0
    if not 0 <= k0 <= N:
        raise ValueError(f"k0 must be between 0 and N, got {k0}")
    return _run(np.random.default_rng(seed), int(N), float(eps), float(delta),
                int(steps), int(k0))
