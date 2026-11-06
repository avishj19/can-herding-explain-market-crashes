"""The Lux-Marchesi (1999) market model, built from the equations in notes/SPEC.md.

N traders are in one of three groups:
    fundamentalists  buy when the price is below the fundamental value, sell above it
    optimists        chartists who buy
    pessimists       chartists who sell
Traders switch groups at random, at rates that depend on the crowd (herding), the
price trend and the profits of each strategy. Excess demand (buyers minus sellers)
moves the price one tick at a time. News moves the fundamental value as Gaussian
noise, so any fat tails or volatility clustering must come from the traders.

Section numbers (sec. 1-5) refer to notes/SPEC.md.

Policy options (an extension, not part of the paper; SPEC.md "Policy experiments"):
circuit breakers that pause or close trading after a big fall within the day, and
one-off policy surprises that move the fundamental value.
"""
import os

import numpy as np

try:
    import numba
except ImportError:                    # still works without numba, just ~50x slower
    numba = None

# Settings. Every one has a fixed meaning in SPEC.md; rates are per unit of time
# (one unit = one trading day).
SETTINGS = {
    "N": "number of traders",
    "v1": "how often chartists reconsider their mood (sec. 1)",
    "v2": "how often traders reconsider their strategy (sec. 2)",
    "a1": "herding: weight of the majority mood (sec. 1)",
    "a2": "trend following: weight of the price trend (sec. 1)",
    "a3": "sensitivity to profit differences (sec. 2)",
    "s": "discount on fundamentalists' future profits, below 1 (sec. 2)",
    "R": "return on other investments per day; dividend yield r/p_f = R (sec. 2)",
    "tc": "trading volume of one chartist (sec. 3)",
    "gamma": "reaction strength of one fundamentalist (sec. 3)",
    "beta": "price speed: relative price change per day per unit of excess demand (sec. 3)",
    "sigma_mu": "standard deviation of the noise in excess demand (sec. 3)",
    "sigma_eps": "daily standard deviation of the news (sec. 4)",
    "tick": "size of one price tick, as a fraction of the price (sec. 3)",
    "dt": "time step, as a fraction of a day (sec. 5)",
    "floor": "minimum number of chartists and of fundamentalists (sec. 5)",
}

# Parameter set I as remembered from Lux & Marchesi (2000), UNVERIFIED (SPEC.md).
# T_c = N tc = 10 and T_f = N gamma = 5.
SET_I = dict(N=500, v1=3.0, v2=2.0, a1=0.6, a2=0.2, a3=0.5, s=0.75, R=0.0004,
             tc=0.02, gamma=0.01, beta=6.0, sigma_mu=0.05, sigma_eps=0.005,
             tick=0.001, dt=0.01, floor=4)

TREND_WINDOW = 0.1     # dp/dt is measured over the last 0.1 day (= 10 steps of 0.01) [S]
REFERENCE_DT = 0.01    # the time step of Lux & Marchesi's own simulations [S]
U_CAP = 30.0           # cap on the exponents U, so exp(U) cannot overflow
MAX_GAP = 1.0          # a runaway: the price more than e = 2.7x away from its value...
MAX_DAYS = 250         # ...for a whole year of trading days in a row,
HARD_GAP = 20.0        # or ever more than e^20 (5 hundred million) times away
START_CHARTISTS = 0.1  # share of chartists at the start, split evenly into the two moods
# US market-wide circuit breakers since 2013 (NYSE Rule 7.12, formerly Rule 80B;
# SEC-approved): a 15-minute pause when the S&P 500 is 7% or 13% below the previous
# close, but only before 3:25 pm (minute 355 of the day); closed for the day at 20%
# at any time.
US_BREAKERS = dict(levels=(0.07, 0.13, 0.20), halt_minutes=15, last_pause_minute=355)
MINUTES_PER_DAY = 390  # 9:30 to 16:00


# ------------------------------------------------------------------ checks

class RunawayError(RuntimeError):
    """The price left its value for good (all traders became chartists)."""

    def __init__(self, message, day):
        super().__init__(message)
        self.day = day


def check(p):
    """Raise ValueError unless p has exactly the known settings with sensible values."""
    if not isinstance(p, dict):
        raise ValueError(f"settings must be a dict, got {type(p).__name__}")
    unknown, missing = set(p) - set(SETTINGS), set(SETTINGS) - set(p)
    if unknown or missing:
        raise ValueError(f"unknown settings {sorted(unknown)}, missing settings {sorted(missing)}")
    for k, v in p.items():
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)) \
                or not np.isfinite(v):
            raise ValueError(f"setting {k} must be a finite number, got {v!r}")
        if v < 0:
            raise ValueError(f"setting {k} must not be negative, got {v}")
    # floor 0 would let the chartists die out: then opinion x = (n+ - n-)/n_c divides by 0
    for k in ("N", "v1", "v2", "beta", "tick", "dt", "floor"):
        if p[k] <= 0:
            raise ValueError(f"setting {k} must be above 0, got {p[k]}")
    for k in ("N", "floor"):
        if int(p[k]) != p[k]:
            raise ValueError(f"setting {k} must be a whole number, got {p[k]}")
    if not p["s"] < 1:
        raise ValueError(f"s must be below 1 (the paper's discount factor), got {p['s']}")
    if not p["tick"] < 1:
        raise ValueError(f"tick must be below 1 (a fraction of the price), got {p['tick']}")
    steps = round(1 / p["dt"])
    if abs(steps * p["dt"] - 1) > 1e-9:
        raise ValueError(f"dt must split one day into whole steps (0.01, 0.005, ...), got {p['dt']}")
    if TREND_WINDOW / p["dt"] < 1 - 1e-9:
        raise ValueError(f"dt must be at most {TREND_WINDOW} (the trend window), got {p['dt']}")
    if p["N"] < 4 * p["floor"]:
        raise ValueError(f"N={p['N']} is too small for floor={p['floor']}: need N >= 4*floor")


STEP_LIMIT = 2.0       # above this, the time step itself makes the price blow up


def step_pull(p):
    """How much of a mispricing the fundamentalists remove in ONE time step, at most
    (all N of them): beta * N * gamma * dt (from SPEC eq. 25 with ED_f = n_f gamma gap).
    Below 1 the price moves part of the way back to its value each step; between 1
    and 2 it overshoots but the swings shrink; at STEP_LIMIT (2) or more every step
    overshoots by more than the last, and the run explodes even without chartists.
    That is a flaw of stepping through time, not a bubble."""
    return p["beta"] * p["N"] * p["gamma"] * p["dt"]


# ------------------------------------------------------------------ one time step

def _leave(g, n, rate_a, rate_b, dt):
    """Of n traders each leaving at rate_a (to A) or rate_b (to B) per day, how many
    go to A and to B within dt. Exact for constant rates: a trader leaves with
    probability 1 - exp(-(rate_a + rate_b) dt), then picks A with chance
    rate_a / (rate_a + rate_b)."""
    total = rate_a + rate_b
    if n <= 0 or total <= 0.0:
        return 0, 0
    k = g.binomial(n, 1.0 - np.exp(-total * dt))
    to_a = g.binomial(k, rate_a / total)
    return to_a, k - to_a


def _cap(u):
    return min(U_CAP, max(-U_CAP, u))


def _simulate(g, days, steps, chartists, N, v1, v2, a1, a2, a3, s, R, tc, gamma,
              beta, sigma_mu, sigma_eps, tick, floor, n_plus, n_minus, max_gap, max_days,
              hard_gap, cb1, cb2, cb3, halt_steps, pause_end, shock):
    """The simulation loop (see SPEC.md). Returns per-day log price, log value,
    chartist share and opinion at each day's close, the circuit-breaker level of
    each day, and the number of days completed (fewer than `days` if the price ran
    away: |log price - log value| above max_gap for max_days days in a row, or ever
    above hard_gap).
    Circuit breakers: 0 > cb1 > cb2 > cb3 are log falls from the previous close
    (-inf = off). Falling to cb1 or cb2 within the first pause_end steps of the day
    freezes the price for halt_steps steps (once each per day; later in the day
    nothing happens); falling to cb3 freezes it for the rest of the day, at any time.
    Traders keep switching while the price is frozen.
    shock[day]: log change of the value at the start of that day (policy surprise)."""
    dt = 1.0 / steps
    window = int(round(TREND_WINDOW * steps))
    up_step, down_step = np.log(1.0 + tick), np.log(1.0 - tick)
    # mu is drawn fresh each step; scaled so its effect per day does not depend on
    # dt (identical to [S] at their dt = 0.01)
    mu_sd = sigma_mu * np.sqrt(REFERENCE_DT / dt)
    lp = 0.0                                   # log price; starts at the value
    lf = 0.0                                   # log fundamental value
    n_f = N - n_plus - n_minus
    past = np.zeros(window)                    # log prices of the last `window` steps
    out_lp = np.empty(days + 1)
    out_lf = np.empty(days + 1)
    out_z = np.empty(days + 1)
    out_x = np.empty(days + 1)
    out_lp[0], out_lf[0] = lp, lf
    nc = n_plus + n_minus
    out_z[0] = nc / N
    out_x[0] = (n_plus - n_minus) / nc if nc > 0 else 0.0
    halts = np.zeros(days, dtype=np.int64)     # 0 none, 1-2 pause levels, 3 closed
    t = 0
    outside = 0                                # days in a row beyond max_gap
    for day in range(days):
        lf += shock[day]
        prev_close = lp
        paused = 0                             # steps of pause left
        for k in range(steps):
            # dp/dt / p: relative price change per day over the trend window
            trend = (np.exp(lp - past[t % window]) - 1.0) / TREND_WINDOW
            past[t % window] = lp
            t += 1
            gap = np.exp(lf - lp) - 1.0        # (p_f - p) / p
            if chartists:
                nc = n_plus + n_minus
                # sec. 1: mood. sec. 2: strategy (profit of chartists vs fundamentalists)
                U1 = _cap(a1 * (n_plus - n_minus) / nc + a2 / v1 * trend)
                gain = R * (1.0 + gap) + trend / v2    # (r + (dp/dt)/v2) / p, r = R p_f
                U21 = _cap(a3 * (gain - R - s * abs(gap)))
                U22 = _cap(a3 * (R - gain - s * abs(gap)))
                mood = v1 * nc / N
                # every rate below is per trader per day (first: where they go)
                p_to_m, p_to_f = _leave(g, n_plus, mood * np.exp(-U1),
                                        v2 * n_f / N * np.exp(-U21), dt)
                m_to_p, m_to_f = _leave(g, n_minus, mood * np.exp(U1),
                                        v2 * n_f / N * np.exp(-U22), dt)
                f_to_p, f_to_m = _leave(g, n_f, v2 * n_plus / N * np.exp(U21),
                                        v2 * n_minus / N * np.exp(U22), dt)
                # floors (sec. 5): undo just enough moves to keep both groups alive
                short_f = floor - (n_f - f_to_p - f_to_m + p_to_f + m_to_f)
                while short_f > 0:
                    if f_to_p >= f_to_m:
                        f_to_p -= 1
                    else:
                        f_to_m -= 1
                    short_f -= 1
                short_c = floor - (nc - p_to_f - m_to_f + f_to_p + f_to_m)
                while short_c > 0:
                    if p_to_f >= m_to_f:
                        p_to_f -= 1
                    else:
                        m_to_f -= 1
                    short_c -= 1
                n_plus += m_to_p - p_to_m + f_to_p - p_to_f
                n_minus += p_to_m - m_to_p + f_to_m - m_to_f
                n_f += p_to_f + m_to_f - f_to_p - f_to_m
            # sec. 3: excess demand -> random up and down ticks
            ed = (n_plus - n_minus) * tc + n_f * gamma * gap
            # ticks per day = beta (ED + mu) / tick, so that on average
            # d(ln p)/dt = beta (ED + mu), SPEC eq. 25
            push = beta * (ed + mu_sd * g.standard_normal()) / tick
            if halts[day] == 3 or paused > 0:  # trading halted: the price is frozen
                paused = max(0, paused - 1)
            else:
                if push > 0.0:
                    lp += g.poisson(push * dt) * up_step
                else:
                    lp += g.poisson(-push * dt) * down_step
                # circuit breakers: each level fires at most once a day
                fall = lp - prev_close
                if fall <= cb3:
                    halts[day] = 3
                elif fall <= cb2 and halts[day] < 2 and k < pause_end:
                    halts[day] = 2
                    paused = halt_steps
                elif fall <= cb1 and halts[day] < 1 and k < pause_end:
                    halts[day] = 1
                    paused = halt_steps
            # sec. 4: news
            lf += sigma_eps * np.sqrt(dt) * g.standard_normal()
        nc = n_plus + n_minus
        out_lp[day + 1], out_lf[day + 1] = lp, lf
        out_z[day + 1] = nc / N
        out_x[day + 1] = (n_plus - n_minus) / nc if nc > 0 else 0.0
        outside = outside + 1 if abs(lp - lf) > max_gap else 0
        if outside >= max_days or abs(lp - lf) > hard_gap:
            return out_lp, out_lf, out_z, out_x, halts, day + 1
    return out_lp, out_lf, out_z, out_x, halts, days


USE_NUMBA = numba is not None and os.environ.get("MODEL_NO_NUMBA") != "1"
if USE_NUMBA:
    _leave = numba.njit(cache=True)(_leave)
    _cap = numba.njit(cache=True)(_cap)
    _simulate = numba.njit(cache=True)(_simulate)


def check_breakers(breakers):
    """Circuit-breaker rules: dict(levels=(l1, l2, l3), halt_minutes=m), and optionally
    last_pause_minute=c. The levels are falls below the previous close (0.07 = 7%),
    increasing; l1 and l2 pause trading for m minutes of a 390-minute day, but only
    before minute c (no limit if c is left out); l3 closes the day at any time.
    None = no breakers."""
    if breakers is None:
        return
    if not isinstance(breakers, dict) or not {"levels", "halt_minutes"} <= set(breakers) \
            or not set(breakers) <= {"levels", "halt_minutes", "last_pause_minute"}:
        raise ValueError("breakers needs 'levels' and 'halt_minutes' (and optionally "
                         f"'last_pause_minute'), got {breakers!r}")
    c = breakers.get("last_pause_minute", MINUTES_PER_DAY)
    if not _is_number(c) or not 0 <= c <= MINUTES_PER_DAY:
        raise ValueError(f"last_pause_minute must be a number from 0 to {MINUTES_PER_DAY}, got {c!r}")
    lv = tuple(breakers["levels"])
    if len(lv) != 3 or not all(_is_number(x) and 0 < x < 1 for x in lv) \
            or not lv[0] < lv[1] < lv[2]:
        raise ValueError(f"breaker levels must be 3 increasing falls between 0 and 1, got {lv}")
    m = breakers["halt_minutes"]
    if not _is_number(m) or not 0 < m <= MINUTES_PER_DAY:
        raise ValueError(f"halt_minutes must be a number from 0 to {MINUTES_PER_DAY}, got {m!r}")


def check_shocks(shocks, days):
    """Policy surprises: {day: fraction}, e.g. {200: -0.1} = at the start of day 200
    a surprise cuts the fundamental value by 10%. None = no surprises."""
    if shocks is None:
        return
    if not isinstance(shocks, dict):
        raise ValueError(f"shocks must be a dict {{day: fraction}}, got {shocks!r}")
    for day, x in shocks.items():
        if isinstance(day, (bool, np.bool_)) or not isinstance(day, (int, np.integer)) \
                or not 0 <= day < days:
            raise ValueError(f"shock day must be a whole number from 0 to {days - 1}, got {day!r}")
        if not _is_number(x) or not x > -1:
            raise ValueError(f"shock on day {day} must be a number above -1 (a fraction), got {x!r}")


def pause_steps(minutes, dt):
    """A pause of `minutes` (of a 390-minute day) in time steps of dt days: at least 1."""
    return max(1, int(round(minutes / MINUTES_PER_DAY / dt)))


def minute_to_step(minute, dt):
    """The time step (counting from 0 at 9:30) that a minute of the day falls in:
    steps 0 .. minute_to_step(c) - 1 are the ones before minute c."""
    return int(round(minute / MINUTES_PER_DAY / dt))


def _is_number(x):
    return isinstance(x, (int, float, np.number)) and not isinstance(x, (bool, np.bool_)) \
        and bool(np.isfinite(x))


def simulate(p, days, seed=None, chartists=True, max_gap=MAX_GAP, max_days=MAX_DAYS,
             breakers=None, shocks=None):
    """Run the market for `days` trading days with settings p.

    Returns a dict of daily series (one value per day's close, days + 1 values):
        lp, lf  log price and log fundamental value (both start at 0)
        z       share of traders who are chartists
        x       chartists' opinion, (optimists - pessimists) / chartists
    plus r, the daily log returns (days values), and halts, the circuit-breaker
    level reached each day (days values: 0 none, 1 or 2 a pause, 3 closed early).
    chartists=False runs the "no chartists" check: everybody is a fundamentalist.
    breakers (check_breakers, e.g. US_BREAKERS) and shocks (check_shocks) are the
    policy options; both off by default, which leaves the model exactly as in SPEC.md.

    Raises RunawayError if the price runs away: more than a factor e^max_gap from
    its value for max_days trading days in a row, or ever more than e^HARD_GAP
    (the paper: no *persistent* pathological mispricing). Bubbles that burst on
    their own are kept. max_gap=None switches the check off.
    """
    check(p)
    if isinstance(days, (bool, np.bool_)) or not isinstance(days, (int, np.integer)) or days < 1:
        raise ValueError(f"days must be a whole number >= 1, got {days!r}")
    if not isinstance(chartists, (bool, np.bool_)):
        raise ValueError(f"chartists must be True or False, got {chartists!r}")
    N, floor = int(p["N"]), int(p["floor"])
    each = max(int(N * START_CHARTISTS) // 2, floor) if chartists else 0
    if max_gap is not None and not (isinstance(max_gap, (int, float, np.number))
                                    and not isinstance(max_gap, (bool, np.bool_)) and max_gap > 0):
        raise ValueError(f"max_gap must be a number above 0, or None, got {max_gap!r}")
    if isinstance(max_days, (bool, np.bool_)) or not isinstance(max_days, (int, np.integer)) \
            or max_days < 1:
        raise ValueError(f"max_days must be a whole number >= 1, got {max_days!r}")
    check_breakers(breakers)
    check_shocks(shocks, days)
    if breakers is None:
        cb, halt_steps, pause_end = (-np.inf, -np.inf, -np.inf), 0, 0
    else:
        cb = tuple(float(np.log(1.0 - x)) for x in breakers["levels"])
        halt_steps = pause_steps(breakers["halt_minutes"], p["dt"])
        pause_end = minute_to_step(breakers.get("last_pause_minute", MINUTES_PER_DAY), p["dt"])
    shock = np.zeros(int(days))
    for day, frac in (shocks or {}).items():
        shock[day] += np.log1p(frac)
    lp, lf, z, x, halts, done = _simulate(
        np.random.default_rng(seed), int(days), int(round(1 / p["dt"])), bool(chartists),
        N, *(float(p[k]) for k in ("v1", "v2", "a1", "a2", "a3", "s", "R", "tc", "gamma",
                                   "beta", "sigma_mu", "sigma_eps", "tick")),
        floor, each, each, np.inf if max_gap is None else float(max_gap), int(max_days),
        np.inf if max_gap is None else HARD_GAP, *cb, int(halt_steps), int(pause_end), shock)
    if done < days:
        ratio = np.exp(min(abs(lp[done] - lf[done]), 700.0))   # 700: largest safe exponent
        raise RunawayError(f"day {done}: the price is {ratio:.3g}x away from its value with "
                           f"{z[done]:.0%} chartists and has not come back (a runaway "
                           f"{'bubble' if lp[done] > lf[done] else 'crash'})", done)
    return dict(r=np.diff(lp), lp=lp, lf=lf, z=z, x=x, halts=halts)
