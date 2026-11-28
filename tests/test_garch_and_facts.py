"""Tests of herding/garch.py (a formula with no traders) and the textbook checklist in herding/explore.py
(FACTS, fact_values, fact_passes, fact_table; Cont 2001), and of check 6 on the Checks page."""
import numpy as np
import pytest

from herding import explore as ex, garch, market_data
from test_web_pages import open_page  # noqa: F401

NUMS = ex.web_numbers()
SPX = market_data.spx_returns().values
TRUE = dict(mu=0.0003, omega=2e-6, alpha=0.03, gamma=0.1, beta=0.9)


# ------------------------------------------------------------------ garch

# Spec: variances() is the recursion written out as a plain loop (it runs it with lfilter).
def test_variances_equal_a_plain_loop():
    r = garch.simulate(TRUE, 3000, seed=2)
    want = np.empty(r.size)
    e = r - TRUE["mu"]
    want[0] = e.var()
    for t in range(1, r.size):
        want[t] = (TRUE["omega"] + (TRUE["alpha"] + TRUE["gamma"] * (e[t - 1] < 0)) * e[t - 1] ** 2
                   + TRUE["beta"] * want[t - 1])
    assert garch.variances(r, *(TRUE[k] for k in garch.NAMES)) == pytest.approx(want, rel=1e-12)


# Spec: the log-likelihood by hand for 3 numbers: sum of log(2 pi s2) + e^2 / s2, halved, with a sign flip.
def test_neg_loglik_by_hand():
    r = np.r_[np.zeros(97), [0.01, -0.02, 0.005]]
    th = (0.0, 1e-5, 0.1, 0.05, 0.8)
    s2 = garch.variances(r, *th)
    want = 0.5 * np.sum(np.log(2 * np.pi * s2) + r ** 2 / s2)
    assert garch.neg_loglik(th, r) == pytest.approx(want, rel=1e-12)


# Spec: fitting data simulated from known settings gets them back. Tolerance |bias| + 5 sd, as in
# helpers.py, measured over seeds 100-119 at 20,000 days (Nov 2026): errors had sd 0.0041 (alpha),
# 0.0044 (gamma), 0.0041 (beta), biases under 0.001.
@pytest.mark.parametrize("seed", [1, 7])
def test_fit_recovers_known_settings(seed):
    got = garch.fit(garch.simulate(TRUE, 20000, seed=seed))
    assert got["alpha"] == pytest.approx(TRUE["alpha"], abs=0.021)
    assert got["gamma"] == pytest.approx(TRUE["gamma"], abs=0.023)
    assert got["beta"] == pytest.approx(TRUE["beta"], abs=0.021)


# Spec: simulate repeats with the same seed; its long-run variance is omega / (1 - alpha - gamma/2 - beta).
def test_simulate_repeats_and_has_the_right_variance():
    a = garch.simulate(TRUE, 200000, seed=3)
    assert np.array_equal(a, garch.simulate(TRUE, 200000, seed=3))
    want = TRUE["omega"] / (1 - TRUE["alpha"] - TRUE["gamma"] / 2 - TRUE["beta"])
    assert a.var() == pytest.approx(want, rel=0.15)


@pytest.mark.parametrize("bad", [dict(TRUE, beta=0.99), dict(TRUE, omega=0.0), {"mu": 0}, dict(TRUE, mu="x")])
def test_simulate_rejects_bad_params(bad):
    with pytest.raises(ValueError):
        garch.simulate(bad, 10)


def test_fit_rejects_bad_returns():
    for bad in ([0.1] * 10, np.r_[np.zeros(200), np.nan]):
        with pytest.raises(ValueError):
            garch.fit(bad)


# Spec: the saved fit is the fit of the S&P 500 file (rerun it).
def test_saved_garch_fit_reruns():
    got = garch.fit(SPX)
    for k in garch.NAMES:
        assert got[k] == pytest.approx(NUMS["facts"]["garch_params"][k], rel=1e-9, abs=1e-12)


# ------------------------------------------------------------------ the checklist

# Spec: a plain bell curve passes only "no trend" and "tails thin out", as FACTS promises.
def test_a_bell_curve_passes_only_the_bell_curve_patterns():
    for seed in range(3):
        r = np.random.default_rng(seed).normal(0, 0.01, 20000)
        got = ex.fact_passes(ex.fact_values(r))
        assert {k for k, v in got.items() if v} <= {"no_trend", "thin_out"}, (seed, got)


# Spec: each measure by hand on a constructed series.
def test_fact_values_by_hand():
    r = np.random.default_rng(1).normal(0, 0.01, 5000)
    v = ex.fact_values(r)
    z = (r - r.mean()) / r.std()
    lo, hi = np.quantile(z, [0.01, 0.99])
    assert v["skew"] == pytest.approx(-lo / hi, rel=1e-12)
    month = r[:5000].reshape(-1, 20).sum(axis=1)
    from herding import stats
    assert v["thin_out"] == pytest.approx(stats.kurtosis(month) - stats.kurtosis(r), rel=1e-12)
    e = r - r.mean()
    lev = np.mean([np.corrcoef(r[:-k], np.abs(e)[k:])[0, 1] for k in range(1, 11)])
    assert v["leverage"] == pytest.approx(lev, rel=1e-12)
    sd = np.array([np.sqrt(np.mean(e[t - 20:t] ** 2)) for t in range(20, r.size)])
    assert v["fat_in_calm"] == pytest.approx(stats.kurtosis(e[20:] / sd), rel=1e-9)


# Spec: a market whose falls are bigger passes "falls bigger than rises"; its mirror image does not.
def test_skew_rule_sees_bigger_falls():
    r = np.random.default_rng(2).standard_t(4, 20000) * 0.01
    r = np.where(r < 0, 1.3 * r, r)
    assert ex.fact_passes(ex.fact_values(r))["skew"]
    assert not ex.fact_passes(ex.fact_values(-r))["skew"]


def test_fact_values_rejects_short_or_bad():
    for bad in ([0.01] * 100, np.r_[np.zeros(600), np.inf]):
        with pytest.raises(ValueError):
            ex.fact_values(bad)


# Spec: the S&P 500 passes every pattern (these are the patterns of real markets).
def test_the_sp500_passes_the_whole_checklist():
    assert all(ex.fact_passes(ex.fact_values(SPX)).values())


# Spec: the saved herding numbers rerun exactly for one held-out seed.
def test_saved_fact_values_rerun():
    p, burn = ex.fitted_settings()
    r, _ = ex.run_mix(p, "herding", p["a1"], NUMS["recipe"]["days"], NUMS["recipe"]["seeds"][0], burn)
    got = ex.fact_values(r)
    for k in ex.FACTS:
        assert got[k] == pytest.approx(NUMS["facts"]["herding"][k][0], rel=1e-12, abs=1e-15)
    g = garch.simulate(NUMS["facts"]["garch_params"], NUMS["recipe"]["days"], seed=NUMS["facts"]["garch_seeds"][0])
    got = ex.fact_values(g)
    for k in ex.FACTS:
        assert got[k] == pytest.approx(NUMS["facts"]["garch"][k][0], rel=1e-12, abs=1e-15)


# Spec: fact_table counts passes from the saved values (Nov 2026: herding 5, GARCH 6 of 8 in most histories).
def test_fact_table_counts():
    rows = ex.fact_table(NUMS, SPX)
    assert [r["key"] for r in rows] == list(ex.FACTS)
    for r in rows:
        test = ex.FACTS[r["key"]][3]
        assert r["herding_pass"] == sum(bool(test(v)) for v in NUMS["facts"]["herding"][r["key"]])
    assert sum(r["herding_pass"] > 20 for r in rows) == 5
    assert sum(r["garch_pass"] > 20 for r in rows) == 6


# Spec: the Checks page shows the textbook checklist (its "Beyond the paper" tab since the redesign) with one
# row per pattern; the scoreboard keeps the proposal's five checks only.
def test_checks_page_shows_the_checklist(open_page):
    at = open_page("web/page_checks.py")
    assert not at.exception
    heads = [h.value for h in at.header]
    assert heads[5].startswith("The textbook checklist")
    table = at.table[1].value
    assert list(table.index) == [v[0] for v in ex.FACTS.values()]
    assert len(at.table[0].value) == 5
    text = " ".join(m.value for m in at.markdown)
    assert "the model passes **5**" in text
