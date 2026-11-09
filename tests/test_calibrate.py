"""Tests for herding/calibrate.py (the two fits: paper and S&P 500)."""
import json

import numpy as np
import pytest

from herding import calibrate, model, stats
from helpers import SerialPool


def test_vector_round_trip_is_exact():
    p = calibrate.draw(np.random.default_rng(0), calibrate.SPX_RANGES)
    back = calibrate.from_vector(calibrate.to_vector(p, calibrate.SPX_RANGES), calibrate.SPX_RANGES)
    for k in calibrate.SPX_RANGES:
        assert back[k] == pytest.approx(p[k], rel=1e-12)


def test_log_settings_use_log_coordinates():
    p = dict(calibrate.base(), beta=np.e ** 2)
    v = calibrate.to_vector(p, calibrate.RANGES)
    assert v[list(calibrate.RANGES).index("beta")] == pytest.approx(2.0)


def test_from_vector_clips_into_the_ranges():
    big = np.full(len(calibrate.RANGES), 50.0)
    p = calibrate.from_vector(big, calibrate.RANGES)
    assert all(p[k] == hi for k, (lo, hi, _) in calibrate.RANGES.items())


def test_draws_stay_inside_the_ranges_and_fill_them():
    rng = np.random.default_rng(1)
    ps = [calibrate.draw(rng, calibrate.RANGES) for _ in range(2000)]
    for k, (lo, hi, kind) in calibrate.RANGES.items():
        v = np.array([p[k] for p in ps])
        assert v.min() >= lo and v.max() <= hi
        mid = np.sqrt(lo * hi) if kind == "log" else (lo + hi) / 2   # the median of the draw
        assert abs(np.mean(v < mid) - 0.5) < 0.05


def test_draws_are_valid_settings_and_keep_the_fixed_ones():
    p = calibrate.draw(np.random.default_rng(2), calibrate.SPX_RANGES)
    model.check(p)
    for k in set(model.SET_I) - set(calibrate.SPX_RANGES):
        assert p[k] == model.SET_I[k]


def test_paper_loss_is_zero_on_target_and_counts_tolerances():
    fp = {k: np.array([v]) for k, (v, _) in calibrate.PAPER_TARGETS.items()}
    fp["kurt_25"] = np.array([0.5])
    assert calibrate.paper_loss(fp) == 0
    fp["H"] = np.array([0.48 + 2 * 0.03])                 # two tolerances off -> 4
    assert calibrate.paper_loss(fp) == pytest.approx(4.0)
    fp["kurt_25"] = np.array([3.0])                       # 2 above the allowed 1 -> + 4
    assert calibrate.paper_loss(fp) == pytest.approx(8.0)


def test_runaways_get_the_worst_score():
    assert calibrate.paper_loss(None) == calibrate.RUNAWAY_LOSS
    assert calibrate.spx_loss(None, {}, {}) == calibrate.RUNAWAY_LOSS


def test_spx_loss_uses_the_median_and_the_fixed_tolerance():
    fp = {k: np.array([0.0, 1.0, 50.0]) for k in calibrate.SPX_TARGETS}   # median 1
    tol = {k: 0.5 for k in calibrate.SPX_TARGETS}
    on = {k: 1.0 for k in calibrate.SPX_TARGETS}
    off = {k: 2.0 for k in calibrate.SPX_TARGETS}                         # 2 tolerances away
    assert calibrate.spx_loss(fp, on, tol) == 0
    assert calibrate.spx_loss(fp, off, tol) == pytest.approx(4.0)


def test_a_wilder_model_does_not_score_better():
    # The bug in the first S&P fit: dividing by the model's own spread rewarded
    # wild models. With a fixed tolerance, more spread around the same median
    # gives the same score, and a worse median gives a worse one.
    tol = {k: 1.0 for k in calibrate.SPX_TARGETS}
    target = {k: 0.0 for k in calibrate.SPX_TARGETS}
    calm = {k: np.array([1.9, 2.0, 2.1]) for k in calibrate.SPX_TARGETS}
    wild = {k: np.array([-40.0, 2.0, 90.0]) for k in calibrate.SPX_TARGETS}
    assert calibrate.spx_loss(calm, target, tol) == calibrate.spx_loss(wild, target, tol)


def test_block_bootstrap_keeps_blocks_whole():
    r = np.arange(1000, dtype=float)      # each value tells where it came from
    import herding.calibrate as c
    seen = []
    orig = c.stats.fingerprint
    c.stats.fingerprint = lambda x: seen.append(np.array(x)) or {}
    try:
        c.block_bootstrap(r, block=100, draws=3)
    finally:
        c.stats.fingerprint = orig
    for x in seen:
        assert len(x) == 1000
        d = np.diff(x.reshape(10, 100), axis=1) % 1000
        assert np.all(d == 1)             # consecutive days inside every block


def test_spx_tolerance_is_positive_and_independent_of_the_model():
    from herding import market_data
    tol = calibrate.spx_tolerance(market_data.spx_returns().values)
    assert set(tol) == set(calibrate.SPX_TARGETS) and all(v > 0 for v in tol.values())
    assert tol["alpha_reg"] == pytest.approx(0.09, abs=0.02)


def test_held_out_seeds_are_never_used_for_fitting():
    assert not set(calibrate.TRAIN_SEEDS) & set(calibrate.HELD_OUT_SEEDS)


def test_unknown_fit_kind_is_rejected():
    with pytest.raises(ValueError, match="paper"):
        calibrate.Fit("nasdaq")


def test_spx_fit_targets_the_whole_fingerprint_of_the_real_data():
    fit = calibrate.Fit("spx")
    assert set(fit.target) == set(stats.FINGERPRINT)
    assert fit.days == 19329 and fit.target["kurtosis"] > 20


def test_runaway_settings_score_as_runaways():
    fit = calibrate.Fit("paper")
    fit.days, fit.seeds = 30000 - calibrate.BURN, (1,)     # set I, seed 1 runs away
    assert fit.loss(dict(model.SET_I)) == calibrate.RUNAWAY_LOSS


def test_load_reports_a_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "SETTINGS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="herding.calibrate paper"):
        calibrate.load("paper")


def test_save_then_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "SETTINGS_DIR", tmp_path)
    p = calibrate.draw(np.random.default_rng(3), calibrate.RANGES)
    path = calibrate.save("paper", 1.5, p)
    assert calibrate.load("paper") == p
    assert json.loads(path.read_text())["loss"] == 1.5


@pytest.mark.parametrize("kind", ["paper", "spx"])
def test_saved_settings_are_valid(kind):
    if not calibrate.settings_path(kind).exists():
        pytest.skip("not fitted yet")
    model.check(calibrate.load(kind))


def test_settings_files_have_descriptive_names():
    assert calibrate.settings_path("paper").name == "paper_matched.json"
    assert calibrate.settings_path("spx").name == "sp500_calibrated.json"
    for kind in ("paper", "spx"):
        assert calibrate.settings_path(kind).exists()     # so the test above never skips


def test_saved_file_records_the_seeds_the_fit_really_used(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "SETTINGS_DIR", tmp_path)
    for kind in ("paper", "spx"):
        path = calibrate.save(kind, 1.0, calibrate.base())
        assert json.loads(path.read_text())["train_seeds"] == list(calibrate.fit_seeds(kind))
    assert calibrate.Fit("paper").seeds == calibrate.fit_seeds("paper") == (100, 101)


def test_parallel_scoring_gives_exactly_the_plain_loss():
    p = dict(model.SET_I, tc=0.002, gamma=0.02)
    plain = calibrate.Fit("paper")
    plain.days, plain.seeds = 3000, (100, 101)
    par = calibrate._PoolFit("paper", SerialPool(), (100, 101))
    par.days = 3000
    assert par.loss(p) == plain.loss(p)


def test_spx_fit_rejects_even_a_bubble_that_bursts():
    assert calibrate.Fit("spx").max_days == 1
    assert calibrate.Fit("paper").max_days == model.MAX_DAYS


def _square(x):
    return (x * x, {"x": x})


def test_checkpointed_saves_each_result_and_resumes_without_recomputing(tmp_path):
    path = tmp_path / "c.jsonl"
    jobs = [1.5, 2.0, 3.0, 4.0]
    first = calibrate.checkpointed(SerialPool(), _square, jobs[:2] + [0.0, 0.0], path)
    assert first == [(2.25, {"x": 1.5}), (4.0, {"x": 2.0}), (0.0, {"x": 0.0}), (0.0, {"x": 0.0})]
    # Pretend the run stopped after two jobs, the second line cut off mid-write
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + "\n" + lines[1][:7])
    calls = []

    def counting(x):
        calls.append(x)
        return _square(x)

    out = calibrate.checkpointed(SerialPool(), counting, jobs, path)
    assert calls == [2.0, 3.0, 4.0]                  # job 0 came from the file
    assert out == [_square(x) for x in jobs]
    again = calibrate.checkpointed(SerialPool(), counting, jobs, path)
    assert again == out and calls == [2.0, 3.0, 4.0]  # nothing recomputed


def test_checkpointed_round_trips_settings_exactly(tmp_path):
    p = calibrate.draw(np.random.default_rng(5), calibrate.SPX_RANGES)
    out = calibrate.checkpointed(SerialPool(), lambda _: (0.123456789012345, p), [0], tmp_path / "c")
    back = calibrate.checkpointed(SerialPool(), lambda _: 1 / 0, [0], tmp_path / "c")
    assert back == out and back[0][1] == p and back[0][0] == 0.123456789012345


def test_runaways_counts_the_histories_that_run_away():
    # Parameter set I runs away on seed 1 before day 30,000 (tests/test_model.py);
    # with no herding pull at all (a1 = a2 = 0, few chartists) nothing runs away.
    assert calibrate.runaways(SerialPool(), model.SET_I, seeds=(1,), days=30_000) == 1
    calm = dict(model.SET_I, a1=0.0, a2=0.0, tc=0.002)
    assert calibrate.runaways(SerialPool(), calm, seeds=(1, 2), days=3000) == 0


def test_screen_puts_a_stable_worse_fit_before_an_unstable_better_one():
    calm = dict(model.SET_I, a1=0.0, a2=0.0, tc=0.002)
    out = calibrate.screen(SerialPool(), [(0.1, model.SET_I), (5.0, calm), (3.0, calm)],
                           seeds=(1,), days=30_000)
    assert [(bad, loss) for bad, loss, _ in out] == [(0, 3.0), (0, 5.0), (1, 0.1)]


def test_save_records_the_stability_screen(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "SETTINGS_DIR", tmp_path)
    path = calibrate.save("spx", 1.5, model.SET_I, runaways=0, draws=3000)
    out = json.loads(path.read_text())
    assert out["draws"] == 3000
    assert out["stability"] == dict(seeds=[2000, 2019], days=100_000, runaways=0)


def test_saved_loss_replays_a_stopped_nelder_mead_exactly(tmp_path):
    from scipy.optimize import minimize
    calls = []

    def bowl(v):
        calls.append(1)
        return float(np.sum((v - np.array([0.3, -1.2])) ** 2))

    path = tmp_path / "scores.jsonl"
    full = minimize(bowl, [1.0, 1.0], method="Nelder-Mead", options=dict(maxfev=60))
    n_full = len(calls)
    # A run that "stopped" after 25 scores (the last line cut off mid-write)...
    calls.clear()
    minimize(calibrate.SavedLoss(bowl, path), [1.0, 1.0], method="Nelder-Mead",
             options=dict(maxfev=25))
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:9])
    # ...then restarted: same answer, and only the scores not saved are computed
    calls.clear()
    again = minimize(calibrate.SavedLoss(bowl, path), [1.0, 1.0], method="Nelder-Mead",
                     options=dict(maxfev=60))
    assert np.array_equal(again.x, full.x) and again.fun == full.fun
    assert len(calls) == n_full - (len(lines) - 1)


class _Bowl:
    """Stands in for a Fit: a loss that is lowest (0) at known settings."""
    ranges = calibrate.RANGES

    def __init__(self, best):
        self.best = calibrate.to_vector(best, self.ranges)

    def loss_of_vector(self, v):
        return float(np.sum((np.asarray(v) - self.best) ** 2))


def test_refine_walks_down_to_the_bottom_of_a_known_bowl():
    best = dict(calibrate.base(), a1=0.9, beta=8.0)            # inside RANGES
    start = dict(calibrate.base(), a1=0.5, beta=5.0)
    fit = _Bowl(best)
    first = fit.loss_of_vector(calibrate.to_vector(start, fit.ranges))   # (0.4)^2 + ln(1.6)^2
    loss, p = calibrate._refine((fit, start, 400))
    assert first == pytest.approx(0.4 ** 2 + np.log(1.6) ** 2)
    assert loss < 0.01 * first
    assert p["a1"] == pytest.approx(0.9, abs=0.02) and p["beta"] == pytest.approx(8.0, rel=0.02)


# ------------------------------------------------------------------ honest reporting
# The paper fit's saved loss (0.06) is a training score: 50,000-day runs on training
# seeds. The documents must also give the score on held-out seeds and long runs,
# where H of |returns| comes out about 0.81, not the paper's 0.85.

DOCS = calibrate.SETTINGS_DIR.parent


def test_results_reports_the_held_out_paper_score_next_to_the_training_score():
    results = (DOCS / "figures" / "RESULTS.md").read_text()
    row = next(line for line in results.splitlines() if line.startswith("| **Paper-matched**"))
    assert "training" in row and "held-out" in row
    assert "2.0" in row          # 8 held-out 100,000-day runs (see the slow test below)
    assert "not reproduced" in results


def test_key_points_does_not_claim_the_whole_paper_is_reproduced():
    key = (DOCS / "docs" / "KEY_POINTS.md").read_text()
    assert "**1. The rebuilt model reproduces the paper**" not in key
    assert "not reproduced" in key


@pytest.mark.slow
def test_held_out_paper_score_matches_what_the_documents_say():
    """8 held-out seeds x 100,000 days, the runs behind Figs 2-3 (about 2 minutes)."""
    fp = calibrate.ensemble(calibrate.load("paper"), calibrate.HELD_OUT_SEEDS[:8], 100_000)
    assert np.mean(fp["H_abs"]) == pytest.approx(0.81, abs=0.005)
    assert calibrate.paper_loss(fp) == pytest.approx(2.0, abs=0.05)
    saved = json.loads(calibrate.settings_path("paper").read_text())["loss"]
    assert calibrate.paper_loss(fp) > 10 * saved     # far worse than the training score


def test_paper_fit_tolerances_are_not_called_the_papers_error_bars():
    """The paper fit divides by the project's own yardsticks (0.15, 0.03, 0.03), which
    are wider than the paper's error bars (0.077, 0.003, 0.010). The documents must say
    so, and must not call the score 'in units of the paper's error bars'."""
    paper_error_bars = {"alpha_reg": 0.077, "H": 0.003, "H_abs": 0.010}   # its Fig 2-3 captions
    for k, (_, tol) in calibrate.PAPER_TARGETS.items():
        assert tol > paper_error_bars[k]
    for doc in (DOCS / "figures" / "RESULTS.md", DOCS / "docs" / "SPEC.md"):
        text = doc.read_text()
        assert "units of the paper's error bars" not in text
        for bar in ("0.077", "0.003", "0.010"):
            assert bar in text, f"{doc.name} should give the paper's error bar {bar}"


# Audit of November 2026, issue 7: an unknown kind used to suggest the invalid command
# "python3 -m herding.calibrate nasdaq"
@pytest.mark.parametrize("kind", ["nasdaq", "", "SPX", None])
def test_load_rejects_an_unknown_kind_and_names_the_choices(kind):
    with pytest.raises(ValueError, match="paper.*spx"):
        calibrate.load(kind)


# parse_args only reads the arguments: a test of the command line must never start a
# real fit (an earlier version of this test ran "spx --refine" for real and re-saved
# settings/sp500_calibrated.json).
@pytest.mark.parametrize("argv", [[], ["nasdaq"], ["paper", "--draws"], ["paper", "--draws", "0"],
                                  ["paper", "--draws", "abc"], ["spx", "--draws", "-5"],
                                  ["spx", "--refine", "--draws", "5"]])
def test_cli_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit, match="usage"):
        calibrate.parse_args(argv)


@pytest.mark.parametrize("argv, want", [(["paper"], ("paper", False, 1200)),
                                        (["spx", "--draws", "3000"], ("spx", False, 3000)),
                                        (["spx", "--refine"], ("spx", True, 1200))])
def test_cli_reads_good_arguments(argv, want):
    assert calibrate.parse_args(argv) == want
