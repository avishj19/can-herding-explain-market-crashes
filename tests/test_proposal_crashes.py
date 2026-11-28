"""Tests of the proposal's crash check (Nov 2026): a bell curve puts a Black Monday 10^86 years apart;
does herding bring crashes within a few decades? Functions in herding/explore.py (bell_crash_years,
crash_counts, big_falls, crash_years, crash_table, crash_check), the Simulated market's zoom
(zoom_window) and its news-versus-moves line (news_and_moves), and the pages that show them."""
import math

import numpy as np
import pytest

from herding import calibrate, explore as ex, market_data, stats
from test_web_explore_fixes import _group
from test_web_pages import _fig, app, open_page  # noqa: F401

P, BURN = ex.fitted_settings()
NUMS = ex.web_numbers()
SPX = market_data.spx_returns().values


# ------------------------------------------------------------------ bell_crash_years

# Spec: 1 / (P(Z < -x) * 252). Exact values: P(Z < -1) = 0.15865525393145707 (tables), so 1 sd falls
# come every 1 / (0.158655... * 252) years; 20 sd is the proposal's 10^86 years (1.44e86).
def test_bell_crash_years_exact():
    assert ex.bell_crash_years(1) == pytest.approx(1 / (0.15865525393145707 * 252), rel=1e-12)
    assert ex.bell_crash_years(20) == pytest.approx(1.4408e86, rel=1e-3)
    assert 86 == math.floor(math.log10(ex.bell_crash_years(20)))
    assert ex.bell_crash_years(2, year_days=1) == pytest.approx(1 / 0.022750131948179195, rel=1e-12)


# Spec: bigger falls are always rarer; a size too big for floating point is "never" (inf), not an error.
def test_bell_crash_years_grows_and_overflows_to_inf():
    ys = [ex.bell_crash_years(s) for s in ex.CRASH_SIZES]
    assert all(a < b for a, b in zip(ys, ys[1:]))
    assert ex.bell_crash_years(60) == math.inf


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), "3", None, True])
def test_bell_crash_years_rejects_bad_sizes(bad):
    with pytest.raises(ValueError):
        ex.bell_crash_years(bad)


# ------------------------------------------------------------------ crash_counts, big_falls, crash_years

# Spec: counted by hand. 98 days of +-1 around 0 (sd 1 after the big days are added in), then falls
# measured in the series' own sd: check against moves_in_sd directly.
def test_crash_counts_by_hand():
    r = np.r_[np.tile([0.01, -0.01], 500), -0.08, -0.06, 0.07]
    z = ex.moves_in_sd(r)
    want = [int(np.sum(z <= -s)) for s in (3, 4, 5, 6, 7)]
    assert ex.crash_counts(r, (3, 4, 5, 6, 7)) == want
    assert want[0] == 2                    # both big falls, not the big rise
    assert ex.big_falls(r, size=3) == sorted([-z[1000], -z[1001]], reverse=True)


# Spec: rises never count as crashes.
def test_crash_counts_ignore_rises():
    r = np.r_[np.tile([0.01, -0.01], 500), 0.2]
    assert ex.crash_counts(r, (3, 5)) == [0, 0]
    assert ex.big_falls(r, size=3) == []


# Spec: years = days / 252 / count; none seen = inf; counts may be one per history.
def test_crash_years_by_hand():
    assert ex.crash_years(2, 2520) == 5.0
    assert ex.crash_years([1, 0, 3], 2520) == 2.5
    assert ex.crash_years(0, 2520) == math.inf
    for bad in (0, -5, 2.5, True):
        with pytest.raises(ValueError):
            ex.crash_years(1, bad)


# ------------------------------------------------------------------ the saved numbers

# Spec: numbers.json's crash counts are the held-out histories' own counts: rerun two seeds.
@pytest.mark.parametrize("i", [0, 39])
def test_saved_crash_counts_rerun_exactly(i):
    seed = calibrate.HELD_OUT_SEEDS[i]
    r, err = ex.run_mix(P, "herding", P["a1"], NUMS["recipe"]["days"], seed, BURN)
    assert err is None                               # no herding history ran away (Checks page, check 3)
    cr = NUMS["crashes"]
    assert cr["sizes"] == list(ex.CRASH_SIZES)
    assert cr["falls"][i] == ex.crash_counts(r)
    assert cr["big_falls"][i] == pytest.approx(ex.big_falls(r), abs=1e-12)
    assert cr["days"][i] == len(r)


# Spec: every saved fall of at least 8 sd is also counted in the 8-sd column (the two lists agree).
def test_saved_big_falls_agree_with_counts():
    cr = NUMS["crashes"]
    i8 = cr["sizes"].index(ex.BIG_FALL)
    for falls, big in zip(cr["falls"], cr["big_falls"]):
        assert falls[i8] == len(big)
        for j, s in enumerate(cr["sizes"]):
            if s >= ex.BIG_FALL:
                assert falls[j] == sum(b >= s for b in big)


# Spec: the table's S&P column is counted from the real data; model counts are the pooled sums.
def test_crash_table_against_the_data():
    rows = ex.crash_table(NUMS, SPX)
    z = ex.moves_in_sd(SPX)
    pooled = sum(NUMS["crashes"]["days"])
    for row, j in zip(rows, range(len(rows))):
        s = row["size"]
        assert row["sp500_count"] == int(np.sum(z <= -s))
        assert row["model_count"] == sum(f[j] for f in NUMS["crashes"]["falls"])
        assert row["model"] == ex.crash_years(row["model_count"], pooled)
        assert row["bell"] == ex.bell_crash_years(s)
        assert row["of"] == 40
    by = {r["size"]: r for r in rows}
    assert by[20]["sp500_count"] == 1          # Black Monday is the S&P 500's only 20-sd fall since 1950


# Spec: the check's two parts with the saved numbers (Nov 2026): 8-sd falls every 35 years (passes),
# Black-Monday-sized falls every 1,534 years (2 in 40 histories; does not pass), so "partly".
def test_crash_check_saved_result():
    cc = ex.crash_check(NUMS, SPX)
    assert cc["worst_sd"] == pytest.approx(-stats.standardize(SPX).min(), rel=1e-12)
    assert cc["row"]["model"] == pytest.approx(34.86, abs=0.01)
    assert cc["decades"] and not cc["black_monday"] and not cc["passed"]
    assert cc["bm_count"] == 2
    assert cc["bm_model"] == pytest.approx(sum(NUMS["crashes"]["days"]) / 252 / 2, rel=1e-12)


# Spec: the rule itself, on made-up numbers: one history of 2,520 days (10 years) with 1 fall of 8+ sd and
# one fall bigger than the S&P's worst passes both parts; none of either fails both.
def test_crash_check_rule_by_hand():
    sizes = list(ex.CRASH_SIZES)
    worst = -stats.standardize(SPX).min()
    i8 = sizes.index(8)
    both = {"crashes": {"sizes": sizes, "days": [2520], "falls": [[3] * i8 + [1] * (len(sizes) - i8)],
                        "big_falls": [[worst + 1]]}}
    cc = ex.crash_check(both, SPX)
    assert cc["row"]["model"] == 10.0 and cc["bm_model"] == 10.0 and cc["passed"]
    none = {"crashes": {"sizes": sizes, "days": [2520], "falls": [[0] * len(sizes)], "big_falls": [[]]}}
    cc = ex.crash_check(none, SPX)
    assert cc["row"]["model"] == math.inf and not cc["decades"] and not cc["passed"]
    with pytest.raises(ValueError):
        ex.crash_check(NUMS, SPX, size=9)


# ------------------------------------------------------------------ zoom_window, news_and_moves

# Spec: by hand. The crash window centres the biggest fall (move i ends on day i + 1) and stays inside.
def test_zoom_window_crash_by_hand():
    r = np.zeros(1000)
    r[600] = -0.1
    assert ex.zoom_window(r, "crash", 100) == (551, 651)
    r = np.zeros(1000); r[10] = -0.1
    assert ex.zoom_window(r, "crash", 100) == (0, 100)
    r = np.zeros(1000); r[995] = -0.1
    assert ex.zoom_window(r, "crash", 100) == (900, 1000)


# Spec: the burst window is the one with the most total movement: a block of wild days is found exactly.
def test_zoom_window_burst_by_hand():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.001, 2000)
    r[1200:1300] = rng.normal(0, 0.05, 100)
    assert ex.zoom_window(r, "burst", 100) == (1200, 1300)
    assert ex.zoom_window(r[:50], "burst", 100) == (0, 50)


@pytest.mark.parametrize("args", [([], "crash"), ([0.1, np.nan], "crash"), ([0.1] * 10, "boom"),
                                  ([0.1] * 10, "crash", 1)])
def test_zoom_window_rejects_bad_input(args):
    with pytest.raises(ValueError):
        ex.zoom_window(*args)


# Spec: the news is bell-curve noise in every mix (excess kurtosis near 0: within 5 standard errors,
# 5 sqrt(24/n)); with value traders only the price is a bell curve too, and with herding it is not.
def test_news_and_moves_answers_the_question():
    days = 8000
    tol = 5 * math.sqrt(24 / days)
    fund = ex.news_and_moves(ex.model_market(P, BURN, "fundamentalists", P["a1"], P["a2"], P["v1"], days, 1000))
    herd = ex.news_and_moves(ex.model_market(P, BURN, "herding", P["a1"], P["a2"], P["v1"], days, 1000))
    for nm in (fund, herd):
        assert abs(nm["news_kurtosis"]) < tol
    assert abs(fund["price_kurtosis"]) < tol and fund["chartists_peak"] == 0
    assert herd["price_kurtosis"] > 3
    assert 0 < herd["chartists_mean"] <= herd["chartists_peak"] <= 1


# ------------------------------------------------------------------ the pages

# Spec: the Checks page follows the proposal's order (Kirman first) and opens with a scoreboard whose
# verdicts are the checks' own: pass, pass, pass, partly, no pass line.
def test_checks_page_scoreboard_in_proposal_order(open_page):
    at = open_page("web/page_checks.py")
    assert not at.exception
    heads = [h.value for h in at.header]
    assert [h[:2] for h in heads[:5]] == ["1.", "2.", "3.", "4.", "5."]
    assert heads[5].startswith("The textbook checklist")            # its own tab since the redesign
    for i in range(1, 6):                                            # the scoreboard links open each check
        assert f"href='#check-{i}'" in " ".join(m.value for m in at.markdown)
    assert heads[0].startswith("1. Kirman") and heads[3].startswith("4. Crashes")
    board = at.table[0].value
    assert list(board["Verdict"])[:5] == ["✅ Passes", "✅ Passes", "✅ Passes", "🟡 Partly", "ℹ️ No pass mark set"]
    assert "10⁸⁶" in board["The proposal expected"].iloc[3]
    text = " ".join(m.value for m in at.markdown)
    assert "every **35 years**" in text and "every **1,534 years**" in text


# Spec: the crash chart's bell-curve line is bell_crash_years at every saved size; the model line skips none.
# Spec (Opus review, 28 Nov): check 4 must not give Black Monday the 20-sd wait. 10⁸⁶ years is a bell curve's wait
# for a 20-sd fall; Black Monday was about 23 sd, so the page names both, each worked out by the backend.
def test_checks_page_black_monday_gets_its_own_wait(open_page):
    import math
    at = open_page("web/page_checks.py")
    text = " ".join(m.value for m in at.markdown)
    spx_r = ex.log_returns(ex.saved_prices("S&P 500")).values
    cc = ex.crash_check(NUMS, spx_r)
    p20 = math.floor(math.log10(ex.bell_crash_years(20)))
    assert p20 == 86                                                   # the proposal's number
    assert "a 20-sd fall comes about once in 10⁸⁶ years" in text
    assert f"Black Monday ({cc['worst_sd']:.0f} sd) even more rarely" in text and round(cc["worst_sd"]) == 23
    assert cc["bm_bell"] > 1e100                                       # 23 sd: far rarer than 10⁸⁶ years
    assert "a fall the size of Black Monday comes once in" not in text  # the false sentence is gone
    assert any(h.value == "4. Crashes come within decades, not 10⁸⁶ years" for h in at.header)


def test_checks_page_crash_chart(open_page):
    at = open_page("web/page_checks.py")
    fig = _fig(at, "c_crash")
    bell = next(t for t in fig["data"] if t.get("name") == "Bell curve")
    assert bell["y"] == pytest.approx([ex.bell_crash_years(s) for s in ex.CRASH_SIZES], rel=1e-12)
    model = next(t for t in fig["data"] if t.get("name") == "Model")
    assert len(model["x"]) == len(ex.CRASH_SIZES)


# Spec: the Story (step 1 and step 6) and Start here tell the 10^86 figure and the model's crash waits.
# The Story writes the power with real superscripts (docs/redesign/WORDING.md rule 10: 10⁸⁶).
def test_story_tells_the_crash_numbers(open_page):
    at = open_page("web/page_story.py")
    assert "10⁸⁶ years" in " ".join(m.value for m in at.markdown)
    for _ in range(5):
        at.button(key="story_next").click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "35 years" in text and "1,534 years" in text


def test_start_here_tells_the_crash_numbers(open_page):
    at = open_page("web/page_guide.py")
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "10⁸⁶" in text and "35 years" in text and "1,534 years" in text


# Spec: "Biggest crash" zooms all three panels to zoom_window(r, "crash") and says so.
def test_simulated_market_zooms_to_the_biggest_crash(app):
    at = app()
    _group(at, "view").set_value("Model market").run()
    assert not at.exception
    assert any("News in, moves out" in m.value for m in at.markdown)
    _group(at, "mm_zoom").set_value("crash").run()
    run = at.session_state["run"]
    mm = ex.model_market(P, BURN, "herding", **{k: run[k] for k in ("a1", "a2", "v1", "days", "seed")},
                         party=run.get("party", 0.0))
    first, last = ex.zoom_window(mm["r"], "crash")
    fig = _fig(at, "model_market_chart")
    assert fig["layout"]["xaxis"]["range"] == [first - 0.5, last + 0.5]
    assert any(f"Showing days {first:,} to {last:,}" in c.value for c in at.caption)
