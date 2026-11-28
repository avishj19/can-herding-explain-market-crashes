"""explore.crowd_counts: the trader crowd on the Start page (value traders, optimists, pessimists)."""
import numpy as np
import pytest

from test_web_pages import open_page  # noqa: F401

from herding import explore as ex


# Spec: optimists z(1+x)/2, pessimists z(1-x)/2, value traders the rest, out of n; each row sums to n.
@pytest.mark.parametrize("z,x,want", [
    (0.0, 0.0, [500, 0, 0]),          # no trend chasers: all value traders
    (1.0, 1.0, [0, 500, 0]),          # all trend chasers, all optimists
    (1.0, -1.0, [0, 0, 500]),         # all pessimists
    (0.4, 0.0, [300, 100, 100]),      # 200 trend chasers, evenly split
    (0.2, 0.5, [400, 75, 25]),        # 100 trend chasers: 75 optimists, 25 pessimists
])
def test_crowd_counts_by_hand(z, x, want):
    assert ex.crowd_counts([z], [x]).tolist() == [want]


def test_crowd_counts_rows_add_up_and_match_the_shares():
    rng = np.random.default_rng(0)
    z, x = rng.uniform(0, 1, 2000), rng.uniform(-1, 1, 2000)
    c = ex.crowd_counts(z, x, n=500)
    assert c.dtype.kind == "i" and c.shape == (2000, 3)
    assert np.all(c.sum(axis=1) == 500) and np.all(c >= 0)
    assert np.all(np.abs(c[:, 1] - z * (1 + x) / 2 * 500) <= 1)      # rounding: at most one trader off
    assert np.all(np.abs(c[:, 2] - z * (1 - x) / 2 * 500) <= 1)


def test_crowd_counts_on_the_models_own_crash():
    # the Start page's picture: the default history around its biggest fall
    p, burn = ex.fitted_settings()
    rec = ex.web_numbers()["recipe"]
    mm = ex.model_market(p, burn, "herding", p["a1"], p["a2"], p["v1"], rec["days"], rec["seeds"][0])
    a, b = ex.zoom_window(mm["r"], "crash")
    c = ex.crowd_counts(mm["z"][a:b + 1], mm["x"][a:b + 1])
    assert np.all(c.sum(axis=1) == 500)
    worst = int(np.argmin(mm["r"])) + 1 - a          # the crash day inside the window
    before = max(worst - 20, 0)
    assert c[before:worst, 1].max() > c[worst:, 1].min()             # optimists before the fall, fewer after


@pytest.mark.parametrize("z,x", [([], []), ([0.5], [0.1, 0.2]), ([1.2], [0.0]), ([0.5], [-1.5]),
                                 ([np.nan], [0.0]), ([[0.5]], [[0.0]])])
def test_crowd_counts_rejects_bad_input(z, x):
    with pytest.raises(ValueError):
        ex.crowd_counts(z, x)


def test_crowd_counts_rejects_bad_n():
    with pytest.raises(ValueError):
        ex.crowd_counts([0.5], [0.0], n=0)


# Spec: the Start page shows the crowd: one canvas, the counts from crowd_counts for the default history's
# crash window, and a caption whose numbers are the counts on the day before and the day of the fall.
def test_start_page_crowd(open_page):
    import json as _json
    import re
    at = open_page("web/page_guide.py")
    html = [h.proto.body for h in at.get("html") if "crowd-cv" in h.proto.body]
    assert len(html) == 1
    data = _json.loads(re.search(r"const D = (\{.*?\}), C = ", html[0]).group(1))
    p, burn = ex.fitted_settings()
    rec = ex.web_numbers()["recipe"]
    mm = ex.model_market(p, burn, "herding", p["a1"], p["a2"], p["v1"], rec["days"], rec["seeds"][0])
    a, b = ex.zoom_window(mm["r"], "crash")
    assert data["first"] == a
    assert data["counts"] == ex.crowd_counts(mm["z"][a:b + 1], mm["x"][a:b + 1], 500).tolist()
    k = data["crash"]
    assert mm["r"][a + k - 1] == mm["r"].min()                       # k: the day the biggest fall ends
    n = data["counts"]
    cap = " ".join(x.value for x in at.caption)
    assert f"optimists go from {n[k - 1][1]} to {n[k][1]} and pessimists from {n[k - 1][2]} to {n[k][2]}" in cap
    assert "prefers-reduced-motion" in html[0] and "isConnected" in html[0]   # still for reduced motion; no leak


# Spec: _loglog_path maps (0.1 sd, share 1) to the box's top-left corner (10, 8) and (30 sd, 1e-5) to its
# bottom-right (190, 82); points outside the box are dropped.
def test_loglog_path_corners_and_clipping():
    from web import page_guide as g
    assert g._loglog_path([0.1, 30.0], [1.0, 1e-5]) == "M10.0 8.0 L190.0 82.0"
    assert g._loglog_path([0.05, 1.0, 40.0], [1.0, 0.1, 1e-3]).count("L") == 0      # only the middle point
    assert g._loglog_path([1.0], [1e-6]) == ""


# Spec: the three steps' pictures use the backend's curves and the model's trader mix.
def test_steps_html_draws_the_real_curves_and_mix():
    from web import page_guide as g
    r = ex.log_returns(ex.saved_prices("S&P 500")).values
    import base64
    import re
    page = g.steps_html(r, [400, 60, 40])
    pics = [base64.b64decode(b).decode() for b in re.findall(r"data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)", page)]
    assert len(pics) == 3 and "<svg" not in page                     # pictures as images (st.html drops <svg>)
    html = page + " ".join(pics)
    tx, ty = ex.tail_curve(r)
    assert g._loglog_path(tx, ty) in html
    grid = np.geomspace(0.1, 30, 120)
    assert g._loglog_path(grid, ex.bell_tail(grid)) in html
    assert html.count("r='3.6'") == round(400 / 500 * 84) + round(60 / 500 * 84)   # filled: value traders + optimists
    assert html.count("stroke-width='1.4'") == 84 - html.count("r='3.6'")           # hollow: pessimists
    for k in ("STEP 1", "STEP 2", "STEP 3"):
        assert k in html
