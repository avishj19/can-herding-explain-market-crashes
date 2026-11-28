"""Stress test of the web app and herding/explore.py on all the data: precise, fast, correct?

    python3 scripts/analysis/web_stress_test.py            # everything (needs internet for part C)
    python3 scripts/analysis/web_stress_test.py --offline  # skip the live Yahoo checks

A. Every saved market (26): each number the app shows is recomputed a second, independent way
   (scipy, plain Python loops, numpy formulas) and must agree to 12 significant digits.
B. The model: herding strength 0-2 x seeds 0-19 at 10,000 days (runaway rate, no NaN, timing);
   the corners of every slider; the same settings must give byte-identical histories.
C. Live data: Yahoo's ^GSPC and GC=F against the saved S&P 500 and gold files on the same dates;
   2,000 random strings through the ticker check (nothing unsafe may reach Yahoo).
D. Every web page run headless (Streamlit AppTest): no exception, cold and warm load times, and
   the workspace's preset -> Run model -> run log flow.
E. The numbers the pages display, read back from the pages, against fresh backend calls and the
   numbers quoted in figures/RESULTS.md and docs/KEY_POINTS.md.
Report: scratch/web_stress/report.md (and printed).
"""
import math
import random
import re
import string
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from herding import explore as ex  # noqa: E402
from herding import model  # noqa: E402

OUT = ROOT / "scratch" / "web_stress"
REL = 1e-12                    # the project's precision rule (tests/conftest.py)
lines, failures = [], []


def say(text=""):
    print(text, flush=True)
    lines.append(text)


def check(ok, what):
    if not ok:
        failures.append(what)
        say(f"  FAIL: {what}")
    return ok


def close(a, b, rel=REL, abs_=0.0):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


# ------------------------------------------------------------------ A. every saved market
def part_a():
    say("## A. Every saved market, recomputed independently\n")
    say("| Market | days | kurtosis | tail | memory | max drawdown | big days | worst diff | ms |")
    say("|---|---|---|---|---|---|---|---|---|")
    for name in sum(ex.catalog().values(), []):
        t0 = time.perf_counter()
        close_ = ex.saved_prices(name)
        r = ex.log_returns(close_)
        s = ex.summary(r.values)
        dd = ex.max_drawdown(r.values)
        big = ex.big_day_list(r)
        tc = ex.tail_curve(r.values)
        hc = ex.histogram(r.values)
        mc = ex.memory_curve(r.values)
        ms = (time.perf_counter() - t0) * 1000
        x, p = r.values, close_.values
        diffs = []
        # returns: log(p1/p0) vs log p1 - log p0
        lr = np.log(p[1:] / p[:-1])
        diffs.append(np.max(np.abs(lr - x)))
        check(np.allclose(lr, x, rtol=0, atol=1e-13), f"{name}: log returns")
        # kurtosis vs scipy (population formula, excess)
        k2 = sps.kurtosis(x, fisher=True, bias=True)
        check(close(s["kurtosis"], k2), f"{name}: kurtosis {s['kurtosis']} vs scipy {k2}")
        # Hill tail by a plain loop
        a = sorted((abs(v) for v in x if v != 0), reverse=True)
        kk = max(int(len(a) * 0.05), 10)
        hill = kk / sum(math.log(a[i] / a[kk]) for i in range(kk))
        check(close(s["tail"], hill), f"{name}: tail {s['tail']} vs loop {hill}")
        # memory vs scipy pearson
        ab = np.abs(x)
        m2 = sps.pearsonr(ab[:-20], ab[20:])[0]
        check(close(s["memory"], m2, 1e-10), f"{name}: memory {s['memory']} vs scipy {m2}")
        # worst day and big days vs scipy zscore
        z = sps.zscore(x)
        check(close(s["worst"], z.min(), 1e-10), f"{name}: worst day")
        check(s["big_days"] == int((np.abs(z) > 4).sum()) == len(big), f"{name}: big-day count")
        check(close(s["bell_big_days"], len(x) * 2 * sps.norm.sf(4)), f"{name}: bell big days")
        # max drawdown by a plain loop over the price path
        peak, worst = 1.0, 0.0
        level = 1.0
        for v in x:
            level *= math.exp(v)
            peak = max(peak, level)
            worst = max(worst, 1 - level / peak)
        check(close(dd, worst, 1e-9), f"{name}: max drawdown {dd} vs loop {worst}")
        # chart data: shapes and invariants
        check(np.all(np.diff(tc[0]) >= 0) and np.all(np.diff(tc[1]) <= 0), f"{name}: tail curve not monotone")
        inside = np.mean(np.abs(z) < 8)
        check(close(np.sum(hc[1] * 0.4), inside, 1e-9), f"{name}: histogram area {np.sum(hc[1]*0.4)} vs {inside}")
        check(np.all(np.isfinite(mc[1])) and abs(mc[1][4] - s["memory"]) < 1e-12, f"{name}: memory curve")
        check(np.all(np.isfinite([s[k] for k in ("kurtosis", "tail", "memory", "worst")])), f"{name}: non-finite")
        say(f"| {name} | {s['days']:,} | {s['kurtosis']:.2f} | {s['tail']:.3f} | {s['memory']:.3f} | {dd:.1%} | "
            f"{s['big_days']} | {max(diffs):.1e} | {ms:.0f} |")
    t0 = time.perf_counter()
    ex.market_table()
    say(f"\nmarket_table() for all 26 markets: {time.perf_counter() - t0:.2f} s (cached after the first call in the app)\n")


# ------------------------------------------------------------------ B. the model
def part_b(days, n):
    say("## B. The model: herding strength x seeds, slider corners, reproducibility\n")
    p, burn = ex.fitted_settings()
    model.simulate(p, 10, seed=0)                     # compile once (numba) before timing
    grid = [(a1, s, days) for a1 in (0.0, 0.3, 0.6, round(p["a1"], 4), 1.5, 2.0) for s in range(n)]
    t0 = time.perf_counter()
    with Pool() as pool:            # the library function itself, so workers can always import it
        out = pool.starmap(ex.run_mix, [(p, "herding", a1, d, s_, burn) for a1, s_, d in grid])
    wall = time.perf_counter() - t0
    say(f"| a1 | ran away (of {n}) | median tail | tail range |")
    say("|---|---|---|---|")
    for a1 in sorted({g[0] for g in grid}):
        runs = [o for g, o in zip(grid, out) if g[0] == a1]
        ok = [r for r, _ in runs if r is not None]
        check(all(np.all(np.isfinite(r)) for r in ok), f"a1={a1}: NaN or inf in returns")
        tails = [ex.summary(r)["tail"] for r in ok]
        gone = sum(1 for r, _ in runs if r is None)
        rng = f"{min(tails):.2f}-{max(tails):.2f}" if tails else "-"
        med = f"{np.median(tails):.2f}" if tails else "-"
        say(f"| {a1} | {gone} | {med} | {rng} |")
    say(f"\n{len(grid)} histories in {wall:.1f} s wall time (all cores).\n")

    # slider corners: every combination of the extremes the page allows, all three mixes
    t0, n, crashes = time.perf_counter(), 0, []
    for a1 in (0.0, 2.0):
        for a2 in (0.0, 1.0):
            for v1 in (0.5, 5.0):
                for d in (2000, min(2 * days, ex.MAX_DAYS)):
                    q = dict(p, a2=a2, v1=v1)
                    for m in ex.MIXES:
                        try:
                            r, err = ex.run_mix(q, m, a1, d, 0, burn)
                            if r is not None:
                                ex.summary(r)
                                check(np.all(np.isfinite(r)), f"corner a1={a1} a2={a2} v1={v1} {m}: NaN")
                        except Exception as e:      # anything but a clean runaway is a bug
                            crashes.append(f"a1={a1} a2={a2} v1={v1} days={d} {m}: {type(e).__name__}: {e}")
                        n += 1
    for c in crashes:
        check(False, "slider corner crashed: " + c)
    say(f"Slider corners: {n} runs, {len(crashes)} crashes, {time.perf_counter() - t0:.1f} s.\n")

    # reproducibility and equality with a direct simulation
    for a1, seed, mix in ((p["a1"], 3, "herding"), (0.6, 11, "herding"), (1.0, 5, "fundamentalists")):
        r1, _ = ex.run_mix(p, mix, a1, 5000, seed, burn)
        r2, _ = ex.run_mix(p, mix, a1, 5000, seed, burn)
        q, ch = ex.mix_settings(p, mix, a1)
        direct = model.simulate(q, 5000 + burn, seed=seed, chartists=ch)["r"][burn:]
        check(r1 is not None and r1.tobytes() == r2.tobytes() == direct.tobytes(),
              f"reproducibility a1={a1} seed={seed} {mix}")
    say("Same settings twice and a direct model.simulate call: byte-identical histories (3 cases).\n")


# ------------------------------------------------------------------ C. live data
def part_c(offline):
    say("## C. Live data and the ticker check\n")
    rng = random.Random(0)
    alphabet = string.ascii_letters + string.digits + " .-=^;/'\"$&|<>!?*\\\t\n" + "ÄéЖ漢"
    seen, accepted = [], 0
    for _ in range(2000):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 16)))
        try:
            t = ex.clean_ticker(s)
            accepted += 1
            check(ex.TICKER.match(t) is not None and t == s.strip().upper(), f"ticker check let through {s!r}")
        except ValueError:
            pass
        try:
            ex.live_prices(s, download=lambda t, st: seen.append(t) or pd.Series(dtype=float))
        except (ValueError, ConnectionError):
            pass
    unsafe = [t for t in seen if not ex.TICKER.match(t)]
    check(not unsafe, f"unsafe strings reached the downloader: {unsafe[:5]}")
    say(f"2,000 random strings: {accepted} accepted as tickers, {len(seen)} reached the (fake) downloader, "
        f"{len(unsafe)} unsafe.\n")
    if offline:
        say("Live Yahoo checks skipped (--offline).\n")
        return
    say("| Live symbol | saved file | days compared | max relative difference | fetch s |")
    say("|---|---|---|---|---|")
    for sym, name in (("^GSPC", "S&P 500"), ("GC=F", "Gold"), ("AAPL", "AAPL")):
        t0 = time.perf_counter()
        try:
            live = ex.live_prices(sym)
        except (ConnectionError, ValueError) as e:
            say(f"| {sym} | {name} | - | unavailable: {e} | - |")
            continue
        dt = time.perf_counter() - t0
        saved = ex.saved_prices(name)
        both = live.index.intersection(saved.index)
        both = both[both >= pd.Timestamp("2005-01-01")]
        diff = np.max(np.abs(live[both].values / saved[both].values - 1)) if len(both) else float("nan")
        check(len(both) > 1000 and diff < 1e-6, f"live {sym} vs saved {name}: max rel diff {diff:.2e}")
        say(f"| {sym} | {name} | {len(both):,} | {diff:.1e} | {dt:.1f} |")
    say("")


# ------------------------------------------------------------------ D. the web pages
PAGE_TIMEOUT = 180     # seconds per page run: a hung page becomes a FAIL line instead of hanging the suite


def _run_page(at, what, page=None):
    """One AppTest run (of `page` if given), True if it finished. A run longer than PAGE_TIMEOUT
    (AppTest raises RuntimeError "timed out") is reported as a failure, so the stress test, and the
    test suite that calls it, always ends.
    Streamlit's script runner installs the page as sys.modules["__main__"] and leaves it there; on
    macOS every later multiprocessing Pool worker (spawn) then re-imports that, the whole app, and
    the pool never returns (TEST-2: the full test suite hung). So the real __main__ is put back."""
    main = sys.modules.get("__main__")
    try:
        (at.switch_page(page) if page else at).run()
        return True
    except RuntimeError as e:
        check(False, f"{what}: {e}")
        return False
    finally:
        sys.modules["__main__"] = main


def part_d():
    from streamlit.testing.v1 import AppTest
    say("## D. Every page, headless\n")
    say("| Page | cold s | warm s | exception |")
    say("|---|---|---|---|")
    pages = ["page_guide", "page_story", "page_explore", "page_checks", "page_sources"]   # five pages since the redesign
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=PAGE_TIMEOUT)
    t0 = time.perf_counter()
    if not _run_page(at, "first visit"):              # opens the default page (Start here) with empty caches
        return None
    say(f"| first visit (Start here, nothing cached) | {time.perf_counter() - t0:.2f} | - | "
        f"{'; '.join(e.value for e in at.exception)[:80] or 'none'} |")
    for pg in pages:
        times = []
        for _ in range(2):
            t0 = time.perf_counter()
            if not _run_page(at, pg, f"web/{pg}.py"):
                return None                             # a timed-out page leaves the app half-run
            times.append(time.perf_counter() - t0)
        exc = "; ".join(e.value for e in at.exception) if at.exception else ""
        check(not at.exception, f"{pg}: {exc[:300]}")
        say(f"| {pg} | {times[0]:.2f} | {times[1]:.2f} | {exc[:80] or 'none'} |")

    # the Test program flow (since the redesign): a preset button -> the model reruns by itself -> the
    # newest history-log line has the preset's a1. (Before: a sidebar radio and a Run model button.)
    if not _run_page(at, "workspace", "web/page_explore.py"):
        return None
    preset = [b for b in at.get("button_group") if "preset" in b.proto.id]
    check(len(preset) == 1, "workspace: one preset control")
    if not preset or not _run_page(preset[0].set_value("No herding"), "workspace preset"):
        return None
    a1 = at.slider(key="in_a1").value
    check(not [b for b in at.button if b.label == "Run model"], "workspace: no Run model button while runs are automatic")
    log = at.session_state["log"]
    check(len(log) >= 1 and log[0][1]["a1"] == 0.0 == a1, f"workspace: newest log line after preset {log[:1]}")
    check(not at.exception, "workspace: exception after the preset")
    say(f"\nWorkspace flow: preset 'No herding' set a1 = {a1}; the model reran by itself; "
        f"{len(log)} log line(s); newest tails {log[0][2] if log else None}.\n")
    return at


def _text(at):
    """Everything the page wrote as markdown, captions and metrics, as one string."""
    parts = [m.value for m in at.markdown] + [m.value for m in at.caption]
    return "\n".join(parts)


def _bold_numbers(text):
    return [float(x.replace(",", "")) for x in re.findall(r"\*\*([+-]?[\d,]*\.?\d+)", text)]


def part_e(at):
    """The numbers the pages display, read back from the pages, against fresh backend calls and the docs."""
    say("## E. What the pages show vs the backend and the docs\n")
    say("| Page | number shown | backend / docs | agrees |")
    say("|---|---|---|---|")
    nums = ex.web_numbers()
    rec = nums["recipe"]
    spx_close = ex.saved_prices("S&P 500")
    spx = ex.summary(ex.log_returns(spx_close).values)
    herd_tails = [t for t in nums["checks"]["herding"]["tail"] if t is not None]
    fund_k = nums["checks"]["fundamentalists"]["kurtosis"]
    p, burn = ex.fitted_settings()
    r1000, _ = ex.run_mix(p, "herding", p["a1"], rec["days"], rec["seeds"][0], burn)
    f1000, _ = ex.run_mix(p, "fundamentalists", p["a1"], rec["days"], rec["seeds"][0], burn)
    gone = sum(d is not None for d in nums["checks"]["chartists"]["runaway_day"])

    def row(page, label, shown, want, tol):
        ok = shown is not None and abs(shown - want) <= tol
        check(ok, f"{page}: {label} shows {shown}, backend {want}")
        say(f"| {page} | {label}: {shown} | {want:.4g} | {'yes' if ok else 'NO'} |")

    # the Story shows one step at a time since the redesign (8 steps, st.session_state["story_step"] 0-7):
    # each number is read from its own step by the words around it, not by its place on one long page
    at.switch_page("web/page_story.py").run()

    def story(step, pattern):
        at.session_state["story_step"] = step
        at.run()
        m = re.search(pattern, _text(at))
        return m and [float(g.replace(",", "")) for g in m.groups()]

    num = r"([+-]?[\d,]*\.?\d+)"
    b = story(0, rf"has had \*\*{num} big days\*\*\. A bell curve expects about \*\*{num}\*\*") or [None, None]
    row("Story", "S&P big days", b[0], spx["big_days"], 0)
    row("Story", "bell-curve big days", b[1], spx["bell_big_days"], 0.05)
    b = story(3, rf"\*\*{num}\*\* in this simulated history and \*\*{num}\*\* \(median\)") or [None, None]
    row("Story", "kurtosis, this history", b[0], ex.summary(f1000)["kurtosis"], 0.005)
    row("Story", "kurtosis, median of 40", b[1], float(np.median(fund_k)), 0.005)
    b = story(4, rf"In \*\*{num} of {num}\*\* simulated histories") or [None, None]
    row("Story", "runaways without herding", b[0], gone, 0)
    b = story(5, rf"is \*\*{num}\*\* in this history and \*\*{num}\*\* \(median\).*?The S&P 500's is \*\*{num}\*\*") \
        or [None, None, None]
    row("Story", "herding tail, this history", b[0], ex.summary(r1000)["tail"], 0.005)
    row("Story", "herding tail, median of 40", b[1], float(np.median(herd_tails)), 0.005)
    row("Story", "S&P tail", b[2], spx["tail"], 0.005)

    at.switch_page("web/page_checks.py").run()
    t = _text(at)
    m = re.search(r"Median \*\*([+-][\d.]+)\*\*", t)
    row("Checks", "kurtosis median", m and float(m.group(1)), float(np.median(fund_k)), 0.0005)
    m = re.search(r"Model median \*\*([\d.]+)\*\*", t)
    row("Checks", "tail median", m and float(m.group(1)), float(np.median(herd_tails)), 0.005)
    # the scoreboard: the proposal's checks 1-3 pass; check 4 is what crash_check says ("partly": crashes
    # within decades, but a Black-Monday-sized fall still rarer than in the S&P 500, as the page explains)
    board = list(at.table[0].value["Verdict"])
    cc = ex.crash_check(nums, ex.log_returns(ex.saved_prices("S&P 500")).values)
    want4 = "✅ Passes" if cc["passed"] else ("🟡 Partly" if cc["decades"] else "❌ Fails")
    check(board[:3] == ["✅ Passes"] * 3 and board[3] == want4,
          f"Checks: scoreboard verdicts {board[:4]}, expected 3 passes then {want4}")
    say(f"| Checks | scoreboard verdicts {board[:4]} | 3 passes, then {want4} | "
        f"{'yes' if board[:3] == ['✅ Passes'] * 3 and board[3] == want4 else 'NO'} |")

    at.switch_page("web/page_explore.py").run()
    df = next(d.value for d in at.dataframe if "Tail exponent" in getattr(d.value, "index", []))   # model vs market
    row("Workspace", "S&P tail in table", float(df.loc["Tail exponent", "S&P 500"]), spx["tail"], 0.005)
    row("Workspace", "S&P max drawdown %", float(df.loc["Max drawdown", "S&P 500"].rstrip("%")),
        100 * ex.max_drawdown(ex.log_returns(spx_close).values), 0.05)

    at.switch_page("web/page_checks.py").run()          # All markets: a tab of Checks since the redesign
    shown = next(d.value for d in at.dataframe if "Tail exponent" in getattr(d.value, "columns", []))
    table = ex.market_table()
    diff = max(abs(float(a) - float(b_)) for a, b_ in zip(shown["Tail exponent"], table["tail"]))
    row("Compare", "largest tail difference over 26 markets", diff, 0.0, 1e-12)

    part_e_findings(at, row, nums, spx, herd_tails, fund_k, gone)

    docs = (ROOT / "figures" / "RESULTS.md").read_text() + (ROOT / "docs" / "KEY_POINTS.md").read_text()
    for want, label in ((f"{np.median(herd_tails):.2f}", "herding tail median"),
                        (f"{np.median(fund_k):.2f}", "no-chartist kurtosis median"),
                        (f"{spx['tail']:.2f}", "S&P tail")):
        ok = want in docs
        check(ok, f"docs don't quote the site's {label} {want}")
        say(f"| docs | {label} {want} quoted in RESULTS.md / KEY_POINTS.md | - | {'yes' if ok else 'NO'} |")
    say("")


def _paper_table(text):
    """{measure: (paper value, our value)} read back from the paper table (Checks, Beyond the paper)."""
    out = {}
    for row in re.findall(r"<tr>(.*?)</tr>", text):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row)
        if len(cells) >= 3:
            out[cells[0]] = tuple(float(re.match(r"[\d.]+", x).group()) for x in cells[1:3])
    return out


def part_e_findings(at, row, nums, spx, herd_tails, fund_k, gone):
    """Findings, the paper section: every number it shows against web/numbers.json (or the S&P file),
    and the paper-matched means against the numbers figures/RESULTS.md quotes."""
    from herding import stats
    if not _run_page(at, "findings", "web/page_checks.py"):   # Beyond the paper tab
        return
    t = _text(at)
    pp = nums["paper"]
    table = _paper_table(t)
    check(len(table) == 3, f"Findings: paper table has {len(table)} rows, expected 3")
    results = (ROOT / "figures" / "RESULTS.md").read_text()
    for (key, label), quoted in zip(ex.PAPER_MEASURES.items(), ("2.62", "0.47", "0.81")):
        mean = float(np.mean([x for x in pp[key] if x is not None]))
        shown = table.get(label, (None, None))[1]
        row("Findings", f"paper-matched {key} (numbers.json mean)", shown, mean, 0.005)
        ok = f"{mean:.2f}" == quoted and quoted in results
        check(ok, f"Findings: paper-matched {key} {mean:.2f} vs RESULTS.md's {quoted}")
        say(f"| Findings | paper-matched {key} {mean:.2f} | RESULTS.md {quoted} | {'yes' if ok else 'NO'} |")
        row("Findings", f"paper's published {key}", table.get(label, (None, None))[0], ex.PAPER_PUBLISHED[key][0], 0.005)
    m = re.search(r"falls from (\d+) for one day to ([\d.]+) for 25 days", t)
    row("Findings", "paper-matched kurtosis, 1 day (median)", m and float(m.group(1)), float(np.median(pp["kurtosis"])), 0.5)
    row("Findings", "paper-matched kurtosis, 25 days (median)", m and float(m.group(2)), float(np.median(pp["kurt_25"])), 0.05)
    ok = m is not None and m.group(1) == "43" and m.group(2) == "0.4"
    check(ok, f"Findings: kurtosis should read 43 -> 0.4, reads {m and m.groups()}")
    say(f"| Findings | kurtosis 1 day -> 25 days: {m and m.groups()} | 43 -> 0.4 | {'yes' if ok else 'NO'} |")
    m = re.search(r"median (?:excess )?kurtosis ([+-]?[\d.]+)", t)
    row("Findings", "no-chartist kurtosis median", m and float(m.group(1)), float(np.median(fund_k)), 0.005)
    m = re.search(r"runs away in (\d+) of", t)
    row("Findings", "runaways without herding", m and float(m.group(1)), gone, 0)
    m = re.search(r"median tail exponent of \*\*([\d.]+)\*\*, against \*\*([\d.]+)\*\*", t)
    row("Findings", "herding tail median", m and float(m.group(1)), float(np.median(herd_tails)), 0.005)
    row("Findings", "S&P tail (Hill)", m and float(m.group(2)), spx["tail"], 0.005)
    m = re.search(r"S&P 500's tail exponent is ([\d.]+)", t)
    spx_reg = stats.alpha_regression(ex.log_returns(ex.saved_prices("S&P 500")).values)
    row("Findings", "S&P tail (paper's method)", m and float(m.group(1)), spx_reg, 0.005)
    ok = f"{spx_reg:.2f}" in results
    check(ok, f"Findings: RESULTS.md does not quote the S&P's paper-method tail {spx_reg:.2f}")
    say(f"| Findings | S&P paper-method tail {spx_reg:.2f} quoted in RESULTS.md | - | {'yes' if ok else 'NO'} |")


def main(days=10000, n=20, offline=True):
    """Run parts A-D; returns the number of failures. days and n (seeds per herding strength)
    size part B; tests call it tiny. Offline by default so tests never need the internet."""
    failures.clear()
    lines.clear()
    t0 = time.perf_counter()
    say(f"# Web app stress test ({time.strftime('%Y-%m-%d %H:%M')})\n")
    part_a()
    part_b(days, n)
    part_c(offline)
    main_module = sys.modules.get("__main__")
    try:
        at = part_d()
        if at is not None:
            part_e(at)
        else:
            say("## E. skipped: a page in part D did not finish\n")
    finally:                      # part E's page runs also install app.py as __main__ (see _run_page)
        sys.modules["__main__"] = main_module
    say(f"## Result\n\n{len(failures)} failures in {time.perf_counter() - t0:.0f} s.")
    for f in failures:
        say(f"- {f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    return len(failures)


if __name__ == "__main__":
    sys.exit(1 if main(offline="--offline" in sys.argv) else 0)
