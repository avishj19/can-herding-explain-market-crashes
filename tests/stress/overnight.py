"""Overnight stress test of the test suite (November 2026): are the tests right, and are
they strong enough to catch bugs? Two experiments, then a report. Both are resumable:
finished jobs are skipped on a rerun. Jobs run in parallel, each worker in its own copy of
the project (scratch/overnight/sandbox/wN), so no two runs ever share a file.

  A  Seed sweep. The whole suite (slow tests too) is rerun with every random seed changed
     (sitecustomize.py in this folder). A correct test must pass for any seed, not just the
     ones it was written with. The tolerance recorder logs each pytest.approx comparison,
     so the spread across shifts gives each noisy number's sd, and its tolerance in sd.
     Shift 0 is the suite as written, run twice to check it repeats bit for bit.
  B  Mutation testing. One small deliberate bug at a time ("+" to "-", "<" to "<=", a
     number plus 1, "and" to "or", a minus sign dropped) is put into a copy of herding/,
     and the tests that use that module are run. A good suite fails ("kills" the bug).
     A bug that survives marks behaviour that no test checks.
  C  Report: scratch/overnight/REPORT.md (summarised in docs/TEST_ACCURACY.md).

Tests marked pinned_seed (conftest.py) describe one documented history and are skipped in
the shifted runs. To recheck a few tests on 40 shifts after a fix:
    python3 tests/stress/overnight.py --shifts 40 --check tests/test_model.py::test_x ...

Run from the project folder (logs progress; Ctrl-C and rerun to resume):
    caffeinate -i -s python3 -u tests/stress/overnight.py [--shifts K] [--until HH:MM]
        [--stage ABC] [--workers-a N] [--workers-b N] [--modules herding/model.py ...]
"""
import argparse
import ast
import csv
import datetime as dt
import json
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scratch" / "overnight"
COPY = ["herding", "tests", "scripts", "settings", "docs", "figures", ".claude", ".vscode",
        ".gitignore", "CLAUDE.md", "README.md", "requirements.txt"]
LOCK = threading.Lock()


def log(msg):
    with LOCK:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---------- sandboxes ----------

def make_sandbox(i):
    """A private copy of the project for worker i (data/ is shared read-only by a link)."""
    d = OUT / "sandbox" / f"w{i}"
    if not (d / "herding").exists():
        d.mkdir(parents=True, exist_ok=True)
        for name in COPY:
            src = ROOT / name
            if src.is_dir():
                shutil.copytree(src, d / name, ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pdf", ".pytest_cache"))
            elif src.exists():
                shutil.copy2(src, d / name)
        (d / "data").symlink_to(ROOT / "data")
        (d / "scratch").mkdir(exist_ok=True)
    return d


def pytest_cmd(args):
    return [sys.executable, "-m", "pytest", *args, "-q", "-p", "no:cacheprovider"]


# ---------- stage A: seed sweep ----------

def shift_names(k):
    """Shift 0 twice (does the suite repeat bit for bit?), then shifts 1..k-1."""
    return ["0", "0r"] + [str(i) for i in range(1, k)]


def run_shift(name, sb):
    xml, tol = OUT / "seeds" / f"shift_{name}.xml", OUT / "seeds" / f"shift_{name}.csv"
    if xml.exists():
        return
    env = dict(os.environ, PYTHONPATH=f"{sb / 'tests' / 'stress'}:{sb / 'tests'}",
               TOLERANCE_CSV=str(tol))
    env.pop("HERDING_SEED_SHIFT", None)
    if not name.startswith("0"):
        env["HERDING_SEED_SHIFT"] = name
    shutil.rmtree(sb / "tmp", ignore_errors=True)
    t = time.time()
    marks = [] if name.startswith("0") else ["-m", "not pinned_seed"]   # see conftest.py
    p = subprocess.run(pytest_cmd(["tests/", *marks, "-p", "tolerance_recorder", f"--junitxml={xml}.part",
                                   f"--basetemp={sb / 'tmp'}"]),
                       cwd=sb, env=env, capture_output=True, text=True, timeout=3 * 3600)
    (OUT / "seeds" / f"shift_{name}.log").write_text(p.stdout[-20000:] + p.stderr[-5000:])
    os.replace(f"{xml}.part", xml)
    shutil.rmtree(sb / "tmp", ignore_errors=True)
    log(f"A shift {name}: {p.stdout.strip().splitlines()[-1] if p.stdout.strip() else '?'} "
        f"({time.time() - t:.0f}s)")


# ---------- stage B: mutation testing ----------

BINOPS = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult,
          ast.FloorDiv: ast.Div, ast.Pow: ast.Mult, ast.Mod: ast.Mult}
CMPOPS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
          ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is, ast.In: ast.NotIn,
          ast.NotIn: ast.In}
BOOLOPS = {ast.And: ast.Or, ast.Or: ast.And}


class Mutator(ast.NodeTransformer):
    """Walks a module; changes only mutation site number `target` (-1: count sites)."""

    def __init__(self, target=-1):
        self.target, self.n, self.sites, self.func = target, 0, [], "<module>"

    def _site(self, node, what):
        hit = self.n == self.target
        self.sites.append((self.n, node.lineno, self.func, what))
        self.n += 1
        return hit

    def visit_FunctionDef(self, node):
        outer, self.func = self.func, node.name
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]                        # leave docstrings alone
        node.body = node.body[:len(node.body) - len(body)] + [self.visit(s) for s in body]
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        self.func = outer
        return node

    def visit_JoinedStr(self, node):               # f-strings: messages, not behaviour
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        new = BINOPS.get(type(node.op))
        if new and self._site(node, f"{type(node.op).__name__} -> {new.__name__}"):
            node.op = new()
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        new = BINOPS.get(type(node.op))
        if new and self._site(node, f"{type(node.op).__name__}= -> {new.__name__}="):
            node.op = new()
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        for i, op in enumerate(node.ops):
            new = CMPOPS.get(type(op))
            if new and self._site(node, f"{type(op).__name__} -> {new.__name__}"):
                node.ops[i] = new()
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        new = BOOLOPS[type(node.op)]
        if self._site(node, f"{type(node.op).__name__} -> {new.__name__}"):
            node.op = new()
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, (ast.USub, ast.Not)) and \
                self._site(node, f"drop {type(node.op).__name__}"):
            return node.operand
        return node

    def visit_Constant(self, node):
        v = node.value
        if isinstance(v, (int, float)) and not isinstance(v, bool) and \
                self._site(node, f"{v!r} -> {v + 1!r}"):
            return ast.copy_location(ast.Constant(v + 1), node)
        return node


def mutate(src, target):
    """(mutated source, site description) for site `target`."""
    m = Mutator(target)
    tree = m.visit(ast.parse(src))
    return ast.unparse(ast.fix_missing_locations(tree)), m.sites[target]


def sites(src):
    m = Mutator()
    m.visit(ast.parse(src))
    return m.sites


def mutant_key(rel, source, target):
    """Name of one mutant. It includes a fingerprint of the file, because site numbers
    move when the file is edited: a resumed run must not reuse results from older code
    (this happened once, Nov 17)."""
    import hashlib
    return f"{rel}@{hashlib.sha1(source.encode()).hexdigest()[:8]}:{target}"


def tests_for(module, durations):
    """Test files that use `module`, its own test file first, then fastest first."""
    files = []
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        if re.search(rf"\b{module}\b", f.read_text()):
            files.append(f.name)
    own = f"test_{module}.py"
    files.sort(key=lambda f: (f != own, durations.get(f, 0)))
    return files


def file_durations():
    """Seconds per test file, from the shift-0 run of stage A."""
    xml = OUT / "seeds" / "shift_0.xml"
    out = {}
    if xml.exists():
        for case in ET.parse(xml).iter("testcase"):
            f = case.get("classname", "").split(".")[-1] + ".py"
            out[f] = out.get(f, 0) + float(case.get("time", 0))
    return out


def run_mutant(job, sb):
    rel, target, files, budget = job              # rel: e.g. "herding/model.py"
    original = (ROOT / rel).read_text()
    key = mutant_key(rel, original, target)
    src_path = sb / rel
    if target == -1:                                # the unmutated code, written the same way
        code, site = ast.unparse(ast.parse(original)), (-1, 0, "-", "none (control)")
    else:
        code, site = mutate(original, target)
    src_path.write_text(code)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", NUMBA_CACHE_DIR=str(sb / "numba_cache"))
    env.pop("HERDING_SEED_SHIFT", None)
    shutil.rmtree(sb / "tmp", ignore_errors=True)
    t = time.time()
    try:
        p = subprocess.run(pytest_cmd([*(f"tests/{f}" for f in files), "-x", "-m", "not slow",
                                       f"--basetemp={sb / 'tmp'}"]),
                           cwd=sb, env=env, capture_output=True, text=True, timeout=budget)
        status = "survived" if p.returncode == 0 else ("killed" if p.returncode == 1 else
                                                         f"error{p.returncode}")
        tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        failed = re.findall(r"^FAILED (\S+)", p.stdout, re.M)[:1]
    except subprocess.TimeoutExpired:
        status, tail, failed = "timeout", "", []
    finally:
        src_path.write_text(original)
        shutil.rmtree(sb / "numba_cache", ignore_errors=True)
    line = original.splitlines()[site[1] - 1].strip() if site[1] else ""
    row = dict(key=key, module=rel, site=target, line=site[1], func=site[2], change=site[3],
               code=line, status=status, seconds=round(time.time() - t, 1), summary=tail,
               caught_by=failed[0] if failed else "")
    with LOCK:
        with (OUT / "mutants.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    return row


# ---------- the parallel runner ----------

def run_all(jobs, fn, workers, deadline=None):
    boxes = queue.Queue()
    for i in range(workers):
        boxes.put(make_sandbox(i))

    def one(job):
        if deadline and time.time() > deadline:
            return None
        sb = boxes.get()
        try:
            return fn(job, sb)
        except Exception as e:                      # one broken job must not stop the night
            log(f"job {job!r} failed: {e!r}")
        finally:
            boxes.put(sb)

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, jobs))


def stage_a(workers, shifts):
    (OUT / "seeds").mkdir(parents=True, exist_ok=True)
    names = [n for n in shift_names(shifts) if not (OUT / "seeds" / f"shift_{n}.xml").exists()]
    log(f"A: {len(names)} suite runs to do on {workers} workers")
    run_all(names, run_shift, workers)


def stage_b(workers, deadline, modules=None):
    durations = file_durations()
    done = set()
    if (OUT / "mutants.jsonl").exists():
        done = {json.loads(l)["key"] for l in (OUT / "mutants.jsonl").read_text().splitlines() if l}
    modules = modules or [str(p.relative_to(ROOT)) for p in
                          sorted((ROOT / "herding").glob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
                          + sorted((ROOT / "scripts" / "analysis").glob("*.py"))
                          if p.stem not in ("__init__", "test_accuracy_audit")]
    jobs, controls = [], []
    for rel in modules:
        files = tests_for(Path(rel).stem, durations)
        if not files:
            continue
        budget = 120 + 3 * sum(durations.get(f, 60) for f in files)
        controls.append((rel, -1, files, budget))
        n = len(sites((ROOT / rel).read_text()))
        jobs += [(rel, i, files, budget) for i in range(n)]
    random.Random(0).shuffle(jobs)                  # any stopping point is a fair sample
    # the model's own code first, then the scripts, the drawing code last
    jobs.sort(key=lambda j: 2 if j[0].endswith("plots.py") else j[0].startswith("scripts"))
    sources = {rel: (ROOT / rel).read_text() for rel in modules}
    todo = [j for j in controls + jobs if mutant_key(j[0], sources[j[0]], j[1]) not in done]
    log(f"B: {len(jobs)} mutants in {len(controls)} modules, {len(todo)} left, "
        f"until {time.strftime('%H:%M', time.localtime(deadline))}")
    return run_all(todo, _progress(run_mutant, len(todo)), workers, deadline)


def _progress(fn, total):
    count = {"n": 0, "killed": 0}

    def wrapped(job, sb):
        row = fn(job, sb)
        with LOCK:
            count["n"] += 1
            count["killed"] += row["status"] != "survived"
            n = count["n"]
        if row["site"] == -1 and row["status"] != "survived":     # control must pass
            log(f"B CONTROL FAILED for {row['module']}: {row['summary']}")
        if n % 25 == 0 or row["site"] == -1:
            log(f"B {n}/{total}: {row['key']} {row['status']} ({row['seconds']}s); "
                f"killed so far {count['killed']}/{n}")
        return row
    return wrapped


# ---------- stage C: report ----------

def report():
    lines = ["# Overnight stress test", "", f"Made {time.strftime('%Y-%m-%d %H:%M')}.", ""]
    lines += seed_report() + [""] + mutant_report()
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    log(f"C: wrote {OUT / 'REPORT.md'}")


def seed_report():
    seeds = OUT / "seeds"
    xmls = sorted(seeds.glob("shift_*.xml"))
    if not xmls:
        return ["## A. Seed sweep", "", "not run"]
    outcome = {}                                    # test -> {shift: "pass"/"fail"/"skip"}
    for x in xmls:
        name = x.stem[6:]
        for c in ET.parse(x).iter("testcase"):
            tid = f"{c.get('classname')}::{c.get('name')}"
            r = "fail" if c.find("failure") is not None or c.find("error") is not None else \
                "skip" if c.find("skipped") is not None else "pass"
            outcome.setdefault(tid, {})[name] = r
    shifted = [x.stem[6:] for x in xmls if not x.stem[6:].startswith("0")]
    out = ["## A. Seed sweep", "",
           f"{len(xmls)} full runs: the suite as written (twice) and {len(shifted)} with every "
           f"seed changed. {len(outcome)} tests.", ""]
    base_fail = [t for t, o in outcome.items() if o.get("0") == "fail" or o.get("0r") == "fail"]
    out.append(f"Failures with the seeds as written: {len(base_fail)} " + str(base_fail[:20]))
    rows = []
    for t, o in outcome.items():
        f = sum(o.get(s) == "fail" for s in shifted)
        if f:
            rows.append((f, t))
    rows.sort(reverse=True)
    out += ["", f"Tests that failed under some changed seeds ({len(rows)}):", "",
            "| failed | test |", "|---|---|"]
    out += [f"| {f}/{len(shifted)} | `{t}` |" for f, t in rows]
    out += [""] + repeat_check() + [""] + spread_report(shifted)
    return out


def _read_rows(path):
    by = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            by.setdefault(r["test"], []).append(r)
    return by


def repeat_check():
    a, b = OUT / "seeds" / "shift_0.csv", OUT / "seeds" / "shift_0r.csv"
    if not (a.exists() and b.exists()):
        return []
    ra, rb = _read_rows(a), _read_rows(b)
    diff = [t for t in ra if [r["actual"] for r in ra[t]] != [r["actual"] for r in rb.get(t, [])]]
    n = sum(len(v) for v in ra.values())
    return [f"Repeatability: the suite as written, run twice: {n} recorded numbers, "
            f"{len(diff)} tests with any difference " + str(diff[:10])]


def spread_report(shifted):
    """Every pytest.approx comparison seen in every shifted run, in two kinds:
    - noisy estimates (the expected answer is the same in every run, the answer is not):
      margin = (tolerance - |average error|) / sd of the error; under 4 is fragile, and
      the tests aim for 5 (docs/TEST_ACCURACY.md);
    - exact answers on random data (the expected answer changes with the seed too): the
      largest share of its tolerance any run used."""
    runs = [_read_rows(OUT / "seeds" / f"shift_{s}.csv") for s in shifted
            if (OUT / "seeds" / f"shift_{s}.csv").exists()]
    if len(runs) < 5:
        return []
    noisy, exact = [], []
    for t in runs[0]:
        k = len(runs[0][t])
        if not all(len(r.get(t, [])) == k for r in runs):
            continue                                # comparison count depends on the seed
        for i in range(k):
            act = np.array([float(r[t][i]["actual"]) for r in runs])
            exp = np.array([float(r[t][i]["expected"]) for r in runs])
            allowed = np.array([float(r[t][i]["allowed"]) for r in runs])
            err = act - exp
            if np.ptp(exp) > 0:
                exact.append((float(np.max(np.abs(err) / allowed)), t, i))
                continue
            sd = err.std(ddof=1)
            if not sd > 1e-6 * allowed.min():
                continue                            # does not depend on the seed
            noisy.append(((allowed.min() - abs(err.mean())) / sd, t, i, exp[0], err.mean(), sd,
                          allowed.min()))
    noisy.sort()
    exact.sort(reverse=True)
    out = [f"Noisy estimates checked with pytest.approx: {len(noisy)} (over {len(runs)} shifts). "
           f"Margin = (tolerance - |average error|) / sd; under 4 is fragile. With {len(runs)} "
           f"runs an sd is only known to about {100 / np.sqrt(2 * (len(runs) - 1)):.0f}%, so "
           f"margins of 4-5 can be tolerances set to 5 sd.", "",
           "| margin (sd) | test | # | expected | avg error | sd | tolerance |", "|---|---|---|---|---|---|---|"]
    out += [f"| {m:.1f} | `{t}` | {i} | {e:.4g} | {b:+.2e} | {sd:.2e} | {a:.2e} |"
            for m, t, i, e, b, sd, a in noisy if m < 4.5]
    out += ["", f"Exact answers on random data: {len(exact)} comparisons; the most any run used "
            f"of its tolerance:", ""]
    out += [f"- {u:.3g} of the tolerance: `{t}` #{i}" for u, t, i in exact[:8]]
    with (OUT / "seed_spread.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["margin_sd", "test", "index", "expected", "avg_error", "sd", "tolerance"])
        w.writerows(noisy)
    return out


def mutant_report():
    path = OUT / "mutants.jsonl"
    if not path.exists():
        return ["## B. Mutation testing", "", "not run"]
    rows = [json.loads(l) for l in path.read_text().splitlines() if l]
    real = [r for r in rows if r["site"] != -1]
    controls = [r for r in rows if r["site"] == -1]
    out = ["## B. Mutation testing", "",
           "Controls (unmutated code, rewritten the same way; must pass): " +
           ", ".join(f"{r['module']} {'passed' if r['status'] == 'survived' else 'FAILED'}"
                     for r in controls), "",
           "| module | mutants | killed | timeout | error | survived | score |", "|---|---|---|---|---|---|---|"]
    for m in sorted({r["module"] for r in real}):
        rs = [r for r in real if r["module"] == m]
        c = {s: sum(r["status"] == s for r in rs) for s in ("killed", "timeout", "survived")}
        err = len(rs) - sum(c.values())
        out.append(f"| {m} | {len(rs)} | {c['killed']} | {c['timeout']} | {err} | {c['survived']} | "
                   f"{100 * (len(rs) - c['survived']) / len(rs):.0f}% |")
    surv = sorted((r for r in real if r["status"] == "survived"), key=lambda r: (r["module"], r["line"]))
    out += ["", f"Survivors ({len(surv)}): bugs no test noticed (some may change nothing, "
            "e.g. a + 1 on an unused default).", ""]
    out += [f"- `{r['module']}.py:{r['line']}` in `{r['func']}`: {r['change']}  —  `{r['code'][:90]}`"
            for r in surv]
    return out


def check(tests, shifts, workers):
    """Run some tests (pytest node ids) under shifts 1..shifts in the project folder, in
    parallel, and print how many shifts each test failed: the quick way to confirm a fix."""
    def one(k):
        env = dict(os.environ, PYTHONPATH=f"{ROOT / 'tests' / 'stress'}:{ROOT / 'tests'}",
                   HERDING_SEED_SHIFT=str(k))
        tmp = OUT / "check_tmp" / str(k)
        p = subprocess.run(pytest_cmd([*tests, "-m", "not pinned_seed", "-rf", f"--basetemp={tmp}"]),
                           cwd=ROOT, env=env,
                           capture_output=True, text=True)
        shutil.rmtree(tmp, ignore_errors=True)
        return k, re.findall(r"^FAILED (\S+)", p.stdout, re.M), p.stdout.strip().splitlines()[-1:]
    fails = {}
    with ThreadPoolExecutor(workers) as ex:
        for k, failed, last in ex.map(one, range(1, shifts + 1)):
            for t in failed:
                fails.setdefault(t, []).append(k)
            if not failed and "passed" not in " ".join(last):
                log(f"shift {k}: {last}")
    log(f"{len(tests)} test target(s) on {shifts} shifts: " +
        ("all passed" if not fails else "; ".join(f"{t} failed on shifts {v}" for t, v in fails.items())))
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs="*", help="only rerun these tests on --shifts shifts")
    # 8 GB of memory, not the 8 cores, is the limit here: a full suite run uses about 3 cores
    # and 1-2 GB, so 4 at once pushed the Mac into swap (trial, 17 Nov). Mutant runs are lighter.
    cores = os.cpu_count() or 2
    ap.add_argument("--workers-a", type=int, default=max(1, min(3, cores // 2)))
    ap.add_argument("--workers-b", type=int, default=max(1, min(5, cores - 2)))
    ap.add_argument("--shifts", type=int, default=40)
    ap.add_argument("--until", default="08:30", help="stop starting mutants after HH:MM")
    ap.add_argument("--stage", default="ABC")
    ap.add_argument("--modules", nargs="*")
    ap.add_argument("--out", help="results folder (default scratch/overnight)")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out).resolve()
    now = dt.datetime.now()
    h, m = map(int, a.until.split(":"))
    stop = now.replace(hour=h, minute=m, second=0)
    if stop < now:
        stop += dt.timedelta(days=1)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.check:
        check(a.check, a.shifts, a.workers_b)
        return
    log(f"overnight stress test: stages {a.stage}, workers A {a.workers_a}, B {a.workers_b}")
    if "A" in a.stage:
        stage_a(a.workers_a, a.shifts)
    if "B" in a.stage:
        stage_b(a.workers_b, stop.timestamp(), a.modules)
    if "C" in a.stage:
        report()


if __name__ == "__main__":
    main()
