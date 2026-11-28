# a4: engine performance (herding/model.py via explore.py), 28 Nov 2026

Machine: MacBook (Mac17,13, 15 cores), Python 3.14, numba cache on. Nothing in herding/, web/ or scripts/ was edited.
A second agent was running simulations during part of this, so single timings jump by up to 10%; I used medians and interleaved min-of-15.
Recipe: web/numbers.json, 19,329 days + 1,000 burn-in = 20,329 days = 2.03 million sub-steps (dt = 0.01), seed 1000, S&P fit.

## 1. Timings (current engine)

| What | Time |
|---|---|
| import herding.explore | 0.25-0.34 s |
| first simulate(3 days), numba cache loaded (normal server restart) | 0.12-0.15 s |
| first simulate(3 days), NO numba cache (fresh Cloud server, true compile) | 1.06-1.18 s here (the code comment says ~2.6 s on Cloud) |
| one history, fundamentalists | 0.065 s (median of 7) |
| one history, chartists without herding | 0.09 s on seed 1000 (it runs away on day 8,282 and stops early; a full run would cost about the herding time) |
| one history, herding | 0.188 s (median of 7; 0.19 on every seed) |
| run_mixes, three mixes (3 threads) | 0.197 s (median of 7); serial sum would be 0.34 s |
| run_mixes with paths=... (site call) | 0.197 s |
| model_market herding / fundamentalists / chartists | 0.19 / 0.07 / 0.18 s |
| numba off (MODEL_NO_NUMBA=1), herding, 2,000 days | 0.63 s, i.e. about 3.3 s per 20k days = ~17x slower |
| memory per history | peak 1.45 MB, 0.77 MB kept as paths (5 arrays) + 0.15 MB returns |

So an S&P-length history is about 93 ns per sub-step with chartists, 32 ns without. run_mixes is limited by the herding history alone.

## 2. Where the time goes

- Python layer: cProfile of run_mix: 0.944 of 0.945 s is inside the numba call. check(), is_number, mix_settings, diff, market_paths, party_shocks(0) are each under 0.1 ms. summary(r) 0.7 ms, fact_values 1.5 ms, tail_curve 0.4 ms, histogram 0.4 ms. The Python layer is negligible; the kernel is everything.
- Kernel, per sub-step (micro-benchmarks of the numba generator, ns per call): standard_normal 3, poisson 3-6, binomial 6-14 (small n p; 84 when n p ~ 20), np.exp 6. A chartist step makes 3 to 6 binomials (3 `_leave` calls, a second draw only if someone left), 2 normals, 1 poisson and 11 np.exp calls. Roughly: exps ~60 ns, RNG draws ~25-30 ns, rest small. Fundamentalists-only does 2 normals, 1 poisson and 2 exps.
- The random draws cannot be reordered or skipped (the project rule is the same numbers for the same seed), and the exp calls have different arguments (exp(-U) is not bit-equal to 1/exp(U)), so there is no common subexpression to remove.
- fastmath or FMA would change the last bits (and the draws that depend on them), so they are excluded by the identity rule.

## 3. Candidates, with identity proof

Prototypes: proto_model.py, proto2_model.py (copies, with the relative import changed); identity.py compares with herding.model.
Identity test: 104 cases, 30 of them runaways: 24 seeds x 3 mixes at recipe length (1000-1023), SET_I with and without chartists, panic=2, US circuit breakers, policy shocks, max_gap=None, max_days=5. Compared BITWISE (every output array's bytes, RunawayError message and day), stricter than 12 digits.

| Candidate | Patch | Identity | Speed-up (min of 15, interleaved) |
|---|---|---|---|
| A. numba error_model="numpy" on `_simulate`, `_leave`, `_cap` (env switch PROTO_EM in proto_model.py) | one keyword per decorator | 104/104 identical | none: herding 0.999x, fundamentalists 0.997x |
| B. ring-buffer index instead of `t % window`, hoist `v2 * n_f / N` | ring_hoist.diff (against herding/model.py with only the import line differing) | 104/104 identical | herding 0.972x, fundamentalists 0.979x, i.e. 2-3%, inside run-to-run noise (an earlier non-interleaved run showed it slower) |

Neither is worth applying: a 2-3% gain on 0.19 s is 5 ms, and every touch of model.py invalidates the numba cache (a 1.1-2.6 s recompile on first load) and re-opens the pinned-seed tests.

## 4. The layer above (web/common.py)

Already well done: the three mixes run on threads (nogil), `_MIX_RESULTS` keys each mix by the settings it depends on (so moving only the herding slider runs one 0.19 s history, not three), `_MARKETS` lets model_market reuse run_mixes' day-by-day paths (model_market("herding") costs 0 after run_mixes), results are cached by fit-file stamp, `warm_model()` compiles on a thread and `_warm_pages()` pre-fills the Story's default history and party tables. Memory is bounded (about 20 MB + 19 MB).

Small things found:
1. A history that RAN AWAY is never put in `_MARKETS` (paths are only filled when not ran away), so model_market(mix="chartists") on the default seed re-simulates it: measured 0.178 s versus 0.089 s for the original run, since model_market runs it once to find the runaway day (a second time) and once more with the guard off. Only paid if a visitor opens the Model market panel for that mix, once per settings (then cached by `_model_market`, 30 entries). Fix, if wanted: have run_mix fill `paths` also for a runaway (market_paths of a rerun is what model_market does anyway), or store the guard-off prefix. Saving: at most about 0.18 s per rare view. Not prototyped; touches explore.py and common.py.
2. `_run_mixes` has max_entries=60 and `_MIX_RESULTS` 120, fine; the "earlier histories" toggle calls run_mixes for up to four old settings, which are mostly cache hits.
3. Nothing is recomputed on every page load that I could find: the tape quotes, party_view, numbers() and fitted() are cache_data. Page loads cost the Python/Streamlit rendering, not the engine (every run is one 0.2 s call).
4. First visit on a fresh server pays the numba compile (1.1 s here, ~2.6 s on Cloud) unless warm_model has finished; it starts in `setup()`, so only the very first visitor, if they press a control within ~3 s, waits. The cache files live in __pycache__ (git-ignored), so every Cloud reboot recompiles. Shipping a cache is not safe (it is keyed to CPU and file mtime), so I would leave this.
5. scripts/web_numbers.py (not run: it writes web/numbers.json): about 120 + 110 histories at 0.2 s plus eight 100,000-day paper runs at about 1 s, on a Pool(): about 60 CPU-seconds, so about 5-10 s on 15 cores. Nothing to gain.

## 5. Recommendation

Do not apply anything. The engine is already at about 93 ns per sub-step (ns-level RNG and exp calls), a history takes 0.19 s, the Python layer is under 1 ms, and the only identical-output tweaks I found (A, B) give 0-3%, below noise. A real speed-up would need fastmath, a cheaper exp, or a different random-number scheme, and each changes the numbers, which the project forbids. Risk of applying A or B is low for results (bitwise identical on 104 cases) but nonzero for process (numba cache invalidation, pinned-seed tests, rule 6 of memory.md: no new features before Dec 1; and this is not a fix). The one worthwhile (still small) item is 4.1, only if the chartists' Model market panel feels slow; I would skip it before the deadline.

Files: bench1.py (timings), prof.py (cProfile and stats timings), micro.py (RNG/exp costs), cold.py (cold start), layer.py (site-layer reruns), identity.py (proof and interleaved timing), orig_model.py / proto_model.py / proto2_model.py (copies), ring_hoist.diff (candidate B).
