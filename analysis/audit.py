import sys, warnings; sys.path.insert(0, "/Users/avishjain/02-120 Project")
import types
import numpy as np
from herding import model, stats
main = types.SimpleNamespace(**{**vars(model), **vars(stats)})   # these probes predate the split into modules
# 1. how often are switching probabilities clipped at 1, or exponents at U_MAX?
calls = {"p": 0, "p_clip": 0}
orig = main._p
def spy(rate):
    calls["p"] += 1
    if rate >= 1 or rate < 0: calls["p_clip"] += 1
    return orig(rate)
main._p = spy   # NOTE: no effect on the numba version; run with MODEL_NO_JIT=1
for N in (500, 250):
    calls.update(p=0, p_clip=0)
    main.simulate(1500, p=dict(main.P, N=N), seed=1)
    print(f"N={N}: probabilities clipped at 1 in {calls['p_clip']} of {calls['p']} draws")
main._p = orig
# 2. dt that does not divide 1 -> day silently has the wrong length
for dt in (0.02, 0.03, 0.3, 0.7):
    print(f"dt={dt}: steps/day={int(round(1/dt))}, simulated day length={int(round(1/dt))*dt:.2f}")
# 3. bad inputs that are silently accepted
for name, kw in [("days=-5", dict(days=-5)), ("N=5", dict(days=10, p=dict(main.P, N=5))),
                 ("dt=2", dict(days=10, p=dict(main.P, dt=2.0))), ("N=0", dict(days=10, p=dict(main.P, N=0)))]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, f = main.simulate(seed=0, **kw)
        print(f"{name}: accepted silently, len={len(r)}, finite={np.isfinite(r).all()}")
    except Exception as e:
        print(f"{name}: {type(e).__name__}: {e}")
# 4. kurtosis / ccdf on constant input
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    k = main.kurtosis(np.ones(5)); z, c = main.ccdf(np.ones(5))
    print(f"constant input: kurtosis={k}, ccdf x={z[:2]}, warnings={len(w)}")
# 5. empty input
for fn in (main.kurtosis, main.ccdf):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); print(fn.__name__, "empty ->", fn(np.array([])))
    except Exception as e: print(fn.__name__, "empty ->", type(e).__name__, e)
# 6. a2 trend term: does it ever matter?
r1, _ = main.simulate(3000, p=dict(main.P, a2=0.0), seed=4)
r2, _ = main.simulate(3000, seed=4)
print(f"a2=0 vs a2=2: kurtosis {main.kurtosis(r1[200:]):.2f} vs {main.kurtosis(r2[200:]):.2f}, "
      f"alpha {main.hill_alpha(r1[200:]):.2f} vs {main.hill_alpha(r2[200:]):.2f}")
