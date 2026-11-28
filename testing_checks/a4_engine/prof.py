import sys, time, cProfile, pstats, statistics as st
sys.path.insert(0, "/Users/avishjain/02-120 Project")
from herding import explore as ex, model
p,burn=ex.fitted_settings(); days=19329
ex.run_mix(p,"herding",p["a1"],600,1,0)
pr=cProfile.Profile(); pr.enable()
for i in range(5): ex.run_mix(p,"herding",p["a1"],days,1000,burn)
pr.disable(); pstats.Stats(pr).sort_stats("cumtime").print_stats(12)
r,_=ex.run_mix(p,"herding",p["a1"],days,1000,burn)
def T(f,n=5):
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); f(); ts.append(time.perf_counter()-t)
    return round(st.median(ts),4)
print("summary(r)", T(lambda: ex.summary(r)))
for nm in ["tail_curve","histogram","memory_curve","fact_values","crash_counts","moves_in_sd","max_drawdown"]:
    try: print(nm, T(lambda: getattr(ex,nm)(r)))
    except Exception as e: print(nm,"err",e)
out=model.simulate(dict(p),days+burn,seed=1000)
print("market_paths", T(lambda: ex.market_paths(out,burn)))
print("sim only herding", T(lambda: model.simulate(dict(p),days+burn,seed=1000)))
print("party_shocks 0", T(lambda: ex.party_shocks(days,burn,0.0)))
