"""Bitwise identity of a prototype model module vs the project's herding.model (and timing, interleaved)."""
import sys, os, time, importlib, tracemalloc, statistics as st
sys.path.insert(0,"/Users/avishjain/02-120 Project"); sys.path.insert(0,".")
import numpy as np
from herding import explore as ex, model as base
pm = importlib.import_module(sys.argv[1])
p,burn=ex.fitted_settings()
def run(m,q,days,seed,**kw):
    try:
        o=m.simulate(q,days,seed=seed,**kw); return ("ok",o)
    except m.RunawayError as e: return ("runaway",(str(e),e.day))
def same(a,b):
    if a[0]!=b[0]: return False
    if a[0]=="runaway": return a[1]==b[1]
    return all(np.array_equal(a[1][k],b[1][k]) and a[1][k].tobytes()==b[1][k].tobytes() for k in a[1])
n=bad=0; ran=0; maxrel=0.0
cases=[]
for seed in range(1000,1024):                                   # 24 seeds x 3 mixes, recipe length
    for mix in ex.MIXES:
        q,ch=ex.mix_settings(p,mix,p["a1"]); cases.append((q,19329+burn,seed,dict(chartists=ch)))
for seed in range(0,6):                                         # short runs, SET_I, other settings
    cases.append((dict(base.SET_I),3000,seed,dict(chartists=True)))
    cases.append((dict(base.SET_I),3000,seed,dict(chartists=False)))
for seed in range(2000,2004):                                   # panic, breakers, shocks, max_gap None
    q,ch=ex.mix_settings(p,"herding",p["a1"])
    cases.append((q,5000,seed,dict(panic=2.0)))
    cases.append((q,5000,seed,dict(breakers=base.US_BREAKERS)))
    cases.append((q,5000,seed,dict(shocks={100:-0.2,2000:0.1})))
    q0,_=ex.mix_settings(p,"chartists",0)
    cases.append((q0,3000,seed,dict(max_gap=None)))
    cases.append((q0,20000,seed,dict(max_days=5)))
for q,days,seed,kw in cases:
    a=run(base,q,days,seed,**kw); b=run(pm,q,days,seed,**kw)
    n+=1; ran+=a[0]=="runaway"
    if not same(a,b): bad+=1; print("MISMATCH",seed,kw)
print(f"{sys.argv[1]}: {n} cases, {ran} of them runaways, mismatches (bitwise): {bad}")
# timing, interleaved, min and median of 15
q,ch=ex.mix_settings(p,"herding",p["a1"]); qf,_=ex.mix_settings(p,"fundamentalists",0)
for name,qq,c in [("herding",q,True),("fundamentalists",qf,False)]:
    tb=[];tp=[]
    for i in range(15):
        t=time.perf_counter(); base.simulate(qq,19329+burn,seed=1000,chartists=c); tb.append(time.perf_counter()-t)
        t=time.perf_counter(); pm.simulate(qq,19329+burn,seed=1000,chartists=c); tp.append(time.perf_counter()-t)
    print(f"  {name}: base median {st.median(tb):.4f} min {min(tb):.4f} | proto median {st.median(tp):.4f} min {min(tp):.4f} | min ratio {min(tp)/min(tb):.3f}")
