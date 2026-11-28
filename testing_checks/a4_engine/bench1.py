import sys, time, statistics as st, subprocess, os
sys.path.insert(0, "/Users/avishjain/02-120 Project")
t0=time.perf_counter()
from herding import explore as ex, model
print("import", round(time.perf_counter()-t0,2))
p,burn=ex.fitted_settings(); print(p,burn)
days=19329
t=time.perf_counter(); ex.run_mix(p,"herding",p["a1"],days,1000,burn); print("first call (cache/compile)", round(time.perf_counter()-t,3))
for mix in ex.MIXES:
    ts=[]
    for i in range(7):
        t=time.perf_counter(); r,e=ex.run_mix(p,mix,p["a1"],days,1000+i,burn); ts.append(time.perf_counter()-t)
    print(mix,"median",round(st.median(ts),3),"min",round(min(ts),3),[round(x,2) for x in ts], "runaway" if r is None else "")
for mix in ex.MIXES:
    ts=[]
    for i in range(7):
        t=time.perf_counter(); r,e=ex.run_mix(p,mix,p["a1"],days,1000,burn); ts.append(time.perf_counter()-t)
    print(mix,"seed1000 median",round(st.median(ts),3), "runaway" if r is None else "")
ts=[]
for i in range(7):
    t=time.perf_counter(); ex.run_mixes(p,burn,p["a1"],p["a2"],p["v1"],days,1000); ts.append(time.perf_counter()-t)
print("run_mixes median",round(st.median(ts),3),[round(x,2) for x in ts])
ts=[]
for i in range(5):
    t=time.perf_counter(); ex.model_market(p,burn,"herding",p["a1"],p["a2"],p["v1"],days,1000); ts.append(time.perf_counter()-t)
print("model_market median",round(st.median(ts),3))
paths={}
ts=[]
for i in range(5):
    paths={}
    t=time.perf_counter(); ex.run_mixes(p,burn,p["a1"],p["a2"],p["v1"],days,1000,paths=paths); ts.append(time.perf_counter()-t)
print("run_mixes with paths median",round(st.median(ts),3))
import resource
print("maxrss MB",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6)
