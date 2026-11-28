import sys, time, statistics as st, importlib
sys.path.insert(0,"/Users/avishjain/02-120 Project"); sys.path.insert(0,".")
import numpy as np
from herding import explore as ex, model as base
import importlib, os; pm = importlib.import_module(os.environ.get("PROTO","proto_model"))
p,burn=ex.fitted_settings(); days=19329+burn
def T(m,mix,seed=1000,n=7):
    ch=mix!="fundamentalists"; q=dict(p); 
    if mix=="chartists": q["a1"]=0.0
    try: m.simulate(q,300,seed=1,chartists=ch)
    except Exception: pass
    ts=[]
    for i in range(n):
        t=time.perf_counter()
        try: m.simulate(q,days,seed=seed,chartists=ch)
        except m.RunawayError: pass
        ts.append(time.perf_counter()-t)
    return st.median(ts)
for mix in ["fundamentalists","herding"]:
    a,b=T(base,mix),T(pm,mix); print(f"{mix}: base {a:.4f}  proto {b:.4f}  ratio {b/a:.3f}")
