import sys, time
sys.path.insert(0, "/Users/avishjain/02-120 Project")
t=time.perf_counter()
from herding import explore as ex, model
ti=time.perf_counter()-t
p,burn=ex.fitted_settings()
t=time.perf_counter(); model.simulate(p,3,seed=0); t3=time.perf_counter()-t
t=time.perf_counter(); ex.run_mix(p,"fundamentalists",0,19329,1000,burn); tf=time.perf_counter()-t
t=time.perf_counter(); ex.run_mix(p,"herding",p["a1"],19329,1000,burn); th=time.perf_counter()-t
t=time.perf_counter(); ex.run_mixes(p,burn,p["a1"],p["a2"],p["v1"],19329,1000); tm=time.perf_counter()-t
print(f"import {ti:.2f}s  first simulate(3d) {t3:.2f}s  then fund {tf:.2f} herd {th:.2f} run_mixes {tm:.2f}")
