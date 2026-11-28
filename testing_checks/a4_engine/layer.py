import sys, time, logging
sys.path.insert(0,"/Users/avishjain/02-120 Project"); sys.path.insert(0,"/Users/avishjain/02-120 Project/web")
logging.disable(logging.CRITICAL)
import numpy as np
from herding import explore as ex
p,burn=ex.fitted_settings(); run=dict(a1=p["a1"],a2=p["a2"],v1=p["v1"],days=19329,seed=1000)
print("default seed ran away day:")
for mix in ex.MIXES:
    t=time.perf_counter(); r,e=ex.run_mix(p,mix,p["a1"] if mix=="herding" else 0,19329,1000,burn); print(" ",mix,"ran" if r is not None else "ran away day %s"%e.day,round(time.perf_counter()-t,3))
for mix in ex.MIXES:
    t=time.perf_counter()
    try: mm=ex.model_market(p,burn,mix,run["a1"],run["a2"],run["v1"],19329,1000)
    except ValueError as e: mm=None
    print(" model_market",mix,round(time.perf_counter()-t,3))
t=time.perf_counter(); ex.runaway_story(p,1000,ex.runaway_day(ex.run_mix(p,"chartists",0,19329,1000,burn)[1],burn) if False else 300,burn); print(" runaway_story(day300)",round(time.perf_counter()-t,3))
