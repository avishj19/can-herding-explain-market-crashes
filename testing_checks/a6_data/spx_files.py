import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from scipy import stats as ss
from herding import market_data as md, stats
for m in ("spx","spx_1928"):
    r=md.returns(m); z=(r-r.mean())/r.std()
    print(m,len(r),"kurt %.2f"%ss.kurtosis(r),"worst z %.1f"%z.min(),"max z %.1f"%z.max(), "big(>4sd) %d"%(z.abs()>4).sum())
    try:
        print("  hill",[round(float(stats.hill(r.values,k)),3) if True else 0 for k in (100,200)])
    except Exception as e: print("  hill err",e)
r=md.returns("spx_1928"); print("pre-1950 days",(r.index<"1950-01-01").sum(), "yearly count 1928-1949 avg", (r.index<"1950-01-01").sum()/22)
print("days per year gaps: Saturday trading pre-1952?", (r.index[r.index<"1953-01-01"].dayofweek==5).sum())
