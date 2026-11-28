import math, json, numpy as np
from scipy.stats import chi2, poisson
from scipy.optimize import brentq
import herding.explore as ex, herding.stats as st
# Garwood by root finding on Poisson CDF
def garwood(k, a=0.025):
    lo = 0 if k==0 else brentq(lambda m: 1-poisson.cdf(k-1,m)-a, 1e-12, 1000)
    hi = brentq(lambda m: poisson.cdf(k,m)-a, 1e-12, 1000)
    return lo, hi
for k in [0,1,2,3,10]:
    lo,hi = garwood(k); Y=3068.0
    w = ex.wait_interval(k, Y)
    print(k, (Y/hi, (Y/lo if lo>0 else math.inf)), w)
nums = ex.web_numbers()
spx = ex.log_returns(ex.saved_prices("S&P 500"))
cc = ex.crash_check(nums, spx.values if hasattr(spx,'values') else spx)
print({k:v for k,v in cc.items() if k!='row'}, cc['row'])
band = ex.tail_interval(spx.values)
print("band", band, "again", ex.tail_interval(spx.values))
print("spx hill", st.hill_alpha(spx.values), "n", len(spx))
for b in (20,50,100,250):
    print("block",b, ex.tail_interval(spx.values, block=b), ex.tail_interval(spx.values, block=b, seed=1))
t = ex.tail_check(nums["checks"]["herding"]["tail"], close=band)
print(t)
# edge cases
print(ex.tail_check([2.6]), ex.tail_check([3.0]), ex.tail_check([3.0000001],close=(2.5,3.5)))
try: print(ex.tail_check([3.1], close=(3.0,)))
except Exception as e: print("ERR", type(e).__name__, e)
try: print(ex.tail_check([3.1], close=(float('nan'),3.5)))
except Exception as e: print("ERR", type(e).__name__, e)
print(ex.tail_check([3.1], close=[2.5,3.5]))
for bad in [(1.5,10),(True,10),(2,0),(2,float('nan')),(2,10,1.0),(2.0,10)]:
    try: print(bad, ex.wait_interval(*bad))
    except Exception as e: print(bad,"ERR",e)
