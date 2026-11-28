import numpy as np, numba, time
@numba.njit
def nrm(g,n):
    s=0.0
    for i in range(n): s+=g.standard_normal()
    return s
@numba.njit
def pois(g,n,lam):
    s=0
    for i in range(n): s+=g.poisson(lam)
    return s
@numba.njit
def binom(g,n,N,p):
    s=0
    for i in range(n): s+=g.binomial(N,p)
    return s
@numba.njit
def ex(n):
    s=0.0
    for i in range(n): s+=np.exp(-1e-3*i%3)
    return s
g=np.random.default_rng(1)
N=2_000_000
for name,f in [("normal",lambda:nrm(g,N)),("poisson lam~0.3",lambda:pois(g,N,0.3)),("poisson lam~0.02",lambda:pois(g,N,0.02)),
  ("binom(5,1e-3)",lambda:binom(g,N,5,1e-3)),("binom(50,0.005)",lambda:binom(g,N,50,0.005)),("binom(450,0.001)",lambda:binom(g,N,450,0.001)),("binom(450,0.05)",lambda:binom(g,N,450,0.05)),("exp",lambda:ex(N))]:
    f(); t=time.perf_counter(); f(); print(f"{name:20s} {(time.perf_counter()-t)/N*1e9:6.1f} ns/call")
