import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from scipy import stats as ss
from herding import market_data as md, stats
spx=md.returns("spx"); rows=[]
for t in md.stock_tickers():
    px=md.load_prices(md.STOCKS/f"{t}.csv")["Close"]; r=md.stock_returns(t)
    x=pd.concat([r,spx],axis=1,join="inner").dropna(); x.columns=["a","m"]
    b_full=np.cov(x.a,x.m)[0,1]/x.m.var(); xs=x.loc["2005":"2025"]; b_5=np.cov(xs.a,xs.m)[0,1]/xs.m.var()
    z=(r-r.mean())/r.std()
    rows.append(dict(t=t,first=px.index[0].date(),last=px.index[-1].date(),n=len(px),beta_2005_25=round(b_5,3),beta_full=round(b_full,3),corr=round(x.a.corr(x.m),2),kurt=round(ss.kurtosis(r),1),hill=round(float(stats.hill_alpha(r.values)),2),big=int((z.abs()>4).sum()),worst_sd=round(z.min(),1),maxabs_move=round(r.abs().max()*100,1)))
d=pd.DataFrame(rows); print(d.to_string(index=False))
s=md.returns("spx").loc["2000":]; print("spx since 2000: kurt %.1f hill %.2f"%(ss.kurtosis(s),stats.hill_alpha(s.values)))
# jumps check: |log ret| >25% on saved stocks
for t in md.stock_tickers():
    r=md.stock_returns(t); big=r[r.abs()>0.2]
    if len(big): print(t,[(str(i.date()),round(v,2)) for i,v in big.items()])
