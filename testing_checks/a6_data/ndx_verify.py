import numpy as np, pandas as pd
rd=lambda p,c=None:pd.read_csv(p,parse_dates=[0],index_col=0).iloc[:,0]
y=pd.to_numeric(rd('data/nasdaq100_daily.csv'),errors='coerce').dropna()
f=pd.to_numeric(rd('scratch/crosscheck/fred_NASDAQ100.csv'),errors='coerce').dropna()
sp=rd('data/sp500_daily.csv')
both=y.index.intersection(f.index)
rel=((y[both]-f[both])/f[both])
bad=rel[rel.abs()>0.001]
print("days >0.1%:",len(bad)," by year:"); print(bad.groupby(bad.index.year).agg(['count','mean']).T.to_string())
print("share of days >0.1% by year (of common days):")
print((rel.abs()>0.001).groupby(rel.index.year).mean().round(2).to_string())
# which source tracks the S&P? daily log returns on common consecutive days
ry,rf,rs=np.log(y[both]).diff(),np.log(f[both]).diff(),np.log(sp).diff()
d=pd.concat([ry,rf,rs],axis=1,keys=['y','f','s']).dropna()
for lab,sl in [("1986-1994",slice("1986","1994")),("1995-2004",slice("1995","2004")),("2005-2026",slice("2005","2026")),("all",slice(None))]:
    x=d.loc[sl]; print(lab,"n=%d corr(y,s)=%.4f corr(f,s)=%.4f  rms resid y-s %.5f  f-s %.5f"%(len(x),x.y.corr(x.s),x.f.corr(x.s),np.sqrt(((x.y-x.s)**2).mean()),np.sqrt(((x.f-x.s)**2).mean())))
dis=d[(d.y-d.f).abs()>0.01]
print("days with |ret diff|>1%%: %d; closer to S&P: yahoo %d, fred %d"%(len(dis),((dis.y-dis.s).abs()<(dis.f-dis.s).abs()).sum(),((dis.f-dis.s).abs()<(dis.y-dis.s).abs()).sum()))
# FRED zero-return (stale) days vs yahoo
print("FRED exactly-zero returns on common days by period:",(d.f==0).groupby(d.index.year).sum().loc[lambda s:s>0].to_dict())
print("Yahoo exactly-zero returns:",(d.y==0).sum())
# is the gap a persistent pattern (alternating sign within few days = noisy vs level offset)?
big=rel[rel.abs()>0.01]; print("days >1%:",len(big)); print(big.round(4).to_string())
