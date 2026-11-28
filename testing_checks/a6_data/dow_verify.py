import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from herding import market_data as md
mw=pd.read_csv('data/dow_jones_daily_from_1896.csv',parse_dates=['Date'],index_col='Date')['Close']
y=pd.read_csv('scratch/crosscheck/yahoo_DJI.csv',parse_dates=[0],index_col=0)
y=y.iloc[:,0] if y.shape[1]==1 else y['Close']
y=pd.to_numeric(y,errors='coerce').dropna()
fr=pd.read_csv('scratch/crosscheck/fred_DJIA.csv',parse_dates=[0],index_col=0).iloc[:,0]
fr=pd.to_numeric(fr,errors='coerce').dropna()
sp=pd.read_csv('data/sp500_daily.csv',parse_dates=['Date'],index_col='Date')['Close']
lr=lambda s:np.log(s).diff()
rm,ry,rs,rf=lr(mw),lr(y),lr(sp),lr(fr)
bad=md.BAD_CLOSES['dow']
both=mw.index.intersection(y.index)
rel=((mw[both]-y[both])/y[both]).abs()
over=rel[rel>0.001]
print("days >0.1% mw vs yahoo:",len(over)); print("in BAD but not >0.1%:",[d for d in bad if pd.Timestamp(d) not in over.index])
print("over 0.1% but not in BAD:",[str(d.date()) for d in over.index if str(d.date()) not in bad])
print("date | mw_close yahoo_close fred | mw_ret yahoo_ret spx_ret | err_mw err_yahoo (vs spx) | sided")
res=[]
for d in bad+("1999-05-04","2009-07-22"):
    t=pd.Timestamp(d)
    nxt=mw.index[mw.index.get_loc(t)+1] if t in mw.index else None
    row=[d, mw.get(t,np.nan), y.get(t,np.nan), fr.get(t,np.nan)]
    def two(s):  # two-day return over d and next day (both returns) 
        return s.get(t,np.nan), s.get(nxt,np.nan)
    a=two(rm);b=two(ry);c=two(rs)
    e_m=abs(a[0]-c[0]); e_y=abs(b[0]-c[0])
    print(d,"| %.2f %.2f %.2f |"%tuple(row[1:]),"d0: mw %+.4f y %+.4f spx %+.4f | d1: mw %+.4f y %+.4f spx %+.4f | |mw-spx| %.4f |y-spx| %.4f -> %s"%(a[0],b[0],c[0],a[1],b[1],c[1],e_m,e_y,"YAHOO" if e_y<e_m else "FILE"))
