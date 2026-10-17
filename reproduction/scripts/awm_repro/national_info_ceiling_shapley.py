"""Order-averaged (Shapley) version of the national information-ceiling diagnostic: the same learner is fitted
for all 16 subsets of the observed target-day groups T (Tmax, Tmin), S (sunshine), H (RH), W (wind) added to the
public-message inputs; per-sequence squared errors are stored for station-cluster bootstrap intervals of the
order-averaged shares of the reducible MSE.
Original description: National information-ceiling diagnostic (no observation history at any level). Same learner
(per-lead HistGradientBoosting, fixed settings) for every level; 150 random stations; train origins
with all targets <= 2018-12-31, evaluate 2020 origins (2020 rolling fold). Levels add target-day
observations to the public-message inputs:
  P    public message (forecast Tmax/Tmin/DTR/rain level, weather & wind text, season, geography, lead)
  P+T  + observed Tmax/Tmin of the target day
  P+TS + observed sunshine duration
  P+TSH + observed relative humidity
  P+all + observed 10-m wind (all PM drivers observed; label = refet PM of the same drivers)
Diagnostic only: observed target-day variables are not available at forecast time."""
import json,sqlite3,sys
from pathlib import Path
import numpy as np,pandas as pd,warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
ROOT=Path(__file__).resolve().parents[2];A=ROOT/'reports/awm_repro_20261015/aligned_shards';OUT=ROOT/'reports/awm_repro_20261015/writing_materials/09_info_ceiling'
conn=sqlite3.connect((ROOT/'data/db/rice_rl_data.db').as_uri()+'?mode=ro',uri=True);conn.execute('PRAGMA query_only=ON')
rng=np.random.default_rng(0);names=sorted(p.name for p in A.glob('station_*.npz'));pick=sorted(rng.choice(names,150,replace=False))
rows=[]
for n in pick:
 z=np.load(A/n);st=str(z['station'][0]);o=pd.to_datetime(z['origin'].astype(str))
 ob=pd.read_sql_query('SELECT date,tmax,tmin,rh,wind10,sunshine_hours FROM historical_weather_daily WHERE station_code=? AND date>=?',conn,params=[st,'2015-01-01']);ob['date']=pd.to_datetime(ob.date);ob=ob.set_index('date')
 for k in range(7):
  t=o+pd.Timedelta(days=k);x=z['x'][:,k];c=z['c'][:,k]
  d=pd.DataFrame(dict(station=st,origin=o,lead=k+1,tmax_f=x[:,0],tmin_f=x[:,1],dtr_f=x[:,2],rain=x[:,3],ss=x[:,4],sc=x[:,5],lat=x[:,6],lon=x[:,7],alt=x[:,8],weather=c[:,0],wind=c[:,1],zone=c[:,2],y=z['y'][:,k]))
  r=ob.reindex(t);d['tmax_o']=r.tmax.to_numpy();d['tmin_o']=r.tmin.to_numpy();d['sun_o']=r.sunshine_hours.to_numpy();d['rh_o']=r.rh.to_numpy();d['wind_o']=r.wind10.to_numpy();rows.append(d)
D=pd.concat(rows,ignore_index=True)
for c in ['tmax_o','tmin_o']:D.loc[~D[c].between(-50,50),c]=np.nan
tr=D[D.origin+pd.Timedelta(days=6)<='2018-12-31'].copy();ev=D[D.origin.dt.year==2020].copy()
cats=['weather','wind','zone']
for c in cats:
 top=tr[c].value_counts().index[:250];voc={s:i for i,s in enumerate(sorted(top))};tr[c]=tr[c].map(voc).astype(float);ev[c]=ev[c].map(voc).astype(float)
base=['tmax_f','tmin_f','dtr_f','rain','ss','sc','lat','lon','alt']+cats
import itertools,math
G={'T':['tmax_o','tmin_o'],'S':['sun_o'],'H':['rh_o'],'W':['wind_o']}
subsets=[c for r in range(5) for c in itertools.combinations('TSHW',r)]
err={};res=[]
for sub in subsets:
 f=base+[v for g in sub for v in G[g]];key=''.join(sub) or '0';E=[]
 for k in range(1,8):
  a=tr[tr.lead==k];b=ev[ev.lead==k]
  m=HGB(max_iter=400,learning_rate=0.08,max_leaf_nodes=63,categorical_features=[i for i,c in enumerate(f) if c in cats],early_stopping=True,validation_fraction=0.1,random_state=0).fit(a[f],a.y)
  e=m.predict(b[f])-b.y.values;E.append(pd.DataFrame(dict(station=b.station.values,lead=k,se=e**2)));res.append(dict(subset=key,lead=k,rmse=float(np.sqrt(np.mean(e**2)))))
 err[key]=pd.concat(E,ignore_index=True);print(key,round(float(np.sqrt(err[key].se.mean())),3),flush=True)
st=err['0'].station.values;S={k:v.se.values for k,v in err.items()}
def shares(w):
 v={k:float((x*w).sum()/w.sum()) for k,x in S.items()};tot=v['0']-v['TSHW'];out={}
 for g in 'TSHW':
  others=[x for x in 'TSHW' if x!=g];phi=0.
  for r in range(4):
   for c in itertools.combinations(others,r):
    a=''.join(x for x in 'TSHW' if x in c) or '0';b=''.join(x for x in 'TSHW' if x in c or x==g)
    phi+=math.factorial(r)*math.factorial(3-r)/24*(v[a]-v[b])
  out[g]=phi/tot
 return out
w0=np.ones(len(st));sh=shares(w0)
codes,uniq=pd.factorize(pd.Series(st));rng=np.random.default_rng(0);bs=[]
for _ in range(500):
 cnt=np.bincount(rng.integers(len(uniq),size=len(uniq)),minlength=len(uniq));bs.append(shares(cnt[codes].astype(float)))
B=pd.DataFrame(bs);T=pd.DataFrame([dict(group=g,share=sh[g],lo95=np.percentile(B[g],2.5),hi95=np.percentile(B[g],97.5)) for g in 'TSHW'])
OUT.mkdir(parents=True,exist_ok=True);pd.DataFrame(res).to_csv(OUT/'national_ladder_all_subsets.csv',index=False);T.to_csv(OUT/'shapley_shares.csv',index=False)
print(T.round(3).to_string(index=False))
