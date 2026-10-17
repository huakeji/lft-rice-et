"""Practical baselines under the national rolling protocol (test 2020; validation 2019; training targets
<= 2018), same stations, sequences and ET0 targets as the neural models:
  HS-cal : Hargreaves-Samani ET0 from forecast Tmax/Tmin and Ra, linearly recalibrated per station and lead
           on the training years (a + b*HS); falls back to pooled coefficients for stations with <200 rows.
  HGB    : histogram gradient boosting per lead on the current public forecast (numerics + weather/wind
           text + zone); learning rate x leaves chosen on 2019 validation from a 6-point grid.
  HGB+lag: the same learner with the six earlier issues for the same valid date (Tmax, Tmin, rain level,
           weather/wind text, availability) and inter-issue spreads.
Outputs per-lead RMSE, pooled RMSE and per-sequence predictions for paired bootstrap against the ensemble."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
R=Path(__file__).resolve().parents[2]/'reports/awm_repro_20261015';OUT=R/'writing_materials/15_baselines';OUT.mkdir(parents=True,exist_ok=True)
L=json.loads((R/'lagged_shards/manifest.json').read_text())['lags']
def ra(lat,doy):
 phi=np.radians(lat);dr=1+0.033*np.cos(2*np.pi*doy/365);d=0.409*np.sin(2*np.pi*doy/365-1.39);ws=np.arccos(np.clip(-np.tan(phi)*np.tan(d),-1,1))
 return 24*60/np.pi*0.0820*dr*(ws*np.sin(phi)*np.sin(d)+np.cos(phi)*np.cos(d)*np.sin(ws))
rows=[]
for p in sorted((R/'lagged_shards').glob('station_*.npz')):
 z=np.load(p);o=z['origin'].astype('datetime64[D]');keep=(o+6<=np.datetime64('2018-12-31'))|((o>=np.datetime64('2019-01-01'))&(o+6<=np.datetime64('2019-12-31')))|((o>=np.datetime64('2020-01-01'))&(o+6<=np.datetime64('2020-12-31')))
 z={k:z[k][keep] for k in ['x','c','y','origin','station']};o=z['origin'].astype('datetime64[D]')
 role=np.where(o+6<=np.datetime64('2018-12-31'),'train',np.where(o<np.datetime64('2020-01-01'),'val','test'))
 rows.append(dict(z=z,role=role,station=str(z['station'][0])))
def frame(k):
 F=[]
 for r in rows:
  z=r['z'];x=z['x'][:,k];c=z['c'][:,k];o=z['origin'].astype('datetime64[D]');doy=((o+k)-(o+k).astype('datetime64[Y]')).astype(int)+1
  d=pd.DataFrame(dict(station=r['station'],origin=z['origin'],role=r['role'],y=z['y'][:,k],tx=x[:,0],tn=x[:,1],dtr=x[:,2],rain=x[:,3],ss=x[:,4],sc=x[:,5],lat=x[:,6],lon=x[:,7],alt=x[:,8],w=c[:,0],v=c[:,1],zone=c[:,2]))
  d['ra']=ra(d.lat.values,doy);d['hs']=0.0023*((d.tx+d.tn)/2+17.8)*np.sqrt(np.clip(d.tx-d.tn,0,None))*d.ra*0.408
  for j in range(1,L+1):
   b=10+4*(j-1);av=x[:,b+3]>0
   d[f'tx{j}']=np.where(av,x[:,b],np.nan);d[f'tn{j}']=np.where(av,x[:,b+1],np.nan);d[f'r{j}']=np.where(av,x[:,b+2],np.nan);d[f'w{j}']=np.where(av,c[:,3+2*(j-1)],None);d[f'v{j}']=np.where(av,c[:,4+2*(j-1)],None)
  d['sprT']=x[:,10+4*L];d['sprt']=x[:,11+4*L];F.append(d)
 return pd.concat(F,ignore_index=True)
cats0=['w','v','zone'];base=['tx','tn','dtr','rain','ss','sc','lat','lon','alt']+cats0
lagc=[f'{v}{j}' for j in range(1,L+1) for v in ('tx','tn','r','w','v')]+['sprT','sprt'];lagcat=[f'{v}{j}' for j in range(1,L+1) for v in ('w','v')]
res=[];pred={}
grid=[(0.05,31),(0.05,63),(0.08,63),(0.08,127),(0.12,63),(0.12,127)]
for k in range(7):
 D=frame(k);tr=D.role=='train';va=D.role=='val';te=D.role=='test'
 # HS calibrated per station
 g=D[tr].groupby('station').apply(lambda s:pd.Series(np.polyfit(s.hs,s.y,1) if len(s)>=200 else [np.nan,np.nan],index=['b','a']))
 pool=np.polyfit(D[tr].hs,D[tr].y,1);g=g.fillna({'b':pool[0],'a':pool[1]})
 phs=D.station.map(g.a)+D.station.map(g.b)*D.hs
 for c in cats0+lagcat:
  top=D.loc[tr,c].value_counts().index[:250];voc={s:i for i,s in enumerate(sorted(top))};D[c]=D[c].map(voc).astype(float)
 out={'HS-cal':phs}
 for name,f in [('HGB',base),('HGB+lag',base+lagc)]:
  best=None
  for lr,lv in grid:
   m=HGB(max_iter=600,learning_rate=lr,max_leaf_nodes=lv,categorical_features=[i for i,c in enumerate(f) if c in cats0+lagcat],early_stopping=True,validation_fraction=0.1,random_state=0).fit(D.loc[tr,f],D.loc[tr,'y'])
   v=np.sqrt(np.mean((m.predict(D.loc[va,f])-D.loc[va,'y'])**2))
   if best is None or v<best[0]:best=(v,lr,lv,m)
  out[name]=pd.Series(best[3].predict(D[f]),index=D.index);res.append(dict(model=name,lead=k+1,val_rmse=best[0],lr=best[1],leaves=best[2]))
 for name,pv in out.items():
  e=(pv-D.y)[te];res.append(dict(model=name,lead=k+1,test_rmse=float(np.sqrt(np.mean(e**2))),test_bias=float(e.mean())))
  pred.setdefault(name,[]).append(pd.DataFrame(dict(station=D.station[te],origin=D.origin[te],lead=k+1,pred=pv[te],y=D.y[te])))
 print('lead',k+1,{n:round(float(np.sqrt(np.mean(((pv-D.y)[te])**2))),3) for n,pv in out.items()},flush=True)
R2=pd.DataFrame(res);R2.to_csv(OUT/'baselines_by_lead.csv',index=False)
for n,v in pred.items():pd.concat(v).to_parquet(OUT/f'pred_{n}.parquet',index=False)
t=R2.dropna(subset=['test_rmse']);print(t.groupby('model').test_rmse.apply(lambda s:np.sqrt((s**2).mean())).round(4))
