"""Quick information test of time-lagged public issues: identical HGB per lead, identical stations and
keys, base aligned features vs base + earlier issues' statements about the same valid date.
Train origins with all targets <=2018, evaluate 2020 origins. Screening only (not the network result)."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
ROOT=Path(__file__).resolve().parents[2];LAG=ROOT/'reports/awm_repro_20261015/lagged_shards'
J=json.loads((LAG/'manifest.json').read_text())['lags']
rng=np.random.default_rng(0);names=sorted(p.name for p in LAG.glob('station_*.npz'));pick=sorted(rng.choice(names,int(sys.argv[1]) if len(sys.argv)>1 else 120,replace=False))
fr=[]
for n in pick:
 z=np.load(LAG/n);x=z['x'];c=z['c'];df=pd.DataFrame({'station':z['station'],'origin':z['origin']})
 for k in range(7):
  for j,nm in enumerate(['tmax','tmin','dtr','rain']):df[f'{nm}{k}']=x[:,k,j]
  df[f'w{k}']=c[:,k,0];df[f'v{k}']=c[:,k,1];df[f'y{k}']=z['y'][:,k]
  for j in range(1,J+1):
   b=10+4*(j-1);av=x[:,k,b+3]>0
   df[f'L{k}_tmax{j}']=np.where(av,x[:,k,b],np.nan);df[f'L{k}_tmin{j}']=np.where(av,x[:,k,b+1],np.nan);df[f'L{k}_rain{j}']=np.where(av,x[:,k,b+2],np.nan)
   df[f'L{k}_dT{j}']=df[f'L{k}_tmax{j}']-x[:,k,0];df[f'L{k}_w{j}']=np.where(av,c[:,k,3+2*(j-1)],None);df[f'L{k}_v{j}']=np.where(av,c[:,k,4+2*(j-1)],None)
  df[f'L{k}_sprT']=x[:,k,10+4*J];df[f'L{k}_sprt']=x[:,k,11+4*J]
 df['lat'],df['lon'],df['alt'],df['ss'],df['sc']=x[:,0,6],x[:,0,7],x[:,0,8],x[:,0,4],x[:,0,5];df['zone']=c[:,0,2];fr.append(df)
D=pd.concat(fr,ignore_index=True);o=pd.to_datetime(D.origin);tr=D[o+pd.Timedelta(days=6)<='2018-12-31'].copy();ev=D[o.dt.year==2020].copy()
iscat=lambda c:(c[0] in 'wv' and c[1:].isdigit()) or c=='zone' or '_w' in c or '_v' in c
for c in [c for c in D if iscat(c)]:
 top=tr[c].value_counts().index[:250];voc={s:i for i,s in enumerate(sorted(top))};tr[c]=tr[c].map(voc).astype(float);ev[c]=ev[c].map(voc).astype(float)
base=[c for c in D if c not in ('station','origin') and not c.startswith(('y','L'))];res={}
for v in ('base','base+lags'):
 out=[]
 for k in range(7):
  feats=base+([c for c in D if c.startswith(f'L{k}_')] if v!='base' else [])
  m=HGB(max_iter=400,learning_rate=0.08,max_leaf_nodes=63,categorical_features=[i for i,c in enumerate(feats) if iscat(c)],early_stopping=True,validation_fraction=0.1,random_state=0)
  m.fit(tr[feats],tr[f'y{k}']);e=m.predict(ev[feats])-ev[f'y{k}'];out.append(float(np.sqrt(np.mean(e**2))))
 res[v]=dict(lead_rmse=out,mean=float(np.mean(out)),n_train=len(tr),n_eval=len(ev));print(v,{k:(round(x,4) if isinstance(x,float) else x) for k,x in res[v].items()},flush=True)
res['stations']=len(pick);(ROOT/'reports/awm_repro_20261015/lag_ab').mkdir(exist_ok=True);(ROOT/'reports/awm_repro_20261015/lag_ab/result.json').write_text(json.dumps(res,indent=1))
