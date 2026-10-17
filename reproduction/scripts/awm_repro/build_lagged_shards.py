"""Aligned same-day shards plus earlier public issues of the same valid dates (time-lagged messages).

For origin O and lead L (valid date T = O+L-1) the base columns are identical to aligned_shards
(raw key L of issue O). Lag block j (j=1..J) holds what issue O-j said about the same date T,
i.e. raw key L+j of issue O-j (raw messages carry 15 keys, so J<=8 covers every lead). Every lag
block was public before issue O, so nothing here is unavailable at forecast time.
Missing earlier issues are filled with the current issue values and flagged avail=0.
Sequences (origins) are exactly those of aligned_shards, so results pair one-to-one.

x columns: 10 base | per lag j: tmax_j, tmin_j, rain_j, avail_j | spread_tmax, spread_tmin (std over
current + available lags).  c columns: weather, wind, zone | per lag j: weather_j, wind_j.
"""
import argparse,hashlib,json,sqlite3,sys,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from forecast_domain import weather_to_rain_signal
from build_aligned_shards import num

def lag_names(J):
 return [f'{v}_lag{j}' for j in range(1,J+1) for v in ('tmax','tmin','rain','avail')]+['spread_tmax','spread_tmin'],[f'{v}_lag{j}' for j in range(1,J+1) for v in ('weather','wind')]

def build_station(st,city,base,conn,out,J):
 b=np.load(base/f'station_{st}.npz');x,c,y,O=b['x'],b['c'],b['y'],b['origin'].astype(str)
 f=pd.read_sql_query("SELECT date,day_weather_json,day_high_json,day_low_json,day_wind_json FROM forecast_raw_daily WHERE city=? ORDER BY date",conn,params=[city])
 m=len(f);H=np.full((m+1,17),np.nan);L=H.copy();W=np.full((m+1,17),'',dtype=object);D=W.copy()   # row m = missing issue
 for r,(w,h,l,wd) in enumerate(f[['day_weather_json','day_high_json','day_low_json','day_wind_json']].itertuples(index=False)):
  w,h,l,wd=(json.loads(v) if v else {} for v in (w,h,l,wd))
  for key in range(1,16):q=str(key);H[r,key]=num(h.get(q));L[r,key]=num(l.get(q));W[r,key]=str(w.get(q) or '').strip();D[r,key]=str(wd.get(q) or '').strip()
 rain={t:float(weather_to_rain_signal(t)) for t in set(W.ravel()) if t}
 idx=pd.Index(pd.to_datetime(f.date));od=pd.to_datetime(O)
 n=len(O);X=np.zeros((n,7,4*J+2),'float32');C=np.empty((n,7,2*J),dtype=object);HS=[x[:,:,0]];LS=[x[:,:,1]]
 for j in range(1,J+1):
  r=idx.get_indexer(od-pd.Timedelta(days=j));r[r<0]=m
  for k in range(7):
   key=k+1+j;hi,lo,wt,wn=H[r,key],L[r,key],W[r,key],D[r,key]
   ok=np.isfinite(hi)&np.isfinite(lo)&(hi>=lo)&~np.isin(np.char.lower(wt.astype(str)),['','none','nan','0'])
   rs=np.array([rain.get(t,0.) for t in wt])
   X[:,k,4*(j-1)]=np.where(ok,hi,x[:,k,0]);X[:,k,4*(j-1)+1]=np.where(ok,lo,x[:,k,1]);X[:,k,4*(j-1)+2]=np.where(ok,rs,x[:,k,3]);X[:,k,4*(j-1)+3]=ok
   C[:,k,2*(j-1)]=np.where(ok,wt,c[:,k,0]);C[:,k,2*(j-1)+1]=np.where(ok&(wn!=''),wn,c[:,k,1])
  HS.append(np.where(X[:,:,4*(j-1)+3]>0,X[:,:,4*(j-1)],np.nan));LS.append(np.where(X[:,:,4*(j-1)+3]>0,X[:,:,4*(j-1)+1],np.nan))
 X[:,:,4*J]=np.nanstd(np.stack(HS),0);X[:,:,4*J+1]=np.nanstd(np.stack(LS),0)
 p=out/f'station_{st}.npz'
 np.savez_compressed(p,x=np.concatenate([x,X],2),c=np.concatenate([c,C.astype(str)],2),y=y,origin=b['origin'],station=b['station'])
 return dict(station=st,sequences=n,lag_available_share=float(X[:,:,3:4*J:4].mean()),shard_sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base',default='reports/awm_repro_20261015/aligned_shards')
 ap.add_argument('--out-dir',default='reports/awm_repro_20261015/lagged_shards');ap.add_argument('--lags',type=int,default=6);ap.add_argument('--n-stations',type=int,default=0)
 a=ap.parse_args();base=ROOT/a.base;out=ROOT/a.out_dir;out.mkdir(parents=True,exist_ok=False);t0=time.time()
 if not 1<=a.lags<=8:raise ValueError('lags must be 1..8 (raw messages have 15 keys)')
 conn=sqlite3.connect((ROOT/'data/db/rice_rl_data.db').as_uri()+'?mode=ro',uri=True);conn.execute('PRAGMA query_only=ON')
 cities=dict(pd.read_sql_query('SELECT station_code,city FROM city_station_mapping',conn).astype(str).itertuples(index=False))
 sts=sorted(p.stem.split('_')[1] for p in base.glob('station_*.npz'))
 if a.n_stations:sts=sts[:a.n_stations]
 inv=pd.DataFrame([build_station(s,cities[s],base,conn,out,a.lags) for s in sts]);inv.to_csv(out/'station_inventory.csv',index=False)
 nn,cn=lag_names(a.lags)
 s=dict(base=a.base,lags=a.lags,extra_numeric=nn,extra_categorical=cn,stations=len(inv),sequences=int(inv.sequences.sum()),
  lag_available_share=float((inv.lag_available_share*inv.sequences).sum()/inv.sequences.sum()),seconds=time.time()-t0,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (out/'manifest.json').write_text(json.dumps(s,indent=1));print(json.dumps(s,indent=1))

if __name__=='__main__':main()
