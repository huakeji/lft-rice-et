"""Same-day shards with forecast inputs and labels on the SAME valid date.

Inputs for lead L are read directly from raw forecast_raw_daily JSON key L (valid origin+L-1,
verified against observed Tmax for all years/stations sampled) and the label is the reference ET0
target on origin+L-1. A physical temperature range check removes importer sentinels such as 100.
"""
import argparse,hashlib,json,sqlite3,sys,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from db_io import ddmm_to_decimal
from forecast_domain import weather_to_rain_signal

def num(v):
 try:v=float(v)
 except(TypeError,ValueError):return np.nan
 return v if -50<=v<=50 else np.nan

def build_station(m,cities,ref,conn,out,match=None):
 st=str(m.station)
 if st not in cities:return dict(station=st,status='NO_CITY',sequences=0)
 f=pd.read_sql_query("SELECT date,day_weather_json,day_high_json,day_low_json,day_wind_json FROM forecast_raw_daily WHERE city=? ORDER BY date",conn,params=[cities[st]])
 if f.date.duplicated().any():raise ValueError(f'duplicate forecast dates {st}')
 r=ref[ref.station.eq(st)].set_index('date')
 X,C,Y,O=[],[],[],[]
 lat,lon,alt=ddmm_to_decimal(m.lat),ddmm_to_decimal(m.lon),float(m.alt)
 for row in f.itertuples(index=False):
  w,h,l,wd=(json.loads(x) if x else {} for x in (row.day_weather_json,row.day_high_json,row.day_low_json,row.day_wind_json))
  origin=pd.Timestamp(row.date);xs,cs,ys=[],[],[]
  for k in range(1,8):
   hi,lo=num(h.get(str(k))),num(l.get(str(k)));wt=str(w.get(str(k)) or '').strip();wn=str(wd.get(str(k)) or '').strip()
   t=origin+pd.Timedelta(days=k-1)
   if t not in r.index or not bool(r.at[t,'mask_main']):break
   y=r.at[t,'et0_refet_candidate']
   if wt.lower() in ('','none','nan','0') or not np.isfinite([hi,lo,y]).all() or hi<lo:break
   doy=t.dayofyear
   xs.append([hi,lo,hi-lo,float(weather_to_rain_signal(wt)),np.sin(2*np.pi*doy/365.0),np.cos(2*np.pi*doy/365.0),lat,lon,alt,float(k)])
   cs.append([wt,wn,str(m.zone)]);ys.append(y)
  if len(ys)==7:X.append(xs);C.append(cs);Y.append(ys);O.append(str(origin.date()))
 if match is not None:
  old=match/f'station_{st}.npz'
  keep=set(np.load(old)['origin'].astype(str)) if old.exists() else set()
  idx=[i for i,o in enumerate(O) if o in keep];X,C,Y,O=[X[i] for i in idx],[C[i] for i in idx],[Y[i] for i in idx],[O[i] for i in idx]
 if not O:return dict(station=st,status='NO_COMPLETE_SEQUENCES',sequences=0)
 p=out/f'station_{st}.npz'
 np.savez_compressed(p,x=np.asarray(X,'float32'),c=np.asarray(C,dtype=str),y=np.asarray(Y,'float32'),origin=np.asarray(O,dtype=str),station=np.repeat(st,len(O)))
 return dict(station=st,status='SHARD_READY',sequences=len(O),shard_sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out-dir',default='reports/awm_repro_20261015/aligned_shards');ap.add_argument('--n-stations',type=int,default=0);ap.add_argument('--match-origins',default='reports/v11_audit_20260909/same_day_shards_20260912_full',help='keep only origins present in these shards (isolates the alignment change); empty = all')
 a=ap.parse_args();out=(ROOT/a.out_dir).resolve();out.mkdir(parents=True,exist_ok=False);t0=time.time()
 man=pd.read_csv(ROOT/'results/integrity/panel_manifest_expanded.csv',dtype={'station':str})
 sites=man[man.build_ok.eq(1)&man.include_flag.eq(1)].sort_values('station')
 if a.n_stations:sites=sites.head(a.n_stations)
 db=ROOT/'data/db/rice_rl_data.db';conn=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);conn.execute('PRAGMA query_only=ON')
 mp=pd.read_sql_query('SELECT city,station_code FROM city_station_mapping',conn)
 if mp.station_code.duplicated().any():raise ValueError('station mapped to several cities')
 cities=dict(zip(mp.station_code.astype(str),mp.city))
 ref=pd.read_parquet(ROOT/'reports/v11_audit_20260909/qc_reference_27/full/reference_and_masks.parquet',columns=['station','date','mask_main','et0_refet_candidate'])
 ref['station']=ref.station.astype(str);ref['date']=pd.to_datetime(ref.date)
 match=(ROOT/a.match_origins) if a.match_origins else None
 rows=[build_station(m,cities,ref,conn,out,match) for m in sites.itertuples(index=False)]
 inv=pd.DataFrame(rows);inv.to_csv(out/'station_inventory.csv',index=False)
 s=dict(match_origins=a.match_origins,mapping='lead L input = raw JSON key L (valid origin+L-1); label = reference ET0 at origin+L-1',stations_with_shards=int((inv.status=='SHARD_READY').sum()),
  sequences=int(inv.sequences.sum()),seconds=time.time()-t0,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (out/'manifest.json').write_text(json.dumps(s,ensure_ascii=False,indent=1));print(json.dumps(s,ensure_ascii=False,indent=1))

if __name__=='__main__':main()
