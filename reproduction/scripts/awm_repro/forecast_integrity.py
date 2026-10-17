"""Station-year integrity of the issue-day public forecast: lead-1 (raw key 1) Tmax/Tmin from the
aligned shards against observed Tmax/Tmin of the same date (historical_weather_daily, read-only)."""
import glob,sqlite3,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
conn=sqlite3.connect((ROOT/'data/db/rice_rl_data.db').as_uri()+'?mode=ro',uri=True);conn.execute('PRAGMA query_only=ON')
rows=[]
for p in sorted(glob.glob(str(ROOT/'reports/awm_repro_20261015/aligned_shards/station_*.npz'))):
 z=np.load(p);st=str(z['station'][0]);d=pd.DataFrame(dict(date=pd.to_datetime(z['origin'].astype(str)),fx=z['x'][:,0,0],fn=z['x'][:,0,1]))
 o=pd.read_sql_query('SELECT date,tmax,tmin FROM historical_weather_daily WHERE station_code=? AND date>=?',conn,params=[st,'2013-01-01'])
 o['date']=pd.to_datetime(o.date);o=o[(o.tmax.between(-50,50))&(o.tmin.between(-50,50))]
 m=d.merge(o,on='date');m['year']=m.date.dt.year
 for y,g in m.groupby('year'):
  if len(g)<60:continue
  rows.append(dict(station=st,year=y,n=len(g),r_tmax=np.corrcoef(g.fx,g.tmax)[0,1],rmse_tmax=float(np.sqrt(((g.fx-g.tmax)**2).mean())),bias_tmax=float((g.fx-g.tmax).mean()),
   r_tmin=np.corrcoef(g.fn,g.tmin)[0,1],rmse_tmin=float(np.sqrt(((g.fn-g.tmin)**2).mean()))))
D=pd.DataFrame(rows);out=ROOT/'reports/awm_repro_20261015/integrity';out.mkdir(exist_ok=True);D.to_csv(out/'station_year_lead1.csv',index=False)
D['bad']=(D.r_tmax<0.85)|(D.rmse_tmax>5)
print(D.groupby('year').agg(stations=('station','size'),bad=('bad','sum'),r_med=('r_tmax','median'),rmse_med=('rmse_tmax','median'),bias_med=('bias_tmax','median')).round(3).to_string())
