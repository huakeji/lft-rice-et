"""Supplementary EC tables: (1) forward Kc,EC calibration counts per site and forecast year (calibration years,
days, months, minimum days per month); (2) forward site-year scores at leads 1 and 7 with 95% intervals from a
block bootstrap of consecutive 7-day periods counted from the first scored day of each site-year (2000 resamples);
pooled rows resample these blocks independently within each site-year. Uses the same prep() as ec_extended."""
import sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ec_validation as V
from ec_actual_forecast import obs_et0
OUT=V.OUT
src=(Path(__file__).parent/'ec_extended.py').read_text();ns={'__file__':str(Path(__file__).parent/'ec_extended.py')};exec(src[:src.index('rows=[];dec=[]')],ns);prep=ns['prep']
cal_rows=[]
for site in V.SITES:
 d,pr,clim,ec=prep(site)
 for y in sorted(set(pr.fold)&set(d.y)):
  cal=d[d.y<min(y,2018)] if site=='Jurong' else d[d.y<y]
  if cal.empty:continue
  g=pr[pr.fold==y];mon=sorted(set(g.date.dt.month)&set(cal.m));c=cal[cal.m.isin(mon)]
  cal_rows.append(dict(site=site,forecast_year=y,cal_years=','.join(map(str,sorted(set(cal.y)))),cal_days=len(c),months=f'{min(mon)}-{max(mon)}',n_months=len(mon),min_days_per_month=int(c.groupby('m').size().min())))
C=pd.DataFrame(cal_rows);C.to_csv(OUT/'tableS6_kc_calibration_counts.csv',index=False);print(C.to_string(index=False))
F=pd.read_csv(OUT/'actual_et_forward_daily.csv',parse_dates=['date']);rng=np.random.default_rng(0);rows=[]
def sc(g):
 e=g.eta-g.et;ec_=g.eta_clim-g.et;return np.sqrt((e**2).mean()),e.mean(),1-(e**2).mean()/(ec_**2).mean()
for (site,y,L),g in F[F.lead.isin([1,7])].groupby(['site','year','lead']):
 g=g.sort_values('date').reset_index(drop=True);n=len(g);wk=((g.date-g.date.min()).dt.days//7).values;blocks=[np.where(wk==w)[0] for w in np.unique(wk)];bs=[]
 for _ in range(2000):
  idx=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),size=len(blocks))]);bs.append(sc(g.iloc[idx]))
 bs=np.array(bs);r,b,s=sc(g)
 rows.append(dict(site=site,year=y,lead=L,n=n,rmse=r,rmse_lo=np.percentile(bs[:,0],2.5),rmse_hi=np.percentile(bs[:,0],97.5),bias=b,bias_lo=np.percentile(bs[:,1],2.5),bias_hi=np.percentile(bs[:,1],97.5),ss=s,ss_lo=np.percentile(bs[:,2],2.5),ss_hi=np.percentile(bs[:,2],97.5)))
# pooled over site-years: resample the 7-day blocks independently within each site-year (strata), 2000 resamples
for L in (1,7):
 G=[]
 for _,g in F[F.lead==L].groupby(['site','year']):
  g=g.sort_values('date').reset_index(drop=True);wk=((g.date-g.date.min()).dt.days//7).values;G.append((g,[np.where(wk==w)[0] for w in np.unique(wk)]))
 cat=lambda gs:pd.concat(gs);bs=np.array([sc(cat([g.iloc[np.concatenate([b[i] for i in rng.integers(0,len(b),size=len(b))])] for g,b in G])) for _ in range(2000)]);r,b,s=sc(cat([g for g,_ in G]))
 rows.append(dict(site='All',year='2018-2020',lead=L,n=sum(len(g) for g,_ in G),rmse=r,rmse_lo=np.percentile(bs[:,0],2.5),rmse_hi=np.percentile(bs[:,0],97.5),bias=b,bias_lo=np.percentile(bs[:,1],2.5),bias_hi=np.percentile(bs[:,1],97.5),ss=s,ss_lo=np.percentile(bs[:,2],2.5),ss_hi=np.percentile(bs[:,2],97.5)))
R=pd.DataFrame(rows);R.to_csv(OUT/'tableS5_site_year_forward_ci.csv',index=False);print(R.round(2).to_string(index=False))
