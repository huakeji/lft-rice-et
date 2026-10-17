"""Forecasting actual paddy ET at the ChinaFLUX sites: monthly EC-derived crop coefficients
Kc_EC(month) = sum(ET_EC)/sum(ET0_obs) from calibration years only (Jurong: 2015-2017 EC, which are not
forecast years; Changling/Panjin: leave-one-year-out), then ETa_fcst = Kc_EC x ET0_fcst (out-of-sample
LFT+GRU ensemble) for the held-out year, verified against EC ET of that year. Reference forecasts:
Kc_EC x station day-of-year ET0 climatology (training years) and persistence of yesterday's EC ET.
Same QC as ec_validation.py (raw EC; the calibrated Kc_EC carries the site's closure level)."""
import sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ec_validation as V
OUT=V.OUT
def obs_et0(st):
 import glob
 rows=[]
 for p in glob.glob(str(V.R/f'aligned_shards/station_{st}.npz')):
  z=np.load(p);o=pd.to_datetime(z['origin'].astype(str))
  for k in range(7):rows.append(pd.DataFrame(dict(date=o+pd.Timedelta(days=k),et0=z['y'][:,k])))
 return pd.concat(rows).drop_duplicates('date').set_index('date').et0
def main():
 EC=V.load_ec();rows=[];daily=[]
 for site,cfg in V.SITES.items():
  st=cfg['station'];ec=EC[site];ec=ec[ec.qc_ok].set_index('date');E0=obs_et0(st)
  d=pd.DataFrame(dict(et=ec.et)).join(E0.rename('et0'),how='inner');d['m']=d.index.month;d['y']=d.index.year
  r=d.groupby([d.y,d.m]).apply(lambda g:g.et.sum()/g.et0.sum());bad={(y,m) for (y,m),v in r.items() if v<0.3};d=d[[ (y,m) not in bad for y,m in zip(d.y,d.m)]]
  pr=V.product(st);pr=pr[pr.in_season]
  clim=E0[E0.index.year<=2016].groupby(E0[E0.index.year<=2016].index.dayofyear).mean().rolling(15,center=True,min_periods=1).mean()
  for y in sorted(set(pr.fold)&set(d.y)):
   cal=d[(d.y!=y)&(d.y<2018)] if site=='Jurong' else d[d.y!=y]
   kc=cal.groupby('m').apply(lambda g:g.et.sum()/g.et0.sum())
   g=pr[pr.fold==y].merge(d[['et']],left_on='date',right_index=True);g['kc_ec']=g.date.dt.month.map(kc);g=g.dropna(subset=['kc_ec','et'])
   g['eta_fcst']=g.kc_ec*g.et0_fcst.clip(lower=0);g['eta_obsdemand']=g.kc_ec*g.et0_obs;g['eta_clim']=g.kc_ec*g.date.dt.dayofyear.map(clim)
   prev=ec.et.reindex(g.date-pd.Timedelta(days=1)).to_numpy();g['eta_persist']=np.where(np.isfinite(prev),prev,np.nan)
   g['site']=site;g['year']=y;daily.append(g)
 D=pd.concat(daily)
 for (site,L),g in D.groupby(['site','lead']):
  e=lambda c:g[c]-g.et;ok=g.eta_persist.notna()
  rows.append(dict(site=site,lead=L,n=len(g),et_mean=g.et.mean(),rmse_fcst=np.sqrt((e('eta_fcst')**2).mean()),bias_fcst=e('eta_fcst').mean(),r_fcst=np.corrcoef(g.eta_fcst,g.et)[0,1],
   rmse_obsdemand=np.sqrt((e('eta_obsdemand')**2).mean()),rmse_clim=np.sqrt((e('eta_clim')**2).mean()),rmse_persist=np.sqrt(((g.eta_persist-g.et)[ok]**2).mean()),
   ss_vs_clim=1-(e('eta_fcst')**2).mean()/(e('eta_clim')**2).mean(),ss_vs_persist=1-((g.eta_fcst-g.et)[ok]**2).mean()/((g.eta_persist-g.et)[ok]**2).mean()))
 R=pd.DataFrame(rows);R.to_csv(OUT/'actual_et_forecast_by_lead.csv',index=False);D.to_csv(OUT/'actual_et_forecast_daily.csv',index=False)
 pd.set_option('display.width',220);print(R.round(3).to_string(index=False))
 # pooled over sites
 P=D.groupby('lead').apply(lambda g:pd.Series(dict(n=len(g),rmse=np.sqrt(((g.eta_fcst-g.et)**2).mean()),rel=np.sqrt(((g.eta_fcst-g.et)**2).mean())/g.et.mean(),bias=(g.eta_fcst-g.et).mean(),r=np.corrcoef(g.eta_fcst,g.et)[0,1],ss_clim=1-((g.eta_fcst-g.et)**2).mean()/((g.eta_clim-g.et)**2).mean())));print(P.round(3))
 P.to_csv(OUT/'actual_et_forecast_pooled.csv')
if __name__=='__main__':main()
