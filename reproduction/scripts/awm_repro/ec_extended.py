"""Extended eddy-covariance analyses requested in review:
 (1) forward-only Kc,EC calibration (only years before the forecast year: Jurong 2015-2017 -> 2018-2020;
     Changling 2019 -> 2020; Panjin 2018 -> 2019 and 2018-2019 -> 2020) vs the retrospective LOYO design;
 (2) site-year metrics (n, bias, RMSE, r, SS vs climatology) for both designs;
 (3) matched-sample comparison of Kc x observed ET0 against EC as measured and closure-corrected;
 (4) MSE decomposition of (Kc x forecast ET0 - EC) into forecast term a, reference term b and 2E[ab];
 (5) sensitivity of the standard-condition comparison to the implausible-month threshold (0.2/0.3/0.4).
Kc,EC estimator: ratio of monthly sums, sum(ET_EC)/sum(ET0_obs), over calibration years."""
import sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ec_validation as V
from ec_actual_forecast import obs_et0
OUT=V.OUT
def prep(site,thr=0.3):
 st=V.SITES[site]['station'];ec=V.load_ec()[site];ec=ec[ec.qc_ok].set_index('date');E0=obs_et0(st)
 d=pd.DataFrame(dict(et=ec.et,et_corr=ec.et_corr)).join(E0.rename('et0'),how='inner');d['m']=d.index.month;d['y']=d.index.year
 r=d.groupby([d.y,d.m]).apply(lambda g:g.et.sum()/g.et0.sum());bad={k for k,v in r.items() if v<thr};d=d[[ (y,m) not in bad for y,m in zip(d.y,d.m)]]
 pr=V.product(st);pr=pr[pr.in_season];clim=E0[E0.index.year<=2016];clim=clim.groupby(clim.index.dayofyear).mean().rolling(15,center=True,min_periods=1).mean()
 return d,pr,clim,ec
def score(g,col='eta'):
 e=g[col]-g.et;ec=g.eta_clim-g.et
 return dict(n=len(g),bias=e.mean(),rmse=np.sqrt((e**2).mean()),r=np.corrcoef(g[col],g.et)[0,1],ss_clim=1-(e**2).mean()/(ec**2).mean())
rows=[];dec=[];mat=[];fwd=[]
for site in V.SITES:
 d,pr,clim,ec=prep(site)
 years=sorted(set(pr.fold)&set(d.y))
 for design in ('forward','loyo'):
  for y in years:
   if design=='forward':cal=d[d.y<min(y,2018)] if site=='Jurong' else d[d.y<y]
   else:cal=d[(d.y!=y)&(d.y<2018)] if site=='Jurong' else d[d.y!=y]
   if cal.empty:continue
   kc=cal.groupby('m').apply(lambda g:g.et.sum()/g.et0.sum())
   g=pr[pr.fold==y].merge(d[['et']],left_on='date',right_index=True);g['kc']=g.date.dt.month.map(kc);g=g.dropna(subset=['kc'])
   g['eta']=g.kc*g.et0_fcst.clip(lower=0);g['eta_obs']=g.kc*g.et0_obs;g['eta_clim']=g.kc*g.date.dt.dayofyear.map(clim)
   for L in (1,7):rows.append(dict(site=site,design=design,year=y,lead=L,cal_years=','.join(map(str,sorted(set(cal.y)))),**score(g[g.lead==L])))
   if design=='forward':fwd.append(g.assign(site=site,year=y))
 # matched-sample closure comparison and MSE decomposition (product with regional Kc, lead 1)
 g=pr[(pr.lead==1)].merge(d[['et','et_corr']],left_on='date',right_index=True)
 m=g.dropna(subset=['et_corr'])
 if len(m):
  for col,lab in (('et','as measured'),('et_corr','closure-corrected')):mat.append(dict(site=site,flux=lab,n=len(m),bias=(m.etc_ref-m[col]).mean(),rmse=np.sqrt(((m.etc_ref-m[col])**2).mean())))
 for col,lab in (('et','as measured'),('et_corr','closure-corrected')):
  h=g.dropna(subset=[col]);a=h.etc_fcst-h.etc_ref;b=h.etc_ref-h[col]
  if len(h):dec.append(dict(site=site,flux=lab,n=len(h),mse_total=((a+b)**2).mean(),Ea2=(a**2).mean(),Eb2=(b**2).mean(),cross=2*(a*b).mean(),bias_a=a.mean(),bias_b=b.mean()))
thr=[]
for t in (0.2,0.3,0.4):
 for site in V.SITES:
  d,pr,clim,ec=prep(site,t);g=pr[pr.lead==1].merge(d[['et']],left_on='date',right_index=True)
  thr.append(dict(threshold=t,site=site,n=len(g),bias=(g.etc_ref-g.et).mean(),rmse=np.sqrt(((g.etc_ref-g.et)**2).mean())))
F=pd.concat(fwd);F.to_csv(OUT/'actual_et_forward_daily.csv',index=False)
bl=[]
for (site,L),g in list(F.groupby(['site','lead']))+[(('All',L),g) for L,g in F.groupby('lead')]:bl.append(dict(site=site,lead=L,**score(g),rmse_clim=np.sqrt(((g.eta_clim-g.et)**2).mean()),rmse_obsdemand=np.sqrt(((g.eta_obs-g.et)**2).mean())))
pd.DataFrame(bl).to_csv(OUT/'actual_et_forward_by_lead.csv',index=False)
for n,v in (('actual_et_site_year',rows),('closure_matched',mat),('mse_decomposition',dec),('threshold_sensitivity',thr)):
 pd.DataFrame(v).to_csv(OUT/f'{n}.csv',index=False)
pd.set_option('display.width',220)
for n in ('actual_et_site_year','closure_matched','mse_decomposition','threshold_sensitivity'):print(n);print(pd.read_csv(OUT/f'{n}.csv').round(3).to_string(index=False))
