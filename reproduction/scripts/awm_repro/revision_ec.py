"""Revision EC analyses (existing records only):
(1) day accounting per site and year (rice-season months of the paired station): EC days, QC removals
    (gap-filled plateaus, ET<=0.05, months below 30% of ET0), retained days, closure-eligible days;
(2) forward actual-ET skill with the uncertainty of the calibrated monthly Kc,EC propagated: each replicate
    resamples calibration years (with replacement) and 7-day blocks within them, recomputes Kc,EC, and
    resamples 7-day blocks of the scored days (2000 replicates; site-years and pooled);
(3) same-day closure comparison (as measured vs Bowen-ratio corrected) with 7-day block bootstrap intervals;
(4) station-pairing sensitivity: Kc x observed ET0 of the nearest vs second-nearest station against EC ET.
Writes writing_materials/17_revision_stats/ec_*.csv"""
import sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ec_validation as V
from ec_actual_forecast import obs_et0
OUT=Path(__file__).resolve().parents[2]/'reports/awm_repro_20261015/writing_materials/17_revision_stats'
src=(Path(__file__).parent/'ec_extended.py').read_text();ns={'__file__':str(Path(__file__).parent/'ec_extended.py')};exec(src[:src.index('rows=[];dec=[]')],ns);prep=ns['prep']
rng=np.random.default_rng(0);B=2000
def blocks(dates):
 d=pd.to_datetime(pd.Series(dates)).reset_index(drop=True);wk=((d-d.min()).dt.days//7).values;return [np.where(wk==w)[0] for w in np.unique(wk)]
def rs(bl):return np.concatenate([bl[i] for i in rng.integers(0,len(bl),size=len(bl))])
acc=[];sk=[];clo=[];pair=[];pooled={1:[],7:[]};pooled0={1:[],7:[]}
EC=V.load_ec()
for site,cfg in V.SITES.items():
 d,pr,clim,ec=prep(site);raw=EC[site].set_index('date')
 st=cfg['station'];E0=obs_et0(st)
 season=pr[pr.lead==1].date.dt.normalize().unique()
 for y in sorted(set(pd.to_datetime(season).year)):
  sd=pd.DatetimeIndex([x for x in season if x.year==y]);r=raw.reindex(sd)
  ok_idx=d.index.intersection(sd)
  acc.append(dict(site=site,year=y,season_days=len(sd),ec_days=int(r.et.notna().sum()),plateau_removed=int(r.qc_plateau.fillna(False).sum()),low_removed=int((r.qc_low.fillna(False)&~r.qc_plateau.fillna(False)).sum()),
   month_screen_removed=int((r.qc_ok.fillna(False)).sum()-len(ok_idx)),retained=len(ok_idx),closure_eligible=int(d.reindex(ok_idx).et_corr.notna().sum()) if site!='Panjin' else 0))
 # (2) forward skill with Kc,EC calibration uncertainty
 for y in sorted(set(pr.fold)&set(d.y)):
  cal=d[d.y<min(y,2018)] if site=='Jurong' else d[d.y<y]
  if cal.empty:continue
  g=pr[pr.fold==y].merge(d[['et']],left_on='date',right_index=True)
  g=g[g.date.dt.month.isin(cal.m.unique())].copy();g['clim']=g.date.dt.dayofyear.map(clim)
  cyears=sorted(cal.y.unique());cb={yy:blocks(cal[cal.y==yy].index) for yy in cyears}
  for L in (1,7):
   h=g[g.lead==L].sort_values('date').reset_index(drop=True);hb=blocks(h.date)
   def score(kc,idx):
    k=h.date.dt.month.map(kc).values[idx];e=k*h.et0_fcst.clip(lower=0).values[idx]-h.et.values[idx];c=k*h.clim.values[idx]-h.et.values[idx]
    return e,c
   kc0=cal.groupby('m').apply(lambda q:q.et.sum()/q.et0.sum(),include_groups=False);e,c=score(kc0,np.arange(len(h)))
   reps=[];raw_e=[]
   for _ in range(B):
    parts=[cal[cal.y==yy].iloc[rs(cb[yy])] for yy in rng.choice(cyears,len(cyears))]
    cc=pd.concat(parts);kc=cc.groupby('m').apply(lambda q:q.et.sum()/q.et0.sum(),include_groups=False).reindex(kc0.index).fillna(kc0)
    ee,cc_=score(kc,rs(hb));reps.append((np.sqrt((ee**2).mean()),ee.mean(),1-(ee**2).mean()/(cc_**2).mean()));raw_e.append((ee,cc_))
   reps=np.array(reps);pooled[L].append(raw_e);pooled0[L].append((e,c))
   sk.append(dict(site=site,year=y,lead=L,n=len(h),cal_years=','.join(map(str,cyears)),rmse=np.sqrt((e**2).mean()),bias=e.mean(),ss=1-(e**2).mean()/(c**2).mean(),
    rmse_lo=np.percentile(reps[:,0],2.5),rmse_hi=np.percentile(reps[:,0],97.5),ss_lo=np.percentile(reps[:,2],2.5),ss_hi=np.percentile(reps[:,2],97.5)))
 # (3) same-day closure comparison
 if site!='Panjin':
  m=pr[pr.lead==1].merge(d[['et','et_corr']],left_on='date',right_index=True).dropna(subset=['et_corr']).sort_values('date').reset_index(drop=True);mb=blocks(m.date)
  for lab,col in (('as measured','et'),('closure-corrected','et_corr')):
   e=(m.etc_ref-m[col]).values;bb=[(e[i].mean(),np.sqrt((e[i]**2).mean())) for i in (rs(mb) for _ in range(B))];bb=np.array(bb)
   clo.append(dict(site=site,flux=lab,n=len(m),bias=e.mean(),bias_lo=np.percentile(bb[:,0],2.5),bias_hi=np.percentile(bb[:,0],97.5),rmse=np.sqrt((e**2).mean()),rmse_lo=np.percentile(bb[:,1],2.5),rmse_hi=np.percentile(bb[:,1],97.5)))
  cf=raw.reindex(m.date).closure;clo.append(dict(site=site,flux='closure factor (H+LE)/(Rn-G), matched days',n=len(m),bias=float(cf.median()),bias_lo=float(cf.quantile(.25)),bias_hi=float(cf.quantile(.75))))
 # (4) station-pairing sensitivity (observed ET0 of nearest vs second-nearest station, regional Kc)
 alt=cfg['alt'];E1=obs_et0(alt)
 q=pr[pr.lead==1][['date','kc']].merge(d[['et']],left_on='date',right_index=True)
 for lab,E in (('nearest '+st,E0),('second '+alt,E1)):
  x=q.assign(e0=q.date.map(E)).dropna(subset=['e0']);e=x.kc*x.e0-x.et
  pair.append(dict(site=site,station=lab,n=len(x),bias=e.mean(),rmse=np.sqrt((e**2).mean()),r=np.corrcoef(x.kc*x.e0,x.et)[0,1]))
# pooled forward skill with calibration uncertainty
for L in (1,7):
 e=np.concatenate([p[0] for p in pooled0[L]]);c=np.concatenate([p[1] for p in pooled0[L]]);reps=[]
 for b in range(B):
  ee=np.concatenate([sy[b][0] for sy in pooled[L]]);cc=np.concatenate([sy[b][1] for sy in pooled[L]]);reps.append((np.sqrt((ee**2).mean()),1-(ee**2).mean()/(cc**2).mean()))
 reps=np.array(reps);sk.append(dict(site='All',year='2018-2020',lead=L,n=len(e),rmse=np.sqrt((e**2).mean()),bias=e.mean(),ss=1-(e**2).mean()/(c**2).mean(),rmse_lo=np.percentile(reps[:,0],2.5),rmse_hi=np.percentile(reps[:,0],97.5),ss_lo=np.percentile(reps[:,1],2.5),ss_hi=np.percentile(reps[:,1],97.5)))
for n,v in (('ec_accounting',acc),('ec_forward_skill_kc_uncertainty',sk),('ec_closure_matched_ci',clo),('ec_station_pairing',pair)):
 pd.DataFrame(v).to_csv(OUT/f'{n}.csv',index=False);print(n);print(pd.DataFrame(v).round(3).to_string(index=False))
