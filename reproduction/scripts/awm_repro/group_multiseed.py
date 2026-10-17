"""Multi-seed / multi-draw summary of the rice-zone transfer runs (outer 2020).
Zone table: per-zone RMSE of the national model (C) and of zero-shot leave-zone-out models (tuned F, untuned U) for
seeds 0-2, their mean and range, and the per-seed relative change against the national model of the same seed.
Scaling: zero-shot RMSE on three random 99-station holdouts (draws s0, d8, d9) by training-set size, with the
three-seed national model evaluated on the same holdout (holdout stations seen in training) as reference."""
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];G=ROOT/'reports/awm_repro_20261015/group'
Z=pd.read_csv(ROOT/'reports/awm_repro_20261015/station_rice_zone.csv',dtype={'station':str}).set_index('station').rice_zone
def load(s,n):
 f=G/s/n/'outer_predictions.npz'
 return np.load(f) if f.exists() else None
r=lambda P,Y:float(np.sqrt(((P-Y)**2).mean()))
def sub(z,stations):k=np.isin(z['station'].astype(str),list(stations));return r(z['prediction'][k],z['target'][k])
def zr(z,zone):k=pd.Series(z['station'].astype(str)).map(Z).values==zone;return r(z['prediction'][k],z['target'][k])
SEEDS=('s0','s1','s2');C={s:load(s,'C_national') for s in SEEDS};rows=[]
for zone in ['R1','R2','R3','R4','R5','R6']:
 c=[zr(C[s],zone) for s in SEEDS];row=dict(zone=zone,n_stations=int((Z==zone).sum()),C_mean=np.mean(c),C_min=min(c),C_max=max(c))
 for lab,n in [('F',f'F_lzo_{zone}'),('U',f'U_lzo_{zone}_default')]:
  v=[];rel=[]
  for s in SEEDS:
   z=load(s,n)
   if z is None:continue
   v.append(zr(z,zone));rel.append(100*(v[-1]/zr(C[s],zone)-1))
  if v:row.update({f'{lab}_n':len(v),f'{lab}_mean':np.mean(v),f'{lab}_min':min(v),f'{lab}_max':max(v),f'{lab}_rel_mean':np.mean(rel),f'{lab}_rel_min':min(rel),f'{lab}_rel_max':max(rel),f'{lab}_rel_vs_Cmean':100*(np.mean(v)/np.mean(c)-1)})
 rows.append(row)
D=pd.DataFrame(rows).set_index('zone')
S=[]
for dr,suf in [('s0',''),('d8','_d8'),('d9','_d9')]:
 h=set((G/'lists'/f'scale_holdout{suf}').read_text().split());ref=[sub(C[s],h) for s in SEEDS]
 for n in [25,50,100,200,400,484]:
  z=load('s0',f'S_scale_{n}{suf}')
  if z is None:continue
  assert set(z['station'].astype(str))<=h,f'{dr} {n}: evaluation outside holdout'
  tr=set((G/'lists'/f'scale_train_{n}{suf}').read_text().split()) if (G/'lists'/f'scale_train_{n}{suf}').exists() else set()
  assert not (tr&h),f'{dr} {n}: train/holdout overlap'
  S.append(dict(draw=dr,train_stations=n,zero_shot_rmse=r(z['prediction'],z['target']),national_ref=np.mean(ref),rel_pct=100*(r(z['prediction'],z['target'])/np.mean(ref)-1)))
S=pd.DataFrame(S);A=S.groupby('train_stations').agg(mean_rmse=('zero_shot_rmse','mean'),min_rmse=('zero_shot_rmse','min'),max_rmse=('zero_shot_rmse','max'),mean_rel=('rel_pct','mean'),min_rel=('rel_pct','min'),max_rel=('rel_pct','max'),draws=('draw','count')).reset_index()
out=ROOT/'reports/awm_repro_20261015/group_summary';out.mkdir(exist_ok=True)
D.to_csv(out/'zone_multiseed.csv');S.to_csv(out/'scaling_draws.csv',index=False);A.to_csv(out/'scaling_draws_agg.csv',index=False)
pd.set_option('display.width',250);print(D.round(3).T.to_string());print(S.round(4).to_string(index=False));print(A.round(4).to_string(index=False))
