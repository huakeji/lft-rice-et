"""Summary of rice-zone group learning / transfer runs (outer 2020). For every zone, RMSE on that zone's
stations under each setting; zero-shot rows come only from runs whose training chain excluded the zone."""
import json,glob,os,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];G=ROOT/'reports/awm_repro_20261015/group/s0'
Z=pd.read_csv(ROOT/'reports/awm_repro_20261015/station_rice_zone.csv',dtype={'station':str}).set_index('station').rice_zone
def load(n):
 f=G/n/'outer_predictions.npz'
 return np.load(f) if f.exists() else None
r=lambda P,Y:float(np.sqrt(((P-Y)**2).mean()))
def zr(z,zone):
 k=pd.Series(z['station'].astype(str)).map(Z).values==zone;return r(z['prediction'][k],z['target'][k]) if k.any() else np.nan
rows=[];C=load('C_national')
for zone in ['R1','R2','R3','R4','R5','R6']:
 row=dict(zone=zone,C_national=zr(C,zone))
 for n,lab in [(f'B_zone_{zone}','B_zone_only'),(f'D_natft_{zone}','D_national_ft'),(f'F_lzo_{zone}','F_zero_shot'),(f'P_lzo_{zone}_phys','P_zero_shot_phys'),(f'U_lzo_{zone}_default','U_zero_shot_untuned')]:
  z=load(n);row[lab]=zr(z,zone) if z is not None else np.nan
 for m in ('full','head','norm','affine'):
  for d in (30,90,365):
   z=load(f'E_lzoft{d}_{zone}' if m=='full' else f'T_{m}{d}_{zone}');row[f'{m}_{d}d']=zr(z,zone) if z is not None else np.nan
 A=[load(Path(p).parent.name) for p in glob.glob(str(G/'A_station_*/outer_predictions.npz'))];A=[a for a in A if a is not None and Z.get(str(a['station'][0]))==zone]
 if A:
  st=[str(a['station'][0]) for a in A];row['A_single_station']=float(np.sqrt(np.mean([((a['prediction']-a['target'])**2).mean() for a in A])))
  k=np.isin(C['station'].astype(str),st);row['C_on_same_stations']=r(C['prediction'][k],C['target'][k])
 rows.append(row)
D=pd.DataFrame(rows).set_index('zone');S=[]
h=set((ROOT/'reports/awm_repro_20261015/group/lists/scale_holdout').read_text().split());k=np.isin(C['station'].astype(str),list(h))
for n in [25,50,100,200,400,484]:
 z=load(f'S_scale_{n}');S.append(dict(draw='s0',train_stations=n,zero_shot_rmse=r(z['prediction'],z['target']) if z is not None else np.nan))
S.append(dict(draw='s0',train_stations='national (holdout seen)',zero_shot_rmse=r(C['prediction'][k],C['target'][k])))
for dr in ('d8','d9'):   # additional random holdout draws (99 stations each), same protocol
 hd=set((ROOT/f'reports/awm_repro_20261015/group/lists/scale_holdout_{dr}').read_text().split());kd=np.isin(C['station'].astype(str),list(hd))
 for n in [200,484]:
  z=load(f'S_scale_{n}_{dr}');S.append(dict(draw=dr,train_stations=n,zero_shot_rmse=r(z['prediction'],z['target']) if z is not None else np.nan))
 S.append(dict(draw=dr,train_stations='national (holdout seen)',zero_shot_rmse=r(C['prediction'][kd],C['target'][kd])))
out=ROOT/'reports/awm_repro_20261015/group_summary';out.mkdir(exist_ok=True);D.to_csv(out/'zone_settings.csv');pd.DataFrame(S).to_csv(out/'scaling.csv',index=False)
pd.set_option('display.width',250);print(D.round(3).T.to_string());print(pd.DataFrame(S).round(4).to_string(index=False))
