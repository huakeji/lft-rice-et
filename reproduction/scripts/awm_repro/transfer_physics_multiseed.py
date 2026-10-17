"""Station-level zero-shot degradation averaged over three seeds (outer 2020). For each station of a withheld zone:
RMSE of the leave-zone-out model and of the national model, each averaged over seeds 0-2 (seed-matched ratio),
plus climate descriptors and novelty as in materials_analyses.transfer_physics. Also the pooled-learning comparisons
(single-station, zone-only) against the three-seed national model. Writes writing_materials/08_transfer_physics/*_3seed.*"""
import glob,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/awm_repro_20261015';W=R/'writing_materials';G=R/'group'
Z=pd.read_csv(R/'station_rice_zone.csv',dtype={'station':str}).set_index('station')
COLS=[0,2,3,6,7,8];CI={c:i for i,c in enumerate(COLS)};SEEDS=('s0','s1','s2')
def st_rmse(f):
 z=np.load(f);s=z['station'].astype(str);e=(z['prediction']-z['target'])**2
 return pd.DataFrame(e).groupby(s).mean().mean(1)**0.5,pd.Series((z['prediction']-z['target']).mean(1)).groupby(s).mean()
C=[st_rmse(G/s/'C_national/outer_predictions.npz')[0] for s in SEEDS]
rows=[]
for zone in ['R1','R2','R3','R4','R5','R6']:
 F=[st_rmse(G/s/f'F_lzo_{zone}/outer_predictions.npz') for s in SEEDS]
 for st in F[0][0].index:
  with np.load(R/f'lagged_shards/station_{st}.npz') as z:x=z['x'][:,0][:,COLS];yv=z['y'][:,0]
  rz=np.mean([f[0][st] for f in F]);rc=np.mean([c[st] for c in C])
  rows.append(dict(station=st,zone=zone,province=Z.province.get(st),rmse_national=rc,rmse_zero_shot=rz,bias_zero_shot=float(np.mean([f[1][st] for f in F])),
   degradation_pct=100*np.mean([F[i][0][st]/C[i][st]-1 for i in range(3)]),lat=float(x[0,CI[6]]),lon=float(x[0,CI[7]]),alt=float(x[0,CI[8]]),
   et0_mean=float(yv.mean()),tmax_mean=float(x[:,CI[0]].mean()),dtr_mean=float(x[:,CI[2]].mean()),rain_day_frac=float((x[:,CI[3]]>0).mean())))
d=pd.DataFrame(rows);f=['alt','et0_mean','tmax_mean','dtr_mean','rain_day_frac'];Zs=(d[f]-d[f].mean())/d[f].std()
d['novelty']=[float(np.min(np.linalg.norm(Zs[d.zone!=d.zone[i]].to_numpy()-Zs.iloc[i].to_numpy(),axis=1))) for i in range(len(d))]
O=W/'08_transfer_physics';d.to_csv(O/'zero_shot_station_descriptors_3seed.csv',index=False)
cor={c:float(d.degradation_pct.corr(d[c],method='spearman')) for c in f+['novelty']};(O/'spearman_degradation_3seed.json').write_text(json.dumps(cor,indent=1))
pd.set_option('display.width',220);print(cor);print(d.groupby('zone')[['degradation_pct','bias_zero_shot','alt','novelty']].median().round(3))
print(d[d.zone=='R3'].groupby('province').agg(n=('station','size'),bias=('bias_zero_shot','mean')).round(2))
print(d[d.zone.isin(['R3','R6'])].groupby('zone').bias_zero_shot.mean().round(3))
# pooled learning vs three-seed national model
Cz=[np.load(G/s/'C_national/outer_predictions.npz') for s in SEEDS]
def rr(z,k):return float(np.sqrt(((z['prediction'][k]-z['target'][k])**2).mean()))
out=[]
for zone in ['R1','R2','R3','R4','R5','R6']:
 kz=[pd.Series(c['station'].astype(str)).map(Z.rice_zone).values==zone for c in Cz];cm=np.mean([rr(c,k) for c,k in zip(Cz,kz)])
 B=np.load(G/'s0'/f'B_zone_{zone}/outer_predictions.npz');b=rr(B,np.ones(len(B['target']),bool))
 A=[np.load(p) for p in glob.glob(str(G/'s0/A_station_*/outer_predictions.npz'))];A=[a for a in A if Z.rice_zone.get(str(a['station'][0]))==zone]
 st=[str(a['station'][0]) for a in A];a_=float(np.sqrt(np.mean([((a['prediction']-a['target'])**2).mean() for a in A])))
 cs=[rr(c,np.isin(c['station'].astype(str),st)) for c in Cz]
 out.append(dict(zone=zone,C_mean=cm,B_zone=b,B_rel=100*(b/cm-1),n_single=len(A),A_single=a_,C_same_mean=np.mean(cs),C_same_min=min(cs),C_same_max=max(cs),A_rel=100*(a_/np.mean(cs)-1)))
P=pd.DataFrame(out);P.to_csv(R/'group_summary/pooled_vs_national_3seed.csv',index=False);print(P.round(3).to_string(index=False))
