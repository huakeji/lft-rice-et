"""Revision statistics on existing 2020 predictions (no training):
(1) interaction test: gain of LFT from earlier issues vs gain of GRU from the same inputs,
    I = [RMSE(LFT) - RMSE(LFT-current)] - [RMSE(GRU-lag) - RMSE(GRU)];
(2) TOST equivalence of LFT-current and GRU with a margin of 1% of the GRU RMSE;
(3) robustness of the primary contrasts to the resampling unit: 2-degree cells (paper), 4-degree cells,
    stations, and issue months (temporal blocks);
(4) Holm-adjusted bootstrap p-values for the three primary contrasts.
Writes writing_materials/17_revision_stats/*.csv"""
import glob
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2]/'reports/awm_repro_20261015';OUT=R/'writing_materials/17_revision_stats';OUT.mkdir(parents=True,exist_ok=True)
def ens(p):
 zs=[np.load(f) for f in sorted(glob.glob(str(R/p)))];return np.mean([z['prediction'] for z in zs],0),zs[0]
L,z=ens('formal/lft_y2020_s*/outer_predictions.npz');G,_=ens('formal/gru_y2020_s*/outer_predictions.npz')
Lc,zc=ens('ablation/lft_nolag_y2020_s*/outer_predictions.npz');Gl,zg=ens('ablation/gru_lag6_y2020_s*/outer_predictions.npz')
for q in (zc,zg):assert (q['station']==z['station']).all() and (q['origin']==z['origin']).all()
Y=z['target'];st=z['station'].astype(str);org=pd.to_datetime(z['origin'].astype(str))
co=pd.read_csv(R/'writing_materials/12_figures/station_metrics_map.csv',dtype={'station':str}).drop_duplicates('station').set_index('station')
lat,lon=co.lat.reindex(st).values,co.lon.reindex(st).values
units={'cell2':np.char.add(np.floor(lat/2).astype(int).astype(str),np.floor(lon/2).astype(int).astype(str)),
       'cell4':np.char.add(np.floor(lat/4).astype(int).astype(str),np.floor(lon/4).astype(int).astype(str)),
       'station':st,'month':org.month.values.astype(str)}
SQ={k:((v-Y)**2).mean(1) for k,v in dict(LFT=L,GRU=G,ENS=(L+G)/2,LFTcur=Lc,GRUlag=Gl).items()}   # per-sequence MSE over leads
def boot(unit,fun,B=2000,seed=0):
 g=pd.Series(unit);codes,uniq=pd.factorize(g);n=np.bincount(codes)
 S={k:np.bincount(codes,weights=v) for k,v in SQ.items()};rng=np.random.default_rng(seed);out=[]
 for _ in range(B):
  w=np.bincount(rng.integers(len(n),size=len(n)),minlength=len(n));N=(w*n).sum()
  r={k:np.sqrt((w*s).sum()/N) for k,s in S.items()};out.append(fun(r))
 r0={k:np.sqrt(s.sum()/n.sum()) for k,s in S.items()};return fun(r0),np.array(out),len(n)
contr={'ENS-GRU':lambda r:r['ENS']-r['GRU'],'LFT-LFTcur':lambda r:r['LFT']-r['LFTcur'],
 'interaction':lambda r:(r['LFT']-r['LFTcur'])-(r['GRUlag']-r['GRU']),'LFTcur-GRU':lambda r:r['LFTcur']-r['GRU'],'LFT-GRUlag':lambda r:r['LFT']-r['GRUlag'],'GRUlag-GRU':lambda r:r['GRUlag']-r['GRU']}
rows=[]
for u,unit in units.items():
 for c,f in contr.items():
  d,bs,nb=boot(unit,f);p=min(1.,2*min((bs<=0).mean(),(bs>=0).mean()))
  rows.append(dict(unit=u,blocks=nb,contrast=c,estimate=d,lo95=np.percentile(bs,2.5),hi95=np.percentile(bs,97.5),lo90=np.percentile(bs,5),hi90=np.percentile(bs,95),p_boot=p))
T=pd.DataFrame(rows)
prim=T[(T.unit=='cell2')&T.contrast.isin(['ENS-GRU','LFT-LFTcur','interaction'])].sort_values('p_boot').copy()
m=len(prim);prim['p_holm']=np.minimum(1,np.maximum.accumulate([(m-i)*p for i,p in enumerate(prim.p_boot)]))
T=T.merge(prim[['unit','contrast','p_holm']],on=['unit','contrast'],how='left')
gru=np.sqrt(SQ['GRU'].mean());margin=0.01*gru;eq=T[(T.contrast=='LFTcur-GRU')].copy();eq['margin']=margin;eq['equivalent_90CI_within_margin']=(eq.lo90>-margin)&(eq.hi90<margin)
T.to_csv(OUT/'contrasts_2020.csv',index=False);eq.to_csv(OUT/'tost_lftcur_gru_2020.csv',index=False)
pd.set_option('display.width',200);print(T.round(4).to_string(index=False));print(eq[['unit','estimate','lo90','hi90','margin','equivalent_90CI_within_margin']].round(4).to_string(index=False))
print({k:round(float(np.sqrt(v.mean())),4) for k,v in SQ.items()})
