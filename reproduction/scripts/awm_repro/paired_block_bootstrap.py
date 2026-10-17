"""Paired 2-degree spatial-block bootstrap of pooled RMSE differences (test year 2020 unless --year).
Compares three-seed ensembles of formal / ablation runs and the national baselines against reference models
(GRU and the LFT+GRU ensemble) on identical (station, origin, lead) keys.
Usage: python3 paired_block_bootstrap.py [--year 2020] [--out writing_materials/16_ablation_baselines]"""
import argparse,glob
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2]/'reports/awm_repro_20261015'
ap=argparse.ArgumentParser();ap.add_argument('--year',type=int,default=2020);ap.add_argument('--out',default='writing_materials/16_ablation_baselines');ap.add_argument('--n-boot',type=int,default=2000)
a=ap.parse_args();OUT=R/a.out;OUT.mkdir(parents=True,exist_ok=True);y=a.year
co=pd.read_csv(R/'writing_materials/12_figures/station_metrics_map.csv',dtype={'station':str}).drop_duplicates('station').set_index('station')
cell=np.floor(co.lat/2).astype(int).astype(str)+'_'+np.floor(co.lon/2).astype(int).astype(str)
def ens(pattern):
 fs=sorted(glob.glob(str(R/pattern)))
 if not fs:return None
 zs=[np.load(f) for f in fs];k0=pd.MultiIndex.from_arrays([zs[0]['station'].astype(str),zs[0]['origin'].astype(str)])
 for z in zs[1:]:assert (z['station'].astype(str)==zs[0]['station'].astype(str)).all() and (z['origin'].astype(str)==zs[0]['origin'].astype(str)).all()
 return pd.DataFrame(np.mean([z['prediction'] for z in zs],0),index=k0),pd.DataFrame(zs[0]['target'],index=k0),len(fs)
G,Y,_=ens(f'formal/gru_y{y}_s*/outer_predictions.npz');L,_,_=ens(f'formal/lft_y{y}_s*/outer_predictions.npz')
M={'GRU':G,'LFT':L,'LFT+GRU':(L+G)/2};seeds={'GRU':3,'LFT':3,'LFT+GRU':3}
for name,pat in [('LFT-current',f'ablation/lft_nolag_y{y}_s*/outer_predictions.npz'),('GRU+lag',f'ablation/gru_lag6_y{y}_s*/outer_predictions.npz')]:
 e=ens(pat)
 if e is not None:assert e[0].index.equals(Y.index);M[name]=e[0];seeds[name]=e[2]
for p in sorted((R/'writing_materials/15_baselines').glob('pred_*.parquet')):
 b=pd.read_parquet(p);b['station']=b.station.astype(str);b['origin']=b.origin.astype(str)
 P=b.pivot_table(index=['station','origin'],columns='lead',values='pred');P.columns=range(7);P=P.reindex(Y.index)
 miss=int(P.isna().any(axis=1).sum())
 if miss:print(p.stem,'missing sequences',miss)
 M[p.stem[5:]]=P;seeds[p.stem[5:]]=1
keep=np.ones(len(Y),bool)
for P in M.values():keep&=~P.isna().any(axis=1).values
Yv=Y.values[keep];cl=pd.Series(Y.index.get_level_values(0)[keep]).map(cell).values
sq={n:((P.values[keep]-Yv)**2) for n,P in M.items()}
rows=[]
for n,s in sq.items():rows.append(dict(model=n,seeds=seeds[n],sequences=int(keep.sum()),pooled_rmse=float(np.sqrt(s.mean())),**{f'L{k+1}':float(np.sqrt(s[:,k].mean())) for k in range(7)}))
T=pd.DataFrame(rows);T.to_csv(OUT/f'rmse_by_lead_{y}.csv',index=False);print(T.round(4).to_string(index=False))
src=(Path(__file__).parent/'multi_metrics.py').read_text();ns={'__file__':str(Path(__file__).parent/'multi_metrics.py')};exec(src[:src.index('rows=[]')],ns)   # reuse metrics() without running the module body
Sv=np.asarray(Y.index.get_level_values(0)[keep]);MM=pd.DataFrame([dict(model=n,**ns['metrics'](P.values[keep],Yv,Sv)) for n,P in M.items()]);MM.to_csv(OUT/f'multi_metrics_{y}.csv',index=False)
print(MM.round(3).to_string(index=False))
def boot(a,b):
 g=pd.DataFrame(dict(k=cl,a=a,b=b)).groupby('k').agg(['sum','count']);la,lb,n=g[('a','sum')].values,g[('b','sum')].values,g[('a','count')].values
 rng=np.random.default_rng(0);d=[]
 for _ in range(a_.n_boot):i=rng.integers(len(n),size=len(n));d.append(np.sqrt(la[i].sum()/n[i].sum())-np.sqrt(lb[i].sum()/n[i].sum()))
 return len(n),float(np.sqrt(a.mean())-np.sqrt(b.mean())),float(np.percentile(d,2.5)),float(np.percentile(d,97.5))
a_=a;rows=[]
pairs=[(m,r) for r in ('GRU','LFT+GRU') for m in sq if m!=r]+[('LFT','LFT-current'),('GRU+lag','GRU'),('LFT','GRU+lag'),('HGB+lag','HGB')]
for m,r in pairs:
 if m not in sq or r not in sq:continue
 for lead in ['all','1-3','4-7']:
  sl={'all':slice(0,7),'1-3':slice(0,3),'4-7':slice(3,7)}[lead]
  n,d,lo,hi=boot(sq[m][:,sl].mean(1),sq[r][:,sl].mean(1));rows.append(dict(year=y,model=m,reference=r,leads=lead,clusters=n,diff=d,rel_pct=100*d/np.sqrt(sq[r][:,sl].mean()),lo=lo,hi=hi))
D=pd.DataFrame(rows).drop_duplicates(['model','reference','leads']);D.to_csv(OUT/f'paired_spatial_block_{y}.csv',index=False);print(D.round(4).to_string(index=False))
