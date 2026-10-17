"""Formal comparison summary: per model x outer year, seed ensemble (mean of seed predictions) metrics,
single-seed spread, per-lead RMSE, rice-zone RMSE, and paired station-block bootstrap vs a reference model."""
import json,glob,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/awm_repro'))
from multi_metrics import metrics
F=ROOT/'reports/awm_repro_20261015/formal';Z=pd.read_csv(ROOT/'reports/awm_repro_20261015/station_rice_zone.csv',dtype={'station':str}).set_index('station').rice_zone
def ens(m,y):
 fs=sorted(glob.glob(str(F/f'{m}_y{y}_s*/outer_predictions.npz')));zs=[np.load(f) for f in fs]
 if not zs:return None
 for z in zs[1:]:assert (z['origin']==zs[0]['origin']).all() and (z['station']==zs[0]['station']).all()
 return dict(P=np.mean([z['prediction'] for z in zs],0),Y=zs[0]['target'],S=zs[0]['station'],O=zs[0]['origin'],n=len(zs),single=[float(np.sqrt(((z['prediction']-z['target'])**2).mean())) for z in zs])
def boot(a,b,S,B=2000):
 da=((a['P']-a['Y'])**2).mean(1);db=((b['P']-b['Y'])**2).mean(1);g=pd.DataFrame(dict(s=S,a=da,b=db)).groupby('s').agg(['sum','count'])
 la,lb,n=g[('a','sum')].values,g[('b','sum')].values,g[('a','count')].values;rng=np.random.default_rng(0);d=[]
 for _ in range(B):i=rng.integers(len(n),size=len(n));d.append(np.sqrt(la[i].sum()/n[i].sum())-np.sqrt(lb[i].sum()/n[i].sum()))
 return float(np.sqrt(da.mean())-np.sqrt(db.mean())),np.percentile(d,[2.5,97.5]).tolist()
def main(ref='gru'):
 rows=[];lead=[];zone=[];pair=[]
 models=sorted({Path(p).name.split('_y')[0] for p in glob.glob(str(F/'*_y*_s*'))})
 for y in (2020,2019,2018):
  E={m:ens(m,y) for m in models};E={k:v for k,v in E.items() if v}
  for m,e in E.items():
   rows.append(dict(model=m,year=y,seeds=e['n'],single_min=min(e['single']),single_max=max(e['single']),**metrics(e['P'],e['Y'],e['S'])))
   lead.append(dict(model=m,year=y,**{f'L{k+1}':float(np.sqrt(((e['P'][:,k]-e['Y'][:,k])**2).mean())) for k in range(7)}))
   zz=pd.Series(e['S']).map(Z).values
   for r in sorted(set(zz)):k=zz==r;zone.append(dict(model=m,year=y,zone=r,rmse=float(np.sqrt(((e['P'][k]-e['Y'][k])**2).mean()))))
   if ref in E and m!=ref and e['n']==E[ref]['n']:d,ci=boot(e,E[ref],e['S']);pair.append(dict(model=m,ref=ref,year=y,diff=d,ci_lo=ci[0],ci_hi=ci[1],seeds=e['n']))
 out=ROOT/'reports/awm_repro_20261015/formal_summary';out.mkdir(exist_ok=True)
 for n,d in [('metrics',rows),('lead',lead),('zone',zone),('paired',pair)]:pd.DataFrame(d).to_csv(out/f'{n}.csv',index=False)
 pd.set_option('display.width',250)
 print(pd.DataFrame(rows)[['model','year','seeds','RMSE','single_min','single_max','MAE','Bias','R2','NSE','KGE','rRMSE_7d_sum_pct','RMSE_top10pct_days','Bias_top10pct_days']].round(3).to_string(index=False))
 print(pd.DataFrame(pair).round(4).to_string(index=False))
 print(pd.DataFrame(zone).pivot_table(index=['year','model'],columns='zone',values='rmse').round(3).to_string())
if __name__=='__main__':main(*sys.argv[1:])
