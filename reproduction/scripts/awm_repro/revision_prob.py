"""Conditional verification of the quantile LFT (2019, 2020): coverage of the central 90% and 50% intervals and
mean 90% width jointly by lead x season x rice zone, and on high-demand days (observed ETo >= 90th percentile of
all test station-days); 95% intervals from a station-cluster bootstrap (1000 resamples).
Writes writing_materials/17_revision_stats/quantile_conditional_*.csv"""
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2]/'reports/awm_repro_20261015';OUT=R/'writing_materials/17_revision_stats'
Z=pd.read_csv(R/'station_rice_zone.csv',dtype={'station':str}).set_index('station').rice_zone
SEAS={12:'DJF',1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'}
def table(y):
 q=np.load(R/f'peak_unc/quant_y{y}/outer_predictions.npz');Qs=np.sort(q['quantiles'],-1);Y=q['target'];n=len(Y)
 st=np.repeat(q['station'].astype(str),7);org=pd.to_datetime(np.repeat(q['origin'].astype(str),7));lead=np.tile(np.arange(1,8),n)
 valid=org+pd.to_timedelta(lead-1,'D');d=pd.DataFrame(dict(station=st,lead=lead,season=valid.month.map(SEAS),zone=pd.Series(st).map(Z).values,
  y=Y.ravel(),lo90=Qs[:,:,0].ravel(),hi90=Qs[:,:,6].ravel(),lo50=Qs[:,:,2].ravel(),hi50=Qs[:,:,4].ravel()))
 d['c90']=((d.y>=d.lo90)&(d.y<=d.hi90)).astype(float);d['c50']=((d.y>=d.lo50)&(d.y<=d.hi50)).astype(float);d['w90']=d.hi90-d.lo90
 d['high']=d.y>=np.quantile(d.y,0.9);return d
def cboot(g,col,B=1000,seed=0):
 s=g.groupby('station')[col].agg(['sum','count']);rng=np.random.default_rng(seed);a,b=s['sum'].values,s['count'].values
 bs=[(a[i].sum()/b[i].sum()) for i in (rng.integers(len(a),size=len(a)) for _ in range(B))];return np.percentile(bs,[2.5,97.5])
for y in (2019,2020):
 d=table(y);rows=[]
 for keys,g in d.groupby(['lead','season','zone']):
  lo,hi=cboot(g,'c90');rows.append(dict(zip(['lead','season','zone'],keys),n=len(g),stations=g.station.nunique(),cov90=g.c90.mean(),cov90_lo=lo,cov90_hi=hi,cov50=g.c50.mean(),width90=g.w90.mean()))
 J=pd.DataFrame(rows);J.to_csv(OUT/f'quantile_conditional_joint_{y}.csv',index=False)
 hd=[]
 for lab,g in [('all',d),('high_demand',d[d.high]),('high_demand_JJA',d[d.high&(d.season=='JJA')])]+[(f'high_lead{L}',d[d.high&(d.lead==L)]) for L in (1,7)]:
  lo,hi=cboot(g,'c90');hd.append(dict(subset=lab,n=len(g),cov90=g.c90.mean(),cov90_lo=lo,cov90_hi=hi,cov50=g.c50.mean(),width90=g.w90.mean(),bias_median=float((((g.lo50+g.hi50)/2)-g.y).mean())))
 H=pd.DataFrame(hd);H.to_csv(OUT/f'quantile_high_demand_{y}.csv',index=False)
 print(y,'joint cells',len(J),'cov90 min/median/max',J.cov90.min().round(3),J.cov90.median().round(3),J.cov90.max().round(3),
  'cells with CI excluding 0.90:',int(((J.cov90_hi<0.9)|(J.cov90_lo>0.9)).sum()),'below:',int((J.cov90_hi<0.9).sum()),'above:',int((J.cov90_lo>0.9).sum()))
 print(J.sort_values('cov90').head(5).round(3).to_string(index=False));print(H.round(3).to_string(index=False))
