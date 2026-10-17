"""Revision: (a) within-zone Spearman correlations of zero-shot degradation with altitude and climatic distance
(R3, R6; three-seed station table); (b) LFT+GRU 2020 metrics for leads 2-7 only (true future leads) vs lead 1;
(c) rice-season ETc errors on high-demand days at rice stations (Kc x forecast ET0 vs Kc x observed ET0)."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
R=Path(__file__).resolve().parents[2]/'reports/awm_repro_20261015';W=R/'writing_materials';OUT=W/'17_revision_stats'
d=pd.read_csv(W/'08_transfer_physics/zero_shot_station_descriptors_3seed.csv',dtype={'station':str});rows=[]
for zone in ('R3','R6','all'):
 g=d if zone=='all' else d[d.zone==zone]
 for v in ('alt','novelty'):
  r,p=spearmanr(g[v],g.degradation_pct);rows.append(dict(zone=zone,variable=v,n=len(g),rho=r,p=p))
A=pd.DataFrame(rows);A.to_csv(OUT/'within_zone_correlations.csv',index=False);print(A.round(3).to_string(index=False))
z=np.load(R/'formal_summary/ens_lft_gru_y2020.npz');P,Y=z['prediction'],z['target'];e=P-Y
for lab,sl in (('lead 1',slice(0,1)),('leads 2-7',slice(1,7)),('all',slice(0,7))):
 ee=e[:,sl];yy=Y[:,sl];print(lab,'RMSE',round(float(np.sqrt((ee**2).mean())),3),'NSE',round(float(1-(ee**2).sum()/((yy-yy.mean())**2).sum()),3))
kc=pd.read_csv(R.parents[1]/'paper_afm_2026/04_final/Supplementary_Data_S1_Kc_station_month.csv',dtype={'station':str}).set_index(['station','month'])
b=np.load(R/'formal/gru_y2020_s0/outer_predictions.npz');assert (b['station'].astype(str)==z['station'].astype(str)).all()
st=np.repeat(b['station'].astype(str),7);org=pd.to_datetime(np.repeat(b['origin'].astype(str),7));lead=np.tile(np.arange(1,8),len(P))
valid=org+pd.to_timedelta(lead-1,'D');k=kc.reindex(list(zip(st,valid.month)))
D=pd.DataFrame(dict(kc=k.Kc.values,season=k.rice_season_month.values,rice=k.rice_station.values if 'rice_station' in k else 1,f=P.ravel(),o=Y.ravel()));D=D[(D.season==1)&(D.rice==1)]
D['ef']=D.kc*D.f.clip(lower=0);D['eo']=D.kc*D.o;hi=D.eo>=np.quantile(D.eo,0.9)
for lab,g in (('rice season, all days',D),('rice season, top-10% ETc days',D[hi])):
 er=g.ef-g.eo;print(lab,'n',len(g),'RMSE',round(float(np.sqrt((er**2).mean())),3),'bias',round(float(er.mean()),3),'mean ETc',round(float(g.eo.mean()),2))
