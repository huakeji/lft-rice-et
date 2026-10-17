"""Analyses for the writing materials (no training): (1) lead-wise skill against station day-of-year
climatology (training years only) and persistence (observed ET0 of the day before issue); (2) error
attribution by season, forecast rain/no-rain on the target day, zone and lead; (3) probabilistic
reliability of the quantile model; (4) zero-shot degradation versus station climate descriptors."""
import glob,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/awm_repro_20261015';W=R/'writing_materials'
Z=pd.read_csv(R/'station_rice_zone.csv',dtype={'station':str}).set_index('station')
rm=lambda e:float(np.sqrt(np.mean(np.square(e))))
COLS=[0,2,3,6,7,8,34]   # tmax, dtr, rain, lat, lon, alt, spread_tmax (lag 6 manifest)
def shards():
 X={}
 for p in sorted((R/'lagged_shards').glob('station_*.npz')):
  with np.load(p) as z:X[str(z['station'][0])]=dict(x=z['x'][:,:,COLS].astype('float32'),y=z['y'].astype('float32'),origin=z['origin'].astype(str))
 return X
X=shards();CI={c:i for i,c in enumerate(COLS)}
def keyed(S,O):   # inputs of evaluated sequences
 idx={};out=[]
 for st in np.unique(S):
  z=X[st];m={o:i for i,o in enumerate(z['origin'])};idx[st]=(z,m)
 xs=np.stack([idx[s][0]['x'][idx[s][1][o]] for s,o in zip(S,O)]);return xs
def obs_series():
 rows=[]
 for st,z in X.items():
  o=pd.to_datetime(z['origin']);d=pd.DataFrame(dict(date=np.concatenate([o+pd.Timedelta(days=k) for k in range(7)]),et0=np.concatenate([z['y'][:,k] for k in range(7)])))
  d=d.drop_duplicates('date');d['station']=st;rows.append(d)
 d=pd.concat(rows);return d.set_index(['station','date']).et0
OBS=obs_series()
def ens(models,y):
 P=[];base=None
 for m in models:
  zs=[np.load(f) for f in sorted(glob.glob(str(R/f'formal/{m}_y{y}_s*/outer_predictions.npz')))]
  P.append(np.mean([z['prediction'] for z in zs],0));base=zs[0]
 return np.mean(P,0),base['target'],base['station'].astype(str),base['origin'].astype(str)
def skill():
 rows=[]
 for y in (2020,2019):
  _,Y,S,O=ens(['gru'],y);od=pd.to_datetime(O)
  tr=OBS[OBS.index.get_level_values('date')<=pd.Timestamp(f'{y-2}-12-31')].reset_index();tr['doy']=tr.date.dt.dayofyear
  cl=tr.groupby(['station','doy']).et0.mean().unstack()
  cl=pd.DataFrame(np.vstack([np.convolve(np.r_[r[-7:],r,r[:7]],np.ones(15)/15,'valid') for r in cl.reindex(columns=range(1,367)).ffill(axis=1).bfill(axis=1).to_numpy()]),index=cl.index,columns=range(1,367))
  C=np.stack([cl.reindex(S).to_numpy()[np.arange(len(S)),((od+pd.Timedelta(days=k)).dayofyear-1).to_numpy()] for k in range(7)],1)
  prev=OBS.reindex(list(zip(S,od-pd.Timedelta(days=1)))).to_numpy();keep=np.isfinite(prev)&np.isfinite(C).all(1)   # stations without training-period history lack a climatology
  for name,models in [('LFT+GRU',['lft','gru']),('LFT',['lft']),('GRU',['gru']),('PFT',['pft']),('MLP',['mlp']),('CBM',['cbm'])]:
   P,_,_,_=ens(models,y)
   for k in range(7):
    e=P[keep,k]-Y[keep,k];ec=C[keep,k]-Y[keep,k];ep=prev[keep]-Y[keep,k]
    rows.append(dict(year=y,model=name,lead=k+1,stations=int(len(np.unique(S[keep]))),rmse=rm(e),rmse_clim=rm(ec),rmse_persist=rm(ep),ss_clim=1-np.mean(e**2)/np.mean(ec**2),ss_persist=1-np.mean(e**2)/np.mean(ep**2),n=int(keep.sum())))
 d=pd.DataFrame(rows);(W/'04_skill').mkdir(parents=True,exist_ok=True);d.to_csv(W/'04_skill/skill_by_lead.csv',index=False);return d
def attribution():
 rows=[]
 for y in (2020,2019):
  P,Y,S,O=ens(['lft','gru'],y);x=keyed(S,O);od=pd.to_datetime(O)
  for k in range(7):
   t=od+pd.Timedelta(days=k);mon=t.month;season=np.select([np.isin(mon,[3,4,5]),np.isin(mon,[6,7,8]),np.isin(mon,[9,10,11])],['MAM','JJA','SON'],'DJF')
   rain=np.where(x[:,k,CI[3]]>0,'rain','no_rain');zone=pd.Series(S).map(Z.rice_zone).values;e=P[:,k]-Y[:,k]
   rows.append(pd.DataFrame(dict(year=y,lead=k+1,season=season,rain=rain,zone=zone,err=e,obs=Y[:,k])))
 D=pd.concat(rows);(W/'05_attribution').mkdir(parents=True,exist_ok=True)
 out=[]
 for keys in (['season'],['rain'],['zone'],['season','rain'],['lead','rain']):
  g=D.groupby(['year']+keys).agg(n=('err','size'),rmse=('err',lambda e:rm(e)),bias=('err','mean'),obs_mean=('obs','mean')).reset_index();g['rel_rmse']=g.rmse/g.obs_mean;g['group']='+'.join(keys);out.append(g)
 A=pd.concat(out);A.to_csv(W/'05_attribution/error_by_group.csv',index=False);return A
def reliability():
 rows=[];pit=[]
 for y in (2020,2019):
  z=np.load(R/f'peak_unc/quant_y{y}/outer_predictions.npz');Q=z['quantiles'];Y=z['target'];q=list(z['q'])
  S=z['station'].astype(str);O=z['origin'].astype(str);x=keyed(S,O);spread=x[:,:,CI[34]]
  for k in range(7):
   qq=Q[:,k];yy=Y[:,k];crps=2*np.mean([np.mean(np.maximum(t*(yy-qq[:,i]),(t-1)*(yy-qq[:,i]))) for i,t in enumerate(q)])
   w90=qq[:,6]-qq[:,0];ae=np.abs(qq[:,3]-yy)
   rows.append(dict(year=y,lead=k+1,cover90=float(((yy>=qq[:,0])&(yy<=qq[:,6])).mean()),cover80=float(((yy>=qq[:,1])&(yy<=qq[:,5])).mean()),cover50=float(((yy>=qq[:,2])&(yy<=qq[:,4])).mean()),
    width90=float(w90.mean()),crps_approx=float(crps),rmse_median=rm(qq[:,3]-yy),spearman_width_abserr=float(pd.Series(w90).corr(pd.Series(ae),method='spearman')),spearman_width_lagspread=float(pd.Series(w90).corr(pd.Series(spread[:,k]),method='spearman'))))
   r=(yy[:,None]>qq).sum(1);pit.append(pd.DataFrame(dict(year=y,lead=k+1,bin=np.arange(8),freq=np.bincount(r,minlength=8)/len(r))))
 (W/'06_uncertainty').mkdir(parents=True,exist_ok=True);d=pd.DataFrame(rows);d.to_csv(W/'06_uncertainty/reliability_by_lead.csv',index=False);pd.concat(pit).to_csv(W/'06_uncertainty/pit_histogram.csv',index=False);return d
def transfer_physics():
 C=np.load(R/'group/s0/C_national/outer_predictions.npz');cs=C['station'].astype(str)
 rows=[]
 for zone in ['R1','R2','R3','R4','R5','R6']:
  F=np.load(R/f'group/s0/F_lzo_{zone}/outer_predictions.npz');fs=F['station'].astype(str)
  for st in np.unique(fs):
   a=fs==st;b=cs==st;z=X[st];x=z['x'];yv=z['y']
   rows.append(dict(station=st,zone=zone,province=Z.province.get(st),rmse_national=rm(C['prediction'][b]-C['target'][b]),rmse_zero_shot=rm(F['prediction'][a]-F['target'][a]),bias_zero_shot=float((F['prediction'][a]-F['target'][a]).mean()),
    lat=float(x[0,0,CI[6]]),lon=float(x[0,0,CI[7]]),alt=float(x[0,0,CI[8]]),et0_mean=float(yv[:,0].mean()),tmax_mean=float(x[:,0,CI[0]].mean()),dtr_mean=float(x[:,0,CI[2]].mean()),rain_day_frac=float((x[:,0,CI[3]]>0).mean())))
 d=pd.DataFrame(rows);d['degradation_pct']=100*(d.rmse_zero_shot/d.rmse_national-1)
 # novelty: distance to nearest station outside the zone in standardized climate-descriptor space
 f=['alt','et0_mean','tmax_mean','dtr_mean','rain_day_frac'];Zs=(d[f]-d[f].mean())/d[f].std()
 d['novelty']=[float(np.min(np.linalg.norm(Zs[d.zone!=d.zone[i]].to_numpy()-Zs.iloc[i].to_numpy(),axis=1))) for i in range(len(d))]
 (W/'08_transfer_physics').mkdir(parents=True,exist_ok=True);d.to_csv(W/'08_transfer_physics/zero_shot_station_descriptors.csv',index=False)
 cor={c:float(d.degradation_pct.corr(d[c],method='spearman')) for c in f+['novelty']}
 (W/'08_transfer_physics/spearman_degradation.json').write_text(json.dumps(cor,indent=1));return d,cor
if __name__=='__main__':
 pd.set_option('display.width',220)
 s=skill();print(s[s.model.isin(['LFT+GRU'])].round(3).to_string(index=False))
 a=attribution();print(a[a.group.isin(['season','rain','season+rain'])&(a.year==2020)].round(3).to_string(index=False))
 r=reliability();print(r.round(3).to_string(index=False))
 d,c=transfer_physics();print(c);print(d.groupby('zone')[['degradation_pct','novelty','alt']].median().round(2))
