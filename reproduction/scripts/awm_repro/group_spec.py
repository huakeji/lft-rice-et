"""Rice-zone group learning / transfer experiments (outer 2020 fold, model = LFT light config lft_c03).
C national | B zone-only | A single-station (10 per zone) | F leave-zone-out (zero-shot) |
D national -> zone fine-tune | E leave-zone-out -> zone fine-tune with last 30/90/365 days |
S scaling: fixed 100-station zero-shot holdout, train on nested random 25..483 stations."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];G='reports/awm_repro_20261015/group'
def main(seed=0):
 seed=int(seed);g=ROOT/G;(g/'lists').mkdir(parents=True,exist_ok=True)
 Z=pd.read_csv(ROOT/'reports/awm_repro_20261015/station_rice_zone.csv',dtype={'station':str})
 have={p.stem.split('_',1)[1] for p in (ROOT/'reports/awm_repro_20261015/lagged_shards').glob('station_*.npz')};Z=Z[Z.station.isin(have)]
 def lst(name,ids):(g/'lists'/name).write_text('\n'.join(sorted(ids)));return f'{G}/lists/{name}'
 base=next(j for j in json.loads((ROOT/'reports/awm_repro_20261015/confirm/lft_jobs.json').read_text())['jobs'] if 'lft_c03' in j['out_dir'])
 cfg=base['config'];ft=dict(cfg,lr=cfg['lr']*0.3,warmup_epochs=1,patience=5)
 common=dict(model='lft',outer_year=2020,seed=seed,shards_dir=base['shards_dir'],lags=base['lags'],order='global')
 J=[];o=lambda n:f'{G}/s{seed}/{n}'
 allf=lst('all',Z.station)
 J.append(dict(common,max_epochs=base['max_epochs'],config=cfg,train_stations=allf,eval_stations=allf,out_dir=o('C_national')))
 rng=np.random.default_rng(100+seed)
 for z,d in Z.groupby('rice_zone'):
  zf=lst(f'zone_{z}',d.station);rest=lst(f'not_{z}',Z.station[Z.rice_zone!=z])
  J.append(dict(common,max_epochs=base['max_epochs'],config=cfg,train_stations=rest,eval_stations=zf,out_dir=o(f'F_lzo_{z}')))
  J.append(dict(common,max_epochs=base['max_epochs'],config=cfg,train_stations=zf,eval_stations=zf,out_dir=o(f'B_zone_{z}')))
  J.append(dict(common,max_epochs=10,config=ft,train_stations=zf,eval_stations=zf,init_from=o('C_national'),out_dir=o(f'D_natft_{z}')))
  for n in (30,90,365):J.append(dict(common,max_epochs=10,config=ft,train_stations=zf,eval_stations=zf,init_from=o(f'F_lzo_{z}'),train_last_days=n,out_dir=o(f'E_lzoft{n}_{z}')))
  for s in sorted(rng.choice(sorted(d.station),10,replace=False)):
   sf=lst(f'st_{s}',[s]);J.append(dict(common,max_epochs=base['max_epochs'],config=cfg,train_stations=sf,eval_stations=sf,out_dir=o(f'A_station_{s}')))
 r2=np.random.default_rng(7);hold=[];pool=[]
 for z,d in Z.groupby('rice_zone'):
  ids=sorted(d.station);k=max(1,round(100*len(ids)/len(Z)));pick=set(r2.choice(ids,k,replace=False));hold+=sorted(pick);pool+=[i for i in ids if i not in pick]
 hf=lst('scale_holdout',hold);order=list(np.random.default_rng(11+seed).permutation(sorted(pool)))
 for k in (25,50,100,200,400,len(order)):
  J.append(dict(common,max_epochs=base['max_epochs'],config=cfg,train_stations=lst(f'scale_train_{k}_s{seed}',order[:k]),eval_stations=hf,out_dir=o(f'S_scale_{k}')))
 big=lambda j:j['out_dir'].split('/')[-1][0] in 'CFS'
 J=[j for j in J if big(j)]+[j for j in J if not big(j)]   # long jobs and dependencies first
 p=g/f'group_s{seed}_jobs.json';p.write_text(json.dumps(dict(stage='group',seed=seed,model='lft_c03 (light LFT)',jobs=J),indent=1));print(p,len(J),'holdout',len(hold),'pool',len(pool))
if __name__=='__main__':main(*sys.argv[1:])
