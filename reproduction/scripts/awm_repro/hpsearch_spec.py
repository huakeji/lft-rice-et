"""Random hyperparameter search spec (development roles only: outer 2020 fold, selection by 2019 val).
Runs on a fixed 195-station subset; the top configurations are then confirmed on all stations."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SPACE={
 'pft':dict(width=[64,128,192,256],layers=[1,2,3,4],heads=[4,8],emb_dim=[8,16,32]),
 'gru':dict(width=[64,128,256,384],layers=[1,2,3]),
 'mlp':dict(width=[128,256,512,1024],layers=[1,2,3,4]),
 'cbm':dict(cbm_width=[32,64,96,128]),
 'lft':dict(width=[64,128,192],layers=[1,2,3],heads=[4,8])}
COMMON=dict(dropout=[0.0,0.05,0.1,0.2],weight_decay=[0.0,1e-5,1e-4,1e-3],batch_size=[256,512,1024,2048],max_epochs=[15,30,60])
def sample(m,rng):
 c={k:v[rng.integers(len(v))] for k,v in {**SPACE[m],**COMMON}.items()};c['lr']=float(np.exp(rng.uniform(np.log(1e-4),np.log(3e-3))))
 if m not in ('pft','lft'):c.update(width=c.get('width',128),layers=c.get('layers',2),heads=4)
 c={k:(v.item() if hasattr(v,'item') else v) for k,v in c.items()};c['warmup_epochs']=1 if c['max_epochs']==15 else 3;c['patience']=15;return c
def main(n,subset,shards,lags,tag,models='pft,gru,cbm,mlp'):
 rng=np.random.default_rng(7);jobs=[]
 for i in range(n):
  for m in models.split(','):
   c=sample(m,rng);me=c.pop('max_epochs')
   jobs.append(dict(model=m,outer_year=2020,seed=0,max_epochs=me,config=c,shards_dir=shards,station_subset=subset,order='global',**({'lags':lags} if lags else {}),
    out_dir=f'reports/awm_repro_20261015/hpsearch/{tag}/{m}_c{i:02d}'))
 spec=dict(stage=f'hpsearch-{tag}',selection='val_rmse on 2019 (outer 2020 fold); outer metrics on the subset are recorded but never used',station_subset=subset,jobs=jobs)
 p=ROOT/f'reports/awm_repro_20261015/hpsearch/{tag}_jobs.json';p.write_text(json.dumps(spec,indent=1));print(p,len(jobs))
if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],int(sys.argv[4]),sys.argv[5],*sys.argv[6:])
