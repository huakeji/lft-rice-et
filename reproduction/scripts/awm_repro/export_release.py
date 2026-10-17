"""Export the frozen 2020-fold models (LFT and GRU, seeds 0-2) for the open release.

For each family the encoder (input scaling, text vocabularies) and the target scaling are refitted on the
training role of the 2020 fold exactly as in pft_fast.py (deterministic), the checkpoint is loaded, and the
predictions on the 2020 test role are compared with the saved outer_predictions.npz of the paper runs.
Writes release/<name>/model/{family}_encoder.json, {family}_s{k}.pt (state_dict) and {family}_config.json.
Usage: python3 export_release.py --family lft|gru --out release/lft-rice-et [--check-stations 40]"""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/audit_v11'));sys.path.insert(0,str(Path(__file__).parent))
import pft_fast
from research_fold_data import FoldData
from research_models import ForecastModel
R=ROOT/'reports/awm_repro_20261015'
ap=argparse.ArgumentParser();ap.add_argument('--family',required=True,choices=['lft','gru','lftq']);ap.add_argument('--out',default='release/lft-rice-et');ap.add_argument('--check-stations',type=int,default=40)
a=ap.parse_args();OUT=ROOT/a.out/'model';OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
runs=[R/'peak_unc/quant_y2020'] if a.family=='lftq' else [R/f'formal/{a.family}_y2020_s{s}' for s in range(3)]
res=[json.loads((r/'result.json').read_text()) for r in runs];cfg=res[0]['config']
shards_dir=res[0]['shards_dir'];lags=res[0]['lags']
assert all(x['shards_dir']==shards_dir and x['lags']==lags and x['config']==cfg for x in res),'runs differ in data or config'
if lags:pft_fast.use_lags(shards_dir,lags)
shards=sorted((ROOT/shards_dir).glob('station_*.npz'))
fold=FoldData(shards,2020,[],-1);enc,target=fold.fit_encoder()
for r in runs:   # where the paper run stored its encoder, it must equal the refit
 if (r/'target.json').exists():
  t=json.loads((r/'target.json').read_text());assert abs(t['mean']-target['mean'])<1e-9 and abs(t['scale']-target['scale'])<1e-9,'target scaling differs'
 if (r/'encoder.pkl').exists():
  import pickle;e=pickle.loads((r/'encoder.pkl').read_bytes())
  assert list(e.ni)==list(enc.ni) and np.allclose(e.mean,enc.mean) and np.allclose(e.scale,enc.scale) and all(e.vocab[j]==enc.vocab[j] for j in enc.ci),'encoder differs'
import matched_baselines as MB
state=dict(numeric_columns=[int(i) for i in enc.ni],numeric_names=[MB.NUM[i] for i in enc.ni],mean=[float(v) for v in enc.mean],scale=[float(v) for v in enc.scale],
 categorical_columns=[int(j) for j in enc.ci],categorical_names=[MB.CATS[j] for j in enc.ci],vocabulary=[list(map(str,enc.vocab[j])) for j in enc.ci],
 unknown_index=0,target_mean=target['mean'],target_scale=target['scale'],fit_scope='training role of the 2020 fold (valid dates 2015-03 to 2018-12)',lags=lags)
(OUT/f'{a.family}_encoder.json').write_text(json.dumps(state,ensure_ascii=False))
(OUT/f'{a.family}_config.json').write_text(json.dumps(dict(family=a.family,lags=lags,config=cfg,seeds=list(range(len(runs))),source_runs=[str(r.relative_to(ROOT)) for r in runs],
 paper_pooled_rmse_2020=[x['pooled_rmse'] for x in res]),indent=1))
# rebuild models and verify against the paper predictions on a subset of test stations
sizes=[len(enc.vocab[j])+1 for j in enc.ci]
rng=np.random.default_rng(0);sub=set(rng.choice([p.stem.split('_')[1] for p in shards],a.check_stations,replace=False))
x0,c0,_,_,_=next(fold.raw_batches('train',8));features=enc.transform(x0,c0).shape[-1]
batches=[b for b in fold.raw_batches('test',4096) if True];X=[];C=[];O=[];S=[]
for x,c,y,o,s in batches:
 k=np.isin(s,list(sub))
 if k.any():X.append(x[k]);C.append(c[k]);O.append(o[k]);S.append(s[k])
X,C,O,S=np.concatenate(X),np.concatenate(C),np.concatenate(O),np.concatenate(S)
Z=torch.from_numpy(enc.transform(X,C))
report=[]
for k,r in enumerate(runs):
 m=pft_fast.LFTModel(enc,sizes,cfg,lags) if a.family in ('lft','lftq') else ForecastModel('gru',enc.state(),features,cfg)
 m.load_state_dict(torch.load(r/'best.pt',map_location='cpu'));m.eval()
 with torch.no_grad():P=np.concatenate([m.predict(Z[i:i+4096]).numpy() for i in range(0,len(Z),4096)])*target['scale']+target['mean']
 ref=np.load(r/'outer_predictions.npz');key={(s_,o_):i for i,(s_,o_) in enumerate(zip(ref['station'].astype(str),ref['origin'].astype(str)))}
 idx=np.array([key[(s_,o_)] for s_,o_ in zip(S.astype(str),O.astype(str))])
 d=float(np.abs(P-(ref['quantiles'] if a.family=='lftq' else ref['prediction'])[idx]).max());report.append(dict(run=r.name,sequences=len(idx),max_abs_diff=d))
 assert d<1e-3,f'{r.name}: export does not reproduce paper predictions ({d})'
 torch.save(m.state_dict(),OUT/f'{a.family}_s{k}.pt')
(OUT/f'{a.family}_export_check.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1))
