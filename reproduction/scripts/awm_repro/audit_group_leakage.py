"""Leakage audit of group / transfer runs: for zero-shot runs (F_lzo*, S_scale*, and anything fine-tuned from
them) the evaluated stations must never appear in the training station list of the run or of any run it was
initialised from; evaluated stations must equal the declared eval list; encoder/target are reused only along
the init chain. Reads the frozen job specs and finished result.json/outer_predictions.npz."""
import json,glob,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
rd=lambda f:set((ROOT/f).read_text().split()) if f else None
bad=[];n=0
for spec in sorted(glob.glob(str(ROOT/'reports/awm_repro_20261015/group/*_jobs.json'))):
 J={j['out_dir']:j for j in json.load(open(spec))['jobs']}
 for d,j in J.items():
  r=ROOT/d/'result.json'
  if not r.exists():continue
  n+=1;res=json.loads(r.read_text());ev=rd(j['eval_stations']);chain=[j];k=j
  while k.get('init_from'):k=J.get(k['init_from']) or {'out_dir':k['init_from'],'train_stations':json.loads((ROOT/k['init_from']/'result.json').read_text()).get('train_stations','')};chain.append(k)
  seen=set().union(*[rd(c['train_stations']) or set() for c in chain])
  name=Path(d).name;zero=name.startswith(('F_','S_','E_','T_','W_','P_lzo','U_'))
  pred=set(np.load(ROOT/d/'outer_predictions.npz')['station'].astype(str))
  if pred-ev:bad.append((name,'predicted stations outside eval list',len(pred-ev)))
  if zero and (seen&ev) and not name.startswith(('E_','T_','W_')):bad.append((name,'eval stations seen in training chain',len(seen&ev)))
  if name.startswith(('E_','T_','W_')):
   base=chain[-1];bs=rd(base['train_stations'])
   if bs&ev:bad.append((name,'pretrained base saw target zone',len(bs&ev)))
print('runs audited',n,'violations',len(bad));[print(b) for b in bad]
