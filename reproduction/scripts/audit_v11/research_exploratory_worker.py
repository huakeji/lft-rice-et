"""Explicit user-authorized exploratory route; never changes formal source gate."""
import argparse,json
from pathlib import Path
from research_full_queue import supervise,R,P
from research_run_manifest import checked_file,digest
from research_full_worker import run as train_public
from research_resource_status import exception_status
from run_conditional_batch import atomic_json

def validate(path):
 path=Path(path).resolve()
 if path not in [R/'configs/research_route_40'/n for n in ('exploratory_resource_v1.json','exploratory_comparison_v1.json','exploratory_compact_tree_v1.json','exploratory_same_day_v1.json','rolling_same_day_2019_v1.json','rolling_same_day_2020_v1.json')]:raise ValueError('Unapproved exploratory specification')
 m=json.loads(path.read_text());protocol=json.loads(P.read_text())
 if m['status']!='EXPLORATORY_USER_AUTHORIZED_UNRESOLVED_TIME_SEMANTICS' or m['protocol_sha256']!=digest(P):raise ValueError('Exploratory identity mismatch')
 if len(m['shards'])!=583:raise ValueError('Exploratory cohort changed')
 if path.name=='exploratory_resource_v1.json':
  if m['configuration']['max_epochs']!=2 or m['models']!=['mlp']:raise ValueError('Resource pilot scope changed')
 elif path.name in ('rolling_same_day_2019_v1.json','rolling_same_day_2020_v1.json'):
  expected_year=int(path.stem.split('_')[3])
  if digest(path)!=path.with_suffix('.sha256').read_text().strip() or m['models']!=['pft'] or m['configurations']!={'pft':protocol['search']['pft'][0]} or m['outer_year']!=expected_year or m.get('target_offset')!=-1:raise ValueError('Frozen rolling comparison changed')
 elif path.name=='exploratory_same_day_v1.json':
  if digest(path)!=path.with_suffix('.sha256').read_text().strip() or m['models']!=protocol['models'] or m['configurations']!={model:protocol['search'][model][0] for model in m['models']}:raise ValueError('Frozen same-day comparison changed')
 else:
  if digest(path)!=path.with_suffix('.sha256').read_text().strip() or m['models']!=(['hgb','catboost'] if path.name=='exploratory_compact_tree_v1.json' else protocol['models']) or m['configurations']!={model:protocol['search'][model][0] for model in m['models']}:raise ValueError('Frozen comparison changed')
 shards=[checked_file(R,r) for r in m['shards']]
 return m,protocol,shards,digest(path)

def search_extension(path, model, candidate, seed):
 spec=(R/'configs/research_route_40/exploratory_same_day_search_v1.json') if Path(path).name=='exploratory_same_day_v1.json' else (R/'configs/research_route_40/exploratory_search_extension_v1.json')
 s=json.loads(spec.read_text())
 if digest(spec)!=spec.with_suffix('.sha256').read_text().strip() or s['protocol_sha256']!=digest(P):raise ValueError('Search extension changed')
 if candidate not in s['candidates'] or seed!=s['seed'] or model not in s['run_order']:raise ValueError('Undeclared exploratory search trial')
 if Path(path).name=='exploratory_same_day_v1.json':
  if s['manifests'].get('exploratory_same_day_v1.json')!=digest(Path(path)):raise ValueError('Same-day manifest identity mismatch')
  return s
 expected='exploratory_compact_tree_v1.json' if model in ('hgb','catboost') else 'exploratory_comparison_v1.json'
 if Path(path).name!=expected or digest(Path(path))!=s['manifests'][expected]:raise ValueError('Search representation mismatch')
 return s

def confirmation_extension(path, model, candidate, seed):
 spec=R/'configs/research_route_40/exploratory_confirmation_v1.json'
 if not spec.is_file() or digest(spec)!=spec.with_suffix('.sha256').read_text().strip():raise ValueError('Exploratory confirmation plan missing or changed')
 s=json.loads(spec.read_text())
 if s.get('status')!='EXPLORATORY_UNVERIFIED_TIME_SEMANTICS' or s.get('protocol_sha256')!=digest(P):raise ValueError('Exploratory confirmation identity mismatch')
 expected='exploratory_compact_tree_v1.json' if model in ('hgb','catboost') else 'exploratory_comparison_v1.json'
 if Path(path).name!=expected or digest(Path(path))!=s['manifests'][expected]:raise ValueError('Confirmation representation mismatch')
 if seed not in s['confirmation_seeds'] or candidate not in s['shortlists'][model]['candidates']:raise ValueError('Undeclared exploratory confirmation trial')
 return s

def run(a):
 m,p,shards,sha=validate(a.manifest)
 if a.model not in m['models']:raise ValueError('Undeclared model')
 if a.seed==0:
  if a.candidate!=0:search_extension(a.manifest,a.model,a.candidate,a.seed)
 elif a.seed in (1,2):confirmation_extension(a.manifest,a.model,a.candidate,a.seed)
 else:raise ValueError('Only search seed or frozen confirmation seeds allowed')
 if a.launch:
  a.stage='search';return supervise(a,p,m,sha,worker='research_exploratory_worker.py')
 out=Path(a.out).resolve()
 if not out.is_relative_to(R):raise ValueError('Project-local output required')
 out.mkdir(parents=True,exist_ok=False)
 cfg=p['search'][a.model][a.candidate] if a.candidate!=0 else (m['configurations'][a.model] if 'configurations' in m else m['configuration'])
 return train_public(a,out,p,cfg,m,shards,sha)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--launch',action='store_true');ap.add_argument('--model',default='mlp');ap.add_argument('--candidate',type=int,default=0);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--manifest',required=True);ap.add_argument('--data');ap.add_argument('--out');ap.add_argument('--seconds',type=int,required=True);ap.add_argument('--resume-from');a=ap.parse_args()
 try:raise SystemExit(run(a))
 except Exception as exc:
  if a.out:
   out=Path(a.out).resolve()
   if out.is_relative_to(R) and out.exists() and not isinstance(exc,FileExistsError):atomic_json(out/'status.json',{'status':exception_status(exc),'reason':repr(exc),'scientific_status':'EXPLORATORY_UNVERIFIED_TIME_SEMANTICS'})
  raise
