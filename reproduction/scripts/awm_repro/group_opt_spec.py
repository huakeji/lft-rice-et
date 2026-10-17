"""Transfer optimisation experiments (outer 2020, lft_c03 light config):
T  fine-tune strategy: init from the leave-zone-out model F_lzo_<z> (never saw zone z), adapt on zone z with
   the last 30/90/365 days using head-only (Bai-style), norm+bias, or per-lead affine recalibration
   (full-network fine-tuning = existing E_ runs).
P  physics covariates (Ra, pressure, Hargreaves ET0; no observations): national and leave-zone-out zero-shot.
U  strict nested check: leave-zone-out zero-shot for R2/R3 with an untuned standard configuration, so the
   held-out zone contributes nothing to model or hyperparameter selection."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];G='reports/awm_repro_20261015/group'
base=json.loads((ROOT/G/'group_s0_jobs.json').read_text())['jobs'];C=next(j for j in base if j['out_dir'].endswith('C_national'))
cfg=C['config'];ft=dict(cfg,lr=cfg['lr']*0.3,warmup_epochs=1,patience=5);J=[];o=lambda n:f'{G}/s0/{n}'
com={k:C[k] for k in ('model','outer_year','seed','lags','order')}
phys=dict(com,shards_dir='reports/awm_repro_20261015/lagged_phys_shards',phys=True)
L=lambda n:f'{G}/lists/{n}'
J.append(dict(phys,max_epochs=C['max_epochs'],config=cfg,train_stations=L('all'),eval_stations=L('all'),out_dir=o('P_national_phys')))
for z in ['R1','R2','R3','R4','R5','R6']:
 J.append(dict(phys,max_epochs=C['max_epochs'],config=cfg,train_stations=L(f'not_{z}'),eval_stations=L(f'zone_{z}'),out_dir=o(f'P_lzo_{z}_phys')))
std=dict(width=128,layers=2,heads=4,dropout=0.1,weight_decay=1e-5,batch_size=1024,lr=1e-3,warmup_epochs=3,patience=15)
for z in ['R2','R3']:J.append(dict(com,shards_dir=C['shards_dir'],max_epochs=30,config=std,train_stations=L(f'not_{z}'),eval_stations=L(f'zone_{z}'),out_dir=o(f'U_lzo_{z}_default')))
for z in ['R1','R2','R3','R4','R5','R6']:
 for n in (30,90,365):
  for m in ('head','norm','affine'):
   J.append(dict(com,shards_dir=C['shards_dir'],max_epochs=10,config=ft if m!='affine' else dict(ft,lr=cfg['lr']),ft_mode=m,train_stations=L(f'zone_{z}'),eval_stations=L(f'zone_{z}'),init_from=o(f'F_lzo_{z}'),train_last_days=n,out_dir=o(f'T_{m}{n}_{z}')))
p=ROOT/G/'group_opt_jobs.json';p.write_text(json.dumps(dict(stage='group-opt',jobs=J),indent=1));print(p,len(J))
