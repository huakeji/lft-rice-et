"""Formal comparison jobs for one frozen configuration per model (frozen by full-station 2019 val in
the confirmation stage). Outer years 2018/2019/2020 x seeds; confirmation runs with the identical
config, seed and outer year are reused by copy (recorded in REUSED_FROM.txt)."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main(model,confirm_name,seeds='0,1,2',years='2020,2019,2018'):
 cj={j['out_dir'].split('/')[-1]:j for f in (ROOT/'reports/awm_repro_20261015/confirm').glob('*_jobs.json') for j in json.loads(f.read_text())['jobs']}
 base=cj[f'{confirm_name}_s0'];jobs=[];out=ROOT/'reports/awm_repro_20261015/formal'
 for y in map(int,years.split(',')):
  for s in map(int,seeds.split(',')):
   j=dict(base);j.update(outer_year=y,seed=s,out_dir=f'reports/awm_repro_20261015/formal/{model}_y{y}_s{s}')
   src=ROOT/f'reports/awm_repro_20261015/confirm/{confirm_name}_s{s}'
   if y==2020 and (src/'result.json').exists() and not (ROOT/j['out_dir']/'result.json').exists():
    (ROOT/j['out_dir']).mkdir(parents=True,exist_ok=True)
    for f in ('result.json','outer_predictions.npz','best.pt','progress.json'):shutil.copy2(src/f,ROOT/j['out_dir']/f)
    (ROOT/j['out_dir']/'REUSED_FROM.txt').write_text(str(src.relative_to(ROOT)))
   jobs.append(j)
 p=out/f'{model}_jobs.json';out.mkdir(exist_ok=True)
 p.write_text(json.dumps(dict(stage='formal',model=model,frozen_from=confirm_name,rule='config with lowest full-station 2019 val in confirmation',jobs=jobs),indent=1));print(p,len(jobs))
if __name__=='__main__':main(*sys.argv[1:])
