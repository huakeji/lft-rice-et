"""Full-station confirmation of the top-k search configurations per model (selection by 2019 val only)."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main(groups,k,seeds,tag):
 jobs=[];sel={}
 for g in groups.split(','):
  spec=json.loads((ROOT/f'reports/awm_repro_20261015/hpsearch/{g}_jobs.json').read_text())['jobs'];by={}
  for j in spec:
   r=ROOT/j['out_dir']/'result.json'
   if r.exists():by.setdefault(j['model'],[]).append((json.loads(r.read_text())['val_rmse'],j))
  for m,v in by.items():
   v.sort(key=lambda t:t[0]);sel[m]=[dict(val_subset=round(a,4),search_dir=j['out_dir']) for a,j in v[:k]]
   for a,j in v[:k]:
    for s in seeds:
     n=dict(j);n.pop('station_subset',None);n['seed']=s;n['out_dir']=f"reports/awm_repro_20261015/confirm/{Path(j['out_dir']).name}_s{s}";jobs.append(n)
 p=ROOT/f'reports/awm_repro_20261015/confirm/{tag}_jobs.json';p.parent.mkdir(exist_ok=True)
 p.write_text(json.dumps(dict(stage=f'confirm-{tag}',selection='top-k by 2019 val on the 195-station search subset; confirmed on all stations, outer 2020 recorded only',selected=sel,jobs=jobs),indent=1));print(p,len(jobs),json.dumps(sel,indent=0))
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]),[int(s) for s in sys.argv[3].split(',')],sys.argv[4])
