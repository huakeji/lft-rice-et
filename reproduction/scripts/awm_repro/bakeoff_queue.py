"""Run a frozen job list of pft_fast.py trainings with bounded parallelism; skip finished jobs.
Selection between candidates must use val_rmse only (outer year is recorded, never used to choose)."""
import json,subprocess,sys,time,os,psutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main(spec,parallel,threads):
 jobs=json.loads(Path(spec).read_text())['jobs'];run=[];busy=lambda j:(ROOT/j['out_dir']/'result.json').exists() or any(j['out_dir'] in ' '.join(q.info['cmdline'] or []) for q in psutil.process_iter(['cmdline']) if 'pft_fast.py' in ' '.join(q.info['cmdline'] or []))
 pending=[j for j in jobs if not busy(j)]   # skip finished jobs and jobs another queue is running
 while pending or run:
  run=[(j,p) for j,p in run if p.poll() is None]
  free=psutil.virtual_memory().available/1024**3;disk=psutil.disk_usage(str(ROOT)).free/1024**3
  if pending and len(run)<parallel and free>2.5 and disk>25:
   ready=[j for j in pending if not j.get('init_from') or (ROOT/j['init_from']/'result.json').exists()]   # dependencies first
   if not ready:time.sleep(30);continue
   j=ready[0];pending.remove(j)
   if busy(j):continue
   (ROOT/j['out_dir']).mkdir(parents=True,exist_ok=True)
   cmd=[sys.executable,str(ROOT/'scripts/awm_repro/pft_fast.py'),'--model',j['model'],'--out-dir',j['out_dir'],'--outer-year',str(j['outer_year']),
    '--seed',str(j['seed']),'--threads',str(threads),'--max-epochs',str(j['max_epochs']),'--config',json.dumps(j['config'])]
   for k in ('shards_dir','lags','order','station_subset','train_stations','eval_stations','init_from','train_last_days','train_window','ft_mode'):
    if k in j:cmd+=['--'+k.replace('_','-'),str(j[k])]
   if j.get('phys'):cmd+=['--phys']
   env=dict(os.environ,OMP_NUM_THREADS=str(threads))
   p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=open(ROOT/j['out_dir']/'stdout.log','w'),stderr=subprocess.STDOUT);run.append((j,p))
   print(time.strftime('%H:%M'),'start',j['out_dir'],flush=True);continue
  time.sleep(30)
 print('queue done',flush=True)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
