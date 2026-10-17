"""Serial, bounded, restartable local queue. No shell commands or implicit full runs."""
from research_resource_status import worker_outcome
from pathlib import Path
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
import psutil
from run_conditional_batch import atomic_json
from validate_data_contract import evaluate
R=Path(__file__).resolve().parents[2];O=R/'reports/v11_audit_20260909/research_route_40';P=R/'configs/research_route_40/protocol.json'

def remaining(ledger,budget):return max(0,budget-sum(t.get('active_seconds',0) for t in ledger['attempts']))
def stage_remaining(ledger,stage,cap):return max(0,cap-sum(t.get('active_seconds',0) for t in ledger['attempts'] if t.get('stage','pilot')==stage))
def threads_for_memory(available):return 2 if available<2*1024**3 else 4 if available<4*1024**3 else 8

def terminate_owned(child):
 # Worker starts its own session: kill only that owned process group, even if
 # sandbox permissions prevent enumerating system processes.
 if child.poll() is not None:return
 try:os.killpg(child.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:child.wait(timeout=10)
 except subprocess.TimeoutExpired:
  try:os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  child.wait()

def run():
 O.mkdir(exist_ok=True,parents=True);lp=O/'queue_ledger.json';protocol=json.loads(P.read_text());limits=protocol['resources']
 ledger=json.loads(lp.read_text()) if lp.exists() else {'version':1,'attempts':[]}
 lock=(O/'queue.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:print('Queue already active');return 0
 # The lock is inherited by each worker, protecting an orphan from duplicate launches.
 for row in ledger['attempts']:
  if row['status']=='RUNNING':
   status_path=R/row['out']/'status.json'
   result=json.loads(status_path.read_text()) if status_path.exists() else {}
   if result.get('status')=='COMPLETE':
    row['status']='COMPLETE';row['recovered_after_supervisor_failure']=True
    row['active_seconds']=max(row['active_seconds'],status_path.stat().st_mtime-row['started_at'])
   else:row['status']='INTERRUPTED';row['reason']='Previous supervisor stopped; incomplete trial retained'
 atomic_json(lp,ledger)
 for model in protocol['models']:
  task='pilot_'+model
  attempts=[x for x in ledger['attempts'] if x['task']==task]
  if any(x['status']=='COMPLETE' for x in attempts):continue
  if attempts and (attempts[-1]['status']!='RESOURCE_STOP' or len(attempts)>=2):continue
  budget_left=min(remaining(ledger,limits['total_active_seconds']),stage_remaining(ledger,'pilot',limits['stages_seconds']['pilot']))
  if budget_left<=0:atomic_json(O/'queue_status.json',{'status':'BUDGET_EXHAUSTED'});return 2
  if psutil.disk_usage(R).free<limits['min_disk_bytes']:atomic_json(O/'queue_status.json',{'status':'WAITING_DISK','minimum_bytes':limits['min_disk_bytes']});return 2
  dest=O/'pilot'/f'{model}_attempt_{len(attempts)}';dest.parent.mkdir(parents=True,exist_ok=True)
  command=[sys.executable,str(R/'scripts/audit_v11/research_trial.py'),'--model',model,'--data',str(R/'reports/v11_audit_20260909/conditional_run_31/data/adaptive_samples.npz'),'--out',str(dest),'--smoke','--seconds','600']
  env=os.environ.copy();threads=2 if attempts else threads_for_memory(psutil.virtual_memory().available)
  for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[key]=str(threads)
  row={'task':task,'stage':'pilot','attempt':len(attempts),'status':'RUNNING','active_seconds':0.,'started_at':time.time(),'out':str(dest.relative_to(R)),'command':command,'worker_sha256':hashlib.sha256((R/'scripts/audit_v11/research_trial.py').read_bytes()).hexdigest()}
  ledger['attempts'].append(row);atomic_json(lp,ledger)
  with (dest.parent/f'{model}_attempt_{len(attempts)}.log').open('w') as log:
   child=subprocess.Popen(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),),start_new_session=True);row['pid']=child.pid;atomic_json(lp,ledger);last=time.monotonic();reason=None;low=0
   try:
    while child.poll() is None:
     time.sleep(1);now=time.monotonic();row['active_seconds']+=min(now-last,5.);last=now
     try:
      proc=psutil.Process(child.pid);tree=[proc]+proc.children(recursive=True);rss=sum(p.memory_info().rss for p in tree if p.is_running())
     except psutil.NoSuchProcess:rss=0
     except (psutil.AccessDenied,PermissionError):
      reason='process-tree monitor permission unavailable';terminate_owned(child);break
     mem=psutil.virtual_memory();low=low+1 if mem.available<.75*1024**3 else 0
     if rss>limits['rss_hard_bytes'] or low>=3:reason='process-tree memory pressure'
     elif remaining(ledger,limits['total_active_seconds'])<=0 or stage_remaining(ledger,'pilot',limits['stages_seconds']['pilot'])<=0 or row['active_seconds']>=600:reason='time limit'
     elif psutil.disk_usage(R).free<limits['min_disk_bytes']:reason='disk headroom'
     row['peak_rss_bytes']=max(row.get('peak_rss_bytes',0),rss);atomic_json(lp,ledger)
     atomic_json(O/'queue_status.json',{'status':'RUNNING','task':task,'active_seconds_used':limits['total_active_seconds']-remaining(ledger,limits['total_active_seconds']),'rss_bytes':rss})
     if reason:terminate_owned(child);break
   except BaseException:
    terminate_owned(child);row['status']='INTERRUPTED';atomic_json(lp,ledger);raise
  status_path=dest/'status.json';result=json.loads(status_path.read_text()) if status_path.exists() else {}
  row['status']=worker_outcome(child.returncode,result,reason)
  row['reason']=reason or result.get('reason');row['exit_code']=child.returncode;row['ended_at']=time.time();atomic_json(lp,ledger)
 gate=evaluate(json.loads((R/protocol['formal_contract']).read_text()),'paired_forecast');atomic_json(O/'source_gate.json',gate)
 complete=len({x['task'] for x in ledger['attempts'] if x['status']=='COMPLETE' and x['task'].startswith('pilot_')})
 atomic_json(O/'queue_status.json',{'status':'PILOT_COMPLETE_FORMAL_BLOCKED' if complete==7 else 'PILOT_INCOMPLETE','completed_pilot_models':complete,'formal_ready':gate['ready'],'next':'source audit and full-data/fold-loader implementation; no full training launched','active_seconds_used':limits['total_active_seconds']-remaining(ledger,limits['total_active_seconds'])})
 return 0 if complete==7 else 1
if __name__=='__main__':sys.exit(run())
