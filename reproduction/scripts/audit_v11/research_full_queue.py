"""One supervised full-fold job; reuse the pilot ledger and process lock.

This entry does not schedule unbounded searches. A bounded job request identifies
one frozen candidate/seed/fold. Source and manifest gates run before launch.
"""
import argparse,fcntl,json,os,subprocess,sys,time,uuid
from research_resource_status import worker_outcome
from pathlib import Path
import psutil
from research_queue import R,O,P,remaining,stage_remaining,threads_for_memory,terminate_owned
from research_run_manifest import validate,digest
from validate_data_contract import evaluate
from run_conditional_batch import atomic_json

BUDGET_AMENDMENT = R/'configs/research_route_40/budget_amendment_same_day_20260914.json'

def effective_limits(protocol,m,data_sha):
 limits=json.loads(json.dumps(protocol['resources']));amendment_sha=None
 if m.get('fold_id')!='exploratory_same_day_temporal_2018':return limits,amendment_sha
 if not BUDGET_AMENDMENT.is_file() or not BUDGET_AMENDMENT.with_suffix('.sha256').is_file():raise ValueError('Same-day budget amendment missing')
 amendment_sha=digest(BUDGET_AMENDMENT)
 if amendment_sha!=BUDGET_AMENDMENT.with_suffix('.sha256').read_text().strip():raise ValueError('Same-day budget amendment changed')
 amendment=json.loads(BUDGET_AMENDMENT.read_text())
 if amendment.get('status')!='USER_AUTHORIZED_BUDGET_EXTENSION':raise ValueError('Budget amendment is not authorized')
 if amendment.get('base_protocol_sha256')!=digest(P):raise ValueError('Budget amendment protocol binding changed')
 scope=amendment.get('scope',{})
 if scope.get('fold_id')!=m.get('fold_id') or scope.get('manifest_sha256')!=data_sha:raise ValueError('Budget amendment scope mismatch')
 changes=amendment.get('resource_changes_seconds',{})
 limits['total_active_seconds']+=int(changes.get('total_active_seconds',0))
 limits['stages_seconds']['search']+=int(changes.get('search',0))
 expected=amendment.get('effective_caps_seconds',{})
 checks={'total_active_seconds':limits['total_active_seconds'],'search':limits['stages_seconds']['search'],'pilot':limits['stages_seconds']['pilot'],'transfer':limits['stages_seconds']['transfer'],'validation':limits['stages_seconds']['validation'],'retry':limits['stages_seconds']['retry']}
 if checks!=expected:raise ValueError('Budget amendment effective caps mismatch')
 return limits,amendment_sha

def preflight(model,candidate,seed,manifest):
 protocol=json.loads(P.read_text())
 if digest(P)!=P.with_suffix('.sha256').read_text().strip():raise ValueError('Protocol changed')
 if model not in protocol['models'] or seed not in protocol['seeds'] or not 0<=candidate<len(protocol['search'][model]):raise ValueError('Undeclared trial')
 gate=evaluate(json.loads((R/protocol['formal_contract']).read_text()),'paired_forecast')
 if not gate['ready']:raise ValueError('BLOCKED_SOURCE: '+json.dumps(gate))
 m,shards,data_sha=validate(manifest,R,digest(P),digest(R/protocol['formal_contract']))
 return protocol,m,data_sha

def job_identity(model,candidate,seed,fold,data_sha):
 return f'full_{fold}_{data_sha[:12]}_{model}_c{candidate}_s{seed}'

def admission(ledger,task,stage,limits,seconds):
 if stage not in ('search','transfer','validation'):raise ValueError('Unknown formal budget stage')
 if seconds<=0:raise ValueError('Positive bounded duration required')
 previous=[a for a in ledger['attempts'] if a['task']==task]
 if any(a['status']=='COMPLETE' for a in previous):raise ValueError('Already complete; no duplicate fit')
 if previous and (previous[-1]['status']!='RESOURCE_STOP' or len(previous)>limits['resource_retry_count']):raise ValueError('Not eligible for automatic retry')
 budget_stage='retry' if previous else stage
 allowance=min(seconds,remaining(ledger,limits['total_active_seconds']),stage_remaining(ledger,budget_stage,limits['stages_seconds'][budget_stage]))
 if allowance<=0:raise ValueError('BUDGET_EXHAUSTED')
 return previous,budget_stage,allowance

def run(args):
 protocol,m,data_sha=preflight(args.model,args.candidate,args.seed,Path(args.manifest).resolve())
 return supervise(args,protocol,m,data_sha)

def supervise(args,protocol,m,data_sha,worker='research_trial.py'):
 # Internal adapter: caller and worker must each enforce their provenance gate.
 # Only owned worker entry points are accepted; this is not an arbitrary command runner.
 completion_paths={'research_exploratory_worker.py':'selection_record.json','research_spatial_worker.py':'selection_record.json','research_distillation_worker.py':'selection_record.json','research_trial.py':'selection_record.json','research_teacher_worker.py':'oof_part/manifest.json','research_oof_assembly_worker.py':'soft_targets/manifest.json'}
 if worker not in completion_paths:raise ValueError('Unapproved training worker')
 worker_path=R/'scripts/audit_v11'/worker
 if not worker_path.is_file():raise ValueError('Training worker not yet implemented')
 limits,budget_amendment_sha256=effective_limits(protocol,m,data_sha);lp=O/'queue_ledger.json'
 with (O/'queue.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise ValueError('Queue or inherited worker still active')
  ledger=json.loads(lp.read_text())
  # An unexplained stopped run needs explicit recovery, not an automatic refit.
  if any(a['status']=='RUNNING' for a in ledger['attempts']):raise ValueError('Stale RUNNING record requires recovery accounting')
  task=job_identity(args.model,args.candidate,args.seed,m['fold_id'],data_sha)
  previous,stage,allowance=admission(ledger,task,args.stage,limits,args.seconds)
  if psutil.disk_usage(R).free<limits['min_disk_bytes']:raise ValueError('WAITING_DISK')
  # Prove process-tree monitoring permissions before starting a training process.
  psutil.Process().children(recursive=True)
  dest=O/('exploratory' if worker=='research_exploratory_worker.py' else 'formal')/task/f'attempt_{len(previous)}';dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():raise ValueError('Existing output requires recovery; never overwrite')
  token=uuid.uuid4().hex;env=os.environ.copy();env['ET0_QUEUE_TOKEN']=token
  threads=2 if previous else threads_for_memory(psutil.virtual_memory().available)
  for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[k]=str(threads)
  command=[sys.executable,str(worker_path),'--model',args.model,'--candidate',str(args.candidate),'--seed',str(args.seed),'--data',args.manifest,'--manifest',args.manifest,'--out',str(dest),'--seconds',str(int(allowance))]
  if previous:command.extend(['--resume-from',str(R/previous[-1]['out'])])
  row={'task':task,'model':args.model,'candidate':args.candidate,'seed':args.seed,'fold_id':m['fold_id'],'stage':stage,'attempt':len(previous),'status':'RUNNING','active_seconds':0.,'started_at':time.time(),'out':str(dest.relative_to(R)),'command':command,'supervisor_pid':os.getpid(),'queue_token':token,'allowed_seconds':allowance,'worker_sha256':digest(worker_path),'worker_entry':worker,'data_sha256':data_sha}
  if budget_amendment_sha256:row['budget_amendment_sha256']=budget_amendment_sha256
  ledger['attempts'].append(row);atomic_json(lp,ledger);reason=None;child=None;last=time.monotonic()
  try:
   with (dest.parent/f'attempt_{len(previous)}.log').open('x') as log:
    child=subprocess.Popen(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),),start_new_session=True)
    row['pid']=child.pid;atomic_json(lp,ledger)
    while child.poll() is None:
     time.sleep(1);now=time.monotonic();row['active_seconds']+=now-last;last=now
     try:
      proc=psutil.Process(child.pid);rss=0
      for p in [proc]+proc.children(recursive=True):
       try:rss+=p.memory_info().rss
       except psutil.NoSuchProcess:pass
     except psutil.NoSuchProcess:rss=0
     row['peak_rss_bytes']=max(row.get('peak_rss_bytes',0),rss)
     if child.poll() is None:
      if rss>limits['rss_hard_bytes']:reason='process-tree memory limit'
      elif row['active_seconds']>=allowance:reason='bounded time or budget limit'
      elif psutil.disk_usage(R).free<limits['min_disk_bytes']:reason='disk headroom'
     atomic_json(lp,ledger);atomic_json(O/'queue_status.json',{'status':'RUNNING','task':task,'rss_bytes':rss,'active_seconds_used':limits['total_active_seconds']-remaining(ledger,limits['total_active_seconds'])})
     if reason:terminate_owned(child);break
   result=json.loads((dest/'status.json').read_text()) if (dest/'status.json').exists() else {}
   row['status']=worker_outcome(child.returncode,result,reason)
   row['reason']=reason or result.get('reason');row['exit_code']=child.returncode
   if row['status']=='COMPLETE':
    record=dest/completion_paths[worker]
    if record.exists():
     row['completion_record_sha256']=digest(record);row['completion_record_relative_path']=completion_paths[worker]
     if worker=='research_trial.py':row['selection_record_sha256']=row['completion_record_sha256']
    else:row['status']='FAILED';row['reason']='Completed worker missing required artifact record'
  except BaseException as exc:
   if child:terminate_owned(child)
   row['status']='INTERRUPTED';row['reason']=repr(exc);raise
  finally:
   row['active_seconds']+=max(0,time.monotonic()-last);row['ended_at']=time.time();atomic_json(lp,ledger)
   atomic_json(O/'queue_status.json',{'status':row['status'],'task':task,'active_seconds_used':limits['total_active_seconds']-remaining(ledger,limits['total_active_seconds'])})
  return 0 if row['status']=='COMPLETE' else 1

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--candidate',type=int,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--stage',choices=['search','transfer','validation'],required=True);ap.add_argument('--seconds',type=int,required=True)
 sys.exit(run(ap.parse_args()))
