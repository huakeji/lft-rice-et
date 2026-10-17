"""Forecast-only neural job, called only after source and manifest checks."""
import hashlib,json,os,shutil,time
from research_resource_status import ResourceStop,SupervisionLost,exception_status
from pathlib import Path
import numpy as np
import psutil,torch
from research_fold_data import FoldData,DevelopmentData
from research_models import ForecastModel
from research_neural_training import train
from run_conditional_batch import atomic_json
from research_run_manifest import digest
R=Path(__file__).resolve().parents[2]
O=R/'reports/v11_audit_20260909/research_route_40'

def supervised_row(out):
 token=os.environ.get('ET0_QUEUE_TOKEN')
 if not token:raise ValueError('Formal worker requires active queue supervision')
 for _ in range(20):
  ledger=json.loads((O/'queue_ledger.json').read_text())
  matches=[r for r in ledger['attempts'] if r.get('queue_token')==token and r['status']=='RUNNING']
  if len(matches)==1 and matches[0].get('pid')==os.getpid():
   row=matches[0]
   if (R/row['out']).resolve()!=out.resolve() or row['supervisor_pid']!=os.getppid():raise ValueError('Queue identity mismatch')
   return row
  time.sleep(.1)
 raise ValueError('No active queue record for worker')

def run(args,out,protocol,cfg,manifest,shards,data_sha,distillation=None,spatial=None):
 if spatial is not None and distillation is not None:raise ValueError('Spatial distillation not integrated yet')
 started=time.monotonic();row=supervised_row(out);limits=protocol['resources']
 if row['data_sha256']!=(distillation['spec_sha256'] if distillation else data_sha):raise ValueError('Queue/data identity mismatch')
 ceiling=int(os.environ.get('OMP_NUM_THREADS','8'))
 def guard():
  if os.getppid()!=row['supervisor_pid']:raise SupervisionLost('Supervisor lost; stop worker')
  mem=psutil.virtual_memory();threads=min(ceiling,2 if mem.available<2*1024**3 else 4 if mem.available<4*1024**3 else 8)
  torch.set_num_threads(threads)
  if psutil.Process().memory_info().rss>limits['rss_hard_bytes'] or psutil.disk_usage(R).free<limits['min_disk_bytes'] or time.monotonic()-started>row['allowed_seconds']:raise ResourceStop('Configured memory/disk/time limit')
 guard()
 if spatial is not None:
  from research_spatial_data import SpatialFoldData
  data=DevelopmentData(SpatialFoldData(shards,manifest['outer_year'],spatial['membership']))
 else:data=DevelopmentData(FoldData(shards,manifest['outer_year'],manifest['excluded_stations'],manifest.get('target_offset',0)))
 if distillation is not None:
  from research_distillation_data import PairedStudentData
  data=PairedStudentData(data,distillation['soft_targets'])
 encoder,target=data.fit_encoder();guard()
 if manifest.get('tree_encoding')=='binary_vocabulary_id':
  if args.model not in ('hgb','catboost'):raise ValueError('Compact encoding only declared for trees')
  from research_compact_tree_encoding import CompactTreeEncoder
  encoder=CompactTreeEncoder(encoder)
 state=encoder.state()
 example=next(data.encoded_batches(encoder,'train',1))[0]
 np.random.seed(args.seed);torch.manual_seed(args.seed)
 model=None if args.model in ('hgb','catboost') else ForecastModel(args.model,state,example.shape[-1],cfg)
 binding={'model':args.model,'seed':args.seed,'fold_id':manifest['fold_id'],'data_sha256':data_sha,'protocol_sha256':digest(R/'configs/research_route_40/protocol.json'),'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()}
 binding['implementation_sha256']=hashlib.sha256(''.join(digest(Path(__file__).with_name(name)) for name in ('research_full_worker.py','research_neural_training.py','research_models.py','research_fold_data.py','research_tree_training.py','research_tree_cache.py')).encode()).hexdigest()
 if distillation is not None:
  binding.update(alpha_protocol_sha256=distillation['alpha_protocol_sha256'],study_arm='paired_oof_distillation',base_candidate=distillation['student_selection']['selected_candidate'],distillation_alpha=distillation['alpha'],oof_content_sha256=distillation['soft_targets'].content_sha256,distillation_spec_sha256=distillation['spec_sha256'],soft_manifest_sha256=distillation['soft_manifest_sha256'])
 if spatial is not None:
  binding.update(spatial_implementation_sha256=hashlib.sha256(''.join(digest(Path(__file__).with_name(n)) for n in ('research_spatial_worker.py','research_spatial_data.py','research_spatial_spec.py','research_spatial_membership.py')).encode()).hexdigest(),spatial_spec_sha256=spatial['spec_sha256'],membership_sha256=spatial['geometry']['membership_sha256'],study_arm='spatial_public')
 if manifest.get('tree_encoding')=='binary_vocabulary_id':binding['tree_encoding_sha256']=digest(Path(__file__).with_name('research_compact_tree_encoding.py'))
 atomic_json(out/'config.json',{'binding':binding,'candidate':args.candidate,'configuration':cfg,'preprocessing':state,'target':target,'scientific_status':manifest.get('scientific_status','RETROSPECTIVE_DEVELOPMENT_VALIDATION'),'initialization_seed':args.seed})
 if args.model in ('hgb','catboost'):
  from research_tree_training import fit
  result=fit(args.model,data,encoder,target,cfg,args.seed,out/'trees',binding,guard,limits['min_disk_bytes'],cache_pool=O/'shared_tree_cache')
  record={**binding,'candidate':args.candidate,'metric_role':'internal_validation','status':'COMPLETE','validation_rmse':result['validation_rmse'],'parameters':None}
  atomic_json(out/'selection_record.json',record)
  atomic_json(out/'status.json',{'status':'COMPLETE','scientific_status':manifest.get('scientific_status','RETROSPECTIVE_DEVELOPMENT_VALIDATION'),'seconds':time.monotonic()-started,'validation_rmse':result['validation_rmse'],'outer_evaluation_performed':False,'resource_retry_mode':'same configuration and seed restarted; no tree checkpoint continuation'})
  return 0
 kernel=out/'training';resume=False
 if getattr(args,'resume_from',None):
  source=Path(args.resume_from).resolve()
  ledger=json.loads((O/'queue_ledger.json').read_text())
  previous=[r for r in ledger['attempts'] if r['task']==row['task'] and r['attempt']==row['attempt']-1 and r['status']=='RESOURCE_STOP']
  if len(previous)!=1 or (R/previous[0]['out']).resolve()!=source:raise ValueError('Resume path is not preceding resource-stopped attempt')
  if (source/'training/last.pt').exists():
   kernel.mkdir()
   for name in ('last.pt','best.pt'):shutil.copy2(source/'training'/name,kernel/name)
   resume=True
 result=train(model,data,encoder,target,cfg,args.seed,kernel,binding,guard=guard,resume=resume,teacher_targets=distillation['soft_targets'] if distillation else None,alpha=distillation['alpha'] if distillation else 0.)
 best=torch.load(kernel/'best.pt',map_location='cpu',weights_only=False)
 if best['binding']!=binding:raise ValueError('Best checkpoint identity mismatch')
 model.load_state_dict(best['model']);model.eval();pred_dir=out/'validation_predictions';pred_dir.mkdir();parts=[];count=0;sse=0.
 with torch.no_grad():
  for index,(x,y,origin,station) in enumerate(data.encoded_batches(encoder,'val',cfg['batch_size'])):
   guard();prediction=model.predict(torch.as_tensor(x,dtype=torch.float32)).numpy()*target['scale']+target['mean']
   if not np.isfinite(prediction).all():raise ValueError('Invalid validation prediction')
   part=pred_dir/f'part_{index:05d}.npz';np.savez_compressed(part,prediction=prediction,target=y,origin=origin,station=station)
   parts.append({'path':str(part.relative_to(out)),'sha256':digest(part),'sequences':len(y)});count+=y.size;sse+=float(np.square(prediction.astype(float)-y).sum())
 if not count:raise ValueError('Empty validation')
 replay_rmse=(sse/count)**.5
 if not np.isclose(replay_rmse,result['validation_rmse'],rtol=1e-6,atol=1e-7):raise ValueError('Best checkpoint replay mismatch')
 atomic_json(out/'validation_predictions.json',{'parts':parts,'target_count':count,'pooled_rmse':replay_rmse})
 record={**binding,'candidate':args.candidate,'metric_role':'internal_validation','status':'COMPLETE','validation_rmse':replay_rmse,'parameters':sum(p.numel() for p in model.parameters())}
 atomic_json(out/'selection_record.json',record)
 atomic_json(out/'status.json',{'status':'COMPLETE','scientific_status':manifest.get('scientific_status','RETROSPECTIVE_DEVELOPMENT_VALIDATION'),'seconds':time.monotonic()-started,'best_epoch':result['best_epoch'],'validation_rmse':replay_rmse,'outer_evaluation_performed':False})
 return 0
