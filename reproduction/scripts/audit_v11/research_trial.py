"""One immutable trial. Smoke is exploratory; formal requires provenance readiness."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'8')
import argparse,hashlib,json,math,sys,time
from research_resource_status import ResourceStop,SupervisionLost,exception_status
from pathlib import Path
import numpy as np
import torch
import psutil
from threadpoolctl import threadpool_limits
import joblib
from matched_baselines import Encoder
from research_models import ForecastModel,distillation_loss
from validate_data_contract import evaluate as evaluate_contract
from run_conditional_batch import atomic_json
R=Path(__file__).resolve().parents[2]

def local(p):
 p=Path(p).resolve()
 if not p.is_relative_to(R):raise ValueError('Project-local path required')
 return p

def metrics(y,p,station):
 error=p.astype(float)-y.astype(float)
 return {'pooled_rmse':float(np.sqrt(np.mean(error**2))),
  'station_mean_rmse':float(np.mean([np.sqrt(np.mean(error[station==s]**2)) for s in np.unique(station)])),
  'mae':float(np.mean(abs(error))),'by_lead_rmse':np.sqrt(np.mean(error**2,axis=0)).tolist(),
  'cumulative_rmse':{str(h):float(np.sqrt(np.mean(error[:,:h].sum(1)**2))) for h in [3,7]}}

def train(args):
 out=local(args.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 protocol_path=R/'configs/research_route_40/protocol.json';protocol=json.loads(protocol_path.read_text())
 if hashlib.sha256(protocol_path.read_bytes()).hexdigest()!=(protocol_path.with_suffix('.sha256')).read_text().strip():raise ValueError('Frozen protocol hash mismatch')
 cfg=protocol['search'][args.model][args.candidate].copy();np.random.seed(args.seed);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
 if not args.smoke:
  gate=evaluate_contract(json.loads((R/protocol['formal_contract']).read_text()),'paired_forecast')
  if not gate['ready']:atomic_json(out/'status.json',{'status':'BLOCKED_SOURCE','gate':gate});return 3
  # Never silently substitute retrospective smoke arrays for audited as-of folds.
  if not args.manifest:
   atomic_json(out/'status.json',{'status':'BLOCKED_FULL_DATA_MANIFEST','reason':'audited as-of fold manifest required'});return 3
  from research_run_manifest import validate,digest
  manifest,shards,data_hash=validate(local(args.manifest),R,digest(protocol_path),digest(R/protocol['formal_contract']))
  atomic_json(out/'manifest_validation.json',{'status':'PASS','data_sha256':data_hash,'fold_id':manifest['fold_id'],'shards':len(shards)})
  from research_full_worker import run as full_run
  return full_run(args,out,protocol,cfg,manifest,shards,data_hash)
 source=local(args.data);a=np.load(source,allow_pickle=False)
 if set(a['split'])!={'train','val'}:raise ValueError('Pilot source must contain train/val only; test inaccessible')
 tr0=np.where(a['split']=='train')[0];va0=np.where(a['split']=='val')[0]
 # Shared deterministic spread: identical keys for every model, independent of seed and errors.
 select=lambda ids,n:ids[np.linspace(0,len(ids)-1,min(n,len(ids)),dtype=int)]
 tr=select(tr0,512);va=select(va0,128);ids=np.r_[tr,va]
 raw=a['x'][ids];cat=a['c'][ids];y=a['y'][ids];station=a['station'][ids];origin=a['origin'][ids]
 if not np.isfinite(raw).all() or not np.isfinite(y).all():raise ValueError('Nonfinite input or target')
 if (raw[:,:,0]<raw[:,:,1]).any():raise ValueError('Inverted temperature units/order')
 roles=np.array(['train']*len(tr)+['val']*len(va));tr=np.arange(len(tr));va=np.arange(len(tr),len(ids))
 # Same station-target-day must never straddle temporal roles.
 key=lambda ii:{(str(station[k]),str(np.datetime64(origin[k],'D')+h)) for k in ii for h in range(1,8)}
 if key(tr)&key(va):raise ValueError('Training and validation target overlap')
 enc=Encoder().fit(raw[tr],cat[tr],'full');state=enc.state();x=enc.transform(raw,cat);del a,raw,cat
 ym=float(y[tr].mean());ys=max(float(y[tr].std()),1e-8);target=((y-ym)/ys).astype('float32')
 cfg['smoke_overrides']={'max_epochs':2,'tree_max_iterations':20,'train_cap':512,'validation_cap':128}
 atomic_json(out/'config.json',{'model':args.model,'candidate':args.candidate,'seed':args.seed,'configuration':cfg,'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'preprocessing':state,'target_mean':ym,'target_scale':ys,'scientific_status':'EXPLORATORY_SMOKE_NOT_A_RANKING'})
 atomic_json(out/'status.json',{'status':'RUNNING'});history=[];peak=0
 def check():
  nonlocal peak
  mem=psutil.virtual_memory();rss=psutil.Process().memory_info().rss;peak=max(peak,rss)
  threads=2 if mem.available<2*1024**3 else 4 if mem.available<4*1024**3 else 8
  threads=min(threads,int(os.environ.get("OMP_NUM_THREADS","8")))
  torch.set_num_threads(threads)
  if rss>12*1024**3 or time.monotonic()-start>args.seconds:raise ResourceStop('Configured memory/disk/time limit')
  return threads
 if args.model in ['hgb','catboost']:
  flat=x.reshape(len(x),-1);threads=check()
  if args.model=='catboost':
   from catboost import CatBoostRegressor
   model=CatBoostRegressor(iterations=20,depth=cfg['depth'],learning_rate=cfg['learning_rate'],l2_leaf_reg=cfg['l2_leaf_reg'],loss_function='MultiRMSE',random_seed=args.seed,thread_count=threads,allow_writing_files=False,verbose=False)
   model.fit(flat[tr],target[tr]);pred=model.predict(flat[va])*ys+ym;model.save_model(str(out/'model.cbm'))
  else:
   from sklearn.ensemble import HistGradientBoostingRegressor
   models=[];pred=np.empty((len(va),7))
   for h in range(7):
    model=HistGradientBoostingRegressor(max_iter=20,max_leaf_nodes=cfg['max_leaf_nodes'],learning_rate=cfg['learning_rate'],l2_regularization=cfg['l2_regularization'],early_stopping=False,random_state=args.seed)
    with threadpool_limits(limits=check()):model.fit(flat[tr],target[tr,h]);pred[:,h]=model.predict(flat[va])*ys+ym
    models.append(model)
   joblib.dump(models,out/'models.joblib')
  parameters=None
 else:
  model=ForecastModel(args.model,state,x.shape[-1],cfg);parameters=sum(p.numel() for p in model.parameters());optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);best=math.inf
  for epoch in range(2):
   model.train();loss_sum=0;order=rng.permutation(tr)
   for b in range(0,len(order),256):
    check();idx=order[b:b+256];optimizer.zero_grad(set_to_none=True)
    loss=distillation_loss(model(torch.from_numpy(x[idx])),torch.from_numpy(target[idx]))
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(idx)
   model.eval()
   with torch.no_grad():prediction=model.predict(torch.from_numpy(x[va])).numpy()*ys+ym
   rmse=float(np.sqrt(np.mean((prediction-y[va])**2)));history.append({'epoch':epoch+1,'validation_rmse':rmse,'training_mse':loss_sum/len(tr)})
   if rmse<best:
    best=rmse;pred=prediction.copy();torch.save({'state_dict':model.state_dict(),'config':cfg,'model':args.model,'preprocessing':state,'target_mean':ym,'target_scale':ys},out/'best.pt')
   # Resume checkpoint includes optimizer and RNG; pilot itself is at most two epochs.
   torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'epoch':epoch+1},out/'last.pt')
   atomic_json(out/'history.json',history)
 if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions')
 np.savez_compressed(out/'predictions.npz',prediction=pred,target=y[va],station=station[va],origin=origin[va]);check()
 result={'status':'COMPLETE','scientific_status':'EXPLORATORY_SMOKE_NOT_A_RANKING','model':args.model,'seed':args.seed,'candidate':args.candidate,'train_sequences':len(tr),'validation_sequences':len(va),'parameters':parameters,'seconds':time.monotonic()-start,'peak_rss_bytes':peak,'metrics':metrics(y[va],pred,station[va])}
 atomic_json(out/'status.json',result);print(json.dumps(result));return 0

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--candidate',type=int,default=0);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--resume-from');ap.add_argument('--manifest');ap.add_argument('--smoke',action='store_true');ap.add_argument('--seconds',type=int,default=600);args=ap.parse_args()
 try:return train(args)
 except Exception as exc:
  out=local(args.out)
  # Never overwrite a prior trial if immutable-directory creation failed.
  if not isinstance(exc,FileExistsError) and out.exists():atomic_json(out/'status.json',{'status':exception_status(exc),'reason':repr(exc)})
  raise
if __name__=='__main__':sys.exit(main())
