"""CPU tree fits on common disk matrices, under an external hard-limit supervisor.

Trees use the declared iteration cap without automatic internal validation splits.
HGB fits seven heads sequentially. CatBoost uses MultiRMSE and equal public input
one-hot features; native-category optimization is not silently substituted.
"""
import json,os
from research_resource_status import ResourceStop,SupervisionLost,exception_status
from pathlib import Path
import joblib,numpy as np,psutil
from threadpoolctl import threadpool_limits
from research_tree_cache import build,load,shared
from research_run_manifest import digest
from run_conditional_batch import atomic_json

def fit(kind,data,encoder,target,cfg,seed,out,binding,guard=lambda:None,min_free_bytes=25*1024**3,cache_pool=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);guard()
 if cache_pool is None:
  build(data,encoder,out/'cache',binding,guard,min_free_bytes);meta,a=load(out/'cache',binding);cache_path=out/'cache';reused=False
 else:meta,a,cache_path,reused=shared(data,encoder,cache_pool,binding,guard,min_free_bytes)
 atomic_json(out/'cache_reference.json',{'path':str(cache_path.resolve()),'manifest_sha256':digest(cache_path/'manifest.json'),'reused':reused})
 guard()
 estimate=meta['memory']['planning_peak_bytes'];available=psutil.virtual_memory().available
 atomic_json(out/'resource_preflight.json',{**meta['memory'],'available_bytes':available,'admission_limit_bytes':min(10*1024**3,int(available*.8))})
 if estimate>min(10*1024**3,int(available*.8)):raise ResourceStop('Tree planning estimate exceeds available memory budget')
 ceiling=int(os.environ.get('OMP_NUM_THREADS','8'));threads=min(ceiling,2 if available<2*1024**3 else 4 if available<4*1024**3 else 8)
 y=(a['train_y']-target['mean'])/target['scale'];prediction=np.empty_like(a['val_y']);artifacts=[]
 if kind=='hgb':
  from sklearn.ensemble import HistGradientBoostingRegressor
  for h in range(7):
   guard();model=HistGradientBoostingRegressor(max_iter=cfg['max_iter'],max_leaf_nodes=cfg['max_leaf_nodes'],learning_rate=cfg['learning_rate'],l2_regularization=cfg['l2_regularization'],early_stopping=False,random_state=seed)
   with threadpool_limits(limits=threads):
    model.fit(a['train_x'],y[:,h]);prediction[:,h]=model.predict(a['val_x'])*target['scale']+target['mean']
   path=out/f'head_{h}.joblib';joblib.dump(model,path);artifacts.append({'path':path.name,'sha256':digest(path)});del model
 elif kind=='catboost':
  from catboost import CatBoostRegressor
  model=CatBoostRegressor(iterations=cfg['iterations'],depth=cfg['depth'],learning_rate=cfg['learning_rate'],l2_leaf_reg=cfg['l2_leaf_reg'],loss_function='MultiRMSE',random_seed=seed,thread_count=threads,allow_writing_files=False,verbose=False,use_best_model=False)
  model.fit(a['train_x'],y);prediction[:]=model.predict(a['val_x'])*target['scale']+target['mean'];path=out/'model.cbm';model.save_model(str(path));artifacts.append({'path':path.name,'sha256':digest(path)})
 else:raise ValueError('Unknown tree family')
 guard()
 # Reopen serialized estimators before declaring a completed trial.
 replay=np.empty_like(prediction)
 if kind=='hgb':
  for h in range(7):
   model=joblib.load(out/f'head_{h}.joblib')
   with threadpool_limits(limits=threads):replay[:,h]=model.predict(a['val_x'])*target['scale']+target['mean']
   del model;guard()
 else:
  from catboost import CatBoostRegressor
  model=CatBoostRegressor();model.load_model(str(out/'model.cbm'));replay[:]=model.predict(a['val_x'],thread_count=threads)*target['scale']+target['mean'];del model
 if not np.allclose(replay,prediction,rtol=1e-6,atol=1e-7):raise ValueError('Serialized tree replay mismatch')
 if not np.isfinite(prediction).all():raise ValueError('Nonfinite tree predictions')
 np.savez_compressed(out/'validation_predictions.npz',prediction=prediction,target=a['val_y'],station=a['val_station'],origin=a['val_origin'])
 rmse=float(np.sqrt(np.mean((prediction.astype(float)-a['val_y'])**2)))
 result={'validation_rmse':rmse,'parameters':None,'iteration_rule':'fixed declared cap; no hidden split or best-iteration selection','artifacts':artifacts,'binding':binding}
 atomic_json(out/'result.json',result);return result
