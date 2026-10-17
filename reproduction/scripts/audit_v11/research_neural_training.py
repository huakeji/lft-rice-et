"""Streaming neural training primitive. Only receives a development data view.

The caller owns source-contract authorization, immutable configuration bindings,
resource supervision and budget accounting. This module does not launch jobs.
"""
import os,random
from pathlib import Path
import numpy as np
import torch
from research_models import distillation_loss

def atomic_save(value,path):
 path=Path(path);temporary=path.with_suffix(path.suffix+'.writing')
 torch.save(value,temporary);os.replace(temporary,path)

def train(model,data,encoder,target,config,seed,out,binding,guard=lambda:None,resume=False,teacher_targets=None,alpha=0.):
 """CPU train/validation only; return best validation checkpoint metadata.

 Binding must identify model/config/data/protocol/fold/seed; resume only loads
 a trusted checkpoint from this run directory after comparing that identity.
 guard raises on budget/resource limits; last complete epoch remains resumable.
 """
 from research_fold_data import DevelopmentData
 if alpha not in (0.,.25,.5,.75):raise ValueError('Undeclared distillation weight')
 from research_distillation_data import PairedStudentData
 if isinstance(data,PairedStudentData) and teacher_targets is None:teacher_targets=data.soft_targets
 if teacher_targets is not None:
  if not isinstance(data,PairedStudentData) or data.soft_targets is not teacher_targets:raise ValueError('Distillation requires matching paired data and OOF source')
  if binding.get('data_sha256')!=teacher_targets.identity['data_sha256'] or binding.get('fold_id')!=teacher_targets.identity['fold_id']:raise ValueError('Student and teacher OOF data/fold mismatch')
  binding={**binding,'distillation_alpha':alpha,'oof_content_sha256':teacher_targets.content_sha256}
 elif alpha:raise ValueError('Teacher targets required for positive distillation weight')
 if not isinstance(data,DevelopmentData):raise TypeError('Restricted development data view required')
 required={'model','config_sha256','data_sha256','protocol_sha256','fold_id','seed'}
 if not required.issubset(binding) or binding['seed']!=seed:raise ValueError('Incomplete run binding')
 if target['scale']<=0:raise ValueError('Invalid target scale')
 out=Path(out)
 if resume:
  if not (out/'last.pt').exists():raise ValueError('No checkpoint to resume')
 else:out.mkdir(parents=True,exist_ok=False)
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config.get('weight_decay',0.0003))
 start=0;best=float('inf');stale=0;history=[];best_epoch=None
 if resume:
  state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
  if state['binding']!=binding:raise ValueError('Checkpoint binding mismatch')
  model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
  torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
  start=state['epoch']+1;best=state['best'];stale=state['stale'];history=state['history'];best_epoch=state['best_epoch']
 for epoch in range(start,config['max_epochs']):
  if stale>=config['patience']:break
  guard();model.train();train_sum=0.;train_n=0
  for x,y,origin,station in data.encoded_batches(encoder,'train',config['batch_size']):
   guard();order=np.random.permutation(len(y));xx=torch.as_tensor(x[order],dtype=torch.float32);yy=torch.as_tensor((y[order]-target['mean'])/target['scale'],dtype=torch.float32)
   teacher=None
   if alpha:
    soft,covered=teacher_targets.lookup(station[order],origin[order],'train')
    if not covered.all():raise ValueError('Unpaired row entered distillation loss')
    teacher=torch.as_tensor((soft-target['mean'])/target['scale'],dtype=torch.float32)
   optimizer.zero_grad(set_to_none=True);loss=distillation_loss(model(xx),yy,teacher,alpha)
   if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
   loss.backward();optimizer.step();train_sum+=float(loss.detach())*yy.numel();train_n+=yy.numel()
  model.eval();sse=0.;count=0
  with torch.no_grad():
   for x,y,_,_ in data.encoded_batches(encoder,'val',config['batch_size']):
    guard();pred=model.predict(torch.as_tensor(x,dtype=torch.float32)).numpy()*target['scale']+target['mean']
    if not np.isfinite(pred).all():raise ValueError('Nonfinite validation prediction')
    sse+=float(np.square(pred.astype(float)-y).sum());count+=y.size
  if not train_n or not count:raise ValueError('Empty development split')
  rmse=(sse/count)**0.5;history.append({'epoch':epoch,'training_standardized_mse':train_sum/train_n,'validation_rmse':rmse})
  if rmse<best:
   best=rmse;stale=0;best_epoch=epoch
   atomic_save({'binding':binding,'model':model.state_dict(),'epoch':epoch,'validation_rmse':rmse},out/'best.pt')
  else:stale+=1
  atomic_save({'binding':binding,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,'history':history,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()},out/'last.pt')
 return {'validation_rmse':best,'best_epoch':best_epoch,'history':history,'metric_role':'internal_validation'}
