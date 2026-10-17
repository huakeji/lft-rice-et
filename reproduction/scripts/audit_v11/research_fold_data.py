"""Bounded-memory fold reader; learning roles are determined before preprocessing."""
from pathlib import Path
import numpy as np
from matched_baselines import Encoder,active
from rolling_manifest import partition

class FoldData:
 def __init__(self,shards,outer_year,excluded_stations=(),target_offset=0):
  self.shards=[Path(p) for p in shards];self.outer_year=outer_year;self.excluded=set(map(str,excluded_stations));self.target_offset=int(target_offset)
 def raw_batches(self,role,batch_size=256):
  if role not in ['train','val','test']:raise ValueError('Unknown role')
  for p in self.shards:
   with np.load(p,allow_pickle=False) as a:
    st=a['station'].astype(str)
    # Excluded sites cannot enter any fitted role through this training reader.
    x,c,y,origin=[a[k] for k in ['x','c','y','origin']];dates=origin.astype('datetime64[D]')
    roles=partition(dates+1+self.target_offset,dates+7+self.target_offset,self.outer_year);allowed=roles==role
    if role in ['train','val']:allowed &= ~np.isin(st,list(self.excluded))
    ids=np.flatnonzero(allowed)
    for start in range(0,len(ids),batch_size):
     i=ids[start:start+batch_size];yield x[i],c[i],y[i],origin[i],st[i]
 def fit_encoder(self,variant='full'):
  enc=Encoder();enc.ni,enc.ci=active(variant);count=0;total=np.zeros(len(enc.ni));squares=total.copy();vocab={j:set() for j in enc.ci};yn=0;ytotal=0.;ysquares=0.
  for x,c,y,_,_ in self.raw_batches('train'):
   z=x[:,:,enc.ni].astype(float).reshape(-1,len(enc.ni))
   if not np.isfinite(z).all() or not np.isfinite(y).all():raise ValueError('Nonfinite training feature or target')
   count+=len(z);total+=z.sum(0);squares+=(z*z).sum(0)
   for j in enc.ci:vocab[j].update(c[:,:,j].ravel().tolist())
   yy=y.astype(float);yn+=yy.size;ytotal+=yy.sum();ysquares+=(yy*yy).sum()
  if count==0:raise ValueError('No permitted training samples')
  enc.mean=total/count;enc.scale=np.sqrt(np.maximum(squares/count-enc.mean**2,0));enc.scale[enc.scale<1e-8]=1.;enc.vocab={j:sorted(v) for j,v in vocab.items()}
  mean=ytotal/yn;scale=max(float(np.sqrt(max(ysquares/yn-mean**2,0))),1e-8)
  return enc,{'mean':float(mean),'scale':scale,'training_target_count':yn}
 def encoded_batches(self,encoder,role,batch_size=256):
  for x,c,y,origin,station in self.raw_batches(role,batch_size):yield encoder.transform(x,c),y,origin,station

class DevelopmentData:
 """Restricted interface supplied to optimizers: no outer evaluation method.

 This is a guard against accidental access, not a security boundary against code
 deliberately reaching into Python object internals.
 """
 def __init__(self,fold):self.__fold=fold
 def raw_batches(self,role,batch_size=256):
  if role not in ('train','val'):raise ValueError('Outer evaluation is forbidden during development')
  yield from self.__fold.raw_batches(role,batch_size)
 def fit_encoder(self,variant='full'):return self.__fold.fit_encoder(variant)
 def encoded_batches(self,encoder,role,batch_size=256):
  for x,c,y,origin,station in self.raw_batches(role,batch_size):yield encoder.transform(x,c),y,origin,station

def evaluation_batches(fold,selection,scope='network',batch_size=256):
 """Explicitly split temporal network evaluation from excluded-site evaluation.
 Callers must additionally validate selection identity against the run manifest.
 """
 if selection.get('selection_stage')!='FROZEN_BEFORE_OUTER_EVALUATION':raise ValueError('Freeze model selection before outer evaluation')
 if scope not in ('network','excluded'):raise ValueError('Unknown evaluation scope')
 for x,c,y,origin,station in fold.raw_batches('test',batch_size):
  excluded=np.isin(station,list(fold.excluded));keep=excluded if scope=='excluded' else ~excluded
  if keep.any():yield x[keep],c[keep],y[keep],origin[keep],station[keep]
