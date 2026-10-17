"""Training-only OOF joins; paired controls preserve full public validation."""
import hashlib,json
import numpy as np
from research_fold_data import DevelopmentData,FoldData
from rolling_manifest import partition

class SoftTargets:
 def __init__(self,station,origin,prediction,covered,outer_year,identity,expected_identity,excluded_stations=()):
  if identity!=expected_identity:raise ValueError('OOF source identity mismatch')
  if identity.get('prediction_units')!='mm/day' or identity.get('prediction_space')!='physical_reference_ET0':raise ValueError('OOF must contain inverse-transformed reference ET0 in mm/day')
  if identity.get('fold_id')!=f'temporal_{outer_year}':raise ValueError('OOF temporal fold identity mismatch')
  required=('data_sha256','plan_sha256','teacher_configuration_sha256')
  if any(not isinstance(identity.get(k),str) or len(identity[k])!=64 for k in required):raise ValueError('Incomplete OOF identity')
  station=np.asarray(station).astype(str);origin=np.asarray(origin,dtype='datetime64[D]');prediction=np.asarray(prediction,dtype=np.float32);covered=np.asarray(covered)
  if origin.ndim!=1 or station.shape!=origin.shape or prediction.shape!=(len(origin),7) or covered.shape!=origin.shape or covered.dtype!=bool or np.isnat(origin).any():raise ValueError('Malformed OOF arrays')
  if not np.isfinite(prediction[covered]).all() or not np.isnan(prediction[~covered]).all():raise ValueError('Coverage and missing OOF values disagree')
  role=partition(origin+1,origin+7,outer_year)
  if ((role!='train')&covered).any() or (np.isin(station,list(map(str,excluded_stations)))&covered).any():raise ValueError('Nontraining or excluded station has OOF target')
  keys=list(zip(station.tolist(),origin.astype(str).tolist()))
  if len(set(keys))!=len(keys):raise ValueError('Duplicate OOF sequence')
  self.__lookup={keys[i]:prediction[i].copy() for i in np.flatnonzero(covered)};self.identity=dict(identity)
  h=hashlib.sha256(json.dumps(identity,sort_keys=True).encode())
  for key in sorted(self.__lookup):h.update(('\t'.join(key)+'\n').encode());h.update(self.__lookup[key].tobytes())
  self.content_sha256=h.hexdigest()
 def lookup(self,station,origin,role):
  if role!='train':raise ValueError('OOF targets are inaccessible outside student training')
  station=np.asarray(station).astype(str);origin=np.asarray(origin,dtype='datetime64[D]')
  if station.shape!=origin.shape:raise ValueError('Student join key shape mismatch')
  result=np.full((len(origin),7),np.nan,dtype=np.float32);covered=np.zeros(len(origin),dtype=bool)
  for i,key in enumerate(zip(station.tolist(),origin.astype(str).tolist())):
   if key in self.__lookup:result[i]=self.__lookup[key];covered[i]=True
  return result,covered

class PairedStudentData(DevelopmentData):
 """Alpha=0 paired baseline and alpha>0 students use identical fitting rows.
 The independent full-data student comparison remains a separate run.
 """
 def __init__(self,data,soft_targets):
  if not isinstance(data,DevelopmentData):raise TypeError('Restricted public development data required')
  self.base=data;self.soft_targets=soft_targets
 def raw_batches(self,role,batch_size=256):
  for x,c,y,origin,station in self.base.raw_batches(role,batch_size):
   if role=='train':
    _,keep=self.soft_targets.lookup(station,origin,'train')
    if keep.any():yield x[keep],c[keep],y[keep],origin[keep],station[keep]
   else:yield x,c,y,origin,station
 def fit_encoder(self,variant='full'):return FoldData.fit_encoder(self,variant)
