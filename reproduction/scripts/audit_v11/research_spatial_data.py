"""Explicit spatial roles layered on the all-seven-target temporal split.

This adapter is not wired into formal jobs yet. Source/admission/coordinate and
100-km-distance evidence must be validated by a separate spatial specification.
"""
import numpy as np
from research_fold_data import FoldData

class SpatialFoldData(FoldData):
 def __init__(self,shards,outer_year,membership,target_offset=0):
  if not membership or set(membership.values())-{'train','validation','test','buffer'}:raise ValueError('Explicit spatial membership required')
  if any(not isinstance(k,str) or not k for k in membership):raise ValueError('Station keys must be nonempty strings')
  if any(role not in membership.values() for role in ('train','validation','test')):raise ValueError('Nonempty train/validation/test groups required')
  # Parent must not pre-filter validation stations; this adapter owns role masks.
  super().__init__(shards,outer_year,target_offset=target_offset)
  self.membership=dict(membership)
  self.excluded={s for s,r in membership.items() if r in ('test','buffer')}
 def raw_batches(self,role,batch_size=256):
  spatial_role={'train':'train','val':'validation','test':'test'}.get(role)
  if spatial_role is None:raise ValueError('Unknown role')
  # Use a fresh temporal reader to avoid its common train/val exclusion policy.
  temporal=FoldData(self.shards,self.outer_year,target_offset=self.target_offset)
  for x,c,y,origin,station in temporal.raw_batches(role,batch_size):
   keys=np.asarray(station).astype(str)
   if any(s not in self.membership for s in keys):raise ValueError('Shard station absent from spatial membership')
   keep=np.array([self.membership[s]==spatial_role for s in keys])
   if keep.any():yield x[keep],c[keep],y[keep],origin[keep],station[keep]
