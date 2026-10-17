"""Lossless bit encoding of training vocabulary IDs, for memory-limited trees.

All known categories retain unique codes; unknown is zero. This changes tree
split geometry versus one-hot and must be reported as a distinct representation.
"""
import numpy as np
class CompactTreeEncoder:
 def __init__(self,base):self.base=base
 def transform(self,x,c):
  b=self.base;parts=[(x[:,:,b.ni]-b.mean)/b.scale]
  for j in b.ci:
   mapping={v:i+1 for i,v in enumerate(b.vocab[j])};bits=max(1,len(mapping).bit_length())
   ids=np.array([mapping.get(v,0) for v in c[:,:,j].ravel()],dtype=np.uint64).reshape(c.shape[:2])
   parts.append(((ids[:,:,None]>>np.arange(bits,dtype=np.uint64))&1).astype(np.float32))
  return np.concatenate(parts,axis=2).astype(np.float32)
 def state(self):return {**self.base.state(),'categorical_encoding':'binary_vocabulary_id_little_endian','information_loss':'none for training-vocabulary categories; unseen maps to zero','split_geometry':'different from one-hot; not original failed representation'}
