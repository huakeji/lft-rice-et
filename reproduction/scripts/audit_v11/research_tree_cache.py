"""Disk-backed common-feature matrices for tree models; no outer-data access.

A memmap bounds cache construction memory, not an estimator's internal copies.
The separate conservative admission estimate must precede every fit, and the
process-tree supervisor still enforces the hard runtime limit.
"""
import hashlib,json,shutil,fcntl,os,uuid
from research_resource_status import ResourceStop,SupervisionLost,exception_status
from pathlib import Path
import numpy as np
from research_fold_data import DevelopmentData
from research_run_manifest import digest
from run_conditional_batch import atomic_json

def memory_estimate(train_rows,val_rows,features):
 # Planning allowance, not a measured peak: float32 inputs, float64 estimator
 # copies, bins, gradients and overhead. Actual estimator behavior may differ.
 matrix=(train_rows+val_rows)*features
 return {'planning_peak_bytes':int(matrix*24+(train_rows+val_rows)*7*32+512*1024**2),'kind':'conservative_planning_estimate_not_measurement'}

def build(data,encoder,out,binding,guard=lambda:None,min_free_bytes=25*1024**3):
 if not isinstance(data,DevelopmentData):raise TypeError('Restricted development data required')
 out=Path(out);out.mkdir(parents=True,exist_ok=False);counts={};width=1;features=None
 for role in ('train','val'):
  count=0
  for x,c,y,origin,station in data.raw_batches(role):
   guard();count+=len(y);width=max(width,max(map(len,station.astype(str)),default=1))
   if features is None:features=encoder.transform(x[:1],c[:1]).shape[-1]*7
  if not count:raise ValueError('Empty development split')
  counts[role]=count
 expected=sum(counts.values())*(features*4+7*4+8+width*4)+4096*8
 if shutil.disk_usage(out).free<expected+min_free_bytes:raise ResourceStop('Insufficient disk for cache plus required headroom')
 result={'status':'COMPLETE','binding':binding,'counts':counts,'flattened_features':features,'station_unicode_width':width,'preprocessing_sha256':hashlib.sha256(json.dumps(encoder.state(),sort_keys=True).encode()).hexdigest(),'memory':memory_estimate(counts['train'],counts['val'],features),'files':{}}
 for role in ('train','val'):
  n=counts[role];spec={'x':('float32',(n,features)),'y':('float32',(n,7)),'origin':('datetime64[D]',(n,)),'station':(f'<U{width}',(n,))}
  arrays={k:np.lib.format.open_memmap(out/f'{role}_{k}.npy',mode='w+',dtype=dtype,shape=shape) for k,(dtype,shape) in spec.items()};offset=0
  for x,y,origin,station in data.encoded_batches(encoder,role):
   guard();end=offset+len(y)
   if end>n:raise ValueError('Sample count changed during cache creation')
   if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite cache data')
   dates=np.asarray(origin,dtype='datetime64[D]')
   if np.isnat(dates).any() or max(map(len,station.astype(str)),default=0)>width:raise ValueError('Invalid dates or changed station identifiers')
   arrays['x'][offset:end]=x.reshape(len(y),features);arrays['y'][offset:end]=y;arrays['origin'][offset:end]=dates;arrays['station'][offset:end]=station;offset=end
  if offset!=n:raise ValueError('Sample count changed during cache creation')
  for k,a in arrays.items():
   a.flush();a._mmap.close();path=out/f'{role}_{k}.npy';guard();result['files'][f'{role}_{k}']={'path':path.name,'sha256':digest(path),'bytes':path.stat().st_size,'fingerprint':fingerprint(path)}
  del arrays
 atomic_json(out/'manifest.json',result);return result

def fingerprint(path):
 s=Path(path).stat();return {'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'inode':s.st_ino}

def load(out,binding,verify_hash=True):
 out=Path(out).resolve();m=json.loads((out/'manifest.json').read_text())
 if m.get('status')!='COMPLETE' or m['binding']!=binding:raise ValueError('Incomplete cache or changed identity')
 arrays={}
 for key,record in m['files'].items():
  path=(out/record['path']).resolve()
  if not path.is_relative_to(out):raise ValueError('Cache path leaves directory')
  if fingerprint(path)!=record.get('fingerprint'):raise ValueError('Cache artifact metadata changed')
  if verify_hash and digest(path)!=record['sha256']:raise ValueError('Cache artifact changed')
  arrays[key]=np.load(path,mmap_mode='r',allow_pickle=False)
 return m,arrays

def shared(data,encoder,pool,run_binding,guard=lambda:None,min_free_bytes=25*1024**3):
 """Reuse immutable derived matrices across model candidates, never across folds.

 Initial construction hashes bytes. Later local reuse checks file identity,
 size/mtime/ctime and the manifest digest instead of rereading large matrices.
 This is mutation detection for a trusted local cache, not hostile-FS protection.
 Interrupted build directories are retained and never treated as complete.
 """
 required=('data_sha256','protocol_sha256','fold_id')
 if any(k not in run_binding for k in required):raise ValueError('Shared cache requires fold/data/protocol identity')
 binding={k:run_binding[k] for k in required}
 binding['preprocessing_sha256']=hashlib.sha256(json.dumps(encoder.state(),sort_keys=True).encode()).hexdigest()
 binding['cache_implementation_sha256']=hashlib.sha256(''.join(digest(Path(__file__).with_name(name)) for name in ('research_tree_cache.py','research_fold_data.py','matched_baselines.py')).encode()).hexdigest()
 key=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest();pool=Path(pool);pool.mkdir(parents=True,exist_ok=True);dest=pool/key
 with (pool/(key+'.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  reused=dest.exists()
  if not reused:
   temporary=pool/(key+'.building-'+uuid.uuid4().hex)
   try:
    build(data,encoder,temporary,binding,guard,min_free_bytes)
    receipt={'manifest_sha256':digest(temporary/'manifest.json')}
    atomic_json(temporary/'receipt.json',receipt)
    if dest.exists():raise ValueError('Unexpected cache destination; refuse overwrite')
    os.rename(temporary,dest)
   except BaseException as exc:
    if temporary.exists():atomic_json(temporary/'build_failure.json',{'status':'INCOMPLETE_RETAINED','reason':repr(exc)})
    raise
  receipt=json.loads((dest/'receipt.json').read_text())
  if digest(dest/'manifest.json')!=receipt['manifest_sha256']:raise ValueError('Shared cache manifest changed')
  metadata,arrays=load(dest,binding,verify_hash=False)
 return metadata,arrays,dest,reused
