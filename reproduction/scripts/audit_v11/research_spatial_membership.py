"""Coordinate-only verification of the preserved five-fold spatial design.

This verifies membership geometry, not historical admission or source semantics.
"""
import csv,json,time
from pathlib import Path
import numpy as np
from research_run_manifest import digest

def distances(a,b):
 a=np.radians(a);b=np.radians(b)
 delta=a[:,None,:]-b[None,:,:]
 h=np.sin(delta[:,:,0]/2)**2+np.cos(a[:,None,0])*np.cos(b[None,:,0])*np.sin(delta[:,:,1]/2)**2
 return 6371.0088*2*np.arcsin(np.sqrt(np.clip(h,0,1)))

def verify(path,buffer_km=100.):
 if buffer_km!=100.:raise ValueError('Declared buffer is 100 km')
 with Path(path).open(newline='') as f:rows=list(csv.DictReader(f))
 if not rows or {int(r['fold']) for r in rows}!=set(range(5)):raise ValueError('All five folds required')
 seen=set();coords={};groups={};test_counts={};cohort=None;summary=[]
 for r in rows:
  key=(int(r['fold']),r['station'])
  if not key[1] or key in seen:raise ValueError('Missing/duplicate fold station')
  seen.add(key);xy=(float(r['latitude_decimal']),float(r['longitude_decimal']))
  if not np.isfinite(xy).all() or not -90<=xy[0]<=90 or not -180<=xy[1]<=180:raise ValueError('Invalid decimal coordinates')
  g=int(r['geographic_group'])
  if g not in range(5) or (key[1] in coords and (coords[key[1]]!=xy or groups[key[1]]!=g)):raise ValueError('Coordinates/group inconsistent across folds')
  coords[key[1]]=xy;groups[key[1]]=g
  if r['role']=='test':test_counts[key[1]]=test_counts.get(key[1],0)+1
 for fold in range(5):
  rr=[r for r in rows if int(r['fold'])==fold];stations={r['station'] for r in rr}
  if cohort is not None and stations!=cohort:raise ValueError('Fold cohorts differ')
  cohort=stations
  roles={role:[r for r in rr if r['role']==role] for role in ('train','validation','test','buffer')}
  if sum(map(len,roles.values()))!=len(rr) or any(not roles[r] for r in ('train','validation','test')):raise ValueError('Unknown role or empty required role')
  for r in rr:
   g=int(r['geographic_group']);role=r['role']
   if (g==fold and role!='test') or (g==(fold+1)%5 and role!='validation') or (g not in (fold,(fold+1)%5) and role not in ('train','buffer')):raise ValueError('Geographic group role mismatch')
  coordinate=lambda rs:np.array([coords[r['station']] for r in rs])
  held=roles['validation']+roles['test'];train=coordinate(roles['train'])
  minimum=distances(train,coordinate(held)).min(axis=1)
  if (minimum<buffer_km-1e-6).any():raise ValueError('Training station violates buffer')
  if roles['buffer'] and (distances(coordinate(roles['buffer']),coordinate(held)).min(axis=1)>=buffer_km+1e-6).any():raise ValueError('Buffered station lacks geometric reason')
  nearest=distances(coordinate(roles['test']),train).min(axis=1)
  summary.append({'fold':fold,**{role:len(v) for role,v in roles.items()},'test_nearest_train_min_km':float(nearest.min()),'test_nearest_train_median_km':float(np.median(nearest)),'test_nearest_train_max_km':float(nearest.max())})
 if set(test_counts)!=cohort or any(n!=1 for n in test_counts.values()):raise ValueError('Each station must appear in exactly one test fold')
 return {'status':'GEOMETRY_VERIFIED_ONLY','membership_sha256':digest(path),'stations':len(cohort),'buffer_km':buffer_km,'earth_radius_km':6371.0088,'summary':summary,'formal_source_ready':False,'as_of_admission_certified':False}

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--membership',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();started=time.monotonic();report=verify(a.membership);report['seconds']=time.monotonic()-started
 with Path(a.output).open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report,indent=2))
