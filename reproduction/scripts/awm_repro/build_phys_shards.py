"""lagged_shards + three deterministic physics covariates per day (no observations): extraterrestrial
radiation Ra (FAO-56 eq. 21, MJ m-2 d-1) from latitude and valid date, atmospheric pressure (FAO-56 eq. 7)
from altitude, and Hargreaves-Samani ET0 from the forecast Tmax/Tmin of the same issue."""
import json,hashlib,shutil,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'reports/awm_repro_20261015/lagged_shards';OUT=ROOT/'reports/awm_repro_20261015/lagged_phys_shards'
def ra(lat,doy):
 phi=np.radians(lat);dr=1+0.033*np.cos(2*np.pi*doy/365);d=0.409*np.sin(2*np.pi*doy/365-1.39);ws=np.arccos(np.clip(-np.tan(phi)*np.tan(d),-1,1))
 return 24*60/np.pi*0.0820*dr*(ws*np.sin(phi)*np.sin(d)+np.cos(phi)*np.cos(d)*np.sin(ws))
OUT.mkdir(exist_ok=False);m=json.loads((SRC/'manifest.json').read_text());n0=10+4*m['lags']+2
for p in sorted(SRC.glob('station_*.npz')):
 z=np.load(p);x=z['x'];o=z['origin'].astype('datetime64[D]')
 doy=np.stack([((o+k)-o.astype('datetime64[Y]')).astype(int)+1 for k in range(7)],1).astype(float)
 R=ra(x[:,:,6],doy);P=101.3*((293-0.0065*x[:,:,8])/293)**5.26;tx,tn=x[:,:,0],x[:,:,1]
 HS=0.0023*((tx+tn)/2+17.8)*np.sqrt(np.clip(tx-tn,0,None))*R*0.408
 np.savez_compressed(OUT/p.name,x=np.concatenate([x,np.stack([R,P,HS],2).astype('float32')],2),c=z['c'],y=z['y'],origin=z['origin'],station=z['station'])
m.update(base='reports/awm_repro_20261015/lagged_shards',extra_numeric=m['extra_numeric']+['Ra','pressure','hs_et0'],phys_index=[n0,n0+1,n0+2],code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'manifest.json').write_text(json.dumps(m,indent=1));print(n0,m['phys_index'])
