"""Ablation copy of lagged_shards in which every earlier issue is marked unavailable: lag values are
replaced by the current issue (exactly as the builder does for missing issues), availability = 0 and the
inter-issue spreads = 0. The LFT then sees only the current issue with identical architecture/capacity."""
import json,shutil
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2]/'reports/awm_repro_20261015';S=R/'lagged_shards';O=R/'lagged_shards_nolag'
O.mkdir(exist_ok=False);m=json.loads((S/'manifest.json').read_text());J=m['lags']
for p in sorted(S.glob('station_*.npz')):
 z=np.load(p);x=z['x'].copy();c=z['c'].copy()
 for j in range(J):
  b=10+4*j;x[:,:,b]=x[:,:,0];x[:,:,b+1]=x[:,:,1];x[:,:,b+2]=x[:,:,3];x[:,:,b+3]=0
  c[:,:,3+2*j]=c[:,:,0];c[:,:,4+2*j]=c[:,:,1]
 x[:,:,10+4*J:12+4*J]=0
 np.savez_compressed(O/p.name,x=x,c=c,y=z['y'],origin=z['origin'],station=z['station'])
m.update(base='lagged_shards',ablation='all earlier issues unavailable');(O/'manifest.json').write_text(json.dumps(m,indent=1));print('ok')
