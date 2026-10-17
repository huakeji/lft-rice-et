"""End-to-end verification of the open release (release/lft-rice-et) against the paper pipeline.
For a random sample of stations: rebuild model inputs from the raw public messages in the (read-only) database
with the release code, compare them with lagged_shards for every 2020 issue date, and compare the release
forecasts (LFT, GRU, ensemble, quantiles) with the paper's 2020 test predictions on identical keys.
Usage: python3 verify_release.py [--stations 30]"""
import argparse,glob,json,sqlite3,sys
from datetime import date
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];REL=ROOT/'release/lft-rice-et';sys.path.insert(0,str(REL))
from lft_et import Message,Forecaster,load_stations,sequences_from_archive
R=ROOT/'reports/awm_repro_20261015'
ap=argparse.ArgumentParser();ap.add_argument('--stations',type=int,default=30);a=ap.parse_args()
conn=sqlite3.connect((ROOT/'data/db/rice_rl_data.db').as_uri()+'?mode=ro',uri=True);conn.execute('PRAGMA query_only=ON')
city=dict(pd.read_sql_query('SELECT station_code,city FROM city_station_mapping',conn).astype(str).itertuples(index=False))
ST=load_stations();rng=np.random.default_rng(1);sample=sorted(rng.choice(sorted(ST),a.stations,replace=False))
def ens(pattern):
 zs=[np.load(f) for f in sorted(glob.glob(str(R/pattern)))];return np.mean([z['prediction'] for z in zs],0),zs[0]
L,z=ens('formal/lft_y2020_s*/outer_predictions.npz');G,_=ens('formal/gru_y2020_s*/outer_predictions.npz');Q=np.load(R/'peak_unc/quant_y2020/outer_predictions.npz')
key={(s,o):i for i,(s,o) in enumerate(zip(z['station'].astype(str),z['origin'].astype(str)))}
qkey={(s,o):i for i,(s,o) in enumerate(zip(Q['station'].astype(str),Q['origin'].astype(str)))}
fc=Forecaster();rep=[];xs_bad=0;cs_bad=0;n_seq=0;dL=[];dG=[];dE=[];dQ=[]
for st in sample:
 f=pd.read_sql_query("SELECT date,day_weather_json,day_high_json,day_low_json,day_wind_json FROM forecast_raw_daily WHERE city=? AND date BETWEEN '2019-12-20' AND '2020-12-31' ORDER BY date",conn,params=[city[st]])
 msgs={date.fromisoformat(r.date):Message(*(json.loads(v) if v else {} for v in (r.day_weather_json,r.day_high_json,r.day_low_json,r.day_wind_json))) for r in f.itertuples()}
 sh=np.load(R/f'lagged_shards/station_{st}.npz');so={o:i for i,o in enumerate(sh['origin'].astype(str))}
 items=[it for it in sequences_from_archive(ST[st],msgs,date(2020,1,1),date(2020,12,25)) if (st,it[1].isoformat()) in key]
 for _,issue,(x,c) in items:
  i=so[issue.isoformat()];n_seq+=1
  xs_bad+=not np.allclose(x,sh['x'][i],atol=1e-4,equal_nan=True);cs_bad+=not (c==sh['c'][i].astype(str)).all()
 if not items:continue
 X=np.stack([it[2][0] for it in items]);C=np.stack([it[2][1] for it in items]);p=fc.predict_arrays(X,C)
 idx=np.array([key[(st,it[1].isoformat())] for it in items]);qi=np.array([qkey[(st,it[1].isoformat())] for it in items])
 dL.append(np.abs(p['eto_lft']-L[idx]).max());dG.append(np.abs(p['eto_gru']-G[idx]).max())
 dE.append(np.abs(p['eto']-np.clip((L[idx]+G[idx])/2,0,None)).max());dQ.append(np.abs(p['eto_quantiles']-np.clip(np.sort(Q['quantiles'][qi],-1),0,None)).max())
out=dict(stations=len(sample),sequences=n_seq,input_x_mismatches=int(xs_bad),input_text_mismatches=int(cs_bad),
 max_abs_diff_lft=float(max(dL)),max_abs_diff_gru=float(max(dG)),max_abs_diff_ensemble=float(max(dE)),max_abs_diff_quantiles=float(max(dQ)))
(REL/'VERIFICATION.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))
