"""National rice crop water requirement (ETc = Kc x ET0) 7-day forecast prototype.

ETc here is the standard-condition crop evapotranspiration requirement, not actual ET and not an
irrigation amount. Kc comes from the station-by-month rice Kc table (external/kc; not distributed)
(product stations, plus nearest-product coefficients applied to local climate for the rest);
months outside the rice season (month_src != month) are not forecast. Rice-region stations follow
the same-day rice-region screening table (county listed in the national rice irrigation quota and
>=2000 degC-days above 10 degC).

Evaluation compares Kc x predicted ET0 with Kc x reference ET0 on the same target days, using the
frozen five-seed 2020 retrospective predictions; relative errors therefore equal ET0 relative errors
weighted by Kc and season, and are reported per rice type, zone and lead.
"""
import argparse,glob,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
KC_DIR=ROOT/'external/kc'
RUNS='reports/v11_audit_20260909/research_route_40/rolling_same_day_2020_pft_seed{}_outer/outer_predictions'

def load_kc():
 kc=pd.read_csv(KC_DIR/'kc_all_stations_monthly.csv',dtype={'station_code':str})
 site=pd.read_csv(KC_DIR/'kc_site_matching.csv',dtype={'station_code':str})
 rice=pd.read_csv(KC_DIR/'rice_zone_stations.csv')
 site=site.merge(rice[['role','zone','site','rice','evidence']],on=['role','zone','site'],how='left',validate='one_to_one')
 kc=kc.merge(site[['station_code','rice','province','lat','lon']],on='station_code',how='left',validate='many_to_one')
 kc['in_season']=kc.month==kc.month_src
 return kc

def load_ensemble(seeds):
 frames=[]
 for s in seeds:
  parts=sorted(glob.glob(str(ROOT/RUNS.format(s)/'part_*.npz')))
  P,Y,O,S=[],[],[],[]
  for f in parts:
   z=np.load(f);P.append(z['prediction']);Y.append(z['target']);O.append(z['origin']);S.append(z['station'])
  frames.append((np.concatenate(P),np.concatenate(Y),np.concatenate(O),np.concatenate(S)))
 for p,y,o,s in frames[1:]:
  if not((o==frames[0][2]).all() and (s==frames[0][3]).all() and np.allclose(y,frames[0][1])):raise ValueError('seed prediction keys differ')
 P=np.mean([f[0] for f in frames],0);_,Y,O,S=frames[0]
 return P,Y,O,S

def main(a):
 out=ROOT/a.out_dir;out.mkdir(parents=True,exist_ok=True)
 kc=load_kc()
 if a.pred_npz:
  z=np.load(ROOT/a.pred_npz);P,Y,S=z['prediction'],z['target'],z['station'].astype(str)
  O=z['origin'] if 'origin' in z.files else np.load(ROOT/a.origin_from)['origin']
 else:P,Y,O,S=load_ensemble(range(5))
 long=pd.DataFrame({'station':np.repeat(S,7),'origin':np.repeat(O,7),'lead':np.tile(np.arange(1,8),len(S)),'et0_pred':P.ravel(),'et0_ref':Y.ravel()})
 long['target']=pd.to_datetime(long.origin)+pd.to_timedelta(long.lead-1,unit='D');long['month']=long.target.dt.month
 m=long.merge(kc[['station_code','month','Kc','sheet','zone','rice','in_season']],left_on=['station','month'],right_on=['station_code','month'],how='left')
 cover=dict(et0_stations=int(long.station.nunique()),with_kc=int(m.dropna(subset=['Kc']).station.nunique()),
  rice_stations=int(m[m.rice==1].station.nunique()))
 r=m[(m.rice==1)&m.in_season].copy()
 r['etc_pred']=r.Kc*r.et0_pred.clip(lower=0);r['etc_ref']=r.Kc*r.et0_ref;r['err']=r.etc_pred-r.etc_ref
 def summ(g):
  e=g.err.to_numpy();return pd.Series(dict(n=len(e),stations=g.station.nunique(),rmse=np.sqrt(np.mean(e**2)),mae=np.mean(np.abs(e)),bias=e.mean(),
   etc_mean=g.etc_ref.mean(),rel_rmse=np.sqrt(np.mean(e**2))/g.etc_ref.mean()))
 by_lead=r.groupby('lead').apply(summ);by_sheet=r.groupby('sheet').apply(summ);by_zone=r.groupby('zone').apply(summ);overall=summ(r)
 # seven-day cumulative crop water requirement per issued batch (the quantity a scheduler uses)
 w=r.groupby(['station','origin']).agg(n=('lead','size'),pred=('etc_pred','sum'),ref=('etc_ref','sum'));w=w[w.n==7]
 cum=dict(batches=len(w),rmse_7day_mm=float(np.sqrt(np.mean((w.pred-w.ref)**2))),mean_7day_mm=float(w.ref.mean()))
 for n,d in [('by_lead',by_lead),('by_sheet',by_sheet),('by_zone',by_zone)]:d.round(4).to_csv(out/f'etc_2020_{n}.csv')
 # example national product for one issue day
 day=a.example_origin;ex=r[r.origin==day].pivot_table(index=['station','sheet','zone'],columns='lead',values='etc_pred').round(2)
 ex.columns=[f'etc_D{c-1}' for c in ex.columns];ex.to_csv(out/f'example_product_{day}.csv')
 res=dict(coverage=cover,overall={k:float(v) for k,v in overall.items()},cumulative_7day=cum,example_origin=day,example_stations=len(ex),
  note='ETc = Kc x ET0 standard-condition requirement; not actual ET or irrigation amount',predictions=a.pred_npz or 'default predictions',
  kc_source=str(KC_DIR),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (out/'summary.json').write_text(json.dumps(res,ensure_ascii=False,indent=1));print(json.dumps(res,ensure_ascii=False,indent=1));print(by_lead.round(3));print(by_sheet.round(3))

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out-dir',default='reports/awm_repro_20261015/etc_product');ap.add_argument('--example-origin',default='2020-07-15');ap.add_argument('--pred-npz',default='');ap.add_argument('--origin-from',default='')
 main(ap.parse_args())
