"""Multi-metric evaluation of saved outer predictions (no retraining). Metrics are pooled over all
station-origin-lead cells unless named st_* (computed per station, then median/mean)."""
import json,sys,glob
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
def kge(p,o):r=np.corrcoef(p,o)[0,1];return 1-np.sqrt((r-1)**2+(p.std()/o.std()-1)**2+(p.mean()/o.mean()-1)**2)
def metrics(P,Y,S):
 e=P-Y;p,o=P.ravel(),Y.ravel();sse=(e**2).sum();out=dict(RMSE=np.sqrt((e**2).mean()),MAE=np.abs(e).mean(),Bias=e.mean(),
  rRMSE_pct=100*np.sqrt((e**2).mean())/Y.mean(),r=np.corrcoef(p,o)[0,1],R2=np.corrcoef(p,o)[0,1]**2,NSE=1-sse/((o-o.mean())**2).sum(),KGE=kge(p,o))
 c7=P.sum(1)-Y.sum(1);out['RMSE_7d_sum']=np.sqrt((c7**2).mean());out['rRMSE_7d_sum_pct']=100*out['RMSE_7d_sum']/Y.sum(1).mean()
 q=np.quantile(Y,0.9);hi=Y>=q;out['RMSE_top10pct_days']=np.sqrt((e[hi]**2).mean());out['Bias_top10pct_days']=e[hi].mean()
 st=[];
 for s in np.unique(S):
  m=S==s;yy=Y[m].ravel();ee=e[m].ravel();st.append((1-(ee**2).sum()/((yy-yy.mean())**2).sum(),kge(P[m].ravel(),yy)))
 st=np.array(st);out['st_NSE_median']=np.median(st[:,0]);out['st_KGE_median']=np.median(st[:,1]);out['st_NSE_lt0.5_n']=int((st[:,0]<0.5).sum())
 return out
rows=[]
for f in sorted(glob.glob(str(ROOT/'reports/awm_repro_20261015/bakeoff/stageB/*/outer_predictions.npz'))+glob.glob(str(ROOT/'reports/awm_repro_20261015/lagfusion/*/outer_predictions.npz'))):
 d=Path(f).parent;r=json.loads((d/'result.json').read_text());z=np.load(f)
 rows.append(dict(run=d.name,model=r['model']+(f"+lag{r['lags']}" if r.get('lags') else ''),year=r['outer_year'],seed=r['seed'],**metrics(z['prediction'],z['target'],z['station'])))
D=pd.DataFrame(rows);out=ROOT/'reports/awm_repro_20261015/multi_metrics';out.mkdir(exist_ok=True);D.to_csv(out/'stageB_multi_metrics.csv',index=False)
pd.set_option('display.width',250);pd.set_option('display.max_columns',30)
print(D.groupby(['year','model']).mean(numeric_only=True).drop(columns='seed').round(3).to_string())
