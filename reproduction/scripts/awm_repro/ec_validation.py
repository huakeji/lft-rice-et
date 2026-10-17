"""Rice-paddy eddy-covariance validation of the ETc product (ChinaFLUX: Jurong, Changling, Panjin).
Pairing rule fixed in advance: nearest network station (second nearest = sensitivity). QC: drop days with ET<=0.05 mm/d and gap-filled plateaus (>=3 identical consecutive ET values).
Advection days (H<0 or LE>Rn-G) are KEPT in the main analysis because they are real paddy ET (flooded
fields routinely draw sensible heat from the air); screening them is a sensitivity variant (ADVECTION_SCREEN).
Rn-G<=0 days are kept for ET but excluded from closure correction.
Months with EC ET / observed ET0 < 0.3 (physically implausible for a ponded paddy; instrument failure) are dropped.
Season = product in-season months (Kc table) within the EC record. Forecasts are the out-of-sample
LFT+GRU ensemble of the rolling fold whose test year is the EC year (2018 = short-history fold).
Three comparisons: Kc*ET0_obs vs EC (Kc/product definition), Kc*ET0_fcst vs EC (end to end),
Kc*ET0_fcst vs Kc*ET0_obs (forecast step). All sites also with Bowen-ratio closure-corrected ET (daily Rn, G, H, LE: Changling daily file; Jurong from half-hourly ChinaFLUX data, >=40/48 valid; Panjin flux + meteorology daily files)."""
import json,glob,os
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/awm_repro_20261015';B=ROOT/'external/chinaflux';import os
OUT=R/('writing_materials/14_ec_validation'+('_advscreen' if os.environ.get('ADV')=='1' else ''))
KC=ROOT/'external/kc/kc_all_stations_monthly.csv'
SITES={'Jurong':dict(station='58238',alt='58345'),'Changling':dict(station='54041',alt='54049'),'Panjin':dict(station='54470',alt='54471')}
ADVECTION_SCREEN=os.environ.get('ADV')=='1'
def plateau(s):g=(s.diff()!=0).cumsum();return s.groupby(g).transform('size')>=3
def load_ec():
 j=pd.read_excel(B/'2015–2020年句容农田生态系统碳水通量观测数据集/句容水稻日数据xin.xlsx',sheet_name='Sheet1');j['date']=pd.to_datetime(j.Date,errors='coerce')
 J=pd.DataFrame(dict(date=j.date,et=pd.to_numeric(j.ET,errors='coerce'),lai=pd.to_numeric(j.LAI,errors='coerce')))
 hh=pd.read_excel(B/'2015–2020年句容农田生态系统碳水通量观测数据集/ChinaFLUX数据集+句容站通量数据-zyl.xlsx',sheet_name='hour',usecols=['Zone','Rn-w/m2','Gs_1-w/m2','LE-w/m2','Hs-w/m2'])
 hh=hh.replace(-99999,np.nan);hh['date']=pd.to_datetime(hh.Zone).dt.normalize()
 dd=hh.groupby('date').agg(rn=('Rn-w/m2','mean'),g=('Gs_1-w/m2','mean'),le=('LE-w/m2','mean'),h=('Hs-w/m2','mean'),n=('LE-w/m2','count'),nh=('Hs-w/m2','count'),nr=('Rn-w/m2','count'))
 dd=dd[(dd.n>=40)&(dd.nh>=40)&(dd.nr>=40)][['rn','g','le','h']]*0.0864   # daily means (W m-2) -> MJ m-2 d-1, >=40 of 48 half-hours
 J=J.merge(dd.reset_index(),on='date',how='left')
 c=pd.read_excel(B/'2018-2020年松嫩平原长岭盐碱稻田碳水通量观测数据集/2018-2020年吉林长岭日数据.xlsx').iloc[1:];c['date']=pd.to_datetime(c.Time,errors='coerce')
 w2mj=0.0864;num=lambda s:pd.to_numeric(s,errors='coerce')
 C=pd.DataFrame(dict(date=c.date,et=num(c.ET),rn=num(c['净辐射'])*w2mj,g=num(c['土壤热通量'])*w2mj,le=num(c.LE)*w2mj,h=num(c.H)*w2mj,lai=num(c.lai)))
 p=pd.read_excel(B/'盘锦稻田碳水通量2018-2020年数据集/ChinaFLUX数据集-盘锦稻田站通量日数据2018-2020.xlsx').iloc[1:,:8];p.columns=['y','m','d','NEE','GEE','RE','LE','H'];p=p.apply(pd.to_numeric,errors='coerce').dropna(subset=['y'])
 P=pd.DataFrame(dict(date=pd.to_datetime(dict(year=p.y,month=p.m,day=p.d)),le=p.LE,h=p.H,lai=np.nan));P['et']=P['le']/2.45   # LE in MJ m-2 d-1
 pm=pd.read_excel(B/'盘锦稻田碳水通量2018-2020年数据集/ChinaFLUX数据集-盘锦稻田站气象日数据2018-2020.xlsx').iloc[1:]
 pm=pd.DataFrame(dict(date=pd.to_datetime(pm.Date,errors='coerce'),rn=pd.to_numeric(pm['净辐射强度平均值'],errors='coerce')*0.0864,g=pd.to_numeric(pm['第一层土壤热通量平均值'],errors='coerce')*0.0864)).dropna(subset=['date'])
 P=P.merge(pm,on='date',how='left')
 out={};site_closure={'Jurong':True,'Changling':True,'Panjin':False}   # Panjin LE/H distributed after energy-balance correction (median closure 1.00)
 for n,d in [('Jurong',J),('Changling',C),('Panjin',P)]:
  d=d.dropna(subset=['date','et']).sort_values('date').reset_index(drop=True);a=d.rn-d.g
  d['qc_plateau']=plateau(d.et.round(6));d['qc_low']=d.et<=0.05;d['qc_energy']=a<=0
  d['qc_advection']=((d['h']/a)<0)|((d['le']/a)>1)
  d['qc_ok']=~(d.qc_plateau|d.qc_low|((d.qc_energy.fillna(False)|d.qc_advection.fillna(False)) if ADVECTION_SCREEN else False))
  d['closure']=np.where(a>0,(d['le']+d['h'])/a,np.nan)
  ok=np.isfinite(d.closure)&(d.closure>0.3)&(d.closure<1.5)&(d['h']>0)   # Bowen-ratio closure needs H>0 and available energy
  d['et_corr']=np.where(ok&(site_closure[n]),d.et/d.closure,np.nan)
  out[n]=d
 return out
def product(st):
 kc=pd.read_csv(KC,dtype={'station_code':str});kc=kc[(kc.station_code==st)];kc=kc.assign(in_season=kc.month==kc.month_src).set_index('month')
 rows=[]
 for y in (2018,2019,2020):
  fs={m:sorted(glob.glob(str(R/f'formal/{m}_y{y}_s*/outer_predictions.npz'))) for m in ('lft','gru')}
  zs={m:[np.load(f) for f in v] for m,v in fs.items()};base=zs['gru'][0];k=base['station'].astype(str)==st
  P=np.mean([np.mean([z['prediction'][k] for z in zs[m]],0) for m in ('lft','gru')],0);Y=base['target'][k];O=pd.to_datetime(base['origin'][k].astype(str))
  for L in range(7):rows.append(pd.DataFrame(dict(lead=L+1,date=O+pd.Timedelta(days=L),et0_fcst=P[:,L],et0_obs=Y[:,L],fold=y)))
 d=pd.concat(rows);d=d[d.date.dt.year==d.fold];d['kc']=d.date.dt.month.map(kc.Kc);d['in_season']=d.date.dt.month.map(kc.in_season).fillna(False).astype(bool)
 d['etc_fcst']=d.kc*d.et0_fcst.clip(lower=0);d['etc_ref']=d.kc*d.et0_obs;return d
def pooled(D):
 rows=[]
 for (site,ec,comp,lead),g in D[D.pair=='nearest'].groupby(['site','ec','comparison',D.lead.fillna(0)]):
  n=g.n.sum();rows.append(dict(site=site,ec=ec,comparison=comp,lead=int(lead),n=int(n),rmse=float(np.sqrt((g.rmse**2*g.n).sum()/n)),bias=float((g.bias*g.n).sum()/n),mean_pred=float((g.mean_pred*g.n).sum()/n),mean_obs=float((g.mean_obs*g.n).sum()/n)))
 return pd.DataFrame(rows)
def stats(a,b):
 e=a-b;return dict(n=int(len(e)),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(np.abs(e))),bias=float(e.mean()),r=float(np.corrcoef(a,b)[0,1]) if len(e)>2 else np.nan,mean_pred=float(a.mean()),mean_obs=float(b.mean()))
def main():
 OUT.mkdir(parents=True,exist_ok=True);EC=load_ec();rows=[];seas=[];kcr=[];qc=[]
 for site,cfg in SITES.items():
  ec=EC[site];qc.append(dict(site=site,days=len(ec),plateau=int(ec.qc_plateau.sum()),low=int(ec.qc_low.sum()),energy=int(ec.qc_energy.fillna(False).sum()),advection=int(ec.qc_advection.fillna(False).sum()),kept=int(ec.qc_ok.sum()),closure_median=float(ec.closure.median()) if ec.closure.notna().any() else np.nan))
  for role,st in [('nearest',cfg['station']),('second',cfg['alt'])]:
   pr=product(st);m=pr.merge(ec[ec.qc_ok],on='date');m=m[m.in_season]
   l1=m[m.lead==1];ratio=l1.groupby(l1.date.dt.to_period('M')).apply(lambda g:g.et.sum()/g.et0_obs.sum());bad=set(ratio[ratio<0.3].index)
   if role=='nearest':qc[-1]['implausible_months']=','.join(str(b) for b in sorted(bad))
   m=m[~m.date.dt.to_period('M').isin(bad)]
   for y,g in m.groupby('fold'):
    one=g[g.lead==1]
    for var,col in [('EC raw','et'),('EC closure-corrected','et_corr')]:
     if g[col].notna().sum()<10:continue
     h=one.dropna(subset=[col]);rows.append(dict(site=site,station=st,pair=role,year=y,ec=var,comparison='Kc*ET0_obs vs EC',**stats(h.etc_ref,h[col])))
     for L,gl in g.dropna(subset=[col]).groupby('lead'):rows.append(dict(site=site,station=st,pair=role,year=y,ec=var,lead=L,comparison='Kc*ET0_fcst vs EC',**stats(gl.etc_fcst,gl[col])))
    for L,gl in g.groupby('lead'):rows.append(dict(site=site,station=st,pair=role,year=y,ec='-',lead=L,comparison='Kc*ET0_fcst vs Kc*ET0_obs',**stats(gl.etc_fcst,gl.etc_ref)))
    if role=='nearest':
     seas.append(dict(site=site,year=y,days=len(one),ec_sum=float(one.et.sum()),ec_corr_sum=float(one.et_corr.sum()) if one.et_corr.notna().all() else np.nan,etc_ref_sum=float(one.etc_ref.sum()),etc_fcst_L1_sum=float(one.etc_fcst.sum()),etc_fcst_L7_sum=float(g[g.lead==7].etc_fcst.sum()) if (g.lead==7).any() else np.nan))
     t=one.assign(month=one.date.dt.month).groupby('month').agg(kc_product=('kc','first'),ec=('et','sum'),et0=('et0_obs','sum'));t['kc_ec']=t.ec/t.et0;t['site']=site;t['year']=y;kcr.append(t.reset_index())
 D=pd.DataFrame(rows);D.to_csv(OUT/'ec_metrics.csv',index=False);pooled(D).to_csv(OUT/'ec_metrics_pooled_years.csv',index=False);pd.DataFrame(seas).to_csv(OUT/'seasonal_totals.csv',index=False);pd.concat(kcr).to_csv(OUT/'monthly_kc_ec_vs_product.csv',index=False);pd.DataFrame(qc).to_csv(OUT/'qc_summary.csv',index=False)
 for n,d in EC.items():d.to_csv(OUT/f'ec_daily_{n}.csv',index=False)
  
 return D
if __name__=='__main__':
 pd.set_option('display.width',250);D=main()
 a=D[(D.pair=='nearest')&((D.lead.isna())|(D.lead.isin([1,7])))]
 print(a[['site','year','ec','comparison','lead','n','rmse','bias','r','mean_pred','mean_obs']].round(3).to_string(index=False))
 print(pd.read_csv(OUT/'qc_summary.csv').round(3).to_string(index=False));print(pd.read_csv(OUT/'seasonal_totals.csv').round(1).to_string(index=False))
