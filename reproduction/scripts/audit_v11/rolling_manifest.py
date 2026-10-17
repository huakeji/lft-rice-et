"""Bounded three-site rolling sequence-design audit; no label reading or training."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2]
STATIONS=['57476','57483','57494']
TEST_YEARS=[2017,2018,2019,2020]

def partition(first,last,test_year,train_start=2013):
    first=pd.to_datetime(first);last=pd.to_datetime(last)
    if (first>last).any():raise ValueError('reversed target interval')
    role=np.full(len(first),'excluded',dtype='<U8')
    role[(first>=pd.Timestamp(f'{train_start}-01-01'))&(last<=pd.Timestamp(f'{test_year-2}-12-31'))]='train'
    role[(first>=pd.Timestamp(f'{test_year-1}-01-01'))&(last<=pd.Timestamp(f'{test_year-1}-12-31'))]='val'
    role[(first>=pd.Timestamp(f'{test_year}-01-01'))&(last<=pd.Timestamp(f'{test_year}-12-31'))]='test'
    return role

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out-dir',required=True);a=ap.parse_args()
    out=(R/a.out_dir).resolve()
    if not out.is_relative_to(R):raise ValueError('project-local outputs only')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[];invalid=[];sources={}
    for st in STATIONS:
        p=R/f'data/processed/forecast_domain/forecast_modeling_rows_{st}.csv'
        sources[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
        df=pd.read_csv(p,usecols=['station_code','origin_date','target_date','lead','input_source','forecast_available'],dtype={'station_code':str})
        if set(df.station_code)!={st}:raise ValueError('station identity mismatch')
        if df.duplicated(['station_code','origin_date','lead']).any():raise ValueError('duplicate issue/lead')
        df['origin_date']=pd.to_datetime(df.origin_date);df['target_date']=pd.to_datetime(df.target_date)
        seq=[]
        for origin,g in df.groupby('origin_date',sort=True):
            g=g.sort_values('lead');reason=None
            if list(g.lead)!=list(range(1,8)):reason='incomplete_leads'
            elif not (g.input_source.eq('forecast')&g.forecast_available.eq(1)).all():reason='not_all_public_forecast'
            elif not (g.target_date.to_numpy()==(origin+pd.to_timedelta(g.lead,unit='D')).to_numpy()).all():reason='target_issue_lead_mismatch'
            if reason:invalid.append({'station':st,'origin_date':str(origin.date()),'reason':reason});continue
            seq.append({'station':st,'origin_date':str(origin.date()),'first_target':g.target_date.min(),'last_target':g.target_date.max()})
        s=pd.DataFrame(seq)
        for year in TEST_YEARS:
            f=s.copy();f['test_year']=year;f['train_end']=year-2;f['val_year']=year-1
            f['role']=partition(f.first_target,f.last_target,year)
            rows.append(f)
    result=pd.concat(rows,ignore_index=True);result.to_csv(out/'sequence_membership.csv',index=False)
    pd.DataFrame(invalid,columns=['station','origin_date','reason']).to_csv(out/'excluded_input_sequences.csv',index=False)
    summary=result.groupby(['test_year','station','role']).size().rename('sequences').reset_index();summary.to_csv(out/'counts.csv',index=False)
    # Actual station/date target sets must not overlap within each fold.
    for (_,st),g in result.groupby(['test_year','station']):
        sets={}
        for role in ['train','val','test']:
            subset=g[g.role.eq(role)]
            sets[role]={str(x.date()) for row in subset.itertuples() for x in pd.date_range(row.first_target,row.last_target)}
        assert not (sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test'])
    report={'status':'DESIGN_COUNTS_ONLY_NOT_FORMAL_BACKTEST','test_years':TEST_YEARS,'stations':STATIONS,'source_hashes':sources,
       'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'membership_rows':len(result),'invalid_input_sequences':len(invalid),
       'target_overlap_check':'passed each station/fold','elapsed_seconds':time.monotonic()-start,
       'not_read':['ET0 targets','forecast errors','legacy clean_train_years','validation/test correlations'],
       'limitations':['fixed three-site software sample, not all-site availability assessment','origin+lead equality only confirms processed-table convention, not real publication semantics','canonical labels and release/observation latency remain unresolved','fold-specific station admission and QC must be frozen using only information available then','2017-2020 are retrospective backtests; 2020 has already been examined','later folds may train on past evaluated years only with fixed chronological protocol, never retroactively tune earlier folds']}
    (out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
