"""Candidate QC masks and corrected reference targets; not a frozen forecast dataset."""
import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from compute_et0 import _ensure_refet
from db_io import station_metadata_to_decimal
from fao56 import extraterrestrial_radiation

POLICY = {
    'version': 'qc_reference_candidate_1',
    'status': 'diagnostic_candidate_not_frozen',
    'main': 'finite Tmax,Tmin,RH,u10,n; Tmax>=Tmin; Tmin>-237.3 (formula domain only); 0<RH<=100; u10>=0; 0<=n<=astronomical daylight',
    'zero_policy': 'Zero wind and sunshine may be real and remain in main. RH=0 is excluded as ambiguous after importer missing-to-zero conversion, not labelled universally impossible.',
    'tmean': 'Observed mean outside extrema is flagged, not excluded: reference uses (Tmax+Tmin)/2.',
    'sensitivity': {'allow_zero_rh': 'Allow RH=0; all other main conditions fixed.',
        'cap_excess_sunshine': 'Allow n>daylight, cap n/N at1 in radiation; all other main conditions fixed.',
        'positive_wind_sunshine': 'Main subset with u10>0 and n>0; selects windy/sunny conditions, not a cleaner representative population.',
        'nonzero_temperature': 'Main subset excluding exact-zero extrema; can exclude genuine freezing conditions; diagnostic selection only.'},
    'no_imputation': True, 'negative_et0': 'Preserve signed output',
    'source_limits': 'Original missing values cannot all be reconstructed. Daily timestamp semantics remain unresolved. Legacy583 admission cannot certify as-of rolling cohorts.',
    'no_performance_selection': 'Rules defined before this run; no model predictions or losses read.'}


def masks(d, daylight):
    a = d[['tmax','tmin','rh','wind10','sunshine_hours']].to_numpy(float)
    finite = np.isfinite(a).all(axis=1)
    domain = (d.tmax >= d.tmin) & (d.tmin > -237.3)
    base = finite & domain & d.rh.between(0,100) & (d.wind10 >= 0) & (d.sunshine_hours >= 0)
    sun_ok = d.sunshine_hours <= daylight
    m = pd.DataFrame(index=d.index)
    m['nonfinite_required'] = ~finite
    m['invalid_temperature_domain'] = ~domain
    m['rh_zero'] = d.rh == 0
    m['sunshine_exceeds_daylight'] = d.sunshine_hours > daylight
    m['wind_zero'] = d.wind10 == 0
    m['sunshine_zero'] = d.sunshine_hours == 0
    m['tmean_outside_extrema'] = (d.t_mean < d.tmin) | (d.t_mean > d.tmax)
    m['joint_six_zero'] = (d[['t_mean','tmax','tmin','rh','wind10','sunshine_hours']] == 0).all(axis=1)
    m['mask_main'] = base & (d.rh > 0) & sun_ok
    m['mask_allow_zero_rh'] = base & sun_ok
    m['mask_cap_excess_sunshine'] = base & (d.rh > 0)
    m['mask_positive_wind_sunshine'] = m.mask_main & (d.wind10 > 0) & (d.sunshine_hours > 0)
    m['mask_nonzero_temperature'] = m.mask_main & (d.tmax != 0) & (d.tmin != 0)
    return m, np.asarray(base, bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--stations', default='', help='Comma separated smoke stations; default legacy583')
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    if not out.is_relative_to(ROOT): raise ValueError('Project-local output required')
    out.mkdir(parents=True, exist_ok=False)
    (out/'policy.json').write_text(json.dumps(POLICY, ensure_ascii=False, indent=2))
    start = time.monotonic(); refet = _ensure_refet()
    db = ROOT/'data/db/rice_rl_data.db'; stat = db.stat()
    conn = sqlite3.connect(db.as_uri()+'?mode=ro', uri=True); conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA query_only=ON'); conn.execute('BEGIN')
    manifest = pd.read_csv(ROOT/'results/integrity/panel_manifest_expanded.csv', dtype={'station':str})
    stations = sorted(args.stations.split(',')) if args.stations else sorted(manifest.loc[manifest.include_flag==1,'station'])
    frames, summaries = [], []
    for station in stations:
        row = conn.execute('SELECT station_code,latitude,longitude,altitude_m FROM stations WHERE station_code=?',(station,)).fetchone()
        meta = station_metadata_to_decimal(row)
        d = pd.read_sql_query('''SELECT date,t_mean,tmax,tmin,rh,wind10,sunshine_hours FROM historical_weather_daily
            WHERE station_code=? AND date BETWEEN '2013-01-01' AND '2020-12-31' ORDER BY date''',conn,params=[station])
        dates = pd.to_datetime(d.date); doy = dates.dt.dayofyear.to_numpy()
        lookup = {int(day):extraterrestrial_radiation(date.date(),meta['latitude']) for day,date in zip(doy,dates)}
        ra = np.array([lookup[int(day)][0] for day in doy]); daylight = np.array([lookup[int(day)][1] for day in doy])
        m, valid = masks(d,daylight)
        eto = np.full(len(d),np.nan)
        tmax = d.tmax.to_numpy(float)[valid]; tmin = d.tmin.to_numpy(float)[valid]
        es = (0.6108*np.exp(17.27*tmax/(tmax+237.3)) + 0.6108*np.exp(17.27*tmin/(tmin+237.3)))/2
        rs = (.25+.50*np.clip(d.sunshine_hours.to_numpy(float)[valid]/daylight[valid],0,1))*ra[valid]
        if valid.any():
            eto[valid] = refet.Daily(tmin=tmin,tmax=tmax,ea=es*d.rh.to_numpy(float)[valid]/100,
                rs=rs,uz=d.wind10.to_numpy(float)[valid],zw=10.0,elev=meta['altitude_m'],
                lat=meta['latitude'],input_units={'lat':'deg'},doy=doy[valid],method='asce').eto()
        assert np.isfinite(eto[valid]).all()
        m.insert(0,'station',station); m.insert(1,'date',d.date)
        m['et0_refet_candidate'] = eto
        m['year'] = dates.dt.year
        cols=[c for c in m if c not in ['station','date','year','et0_refet_candidate']]
        for year,g in m.groupby('year'):
            summaries.append({'station':station,'year':int(year),'rows':len(g),**{c:int(g[c].sum()) for c in cols},
                'negative_reference_main':int(((g.et0_refet_candidate<0)&g.mask_main).sum())})
        frames.append(m.drop(columns='year'))
    conn.rollback();conn.close()
    full=pd.concat(frames,ignore_index=True)
    assert not full.duplicated(['station','date']).any()
    full.to_parquet(out/'reference_and_masks.parquet',index=False)
    sm=pd.DataFrame(summaries);sm.to_csv(out/'by_station_year.csv',index=False)
    sm.groupby('year').sum(numeric_only=True).to_csv(out/'by_year.csv')
    result={'status':POLICY['status'],'stations':len(stations),'rows':len(full),
        'counts':{c:int(sm[c].sum()) for c in sm if c not in ['station','year','rows']},
        'refet_version':refet.__version__,'elapsed_seconds':time.monotonic()-start,
        'parquet_bytes':(out/'reference_and_masks.parquet').stat().st_size,
        'db_size_mtime_unchanged':(stat.st_size,stat.st_mtime_ns)==(db.stat().st_size,db.stat().st_mtime_ns),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'policy_sha256':hashlib.sha256((out/'policy.json').read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
