"""Bounded same-information baseline smoke runner, never a formal research run.

Synthetic mode needs no archive. Archive smoke uses three predeclared sites and
legacy labels, with every artifact marked non-evidential. Formal preflight fails
closed on the data contract and is not a training bypass.
"""
from pathlib import Path
import argparse, copy, hashlib, json, os, resource, sys, time
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(k, '1')
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import sklearn
import joblib
from validate_data_contract import evaluate
from check_asof_admission import evaluate as evaluate_admission
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
from db_io import ddmm_to_decimal
NUM = ['tmax', 'tmin', 'dtr', 'rain_from_weather', 'season_sin', 'season_cos', 'lat', 'lon', 'alt', 'lead']
CATS = ['weather', 'wind', 'zone']
STATIONS = ['57476', '57483', '57494']
VARIANTS = ['full', 'no_weather', 'no_wind', 'temperature_date_geo', 'no_geo_season', 'single_lead']
BUDGET = {'seed': 0, 'candidates_per_model': 2, 'nn_epochs_per_candidate': 3,
          'nn_widths': [16, 32], 'tree_iterations': [16, 32], 'cpu_threads': 1,
          'archive_sequences_per_station': {'train': 64, 'val': 16, 'test': 16}}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def synthetic():
    rng = np.random.default_rng(713)
    n = 144
    x = rng.normal(size=(n, 7, len(NUM))).astype('float32')
    x[:, :, 0] = 20 + 5*x[:, :, 0]; x[:, :, 1] = x[:, :, 0]-abs(5+x[:, :, 1])
    x[:, :, 2] = x[:, :, 0]-x[:, :, 1]; x[:, :, 3] = rng.integers(0, 4, (n, 7))
    x[:, :, 9] = np.arange(1, 8)
    c = np.empty((n, 7, 3), dtype='<U24')
    c[:, :, 0] = rng.choice(['晴', '小雨'], (n, 7)); c[:, :, 1] = rng.choice(['微风', '3-4级'], (n, 7))
    c[:, :, 2] = 'CC'
    y = (.1*x[:, :, 0] + .2*(c[:, :, 0]=='晴') + rng.normal(0, .1, (n, 7))).astype('float32')
    return {'x': x, 'c': c, 'y': y, 'split': np.array(['train']*96+['val']*24+['test']*24),
            'station': np.array(['S'+str(i%3) for i in range(n)]),
            'origin': np.array(['synthetic-'+str(i) for i in range(n)]),
            'source_hashes': {}, 'label_status': 'synthetic_fixture_not_et0_research'}

def archive():
    manifest = R/'results/integrity/panel_manifest_expanded.csv'
    meta = pd.read_csv(manifest, dtype={'station': str}).set_index('station')
    result = {k: [] for k in ['x', 'c', 'y', 'split', 'station', 'origin']}
    sources = {str(manifest.relative_to(R)): sha(manifest)}
    counts = []
    required = ['station_code', 'origin_date', 'target_date', 'lead', 'input_source',
                'forecast_available', 'weather_text', 'wind_text', 'tmax_input',
                'tmin_input', 'rain_signal_input', 'y']
    for st in STATIONS:
        path = R/f'data/processed/forecast_domain/forecast_modeling_rows_{st}.csv'
        sources[str(path.relative_to(R))] = sha(path)
        frame = pd.read_csv(path, usecols=required, dtype={'station_code': str}, keep_default_na=False)
        if set(frame.station_code) != {st}: raise ValueError('station identity mismatch')
        if frame.duplicated(['station_code', 'origin_date', 'lead']).any(): raise ValueError('duplicate sequence key')
        frame['target_date'] = pd.to_datetime(frame.target_date)
        frame['origin_date'] = pd.to_datetime(frame.origin_date)
        # Existing manifests may use a comma/pipe-delimited representation.
        import re
        clean = set(re.findall(r'20\d{2}', str(meta.loc[st, 'clean_train_years'])))
        pools = {'train': [], 'val': [], 'test': []}
        reasons = {'incomplete_or_not_public': 0, 'invalid_numeric': 0, 'outside_split_or_clean_year': 0}
        for origin, g in frame.groupby('origin_date', sort=True):
            g = g.sort_values('lead')
            if list(g.lead) != list(range(1, 8)) or not (g.forecast_available.eq(1) & g.input_source.eq('forecast')).all():
                reasons['incomplete_or_not_public'] += 1; continue
            if not (g.target_date.to_numpy() == (origin + pd.to_timedelta(g.lead, unit='D')).to_numpy()).all():
                raise ValueError('origin/target/lead mismatch; source semantics still unverified')
            yr = g.target_date.dt.year
            if yr.between(2013, 2018).all() and str(origin.year) in clean: split = 'train'
            elif yr.eq(2019).all(): split = 'val'
            elif yr.eq(2020).all(): split = 'test'
            else: reasons['outside_split_or_clean_year'] += 1; continue
            vals = g[['tmax_input', 'tmin_input', 'rain_signal_input', 'y']].apply(pd.to_numeric, errors='coerce').to_numpy()
            if not np.isfinite(vals).all() or (vals[:, 0] < vals[:, 1]).any():
                reasons['invalid_numeric'] += 1; continue
            pools[split].append(g)
        for split, pool in pools.items():
            cap = BUDGET['archive_sequences_per_station'][split]
            if len(pool) < cap: raise ValueError(f'{st}/{split}: fewer than fixed smoke cap')
            # Spread deterministic smoke samples across time; no model-score selection.
            chosen = np.linspace(0, len(pool)-1, cap, dtype=int)
            for idx in chosen:
                g = pool[idx]; doy = g.target_date.dt.dayofyear.to_numpy()
                hi = g.tmax_input.to_numpy(float); lo = g.tmin_input.to_numpy(float)
                lat = ddmm_to_decimal(meta.loc[st, 'lat']); lon = ddmm_to_decimal(meta.loc[st, 'lon'])
                x = np.column_stack([hi, lo, hi-lo, g.rain_signal_input.to_numpy(float),
                     np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365),
                     np.full(7, lat), np.full(7, lon), np.full(7, meta.loc[st, 'alt']), np.arange(1, 8)])
                c = np.column_stack([g.weather_text.astype(str), g.wind_text.astype(str), np.full(7, str(meta.loc[st, 'zone']))])
                for k,v in {'x': x, 'c': c, 'y': g.y.to_numpy(float), 'split': split,
                            'station': st, 'origin': g.origin_date.iloc[0].strftime('%Y-%m-%d')}.items(): result[k].append(v)
            counts.append({'station': st, 'split': split, 'eligible_sequences': len(pool), 'selected': cap, **reasons})
    result = {k: np.asarray(v) for k,v in result.items()}
    # Pandas strings can become object arrays; persist text without pickle.
    for k in ['c', 'split', 'station', 'origin']: result[k] = result[k].astype(str)
    # Label station-date sets must be separated even for overlapping seven-day windows.
    keys = {}
    for split in ['train', 'val', 'test']:
        mask = result['split'] == split
        keys[split] = {(st, str(pd.Timestamp(str(day))+pd.Timedelta(days=h))) for st,day in zip(result['station'][mask], result['origin'][mask]) for h in range(1,8)}
    assert not(keys['train'] & keys['val'] or keys['train'] & keys['test'] or keys['val'] & keys['test'])
    result.update(source_hashes=sources, screening=counts,
                  label_status='legacy_latitude_bug_not_formal; DDMM_confirmed_by_author_station_source; issue_semantics_unverified')
    return result

def active(variant):
    ni = list(range(len(NUM))); ci = list(range(len(CATS)))
    if variant in ['no_weather', 'temperature_date_geo']: ni.remove(3); ci.remove(0)
    if variant in ['no_wind', 'temperature_date_geo']: ci.remove(1)
    if variant == 'no_geo_season': ni = [0,1,2,3,9]; ci = [0,1]
    return ni, ci

class Encoder:
    def fit(self, x, c, variant):
        self.ni, self.ci = active(variant)
        z = x[:, :, self.ni].astype(float)
        self.mean = z.mean(axis=(0,1)); self.scale = z.std(axis=(0,1)); self.scale[self.scale < 1e-8] = 1
        self.vocab = {j: sorted(set(c[:, :, j].ravel())) for j in self.ci}
        return self
    def transform(self, x, c):
        z = [(x[:, :, self.ni]-self.mean)/self.scale]
        for j in self.ci:
            mapping = {v: i+1 for i,v in enumerate(self.vocab[j])}
            ids = np.array([mapping.get(v, 0) for v in c[:, :, j].ravel()]).reshape(c.shape[:2])
            z.append(np.eye(len(mapping)+1, dtype='float32')[ids])
        return np.concatenate(z, axis=2).astype('float32')
    def state(self):
        return {'numeric_names': [NUM[i] for i in self.ni], 'categorical_names': [CATS[i] for i in self.ci],
                'mean': self.mean.tolist(), 'scale': self.scale.tolist(),
                'vocabulary': {CATS[j]: v for j,v in self.vocab.items()}, 'unknown_index': 0,
                'fit_scope': 'training partition only'}

class Net(nn.Module):
    def __init__(self, kind, steps, features, width):
        super().__init__(); self.kind = kind
        if kind == 'mlp': self.body = nn.Sequential(nn.Flatten(), nn.Linear(steps*features, width), nn.ReLU(), nn.Linear(width, steps))
        else: self.gru = nn.GRU(features, width, batch_first=True); self.head = nn.Linear(width, 1)
    def forward(self, x):
        if self.kind == 'mlp': return self.body(x)
        # Bidirectional context is not implied here: sequence comparison uses the
        # whole issued batch for trees/MLP; GRU uses final state for every lead below.
        _, h = self.gru(x)
        return self.head(h[-1]) if self.steps == 1 else self.output(h[-1])
    def configure_steps(self, steps, width):
        self.steps = steps
        if self.kind == 'gru' and steps > 1: self.output = nn.Linear(width, steps)
        return self

def rmse(y, p): return float(np.sqrt(np.mean((y-p)**2)))

def fit_select(kind, x, y, split, model_factory=None):
    if kind not in ['hgb','mlp','gru','pft']: raise ValueError('unknown model kind')
    if kind=='pft' and model_factory is None: raise ValueError('PFT requires explicit adapter factory')
    tr = split=='train'; va = split=='val'
    ym = float(y[tr].mean()); ys = max(float(y[tr].std()), 1e-8)
    z = (y-ym)/ys
    trials=[]; best=None; bestscore=float('inf')
    for candidate in range(2):
        started=time.monotonic(); torch.manual_seed(0); np.random.seed(0)
        if kind=='hgb':
            model=MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=BUDGET['tree_iterations'][candidate],
                max_leaf_nodes=[7,15][candidate], min_samples_leaf=10, learning_rate=.08, early_stopping=False, random_state=0), n_jobs=1)
            model.fit(x[tr].reshape(tr.sum(),-1), z[tr])
            pred=model.predict(x[va].reshape(va.sum(),-1))*ys+ym; score=rmse(y[va],pred); epoch=None
        else:
            width=BUDGET['nn_widths'][candidate]
            model=(model_factory(x.shape[1],x.shape[2],width) if kind=='pft'
                   else Net(kind,x.shape[1],x.shape[2],width).configure_steps(x.shape[1],width))
            opt=torch.optim.Adam(model.parameters(),lr=.001)
            xt=torch.from_numpy(x); yt=torch.from_numpy(z.astype('float32'))
            beststate=None; score=float('inf'); epoch=None
            for ep in range(BUDGET['nn_epochs_per_candidate']):
                model.train()
                for ids in np.array_split(np.random.permutation(np.where(tr)[0]), max(1,int(np.ceil(tr.sum()/64)))):
                    opt.zero_grad(); loss=((model(xt[ids])-yt[ids])**2).mean(); loss.backward(); opt.step()
                model.eval()
                with torch.no_grad(): value=rmse(y[va], model(xt[va]).numpy()*ys+ym)
                if value<score: score=value; beststate=copy.deepcopy(model.state_dict()); epoch=ep+1
            model.load_state_dict(beststate)
        trials.append({'candidate':candidate,'validation_rmse_smoke_only':score,'selected_epoch':epoch,'elapsed_seconds':time.monotonic()-started})
        if score<bestscore: bestscore=score; best=(model,candidate)
    model,candidate=best
    # No test labels have been accessed for fitting or selection.
    if kind=='hgb': pred=model.predict(x.reshape(len(x),-1))*ys+ym
    else:
        model.eval()
        with torch.no_grad(): pred=model(torch.from_numpy(x)).numpy()*ys+ym
    return model,pred,{'kind':kind,'candidate':candidate,'trials':trials,'target_mean':ym,'target_scale':ys,
                       'target_inverse_before_metric':True,'clipping':'none for every method', 'selection':'validation only'}

def run(data, out, variant, model_kinds=('hgb','mlp','gru'), model_factory=None):
    started=time.monotonic(); torch.set_num_threads(1)
    train=data['split']=='train'; enc=Encoder().fit(data['x'][train],data['c'][train],variant)
    x=enc.transform(data['x'],data['c']); y=data['y'].astype('float32'); split=data['split']
    if variant=='single_lead': x=x.reshape(-1,1,x.shape[-1]); y=y.reshape(-1,1); split=np.repeat(split,7)
    out.mkdir(parents=True,exist_ok=False)
    (out/'preprocessing.json').write_text(json.dumps(enc.state(),ensure_ascii=False,indent=2))
    rows=[]; selections={}
    for kind in model_kinds:
        model,pred,selection=fit_select(kind,x,y,split,model_factory=model_factory); selections[kind]=selection
        if kind=='hgb': joblib.dump(model,out/'hgb.joblib')
        else: torch.save(model.state_dict(),out/f'{kind}.pt')
        pred=pred.reshape(-1,7)
        for part in ['val','test']:
            take=data['split']==part
            for idx in np.where(take)[0]:
                for h in range(7): rows.append({'model':kind,'split':part,'station':data['station'][idx],
                    'origin_date':data['origin'][idx],'lead':h+1,'y':float(data['y'][idx,h]),'prediction':float(pred[idx,h]),
                    'status':'SMOKE_ONLY_NOT_RESEARCH_RESULT'})
    pd.DataFrame(rows).to_csv(out/'predictions_smoke_only.csv',index=False)
    (out/'selection.json').write_text(json.dumps(selections,indent=2))
    (out/'run.json').write_text(json.dumps({'variant':variant,'sequences':{k:int((data['split']==k).sum()) for k in ['train','val','test']},
        'feature_tensor_shape':list(x.shape),'elapsed_seconds':time.monotonic()-started,
        'process_peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'label_status':data['label_status'],'purpose':'pipeline smoke; no model ranking or research claim',
        'budget':BUDGET,'runtime':{'python':sys.version,'torch':torch.__version__,'sklearn':sklearn.__version__},
        'source_hashes':data['source_hashes'],'screening':data.get('screening',[]),
        'script_sha256':sha(__file__)},ensure_ascii=False,indent=2))
    return {'variant':variant,'elapsed_seconds':time.monotonic()-started,'outputs':len(rows)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['synthetic','archive-smoke','formal-preflight'],required=True)
    ap.add_argument('--out-dir',required=True);ap.add_argument('--variant',choices=VARIANTS+['all'],default='full')
    ap.add_argument('--contract',default='reports/v11_audit_20260909/source_05/data_contract.json')
    ap.add_argument('--admission');args=ap.parse_args()
    out=(R/args.out_dir).resolve()
    if not out.is_relative_to(R): raise ValueError('output must be project-local')
    if out.exists(): raise FileExistsError('refusing to overwrite output')
    out.mkdir(parents=True)
    contract_path=(R/args.contract).resolve()
    if not contract_path.is_relative_to(R): raise ValueError('contract must be project-local')
    contract=json.loads(contract_path.read_text())
    gate=evaluate(contract,'paired_forecast')
    gate.update(contract_path=str(contract_path.relative_to(R)), contract_sha256=sha(contract_path))
    (out/'preflight.json').write_text(json.dumps(gate,indent=2))
    (out/'planned_budget.json').write_text(json.dumps({'mode':args.mode,'variant':args.variant,**BUDGET},indent=2))
    if args.mode=='formal-preflight':
        admission={'ready':False,'issues':[{'reason':'missing_admission_protocol'}]}
        if args.admission:
            admission_path=(R/args.admission).resolve()
            if not admission_path.is_relative_to(R): raise ValueError('admission must be project-local')
            admission=evaluate_admission(json.loads(admission_path.read_text()))
            admission.update(path=str(admission_path.relative_to(R)),sha256=sha(admission_path))
        (out/'admission_preflight.json').write_text(json.dumps(admission,indent=2))
        (out/'status.json').write_text(json.dumps({'formal_training_started':False,'ready':False,'data_contract_ready':gate['ready'],
            'admission_ready':admission['ready'],
            'remaining':['frozen common sample/version manifest','approved measured full-run budget','same-input PFT integration','multi-seed final protocol']} ,indent=2))
        print('FORMAL RUN NOT STARTED: see preflight.json and status.json');return 2
    data=synthetic() if args.mode=='synthetic' else archive()
    arrays={k:v for k,v in data.items() if isinstance(v,np.ndarray)}
    if any(v.dtype.hasobject for v in arrays.values()): raise ValueError('smoke archive must not require pickle')
    np.savez_compressed(out/'smoke_sample.npz',**arrays)
    variants=VARIANTS if args.variant=='all' else [args.variant]
    summary=[run(data,out/v,v) for v in variants]
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':sys.exit(main())
