"""Approved bounded exploratory run with cooperative memory/time controls."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='8'
import argparse,copy,gc,json,math,sys,time,traceback
from pathlib import Path
import numpy as np
import psutil
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import joblib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/audit_v11'))
from matched_baselines import Encoder,Net
from pft_matched_adapter import PFTAdapter


class BudgetStop(RuntimeError):pass


class Monitor:
    def __init__(self,out,seconds):
        self.out=out;self.started=time.monotonic();self.seconds=seconds;self.last=0;self.threads=8
        self.proc=psutil.Process();self.peak=0;self.low_count=0
    def check(self,force=False):
        now=time.monotonic()
        if not force and now-self.last<3:return
        self.last=now;mem=psutil.virtual_memory();rss=self.proc.memory_info().rss;self.peak=max(self.peak,rss)
        self.threads=2 if mem.available<1.5e9 else 4 if mem.available<2.5e9 else 8
        torch.set_num_threads(self.threads)
        self.low_count=self.low_count+1 if mem.available<.75e9 else 0
        used=sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        event={'elapsed_seconds':now-self.started,'rss':rss,'system_available':mem.available,'system_memory_percent':mem.percent,
            'cpu_threads':self.threads,'cpu_percent_process':self.proc.cpu_percent(),'artifact_bytes':used}
        with (self.out/'resources.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        if rss>8e9:raise BudgetStop('process RSS >8GB')
        if used>2e9:raise BudgetStop('run artifacts >2GB')
        if now-self.started>self.seconds:raise BudgetStop('wall-clock budget reached')
        if self.low_count>=3:raise BudgetStop('system available memory <0.75GB on3 checks')


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def evaluate(model,x,ids,ym,ys,monitor):
    model.eval();pred=[]
    with torch.no_grad():
        for begin in range(0,len(ids),256):
            monitor.check();batch=torch.from_numpy(x[ids[begin:begin+256]])
            pred.append(model(batch).numpy()*ys+ym)
    return np.concatenate(pred)


def train_neural(kind,candidate,x,y,tr,va,state,dest,monitor,ym,ys):
    width={'mlp':[128,256],'gru':[64,128],'pft':[64,128]}[kind][candidate]
    torch.manual_seed(0);rng=np.random.default_rng(0)
    model=PFTAdapter(state,7,width) if kind=='pft' else Net(kind,7,x.shape[-1],width).configure_steps(7,width)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    best=math.inf;patience=0;history=[];best_pred=None
    target=((y-ym)/ys).astype('float32')
    for epoch in range(1,31):
        monitor.check(True);model.train();order=rng.permutation(tr);loss_sum=0
        for begin in range(0,len(order),256):
            monitor.check();ids=order[begin:begin+256]
            optimizer.zero_grad(set_to_none=True)
            p=model(torch.from_numpy(x[ids]));loss=((p-torch.from_numpy(target[ids]))**2).mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(ids)
        pred=evaluate(model,x,va,ym,ys,monitor);score=float(np.sqrt(np.mean((y[va]-pred)**2)))
        if not np.isfinite(pred).all():raise ValueError('nonfinite validation prediction')
        history.append({'epoch':epoch,'training_standardized_mse':loss_sum/len(tr),'validation_rmse':score})
        if score<best:
            best=score;patience=0;best_pred=pred.copy()
            torch.save({'state_dict':model.state_dict(),'width':width,'kind':kind,'epoch':epoch,
                'target_mean':ym,'target_scale':ys,'preprocessing':state},dest/'best.pt')
            np.save(dest/'validation_predictions.npy',best_pred,allow_pickle=False)
        else:patience+=1
        atomic_json(dest/'history.json',history)
        print(dest.parent.name,kind,candidate,'epoch',epoch,'val',round(score,6),'threads',monitor.threads,flush=True)
        if patience>=5:break
    return best,{'width':width,'epochs':len(history),'early_stopping_patience':5}


def train_hgb(candidate,x,y,tr,va,dest,monitor,ym,ys):
    count=[200,400][candidate];leaves=[31,63][candidate];models=[];pred=np.empty((len(va),7))
    flat=x.reshape(len(x),-1);xt=flat[tr];xv=flat[va]
    for h in range(7):
        model=HistGradientBoostingRegressor(max_iter=25,max_leaf_nodes=leaves,learning_rate=.08,
            early_stopping=False,random_state=0,warm_start=True)
        for it in range(25,count+1,25):
            monitor.check(True);model.set_params(max_iter=it)
            with threadpool_limits(limits=monitor.threads):model.fit(xt,(y[tr,h]-ym)/ys)
            atomic_json(dest/'progress.json',{'lead':h+1,'iterations':it,'maximum':count})
            print(dest.parent.name,'hgb',candidate,'lead',h+1,'iterations',it,flush=True)
        with threadpool_limits(limits=monitor.threads):pred[:,h]=model.predict(xv)*ys+ym
        models.append(model);joblib.dump(models,dest/'lead_models_partial.joblib')
    assert np.isfinite(pred).all();np.save(dest/'validation_predictions.npy',pred,allow_pickle=False)
    return float(np.sqrt(np.mean((y[va]-pred)**2))),{'iterations':count,'leaves':leaves,'completed_leads':7}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',required=True);ap.add_argument('--out-dir',required=True)
    ap.add_argument('--seconds',type=int,default=7200);args=ap.parse_args()
    out=Path(args.out_dir).resolve();source=Path(args.data).resolve()
    if not out.is_relative_to(ROOT) or not source.is_relative_to(ROOT):raise ValueError('Project-local paths required')
    out.mkdir(parents=True,exist_ok=False);monitor=Monitor(out,args.seconds)
    plan={'status':'AUTHORIZED_RUNNING','authorization':'User2026-09-10:10cores/16GB,80-90%performance,dynamic adjustment',
        'hard_caps':{'seconds':args.seconds,'process_rss':8e9,'artifacts':2e9},'cpu_threads':'8 preferred;4/2 under memory pressure',
        'safety':'Cooperative checks each minibatch and25tree iterations; no OS hard-real-time cap claimed',
        'scientific_status':'Conditional exploration; date semantics and legacy admission unresolved',
        'models':['mlp','gru','pft','hgb'],'variants':['full','temperature_date_geo'],'seed':0,'test_samples':0}
    atomic_json(out/'plan.json',plan);results=[]
    try:
        monitor.check(True)
        data=np.load(source,allow_pickle=False)
        raw=data['x'];cat=data['c'];y=data['y'];split=data['split'];station=data['station'];origin=data['origin']
        assert set(split)=={'train','val'}
        tr=np.where(split=='train')[0];va=np.where(split=='val')[0];ym=float(y[tr].mean());ys=max(float(y[tr].std()),1e-8)
        np.savez_compressed(out/'validation_keys_targets.npz',station=station[va],origin=origin[va],y=y[va])
        for variant in ['full','temperature_date_geo']:
            enc=Encoder().fit(raw[tr],cat[tr],variant);state=enc.state()
            features=len(enc.ni)+sum(len(v)+1 for v in enc.vocab.values())
            estimated=len(raw)*7*features*4
            monitor.check(True)
            if estimated*3>psutil.virtual_memory().available:raise BudgetStop('encoding temporary-memory estimate exceeds current headroom')
            x=enc.transform(raw,cat)
            vd=out/variant;vd.mkdir();atomic_json(vd/'preprocessing.json',state)
            for kind in ['mlp','gru','pft','hgb']:
                for candidate in range(2):
                    monitor.check(True);dest=vd/f'{kind}_{candidate}';dest.mkdir();t=time.monotonic()
                    atomic_json(dest/'status.json',{'status':'RUNNING','model':kind,'candidate':candidate})
                    if kind=='hgb':score,detail=train_hgb(candidate,x,y,tr,va,dest,monitor,ym,ys)
                    else:score,detail=train_neural(kind,candidate,x,y,tr,va,state,dest,monitor,ym,ys)
                    item={'variant':variant,'model':kind,'candidate':candidate,'validation_rmse':score,
                          'elapsed_seconds':time.monotonic()-t,'status':'COMPLETE','detail':detail}
                    atomic_json(dest/'status.json',item);results.append(item);atomic_json(out/'results.json',results);gc.collect()
            del x;gc.collect()
        atomic_json(out/'status.json',{'status':'COMPLETE','candidate_fits':len(results),'peak_rss_observed':monitor.peak})
    except Exception as exc:
        atomic_json(out/'status.json',{'status':'BUDGET_STOP' if isinstance(exc,BudgetStop) else 'FAILED',
            'reason':str(exc),'completed_candidates':len(results),'peak_rss_observed':monitor.peak})
        if not isinstance(exc,BudgetStop):traceback.print_exc()
        return 2
    return 0

if __name__=='__main__':sys.exit(main())
