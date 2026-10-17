"""Fast national trainer: same model classes, encoder and recipe as the reference rolling trainer (research_models), but the
training partition is encoded once into memory (normalised numerics + category ids; one-hot built
per batch, verified identical to Encoder.transform) and batches can be shuffled across all stations.

--order station  reproduces the reference batch order (one shard after another, batches of one station).
--order global   shuffles sequences across the whole national training partition every epoch.
Roles come from research_fold_data.FoldData (target_offset=-1: lead 1 = issue day); the outer year is
read only after the checkpoint is frozen.
"""
import argparse,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/audit_v11'))
from research_fold_data import FoldData,DevelopmentData,evaluation_batches
from research_models import ForecastModel
from research_legacy_pft_replay import CONFIG as PFT_CONFIG
import matched_baselines,research_fold_data

class CBMModel(torch.nn.Module):
 ids_input=False
 """Bai et al. (2026) CNN-BiLSTM-MHA adapted to public messages only (no observations at inference).
 TFT-style variable selection over variables (10 numerics + weather/wind/zone embeddings), 1x1 conv +
 BiLSTM context over the 7 issued steps, step queries = gated step predictors + position, multi-head
 cross-attention, gated residual network, per-step linear head. Widths are local choices."""
 def __init__(self,n_num,sizes,cfg):
  super().__init__();w=cfg.get('cbm_width',64);d=cfg['dropout'];self.n_num=n_num;self.sizes=sizes;nv=n_num+len(sizes)
  self.num_w=torch.nn.Parameter(torch.randn(n_num,w)*0.1);self.num_b=torch.nn.Parameter(torch.zeros(n_num,w))
  self.emb=torch.nn.ModuleList(torch.nn.Embedding(n,w) for n in sizes)
  self.sel=torch.nn.Sequential(torch.nn.Linear(nv*w,w),torch.nn.ELU(),torch.nn.Linear(w,nv))
  self.conv=torch.nn.Conv1d(w,w,1);self.rnn=torch.nn.LSTM(w,w//2,batch_first=True,bidirectional=True)
  self.pos=torch.nn.Parameter(torch.randn(1,7,w)*0.02);self.att=torch.nn.MultiheadAttention(w,4,dropout=d,batch_first=True)
  self.n1=torch.nn.LayerNorm(w);self.gv=torch.nn.Sequential(torch.nn.Linear(w,w),torch.nn.ELU(),torch.nn.Dropout(d),torch.nn.Linear(w,w))
  self.gg=torch.nn.Sequential(torch.nn.Linear(w,w),torch.nn.Sigmoid());self.n2=torch.nn.LayerNorm(w);self.head=torch.nn.Linear(w,1)
 def forward(self,x):
  num=x[:,:,:self.n_num];vs=[num.unsqueeze(-1)*self.num_w+self.num_b];o=self.n_num
  for k,(e,n) in enumerate(zip(self.emb,self.sizes)):
   i=x[:,:,self.n_num+k].long() if self.ids_input else x[:,:,o:o+n].argmax(-1);o+=n;vs.append(e(i).unsqueeze(2))
  v=torch.cat(vs,2)                                   # [B,7,variables,w]
  a=self.sel(v.flatten(2)).softmax(-1).unsqueeze(-1);z=(v*a).sum(2)
  mem=self.rnn(torch.relu(self.conv(z.transpose(1,2))).transpose(1,2))[0]
  q=z+self.pos;h=self.n1(q+self.att(q,mem,mem,need_weights=False)[0])
  h=self.n2(h+self.gg(h)*self.gv(h));return self.head(h).squeeze(-1)
 def predict(self,x):return self(x)

class LFTModel(torch.nn.Module):
 """Lag-fusion transformer. Each day carries J+1 public issues about the same valid date (current + J
 earlier). Issue tokens = [Tmax, Tmin, rain, dTmax, dTmin, avail] + shared weather/wind text embeddings +
 issue-age embedding; a per-day attention pools issues (unavailable ones masked), day-level context
 (season, geography, lead, DTR, spread, zone) is added, and a transformer runs across the 7 days."""
 def __init__(self,enc,sizes,cfg,n_lags):
  super().__init__();w=cfg['width'];d=cfg['dropout'];J=n_lags;self.J=J;ni=list(enc.ni);pos={c:i for i,c in enumerate(ni)}
  self.cur=[pos[0],pos[1],pos[3]];self.lag=[[pos[10+4*(j-1)+v] for v in range(4)] for j in range(1,J+1)]
  self.pool_mode=cfg.get('pool','attn');self.no_age=bool(cfg.get('no_age',False));self.no_revision=bool(cfg.get('no_revision',False))   # ablation switches (defaults = paper model)
  self.ctx=[pos[c] for c in (2,4,5,6,7,8,9)]+([] if self.no_revision else [i for i,c in enumerate(ni) if c>=10+4*J])
  av=[10+4*(j-1)+3 for j in range(1,J+1)];self.register_buffer('av_m',torch.tensor([enc.mean[pos[c]] for c in av],dtype=torch.float32));self.register_buffer('av_s',torch.tensor([enc.scale[pos[c]] for c in av],dtype=torch.float32))
  ci=list(enc.ci);self.sizes=sizes;self.n_num=len(ni);base={0:{v:i+1 for i,v in enumerate(enc.vocab[0])},1:{v:i+1 for i,v in enumerate(enc.vocab[1])}}
  for k,j in enumerate(ci):   # map every issue's text ids onto the current-issue vocabularies (shared embeddings)
   kind=0 if j==0 or (j>=3 and (j-3)%2==0) else 1 if j!=2 else None
   if kind is not None:self.register_buffer(f'map{k}',torch.tensor([0]+[base[kind].get(v,0) for v in enc.vocab[j]],dtype=torch.long))
  self.ci=ci;self.wemb=torch.nn.Embedding(sizes[0],w);self.vemb=torch.nn.Embedding(sizes[1],w);self.zemb=torch.nn.Embedding(sizes[2],w)
  self.tok=torch.nn.Linear(6,w);self.age=torch.nn.Parameter(torch.randn(J+1,w)*0.02);self.q=torch.nn.Parameter(torch.randn(1,1,w)*0.02)
  self.pool=torch.nn.MultiheadAttention(w,cfg['heads'],dropout=d,batch_first=True);self.n0=torch.nn.LayerNorm(w)
  self.ctxp=torch.nn.Linear(len(self.ctx),w);self.pos=torch.nn.Parameter(torch.randn(1,7,w)*0.02)
  self.enc=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(w,cfg['heads'],4*w,d,batch_first=True,norm_first=True),cfg['layers'])
  self.nq=len(cfg.get('quantiles',[]) or [0]);self.head=torch.nn.Sequential(torch.nn.LayerNorm(w),torch.nn.Linear(w,self.nq))
 ids_input=False
 def ids(self,x):
  o=self.n_num;out=[]
  for k,n in enumerate(self.sizes):
   if self.ids_input:i=x[:,:,self.n_num+k].long()
   else:i=x[:,:,o:o+n].argmax(-1);o+=n
   out.append(getattr(self,f'map{k}')[i] if hasattr(self,f'map{k}') else i)
  return out
 def forward(self,x):
  B=x.shape[0];num=x[:,:,:self.n_num];I=self.ids(x);cur=num[:,:,self.cur]
  toks=[torch.cat([cur,torch.zeros_like(cur[...,:2]),torch.ones_like(cur[...,:1])],-1)];mask=[torch.zeros(B,7,dtype=torch.bool,device=x.device)]
  for j,c in enumerate(self.lag):
   v=num[:,:,c[:3]];a=(num[:,:,c[3]]*self.av_s[j]+self.av_m[j])>0.5
   dv=torch.zeros_like(v[...,:2]) if self.no_revision else v[...,:2]-cur[...,:2]
   toks.append(torch.cat([v,dv,a.unsqueeze(-1).float()],-1));mask.append(~a)
  T=torch.stack(toks,2)                                                    # [B,7,J+1,6]
  wv=torch.stack([self.wemb(I[0])+self.vemb(I[1])]+[self.wemb(I[3+2*j])+self.vemb(I[4+2*j]) for j in range(self.J)],2)
  z=(self.tok(T)+wv+(0 if self.no_age else self.age)).flatten(0,1);m=torch.stack(mask,2).flatten(0,1)
  if self.pool_mode=='attn':pooled=self.pool(self.q.expand(z.shape[0],1,-1)+z[:,:1],z,z,key_padding_mask=m,need_weights=False)[0][:,0]
  else:   # fixed pooling over available issues: uniform mean, or recency weights 2^-age
   wt=(~m).float()*(1.0 if self.pool_mode=='mean' else torch.tensor([2.0**-j for j in range(self.J+1)],device=z.device));wt=wt/wt.sum(1,keepdim=True);pooled=(z*wt.unsqueeze(-1)).sum(1)
  h=self.n0(pooled+z[:,0]).view(B,7,-1)+self.ctxp(num[:,:,self.ctx])+self.zemb(I[2]).mean(1,keepdim=True)+self.pos
  o=self.head(self.enc(h));return o.squeeze(-1) if self.nq==1 else o   # [B,7] or [B,7,Q]
 def predict(self,x):return self(x)

def use_lags(shards_dir,n,phys=False):
 """Expose the first n time-lagged issue blocks of lagged_shards to the legacy Encoder (names appended to NUM/CATS)."""
 m=json.loads((ROOT/shards_dir/'manifest.json').read_text());J=m['lags']
 if not 0<n<=J:raise ValueError(f'--lags must be 1..{J}')
 matched_baselines.NUM.extend(m['extra_numeric']);matched_baselines.CATS.extend(m['extra_categorical'])
 ni=list(range(10))+[10+4*(j-1)+v for j in range(1,n+1) for v in range(4)]+([10+4*J,11+4*J] if n==J else [])
 ci=[0,1,2]+[3+2*(j-1)+v for j in range(1,n+1) for v in range(2)]
 if phys:ni+=m['phys_index']   # deterministic physics covariates (no observations)
 research_fold_data.active=lambda variant:(list(ni),list(ci))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Cache:
 """Normalised numerics and category ids for one role; one-hot expanded per batch."""
 def __init__(self,batches,enc):
  self.enc=enc;X,C,Y,O,S,G=[],[],[],[],[],[]
  self.maps=[{v:i+1 for i,v in enumerate(enc.vocab[j])} for j in enc.ci]
  for g,(x,c,y,o,s) in enumerate(batches):
   X.append(((x[:,:,enc.ni]-enc.mean)/enc.scale).astype('float32'))
   C.append(np.stack([np.vectorize(m.get,otypes=[np.int32])(c[:,:,j],0) for m,j in zip(self.maps,enc.ci)],2))
   Y.append(y.astype('float32'));O.append(o);S.append(s);G.append(np.full(len(y),g))
  self.x=torch.from_numpy(np.concatenate(X));self.c=torch.from_numpy(np.concatenate(C).astype('int32'));self.y=np.concatenate(Y)
  self.origin=np.concatenate(O);self.station=np.concatenate(S);self.group=np.concatenate(G)
  self.sizes=[len(m)+1 for m in self.maps]
 def keep(self,m):
  m=np.asarray(m,bool);mt=torch.from_numpy(m);self.x,self.c=self.x[mt],self.c[mt];self.y,self.origin,self.station,self.group=self.y[m],self.origin[m],self.station[m],self.group[m];return self
 def batch_ids(self,idx):return torch.cat([self.x[idx],self.c[idx].float()],2)   # numerics + raw category ids (embedding models)
 def batch(self,idx):
  c=self.c[idx].long();parts=[self.x[idx]]+[torch.nn.functional.one_hot(c[:,:,k],n).float() for k,n in enumerate(self.sizes)]
  return torch.cat(parts,2)

def order_batches(cache,mode,bs,rng):
 if mode=='global':
  p=rng.permutation(len(cache.y));return [p[i:i+bs] for i in range(0,len(p),bs)]
 out=[]   # legacy: shard order fixed, contiguous chunks per raw batch group
 for g in np.unique(cache.group):
  ids=np.flatnonzero(cache.group==g);out.append(ids[rng.permutation(len(ids))])
 return out

def rmse(e):return float(np.sqrt(np.mean(np.square(e))))

def main(a):
 out=ROOT/a.out_dir;out.mkdir(parents=True,exist_ok=True);t0=time.time();torch.set_num_threads(a.threads)
 if a.lags:use_lags(a.shards_dir,a.lags,a.phys)
 cfg=dict(PFT_CONFIG);cfg.update(json.loads(a.config));cfg['max_epochs']=a.max_epochs
 shards=sorted((ROOT/a.shards_dir).glob('station_*.npz'))
 if a.station_subset:shards=sorted(np.random.default_rng(20261015).choice(shards,a.station_subset,replace=False))   # fixed tuning subset
 sid=lambda q:q.stem.split('_',1)[1];allst={sid(q) for q in shards}
 rd=lambda f:set((ROOT/f).read_text().split()) if f else None
 trs=rd(a.train_stations) or allst;evs=rd(a.eval_stations) or trs
 shards=[q for q in shards if sid(q) in trs|evs];excl=sorted(evs-trs)   # eval-only stations never enter train/val
 fold=FoldData(shards,a.outer_year,excl,-1);data=DevelopmentData(fold)
 if a.init_from:
  import pickle;ini=ROOT/a.init_from;enc=pickle.loads((ini/'encoder.pkl').read_bytes());target=json.loads((ini/'target.json').read_text())
 else:enc,target=data.fit_encoder()
 import pickle;(out/'encoder.pkl').write_bytes(pickle.dumps(enc));(out/'target.json').write_text(json.dumps(target))
 tr=Cache(data.raw_batches('train',cfg['batch_size']),enc);va=Cache(data.raw_batches('val',cfg['batch_size']),enc)
 if a.train_last_days:   # few-shot: keep only the last N days of the training period
  o=tr.origin.astype('datetime64[D]');tr.keep(o>=o.max()-np.timedelta64(a.train_last_days-1,'D'))
 if a.train_window:   # few-shot from a calendar window of the training period (origins inside [start,end])
  w0,w1=(np.datetime64(t) for t in a.train_window.split(','));o=tr.origin.astype('datetime64[D]');tr.keep((o>=w0)&(o<=w1))
 if len(tr.y)==0 or len(va.y)==0:raise ValueError('empty train or validation role')
 # equivalence check against the legacy encoder on the first training batch
 x0,c0,_,_,_=next(data.raw_batches('train',64));ref=enc.transform(x0,c0);mine=Cache([(x0,c0,np.zeros(c0.shape[:2]),np.zeros(len(x0)),np.zeros(len(x0)))],enc).batch(np.arange(len(x0))).numpy()
 if not np.allclose(ref,mine,atol=1e-6):raise ValueError('cached encoding differs from Encoder.transform')
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
 if a.model=='lft' and not a.lags:raise ValueError('lft needs --lags')
 model=CBMModel(len(enc.ni),tr.sizes,cfg) if a.model=='cbm' else LFTModel(enc,tr.sizes,cfg,a.lags) if a.model=='lft' else ForecastModel(a.model,enc.state(),ref.shape[-1],cfg)
 if a.model=='pft' and cfg.get('emb_dim',8)!=8:   # embedding width of weather/wind texts (PanelE2ENet default 8)
  from panel_dl import PanelE2ENet;nt=model.net;b=nt.blocks;size=lambda k:b[k][1]-b[k][0] if k in b else 1
  nt.native=PanelE2ENet(past_dim=1,future_num_dim=nt.numeric,weather_n=size('weather'),wind_n=size('wind'),hidden=cfg['width'],horizon=7,station_n=0,zone_n=size('zone') if 'zone' in b else 0,forecast_only=True,num_layers=cfg['layers'],nhead=cfg['heads'],dropout=cfg['dropout'],future_encoder='transformer',emb_dim=cfg['emb_dim'])
 if a.init_from:model.load_state_dict(torch.load(ROOT/a.init_from/'best.pt'))
 B=lambda c,i:c.batch(i)
 if a.model in ('lft','cbm'):   # embedding models read category ids directly; verified identical to the one-hot path
  i0=np.arange(min(256,len(tr.y)));model.eval()
  with torch.no_grad():o1=model(tr.batch(i0));model.ids_input=True;o2=model(tr.batch_ids(i0))
  if not torch.allclose(o1,o2,atol=1e-5):raise ValueError('id input path differs from one-hot path')
  B=lambda c,i:c.batch_ids(i)
 if a.ft_mode=='affine':   # frozen network + per-lead output scale/shift
  class Affine(torch.nn.Module):
   def __init__(s,m):super().__init__();s.m=m;s.a=torch.nn.Parameter(torch.ones(7));s.b=torch.nn.Parameter(torch.zeros(7))
   def forward(s,x):return s.m(x)*s.a+s.b
   def predict(s,x):return s(x)
  for q in model.parameters():q.requires_grad=False
  model=Affine(model)
 elif a.ft_mode in ('head','norm'):
  for n_,q in model.named_parameters():
   q.requires_grad=n_.startswith('head') if a.ft_mode=='head' else ('norm' in n_ or n_.startswith(('n0.','head.0.')) or n_.endswith('bias'))
 trainable=[q for q in model.parameters() if q.requires_grad]
 opt=torch.optim.Adam(trainable,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
 sch=torch.optim.lr_scheduler.SequentialLR(opt,[torch.optim.lr_scheduler.LinearLR(opt,1e-8,1.0,total_iters=cfg['warmup_epochs']),
  torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['max_epochs']-cfg['warmup_epochs'],eta_min=0.0)],milestones=[cfg['warmup_epochs']])
 yt=torch.from_numpy((tr.y-target['mean'])/target['scale']).float()
 Q=cfg.get('quantiles') or [];qt=torch.tensor(Q,dtype=torch.float32);mid=Q.index(0.5) if Q else None
 if Q and a.model!='lft':raise ValueError('quantile heads implemented for lft only')
 thr=float(np.quantile(tr.y,0.9));wt=torch.from_numpy(1+cfg.get('peak_alpha',0.)*(tr.y>=thr)).float()   # peak weighting (training targets only)
 def lossf(o,y,w):
  if Q:e=y.unsqueeze(-1)-o;return (torch.maximum(qt*e,(qt-1)*e).mean(-1)*w).mean()
  return ((o-y).square()*w).mean()
 point=lambda o:o[...,mid] if Q else o
 best,best_ep,stale,hist=np.inf,None,0,[]
 for ep in range(cfg['max_epochs']):
  if stale>=cfg['patience']:break
  model.train();ls=0.
  for idx in order_batches(tr,a.order,cfg['batch_size'],rng):
   opt.zero_grad(set_to_none=True);loss=lossf(model(B(tr,idx)),yt[idx],wt[idx]);loss.backward();opt.step();ls+=float(loss)*len(idx)
  model.eval();P=[]
  with torch.no_grad():
   for i in range(0,len(va.y),4096):P.append(point(model.predict(B(va,np.arange(i,min(i+4096,len(va.y)))))).numpy())
  v=rmse(np.concatenate(P)*target['scale']+target['mean']-va.y)
  hist.append(dict(epoch=ep,val_rmse=v,train_mse=ls/len(tr.y),lr=opt.param_groups[0]['lr'],seconds=time.time()-t0))
  if v<best:best,best_ep,stale=v,ep,0;torch.save(model.state_dict(),out/'best.pt')
  else:stale+=1
  sch.step();(out/'progress.json').write_text(json.dumps(hist[-1]|dict(best_epoch=best_ep,best_val=best)))
 model.load_state_dict(torch.load(out/'best.pt'));model.eval()
 sel={'selection_stage':'FROZEN_BEFORE_OUTER_EVALUATION'};import itertools
 te=Cache(itertools.chain(evaluation_batches(fold,sel,'network',4096),evaluation_batches(fold,sel,'excluded',4096)),enc);te.keep(np.isin(te.station.astype(str),sorted(evs)))
 def full(c):
  with torch.no_grad():return np.concatenate([model.predict(B(c,np.arange(i,min(i+4096,len(c.y))))).numpy() for i in range(0,len(c.y),4096)])*target['scale']+target['mean']
 RV=full(va);RT=full(te);P=RT[...,mid] if Q else RT
 np.savez_compressed(out/'val_predictions.npz',prediction=(RV[...,mid] if Q else RV).astype('float32'),target=va.y,origin=va.origin,station=va.station,**({'quantiles':RV.astype('float32'),'q':np.array(Q)} if Q else {}))
 E=P-te.y;S=te.station
 np.savez_compressed(out/'outer_predictions.npz',prediction=P.astype('float32'),target=te.y,origin=te.origin,station=S,**({'quantiles':RT.astype('float32'),'q':np.array(Q)} if Q else {}))
 res=dict(phys=a.phys,ft_mode=a.ft_mode,n_trainable=int(sum(q.numel() for q in trainable)),model=a.model,order=a.order,lags=a.lags,station_subset=a.station_subset,train_stations=a.train_stations,eval_stations=a.eval_stations,n_train_stations=len(trs),n_eval_stations=len(evs),init_from=a.init_from,train_last_days=a.train_last_days,train_window=a.train_window,outer_year=a.outer_year,seed=a.seed,shards_dir=a.shards_dir,best_epoch=best_ep,epochs_ran=len(hist),val_rmse=best,
  pooled_rmse=rmse(E),station_mean_rmse=float(np.mean([rmse(E[S==s]) for s in np.unique(S)])),lead_rmse=[rmse(E[:,k]) for k in range(7)],
  sequences=int(len(te.y)),stations=int(len(np.unique(S))),train_sequences=int(len(tr.y)),config=cfg,history=hist,seconds=time.time()-t0,code_sha256=sha(__file__))
 (out/'result.json').write_text(json.dumps(res,indent=1));print(json.dumps({k:v for k,v in res.items() if k!='history'},indent=1))

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--model',default='pft',choices=['pft','gru','mlp','cgt','cbm','lft']);ap.add_argument('--order',default='global',choices=['global','station'])
 ap.add_argument('--shards-dir',default='reports/awm_repro_20261015/aligned_shards');ap.add_argument('--out-dir',required=True)
 ap.add_argument('--outer-year',type=int,default=2020);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--threads',type=int,default=3)
 ap.add_argument('--max-epochs',type=int,default=200);ap.add_argument('--config',default='{}',help='JSON overrides of the legacy PFT recipe')
 ap.add_argument('--station-subset',type=int,default=0,help='train/val/test on a fixed random subset of stations (tuning only)')
 ap.add_argument('--train-stations',default='',help='file with station ids used for train/val (default all)')
 ap.add_argument('--eval-stations',default='',help='file with station ids evaluated in the outer year (default = train stations); eval-only stations are excluded from fitting')
 ap.add_argument('--init-from',default='',help='run dir to fine-tune from (reuses its encoder, target scaling and weights)')
 ap.add_argument('--train-window',default='',help='few-shot: keep training origins in start,end (YYYY-MM-DD,YYYY-MM-DD)')
 ap.add_argument('--train-last-days',type=int,default=0,help='few-shot: keep only the last N days of training origins')
 ap.add_argument('--phys',action='store_true',help='add Ra, pressure and Hargreaves ET0 from forecast temperatures (needs lagged_phys_shards)')
 ap.add_argument('--ft-mode',default='full',choices=['full','head','norm','affine'],help='which parameters to update (fine-tuning)')
 ap.add_argument('--lags',type=int,default=0,help='use n earlier-issue blocks (requires --shards-dir lagged_shards)')
 main(ap.parse_args())
