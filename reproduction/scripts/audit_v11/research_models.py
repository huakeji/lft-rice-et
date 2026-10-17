"""Forecast-only model adapters and explicitly separate training-only teacher.

No adapter reconstructs unavailable observations at inference. TabM is the official
implementation, not a renamed custom MLP. All model configuration is serialized.
"""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from pft_matched_adapter import PFTAdapter
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'.tools/research_route_40'))
sys.path.insert(0,str(R/'scripts'))
from panel_dl import PanelE2ENet

class ForecastModel(nn.Module):
 def __init__(self,kind,state,features,config):
  super().__init__();self.kind=kind;self.config=dict(config);w=config['width'];n=config['layers'];drop=config['dropout']
  if kind in ['pft','cgt']:
   self.net=PFTAdapter(state,7,w);s=self.net.blocks
   size=lambda k:s[k][1]-s[k][0] if k in s else 1
   self.net.native=PanelE2ENet(past_dim=1,future_num_dim=self.net.numeric,weather_n=size('weather'),wind_n=size('wind'),hidden=w,horizon=7,station_n=0,zone_n=size('zone') if 'zone' in s else 0,forecast_only=True,num_layers=n,nhead=config['heads'],dropout=drop,future_encoder='transformer' if kind=='pft' else 'cnn_gru_transformer')
  elif kind=='mlp':
   blocks=[nn.Flatten(),nn.Linear(7*features,w),nn.ReLU(),nn.Dropout(drop)]
   for _ in range(n-1):blocks.extend([nn.Linear(w,w),nn.ReLU(),nn.Dropout(drop)])
   self.net=nn.Sequential(*blocks,nn.Linear(w,7))
  elif kind=='gru':
   self.net=nn.GRU(features,w,num_layers=n,dropout=drop if n>1 else 0,batch_first=True);self.head=nn.Sequential(nn.Dropout(drop),nn.Linear(w,7))
  elif kind=='tabm':
   from tabm import TabM
   self.net=TabM.make(n_num_features=7*features,d_out=7,n_blocks=n,d_block=w,dropout=drop,k=config['k'],arch_type='tabm')
  else:raise ValueError('Unknown neural model: '+kind)
 def forward(self,x):
  if x.ndim!=3 or x.shape[1]!=7:raise ValueError('Expected complete seven-step forecast tensor')
  if self.kind=='gru':return self.head(self.net(x)[1][-1])
  if self.kind=='tabm':return self.net(x.flatten(1)) # [B,k,7]; separate member training loss
  return self.net(x)
 def predict(self,x):
  p=self(x);return p.mean(1) if p.ndim==3 else p

class HistoryTeacher(nn.Module):
 """Training-only CNN-GRU + public-forecast Transformer. Never deployed as student."""
 def __init__(self,state,features,config,history_features,history_days=90):
  super().__init__();self.history_days=history_days;self.history_features=history_features
  self.adapter=PFTAdapter(state,7,config['width']);s=self.adapter.blocks
  size=lambda k:s[k][1]-s[k][0] if k in s else 1
  self.adapter.native=PanelE2ENet(past_dim=history_features,future_num_dim=self.adapter.numeric,weather_n=size('weather'),wind_n=size('wind'),hidden=config['width'],horizon=7,station_n=0,zone_n=size('zone') if 'zone' in s else 0,forecast_only=False,num_layers=config['layers'],nhead=config['heads'],dropout=config['dropout'])
 def forward(self,x,history):
  if history.shape[1:]!=(self.history_days,self.history_features):raise ValueError('Historical window mismatch')
  num,weather,wind,zone=self.adapter.inputs(x)
  return self.adapter.native(history,num,weather,wind,None,zone)

def distillation_loss(prediction,target,teacher=None,alpha=0.):
 if not 0<=alpha<=1:raise ValueError('alpha outside [0,1]')
 if prediction.ndim==3:target=target[:,None,:]
 loss=(prediction-target).square()
 if alpha:
  if teacher is None or not torch.isfinite(teacher).all():raise ValueError('Finite cross-fitted teacher targets required')
  t=teacher.detach();t=t[:,None,:] if prediction.ndim==3 else t
  loss=(1-alpha)*loss+alpha*(prediction-t).square()
 return loss.mean()

def purged_crossfit(origins,eligible,folds=3,history_days=0):
 """Partition by global origin dates, purge every shared target date across stations.
 Only explicitly eligible outer-training rows can appear on either side.
 """
 origins=np.asarray(origins,dtype='datetime64[D]');eligible=np.asarray(eligible,dtype=int)
 dates=np.unique(origins[eligible]);splits=[]
 if len(dates)<folds:raise ValueError('Too few origin dates for cross-fitting')
 for dates_held in np.array_split(dates,folds):
  held=eligible[np.isin(origins[eligible],dates_held)]
  held_days=np.unique((origins[held,None]+np.arange(1,8)).ravel())
  if history_days<0:raise ValueError('Negative history length')
  # Historical ET0/meteorology in teacher inputs must not expose held targets.
  offsets=np.r_[np.arange(-history_days,0),np.arange(1,8)]
  overlaps=np.isin(origins[eligible,None]+offsets,held_days).any(1)
  train=eligible[~overlaps]
  if not len(train) or not len(held):raise ValueError('Purging leaves empty cross-fit fold')
  splits.append((train,held))
 return splits

def assert_history_precedes_issue(history_dates,origins):
 dates=np.asarray(history_dates,dtype='datetime64[D]');origin=np.asarray(origins,dtype='datetime64[D]')
 if dates.ndim!=2 or dates.shape[0]!=len(origin) or np.isnat(dates).any():raise ValueError('Invalid history dates')
 if not (dates<origin[:,None]).all():raise ValueError('History includes issue date or future observation')
 # Date-only comparison is conservative, not proof of original publication clock.
 return True
