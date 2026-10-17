"""Native PanelE2ENet adapter for the bounded same-information smoke pipeline."""
from pathlib import Path
import sys,argparse,json
import numpy as np
import torch
from torch import nn
from matched_baselines import R,Encoder,synthetic,run,VARIANTS,sha
sys.path.insert(0,str(R/'scripts'))
from panel_dl import PanelE2ENet

class PFTAdapter(nn.Module):
    def __init__(self,state,steps,width):
        super().__init__();self.numeric=len(state['numeric_names']);self.blocks={};offset=self.numeric
        for name in state['categorical_names']:
            n=len(state['vocabulary'][name])+1;self.blocks[name]=(offset,offset+n);offset+=n
        self.features=offset;self.steps=steps
        size=lambda name:self.blocks[name][1]-self.blocks[name][0] if name in self.blocks else 1
        self.native=PanelE2ENet(past_dim=1,future_num_dim=self.numeric,weather_n=size('weather'),wind_n=size('wind'),
          hidden=width,horizon=steps,station_n=0,zone_n=size('zone') if 'zone' in self.blocks else 0,
          forecast_only=True,num_layers=2,nhead=4,dropout=.1,future_encoder='transformer')
    def inputs(self,x):
        if x.shape[1:]!=(self.steps,self.features):raise ValueError('encoded feature layout mismatch')
        def ids(name):
            if name not in self.blocks:return torch.zeros(x.shape[:2],dtype=torch.long,device=x.device)
            a,b=self.blocks[name];return x[:,:,a:b].argmax(dim=-1)
        zone=ids('zone')
        if 'zone' in self.blocks and not torch.equal(zone,zone[:,:1].expand_as(zone)):raise ValueError('zone must be constant within issued sequence')
        return x[:,:,:self.numeric],ids('weather'),ids('wind'),zone[:,0] if 'zone' in self.blocks else None
    def forward(self,x):
        num,weather,wind,zone=self.inputs(x)
        return self.native(None,num,weather,wind,None,zone)

def factory(state):
    def create(steps,features,width):
        model=PFTAdapter(state,steps,width)
        if features!=model.features:raise ValueError('adapter feature count mismatch')
        return model
    return create

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out-dir',required=True);a=ap.parse_args()
    out=(R/a.out_dir).resolve()
    if not out.is_relative_to(R):raise ValueError('project-local output required')
    out.mkdir(parents=True,exist_ok=False);data=synthetic()
    data['source_hashes']={str(p.relative_to(R)):sha(p) for p in [Path(__file__),R/'scripts/panel_dl.py',R/'scripts/audit_v11/matched_baselines.py']}
    summaries=[]
    for variant in VARIANTS:
        state=Encoder().fit(data['x'][data['split']=='train'],data['c'][data['split']=='train'],variant).state()
        summaries.append(run(data,out/variant,variant,model_kinds=('hgb','mlp','gru','pft'),model_factory=factory(state)))
    np.savez_compressed(out/'synthetic_sample.npz',**{k:v for k,v in data.items() if isinstance(v,np.ndarray)})
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    (out/'adapter_protocol.json').write_text(json.dumps({'status':'SYNTHETIC_ONLY_NOT_PAPER_PFT_RESULT','native_class':'scripts/panel_dl.py:PanelE2ENet',
        'numeric':'same train-scaled numeric channels as other models','categories':'recover train-vocabulary IDs from shared one-hot blocks; use native learned embeddings',
        'history_input':False,'station_embedding':False,'heads':4,'layers':2,'dropout':.1,'width_candidates':[16,32],
        'candidate_count':2,'epochs_per_candidate':3,'seed':0,'same_information_not_identical_representation':True,
        'limitations':['smoke architecture sizes not manuscript optimal configuration','not old checkpoint replay or formal retraining','same candidate count does not establish equal compute or sufficient tuning','paired_forecast contract and station admission unresolved']},indent=2))
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
