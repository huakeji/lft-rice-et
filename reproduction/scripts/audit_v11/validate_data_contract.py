"""Check provenance completeness before freezing a new research data version.
This verifies evidence files and declared statuses, not their scientific truth.
"""
from pathlib import Path
import argparse,hashlib,json,sys
R=Path(__file__).resolve().parents[2]
REQUIRED={
 'reference':['observation_units','normalization_lineage','coordinate_encoding','observation_day_boundary','missing_codes','reference_algorithm','negative_target_policy'],
 'paired_forecast':['observation_units','normalization_lineage','coordinate_encoding','observation_day_boundary','missing_codes','reference_algorithm','negative_target_policy','forecast_publication_semantics'],
}
def evaluate(contract,stage,root=R):
 issues=[]
 for key in REQUIRED[stage]:
  item=contract.get('decisions',{}).get(key,{})
  if item.get('status') not in ['verified','specified_policy']:
   issues.append({'field':key,'reason':'unresolved_source_or_policy','status':item.get('status','missing')});continue
  refs=item.get('evidence',[])
  if not refs:issues.append({'field':key,'reason':'missing_evidence'});continue
  for ref in refs:
   p=root/ref['path']
   if not p.resolve().is_relative_to(root.resolve()) or not p.is_file():issues.append({'field':key,'reason':'missing_or_outside_evidence'});continue
   if hashlib.sha256(p.read_bytes()).hexdigest()!=ref.get('sha256'):issues.append({'field':key,'reason':'evidence_changed'})
 return {'stage':stage,'ready':not issues,'issues':issues,'scope':'checks declarations and file identities; not independent certification of source contents'}
def main():
 a=argparse.ArgumentParser();a.add_argument('--contract',default='reports/v11_audit_20260909/source_05/data_contract.json');a.add_argument('--stage',choices=REQUIRED,default='reference');a.add_argument('--require-ready',action='store_true');args=a.parse_args()
 result=evaluate(json.loads((R/args.contract).read_text()),args.stage);print(json.dumps(result,ensure_ascii=False,indent=2));return 2 if args.require_ready and not result['ready'] else 0
if __name__=='__main__':sys.exit(main())
