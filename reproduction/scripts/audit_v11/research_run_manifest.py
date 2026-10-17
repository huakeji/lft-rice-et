"""Validate explicit full-fold provenance before a formal training job starts.

A JSON assertion is not source evidence: gate evidence and admission review must
be independently completed. This validator verifies their recorded identities;
it does not infer historical publication semantics from forecast correlations.
"""
import hashlib,json
from research_resource_status import ProvenanceBlocked
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def checked_file(root,record):
 p=(root/record['path']).resolve()
 if not p.is_relative_to(root.resolve()):raise ProvenanceBlocked('Manifest path leaves project')
 if not p.is_file() or digest(p)!=record['sha256']:raise ProvenanceBlocked('Missing or changed manifest artifact')
 return p

def validate(path,root,protocol_sha256,contract_sha256):
 root=Path(root).resolve();path=Path(path).resolve()
 if not path.is_relative_to(root):raise ProvenanceBlocked('Manifest outside project')
 m=json.loads(path.read_text())
 if m.get('status')!='FORMAL_AUDITED':raise ProvenanceBlocked('Conditional/exploratory data cannot enter formal fitting')
 if m.get('protocol_sha256')!=protocol_sha256 or m.get('contract_sha256')!=contract_sha256:raise ProvenanceBlocked('Protocol or source-contract identity mismatch')
 if m.get('outer_year') not in (2018,2019,2020):raise ProvenanceBlocked('Undeclared retrospective fold')
 if m.get('fold_id')!=f"temporal_{m['outer_year']}":raise ProvenanceBlocked('Temporal fold identity mismatch')
 if m.get('target_partition')!='all_seven_target_dates_within_role_years':raise ProvenanceBlocked('Cross-boundary target rule not certified')
 if m.get('admission_cutoff_year')!=m['outer_year']-2:raise ProvenanceBlocked('Admission must use training cutoff')
 review=json.loads(checked_file(root,m['admission_review']).read_text())
 if review.get('status')!='PASS' or review.get('uses_future_coverage') is not False or review.get('cutoff_year')!=m['admission_cutoff_year']:raise ProvenanceBlocked('As-of admission review incomplete')
 excluded=m.get('excluded_stations')
 if not isinstance(excluded,list) or not all(isinstance(s,str) for s in excluded) or len(excluded)!=len(set(excluded)):raise ProvenanceBlocked('Explicit unique exclusion list required')
 items=m.get('shards',[])
 if not items:raise ProvenanceBlocked('No shards')
 if len({s['path'] for s in items})!=len(items):raise ProvenanceBlocked('Duplicate shard')
 shards=[checked_file(root,r) for r in items]
 return m,shards,digest(path)
