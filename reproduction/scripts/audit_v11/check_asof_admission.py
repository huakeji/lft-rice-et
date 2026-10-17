"""Fail closed on missing/late station-admission evidence. Checks declarations, not source truth."""
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json
R=Path(__file__).resolve().parents[2]

def stamp(value):
    t=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if t.tzinfo is None:raise ValueError('timezone required')
    return t

def evaluate(protocol,root=R):
    issues=[];seen=set()
    try:cutoff=stamp(protocol['admission_cutoff'])
    except (KeyError,TypeError,ValueError):return {'ready':False,'issues':[{'reason':'invalid_admission_cutoff'}]}
    items=protocol.get('decisions',[])
    if not items:issues.append({'reason':'no_admission_decisions'})
    for row in items:
        identity=row.get('decision_id');tag={'decision_id':identity}
        if not identity or identity in seen:issues.append({**tag,'reason':'missing_or_duplicate_decision_id'})
        seen.add(identity)
        if row.get('purpose')!='station_admission':issues.append({**tag,'reason':'unrecognized_purpose'})
        if not row.get('rule'):issues.append({**tag,'reason':'missing_rule'})
        for key in ['dependency_latest_time','available_at']:
            try:
                t=stamp(row[key])
                if t>cutoff:issues.append({**tag,'reason':key+'_after_cutoff'})
            except (KeyError,TypeError,ValueError):issues.append({**tag,'reason':key+'_unverified'})
        try:
            if stamp(row['available_at']) < stamp(row['dependency_latest_time']):
                issues.append({**tag,'reason':'availability_precedes_dependency'})
        except (KeyError,TypeError,ValueError):pass
        refs=row.get('evidence',[])
        if not refs:issues.append({**tag,'reason':'missing_evidence'})
        for ref in refs:
            p=(root/ref.get('path','')).resolve()
            if not p.is_relative_to(root.resolve()) or not p.is_file():issues.append({**tag,'reason':'missing_or_external_evidence'});continue
            if hashlib.sha256(p.read_bytes()).hexdigest()!=ref.get('sha256'):issues.append({**tag,'reason':'evidence_hash_mismatch'})
    return {'ready':not issues,'issues':issues,'scope':'declarations and evidence identity only; does not certify observations, release latency or completeness of listed decision dependencies'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);a=ap.parse_args()
    p=(R/a.protocol).resolve()
    if not p.is_relative_to(R):raise ValueError('project-local protocol required')
    result=evaluate(json.loads(p.read_text()));print(json.dumps(result,indent=2));return 0 if result['ready'] else 2
if __name__=='__main__':raise SystemExit(main())
