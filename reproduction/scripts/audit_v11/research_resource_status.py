"""Explicit resource stops, distinct from numerical/data/algorithm failures."""
class ResourceStop(RuntimeError):pass
class SupervisionLost(RuntimeError):pass
class ProvenanceBlocked(ValueError):pass

def exception_status(exc):
 if isinstance(exc,(ResourceStop,MemoryError)):return 'RESOURCE_STOP'
 if isinstance(exc,ProvenanceBlocked):return 'BLOCKED_PROVENANCE'
 if isinstance(exc,SupervisionLost):return 'INTERRUPTED'
 return 'FAILED'

def worker_outcome(exit_code,result,supervisor_reason=None):
 if supervisor_reason:return 'RESOURCE_STOP'
 if str(result.get('status','')).startswith('BLOCKED_'):return result['status']
 if result.get('status') in ('RESOURCE_STOP','INTERRUPTED'):return result['status']
 if exit_code==0 and result.get('status')=='COMPLETE':return 'COMPLETE'
 return 'FAILED'
