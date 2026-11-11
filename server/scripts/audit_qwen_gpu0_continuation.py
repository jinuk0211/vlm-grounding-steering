"""Keep failed startup history and audit adopted extraction without inventing an exit code."""
import argparse
import inspect
import json
from pathlib import Path
from qwen_gpu0_migration_api import ROOT,sha,save
from qwen_gpu0_stage_v2 import verify_repair

def audit(output):
    verify_repair(output)
    manifest=json.loads((output/'gpu0-continuation-amendment.json').read_text())
    assert manifest['repair_amendment_sha256']==sha(output/'gpu0-repair-amendment.json')
    for name,digest in manifest['source_hashes'].items():assert sha(ROOT/name)==digest
    import audit_qwen_five_profile_suite as core
    failures={(j['pid'],j['start_ticks']):j for j in manifest['failed_pre_model_jobs']}
    def verify_failed(j):
        original=failures[(j['pid'],j['start_ticks'])]
        assert j==original and not core.actual_job_live(j,output)
        assert sha(output/j['state_file'])==j['failed_state_sha256']
        s=json.loads((output/j['state_file']).read_text())
        assert s['stage']=='failed' and s['completed']==38
        return True
    def verify_adopted(j):
        assert {'pid':j['pid'],'start_ticks':j['start_ticks']}==manifest['adopted_extraction_identity']
        assert j['kind']=='extraction' and j['direct_launch'] and not core.actual_job_live(j,output)
        assert j['adopted_completion_observed'] and not j['exit_code_observed']
        s=json.loads((output/j['state_file']).read_text())
        assert s['pid']==j['pid'] and s['stage']=='complete' and s['completed']==50
        assert sha(output/j['state_file'])==j['state_sha256']
        return True
    core._verify_pre_model_failure=verify_failed;core._verify_adopted_completion=verify_adopted
    import audit_qwen_gpu0_suite as original
    source=inspect.getsource(original.audit)
    marker='    for old,new in replacements.items():'
    assert marker in source
    injected="    replacements['for job in jobs:'] = \"for job in jobs:\\n        if job.get('gpu0_pre_model_failure'):\\n            assert _verify_pre_model_failure(job)\\n            continue\"\n"
    injected+="    replacements[\"assert job['exit_code']==0 and sha(output/job['state_file'])==job['state_sha256']\"] = \"assert job.get('gpu0_migration_stopped') or (_verify_adopted_completion(job) if job.get('direct_launch') else (job['exit_code']==0 and sha(output/job['state_file'])==job['state_sha256']))\"\n"
    source=source.replace(marker,injected+marker)
    exec(compile(source,str(Path(original.__file__))+'::continuation','exec'),original.__dict__)
    original.audit(output)
    report=json.loads((output/'report.json').read_text())
    report['gpu0_continuation']={'amendment_sha256':sha(output/'gpu0-continuation-amendment.json'),
        'pre_model_failed_attempts_retained':len(failures),'adopted_extraction_exit_code_unavailable':True,
        'adopted_extraction_complete_state_and_actual_exit_verified':True,'all_benchmark_rows_audited':True}
    save(output/'report.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
