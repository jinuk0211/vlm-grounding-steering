"""Retain a failed pre-model attempt as history while auditing all real data."""
import argparse
import inspect
import json
from pathlib import Path
from qwen_gpu0_migration_api import ROOT,sha,save
from qwen_gpu0_stage_v2 import verify_repair

def audit(output):
    repair=verify_repair(output)
    import audit_qwen_five_profile_suite as core
    failures={(j['pid'],j['start_ticks']):j for j in repair['failed_pre_model_jobs']}
    def verify_failed(job):
        assert (job['pid'],job['start_ticks']) in failures and not core.actual_job_live(job,output)
        assert sha(output/job['state_file'])==job['failed_state_sha256']
        state=json.loads((output/job['state_file']).read_text())
        assert state['stage']=='failed' and state['completed']==38 and 'Finish Gemma before any Qwen GPU run' in state['error']
        return True
    core._verify_pre_model_failure=verify_failed
    import audit_qwen_gpu0_suite as original
    source=inspect.getsource(original.audit)
    marker='    for old,new in replacements.items():'
    assert marker in source
    added="    replacements['for job in jobs:'] = \"for job in jobs:\\n        if job.get('gpu0_pre_model_failure'):\\n            assert _verify_pre_model_failure(job)\\n            continue\"\n"
    source=source.replace(marker,added+marker)
    exec(compile(source,str(Path(original.__file__))+'::failed_attempt_history_v2','exec'),original.__dict__)
    original.audit(output)
    report=json.loads((output/'report.json').read_text())
    report['gpu0_guard_repair']={'sha256':sha(output/'gpu0-repair-amendment.json'),'failed_before_model_load':True,
        'all_benchmark_rows_still_audited':True,'new_auditor_sha256':sha(__file__)}
    save(output/'report.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
