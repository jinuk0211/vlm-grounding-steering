"""Full original audit, with explicit GPU0 migration/old stopped-IG evidence."""
import argparse
import inspect
import json
from pathlib import Path

from qwen_gpu0_migration_api import ROOT,GPU0,verify_protocol,sha,save

def audit(output):
    import audit_qwen_five_profile_suite as module
    protocol=verify_protocol(output);amend=json.loads((output/'gpu0-hardware-amendment.json').read_text())
    stop=json.loads(Path(amend['stop_proof']).read_text())
    stopped={(j['pid'],j['start_ticks']) for j in stop['stopped_instances']}
    def state_complete_or_stopped(job):
        state=json.loads((output/job['state_file']).read_text())
        if job.get('gpu0_migration_stopped'):
            assert (job['pid'],job['start_ticks']) in stopped and job['kind']=='extraction'
            assert state['completed']==stop['completed_ig_records']
            assert not module.actual_job_live(job,output)
            return True
        assert state['stage']=='complete'
        return True
    module.verify_run_protocol=verify_protocol;module.GPU0=GPU0;module._state_complete_or_stopped=state_complete_or_stopped
    source=inspect.getsource(module.audit)
    replacements={
        "assert json.loads((output / job['state_file']).read_text())['stage'] == 'complete'":"assert _state_complete_or_stopped(job)",
        "assert device['visible_devices'] == PAIR":"assert device['visible_devices'] == (GPU0 if job.get('gpu0_migration') else PAIR)",
        "device['parameter_devices'] == ['cuda:0', 'cuda:1']":"device['parameter_devices'] == (['cuda:0'] if job.get('gpu0_migration') else ['cuda:0','cuda:1'])",
        "device['visible_devices'] in (GPU2, GPU3)":"device['visible_devices'] in ((GPU0,) if job.get('gpu0_migration') else (GPU2,GPU3))",
        "assert job['exit_code']==0 and sha(output/job['state_file'])==job['state_sha256']":"assert job.get('gpu0_migration_stopped') or (job['exit_code']==0 and sha(output/job['state_file'])==job['state_sha256'])",
        "assert sha(output/job['device_proof'])==job['device_proof_sha256']":"assert job.get('gpu0_migration_stopped') or sha(output/job['device_proof'])==job['device_proof_sha256']",
        "assert row['gpu_uuid'] in (GPU2,GPU3)":"assert row['gpu_uuid'] == GPU0",
    }
    for old,new in replacements.items():assert old in source,old;source=source.replace(old,new)
    exec(compile(source,str(Path(module.__file__))+'::gpu0_migration','exec'),module.__dict__)
    score_source=inspect.getsource(module.score_and_report)
    assert "'only_gpu2_gpu3_used': True" in score_source
    score_source=score_source.replace("'only_gpu2_gpu3_used': True","'only_gpu0_used_for_benchmark_responses': True")
    score_source=score_source.replace('GPU2·3만 사용했다.','벡터 추출 일부는 GPU2·3에서 보존했고, 사용자 요청 후 모든 평가 응답은 GPU0에서 생성했다.')
    exec(compile(score_source,str(Path(module.__file__))+'::gpu0_scoring_note','exec'),module.__dict__)
    module.audit(output)
    report=json.loads((output/'report.json').read_text())
    journal=(output/'extraction.jsonl').read_bytes()
    assert sha(Path(amend['stop_proof']))==amend['stop_proof_sha256']
    import hashlib
    assert hashlib.sha256(journal[:stop['committed_journal_bytes']]).hexdigest()==stop['journal_prefix_sha256']
    entries=[json.loads(s) for s in journal.splitlines() if s.strip()]
    assert len(entries)==50 and all(e['gpu_uuid']==GPU0 for e in entries[stop['completed_ig_records']:])
    report['hardware_amendment']={'effective_gpu_indices':[0],'sha256':sha(output/'gpu0-hardware-amendment.json'),
        'retained_gpu23_ig_records':stop['completed_ig_records'],'all_benchmark_responses_gpu0':True,
        'parent_protocol_hardware_is_historical':True,'new_auditor_sha256':sha(__file__),
        'original_auditor_sha256':sha(ROOT/'scripts/audit_qwen_five_profile_suite.py')}
    save(output/'report.json',report)
    with (output/'REPORT_KO.md').open('a') as stream:
        stream.write('\n사용자 GPU0 전환 요청을 기록한 hardware amendment를 적용했다. 기존 IG38장과 source/protocol을 보존하고 나머지 추출·FP16 사전 검증·모든 응답 생성을 GPU0에서 수행했다. 원저자 수학/고정 데이터/정밀도/512토큰 설정은 유지했다.\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
