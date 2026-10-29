"""Resume the source-sealed GPU0 controller after a pre-model guard failure."""
import argparse
import inspect
import json
import shutil
from pathlib import Path
from qwen_gpu0_migration_api import ROOT,sha,save,verify_protocol

SOURCES=['scripts/qwen_gpu0_stage_v2.py','scripts/run_qwen_gpu0_suite_v2.py','scripts/audit_qwen_gpu0_suite_v2.py']

def run(output):
    verify_protocol(output)
    amendment=json.loads((output/'gpu0-hardware-amendment.json').read_text());migration=Path(amendment['migration'])
    assert not (output/'gpu0-repair-amendment.json').exists()
    old_status=json.loads((output/'extraction-gpu0-status.json').read_text())
    assert old_status['stage']=='failed' and old_status['completed']==38 and 'Finish Gemma before any Qwen GPU run' in old_status['error']
    assert not (output/'grains-extraction-gpu0-device-map.json').exists()
    assert not (migration/'remaining-extraction').exists()
    failed=migration/'failed-gpu0-attempt-v1';failed.mkdir()
    for name in ['status.json','jobs.json','extraction-gpu0-status.json','gpu0-controller.log','gpu0-grains-extraction.log']:
        shutil.copy2(output/name,failed/name)
    jobs=json.loads((output/'jobs.json').read_text());failed_jobs=[]
    for job in jobs:
        if job.get('gpu0_migration') and job['kind']=='extraction':
            job['gpu0_pre_model_failure']=True
            job['state_file']=str((failed/'extraction-gpu0-status.json').relative_to(output))
            job['failed_state_sha256']=sha(failed/'extraction-gpu0-status.json');failed_jobs.append(job)
    assert len(failed_jobs)==1
    save(output/'jobs.json',jobs)
    repair={'reason':'Legacy extraction helper was still checking the superseded six-profile Gemma run; use the completed five-profile guard',
        'hardware_amendment_sha256':sha(output/'gpu0-hardware-amendment.json'),
        'source_hashes':{name:sha(ROOT/name) for name in SOURCES},'failed_pre_model_jobs':failed_jobs,
        'journal_sha256_before_retry':sha(output/'extraction.jsonl'),'saved_ig_records':38,
        'model_and_author_math_unchanged':True,'failure_before_model_load_and_before_new_ig':True}
    save(output/'gpu0-repair-amendment.json',repair)
    for name in SOURCES:
        dest=migration/'repair-source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    import run_qwen_gpu0_suite as original
    source=inspect.getsource(original.run)
    replacements={
        "assert not (output/'gpu0-hardware-amendment.json').exists()":"assert (output/'gpu0-hardware-amendment.json').exists()",
        "save(output/'gpu0-hardware-amendment.json',amendment)":"amendment=json.loads((output/'gpu0-hardware-amendment.json').read_text())",
        "migration/'source-snapshot'/name":"migration/'source-snapshot-v2-parent'/name",
        "migration/'previous-launch.json'":"migration/'previous-launch-v2.json'",
        "scripts/qwen_gpu0_stage_entry.py":"scripts/qwen_gpu0_stage_v2.py",
        "scripts/run_qwen_gpu0_suite.py":"scripts/run_qwen_gpu0_suite_v2.py",
        "scripts/audit_qwen_gpu0_suite.py":"scripts/audit_qwen_gpu0_suite_v2.py",
        "gpu0-controller.log":"gpu0-controller-v2.log",
        "'gpu0-grains-extraction'":"'gpu0-v2-grains-extraction'",
    }
    for old,new in replacements.items():assert old in source,old;source=source.replace(old,new)
    exec(compile(source,str(Path(original.__file__))+'::guard_repair_v2','exec'),original.__dict__)
    original.run(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
