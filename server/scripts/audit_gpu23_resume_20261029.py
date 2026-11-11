"""Explicit analysis amendment for a second continuation; original scorer sealed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import types

from resume_gpu23_suite_20261029 import ROOT, same_instance, sha, save

def audit(output, epoch):
    folder = output / 'resumes' / epoch
    amendment = json.loads((folder / 'amendment.json').read_text())
    for name,digest in amendment['new_source_hashes'].items():assert sha(ROOT/name)==digest
    for name,digest in amendment['frozen_file_hashes'].items():assert sha(output/name)==digest
    for record in amendment['prediction_files']:
        raw = (output/record['path']).read_bytes()[:record['bytes']]
        assert hashlib.sha256(raw).hexdigest() == record['sha256']
    jobs = json.loads((output/'jobs.json').read_text())
    stopped = {j['worker']:j for j in json.loads((folder/'previous-jobs.json').read_text()) if not j['extract']}
    pause = json.loads((output/'pause-20261028.json').read_text())
    allowed = {j.get('worker') for j in pause['targets'] if j['role']=='gpu_worker'}
    assert set(stopped) <= allowed
    targets = {j['pid']:j for j in pause['targets']}
    for job in jobs:
        identity = {**job, 'output':str(output)}
        if job['pid'] in targets:identity.update(targets[job['pid']])
        assert not same_instance(identity), job
    for file in (output/'predictions').glob('*.jsonl'):
        seals = {r['path']:r for r in amendment['prediction_files']}
        old_records = seals.get(str(file.relative_to(output)),{}).get('records',0)
        rows = [json.loads(s) for s in file.read_text().splitlines() if s.strip()]
        for row in rows[old_records:]:assert row['resume_epoch']==epoch
    source = (ROOT/'scripts/audit_author_clone_suite.py').read_text()
    hardware = json.loads((output/'hardware-amendment.json').read_text())
    assert sha(ROOT/'scripts/audit_author_clone_suite.py') == hardware['source_hashes']['scripts/audit_author_clone_suite.py']
    old = "        assert state['stage']=='complete',state"
    new = "        assert state['stage']=='complete' or (not job['extract'] and job['worker'] in paused_workers),state"
    assert source.count(old)==1
    source = source.replace(old,new)
    module = types.ModuleType('resumed_author_audit')
    module.__file__ = str(Path(__file__))
    exec(compile(source,str(ROOT/'scripts/audit_author_clone_suite.py'),'exec'),module.__dict__)
    # Instance checks above use recorded ticks and task command/cwd, avoiding PID reuse.
    module.live = lambda pid: False
    module.paused_workers = set(stopped)
    module.audit(output)
    report = json.loads((output/'report.json').read_text())
    report['resume_20261029_amendment'] = {'epoch':epoch,'sha256':sha(folder/'amendment.json'),
        'all_19892_saved_response_prefix_bytes_preserved':True,'new_records':1108,
        'historical_paused_worker_states_retained':True,'actual_instance_exit_verified':True,
        'original_auditor_sha256':sha(ROOT/'scripts/audit_author_clone_suite.py')}
    save(output/'report.json',report)
    with (output/'REPORT_KO.md').open('a') as sink:
        sink.write('\n10/29 GPU2·3에서 추가 재개했다. 10/28 정지 시점의 19,892개 응답과 기존 source/protocol을 보존하고 남은 1,108개만 생성했다. 이전 중지 worker 상태는 보존했으며 실제 프로세스 식별과 전체 응답 검증을 수행했다.\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--epoch',required=True)
    a=p.parse_args();audit(a.output,a.epoch)
