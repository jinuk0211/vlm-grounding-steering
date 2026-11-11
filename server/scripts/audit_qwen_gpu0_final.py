"""Full Qwen audit including four BF16 replicas and FP16 attention amendment."""
import argparse,hashlib,inspect,json,os
from pathlib import Path
from qwen_gpu0_migration_api import ROOT,GPU0,sha,save,verify_protocol
from qwen_fp16_stable_stage import verify_attention_amendment
from qwen_fp16_attention_guard import verify_arithmetic

def verify_final_sources(output):
    a=json.loads((output/'gpu0-finalization-amendment.json').read_text())
    assert a['protocol_sha256']==verify_protocol(output)['sha256']
    for path,digest in a['source_hashes'].items():
        assert sha(ROOT/path)==digest and sha(output/'finalization-source-snapshot'/path)==digest
    for path,digest in a['amendment_hashes'].items():assert sha(output/path)==digest
    hardware=json.loads((output/'gpu0-hardware-amendment.json').read_text());migration=Path(hardware['migration'])
    for manifest,snapshot in [('gpu0-hardware-amendment.json',migration/'source-snapshot'),
        ('gpu0-repair-amendment.json',migration/'repair-source-snapshot'),
        ('gpu0-continuation-amendment.json',migration/'continuation-source-snapshot')]:
        for path,digest in json.loads((output/manifest).read_text())['source_hashes'].items():
            assert sha(ROOT/path)==digest and sha(snapshot/path)==digest
    bf16=json.loads((output/'gpu0-bf16-priority-amendment.json').read_text())
    assert sha(ROOT/bf16['source_file'])==bf16['source_sha256']
    assert sha(Path(bf16['archive'])/Path(bf16['source_file']).name)==bf16['source_sha256']
    extra=json.loads((output/'gpu0-extra-replicas-amendment.json').read_text())
    assert sha(ROOT/'scripts/run_qwen_gpu0_extra_workers.py')==sha(output/'gpu0-extra-replicas-source.py')==extra['source_sha256']
    fp16=json.loads((output/'fp16-stable-controller-amendment.json').read_text())
    assert sha(ROOT/'scripts/run_qwen_fp16_stable_workers.py')==sha(output/'fp16-stable-controller-source.py')==fp16['controller_source_sha256']
    return a

def audit(output):
    verify_final_sources(output);attention=verify_attention_amendment(output)
    import audit_qwen_five_profile_suite as core
    for path,key,allowed in [('status.json','controller_pid',['bf16_generation_complete_fp16_pending']),
        ('supplemental-status.json','controller_pid',['complete']),('fp16-stable-status.json','controller_pid',['complete'])]:
        state=json.loads((output/path).read_text());assert state['stage'] in allowed,state
        process=Path('/proc')/str(state[key])
        if process.exists():
            fields=(process/'stat').read_text().split(') ',1)[1].split()
            assert fields[0] in ('Z','X') or int(fields[19])!=state['controller_start_ticks'],state
    jobs=[];inputs={};seen=set()
    for name in ['jobs.json','supplemental-jobs.json','fp16-stable-jobs.json']:
        inputs[name]=sha(output/name)
        for job in json.loads((output/name).read_text()):
            identity=(job['pid'],job['start_ticks']);assert identity not in seen
            seen.add(identity);jobs.append(job)
    assert len([j for j in jobs if j['kind']=='generation'])==5
    assert len([j for j in jobs if j['kind']=='generation' and j['dtype']=='float16'])==1
    save(output/'all-final-jobs.json',jobs)
    save(output/'final-job-merge.json',{'source_hashes':inputs,'merged_sha256':sha(output/'all-final-jobs.json'),
        'no_source_job_files_modified':True,'workers':5})
    continuation=json.loads((output/'gpu0-continuation-amendment.json').read_text())
    failed={(j['pid'],j['start_ticks']):j for j in continuation['failed_pre_model_jobs']}
    def verify_failed(j):
        assert j==failed[(j['pid'],j['start_ticks'])] and not core.actual_job_live(j,output)
        assert sha(output/j['state_file'])==j['failed_state_sha256']
        s=json.loads((output/j['state_file']).read_text());assert s['stage']=='failed' and s['completed']==38
        return True
    def verify_adopted(j):
        assert {'pid':j['pid'],'start_ticks':j['start_ticks']}==continuation['adopted_extraction_identity']
        assert j['kind']=='extraction' and j['direct_launch'] and not core.actual_job_live(j,output)
        assert j['adopted_completion_observed'] and not j['exit_code_observed']
        s=json.loads((output/j['state_file']).read_text())
        assert s['pid']==j['pid'] and s['stage']=='complete' and s['completed']==50
        assert sha(output/j['state_file'])==j['state_sha256']
        return True
    cpu=json.loads((output/'fp16-cpu-diagnostic.json').read_text())
    assert cpu['stage']=='first_nonfinite_located' and not cpu['nonfinite_parameters']
    assert cpu['first_module']=='model.language_model.layers.27.self_attn.o_proj'
    assert cpu['benchmark_predictions_written']==0
    numerical=json.loads((output/'gpu0-bf16-priority-amendment.json').read_text())
    archive=Path(numerical['archive'])
    def verify_numerical(j):
        assert j['kind']=='preflight_fp16' and not core.actual_job_live(j,output)
        assert output/j['state_file']==archive/'fp16-preflight.json'
        assert sha(output/j['state_file'])==j['failed_state_sha256']==numerical['failed_preflight_state_sha256']
        state=json.loads((output/j['state_file']).read_text())
        assert state['pid']==j['pid'] and state['stage']==state['status']=='failed'
        assert 'torch.isfinite(tensor).all()' in state['error']
        device=json.loads((archive/'fp16-preflight-device-map.json').read_text())
        assert device['visible_devices']==GPU0 and device['parameter_dtypes']==['torch.float16']
        assert not device['quantization'] and not device['cpu_or_disk_offload']
        return True
    assert len([j for j in jobs if j.get('numerical_preflight_failed')])==1
    stable=json.loads((output/'fp16-stable-preflight.json').read_text())
    assert stable['status']=='passed' and stable['attention_score_dtype']=='torch.float32'
    assert stable['attention_amendment_sha256']==sha(output/'fp16-attention-amendment.json')
    assert any(r['layer']==27 and r['elements']>0 for r in stable['attention_diagnostics']['native_half_score_overflows'])
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen
    assert sha(qwen.__file__)==stable['attention_diagnostics']['transformers_model_source_sha256']
    assert hashlib.sha256(inspect.getsource(qwen.eager_attention_forward).encode()).hexdigest()==stable['attention_diagnostics']['original_attention_source_sha256']
    arithmetic=verify_arithmetic();assert arithmetic==attention['cpu_arithmetic_check']
    fp16_rows=0
    for path in (output/'predictions').glob('*.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['dtype']=='float16':
                assert row['method']=='grains_clone' and row['attention_score_dtype']=='torch.float32'
                assert row['fp16_attention_amendment_sha256']==sha(output/'fp16-attention-amendment.json')
                fp16_rows+=1
    assert fp16_rows==3500
    for j in jobs:
        if j.get('fp16_stable_attention'):
            device=json.loads((output/j['device_proof']).read_text())
            assert device['attention_score_dtype']=='torch.float32' and device['attention_output_dtype']=='torch.float16'
            assert device['fp16_attention_amendment_sha256']==sha(output/'fp16-attention-amendment.json')
    core._verify_pre_model_failure=verify_failed;core._verify_adopted_completion=verify_adopted;core._verify_numerical_failure=verify_numerical
    import audit_qwen_gpu0_suite as original
    source=inspect.getsource(original.audit);marker='    for old,new in replacements.items():';assert marker in source
    extra="    replacements['for job in jobs:'] = \"for job in jobs:\\n        if job.get('gpu0_pre_model_failure'):\\n            assert _verify_pre_model_failure(job)\\n            continue\\n        if job.get('numerical_preflight_failed'):\\n            assert _verify_numerical_failure(job)\\n            continue\"\n"
    extra+="    replacements[\"assert job['exit_code']==0 and sha(output/job['state_file'])==job['state_sha256']\"] = \"assert job.get('gpu0_migration_stopped') or (_verify_adopted_completion(job) if job.get('direct_launch') else (job['exit_code']==0 and sha(output/job['state_file'])==job['state_sha256']))\"\n"
    extra+="    replacements[\"(output / 'jobs.json')\"] = \"(output / 'all-final-jobs.json')\"\n"
    extra+="    replacements[\"(output/'fp16-preflight.json')\"] = \"(output/'fp16-stable-preflight.json')\"\n"
    source=source.replace(marker,extra+marker)
    exec(compile(source,str(Path(original.__file__))+'::full_gpu0_final','exec'),original.__dict__)
    original.audit(output)
    report=json.loads((output/'report.json').read_text())
    report['gpu0_parallelism']={'bf16_replicas':4,'fp16_replicas':1,'job_merge_sha256':sha(output/'final-job-merge.json'),
        'extra_replicas_amendment_sha256':sha(output/'gpu0-extra-replicas-amendment.json')}
    report['fp16_attention_amendment']={'sha256':sha(output/'fp16-attention-amendment.json'),'weights':'float16',
        'QK_mask_softmax_scores':'float32','attention_outputs':'float16','steering_vector':'float32',
        'original_author_steering_unchanged':True,'all_3500_fp16_rows_verified':True,
        'native_fp16_failure_preserved':True,'causal_mask_and_fp32_reference_check':arithmetic}
    report['execution_history']={'pre_model_failures_retained':2,'native_fp16_preflight_failure_retained':1,
        'adopted_extraction_actual_exit_and_complete_state_verified':True,'adopted_extraction_exit_code_unavailable':True,
        'finalization_sources_amendment_sha256':sha(output/'gpu0-finalization-amendment.json')}
    save(output/'report.json',report)
    with (output/'REPORT_KO.md').open('a') as stream:
        stream.write('\nQwen FP16 native eager attention은 steering 이전 layer27 QK에서 overflow가 발생했다. 실패 증명을 보존하고, 모델 가중치·attention 출력은 FP16으로 유지하면서 QK·mask·softmax 점수 계산을 FP32로 승격했다. 이 산술 변경과 FP32 steering 벡터를 명시하므로 내부 연산 전체가 FP16인 재현 결과로 표현하지 않는다. CPU reference·causal mask·실제 FP16 활성값·원본 GrAInS 호출 사전 검증 및 전체3,500개 FP16 응답의 변경 provenance를 감사했다. BF16 네 방법과의 비교는 방법 효과만 분리하지 못한다. 모든 평가 응답은 물리 GPU0에서 BF16 replica4개·GrAInS FP16 replica1개로 생성했다.\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
