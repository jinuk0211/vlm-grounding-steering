"""Independently audit coverage, author calls, extraction and official scoring."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import zipfile

from author_clone_bridge import ROOT, load_authors, verify_clone_sources
import numpy as np
import torch
from vlm_grounding.evaluation import _load_external_module, attach_chair_scores, evaluate_predictions


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def live(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1][0] not in ['Z','X']
    except FileNotFoundError:return False


def audit(output):
    protocol=json.loads((output/'protocol.json').read_text())
    body={k:v for k,v in protocol.items() if k!='sha256'}
    assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==protocol['sha256']
    assert protocol['gpu_uuid']=='GPU-4a6e1799-a39c-86ed-abdf-afad03791c12' and not protocol['other_gpu_fallback']
    hardware_path=output/'hardware-amendment.json'
    hardware=json.loads(hardware_path.read_text()) if hardware_path.exists() else None
    historical_records={}
    worker_devices={}
    if hardware:
        assert hardware['current_gpu_indices']==[2,3] and not hardware['automatic_gpu_fallback']
        assert hardware['historical_gpu_uuid']==protocol['gpu_uuid']
        assert hardware['dtype_and_generation_settings_unchanged'] and hardware['generation_body_identical']
        assert hardware['bf16_visible_devices']==['GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde','GPU-4c593ff4-fcc4-39c9-d7b4-981dabee1cb3']
        assert hardware['fp32_visible_devices']==','.join(hardware['bf16_visible_devices'])
        for f,h in hardware['source_hashes'].items():assert sha(ROOT/f)==h,f
        assert sha(output/'gpu23-preflight.json')==hardware['preflight_sha256']
        assert json.loads((output/'gpu23-preflight.json').read_text())['status']=='passed'
        paused_hardware=json.loads((output/'PAUSED_CHECKPOINT.json').read_text())
        historical_records={r['path']:r['records'] for r in paused_hardware['prediction_files']}
        old_source=(ROOT/'scripts/run_author_clone_worker.py').read_text()
        gpu_source=(ROOT/'scripts/continue_gpu23_worker.py').read_text()
        first='                    key=(task'
        assert old_source[old_source.index(first):old_source.index('                    sink.write')]==gpu_source[gpu_source.index(first):gpu_source.index("                    row['hardware_execution']")]
    for f,h in protocol['source_hashes'].items():assert sha(ROOT/f)==h,f
    amendment=json.loads((output/'execution-amendment.json').read_text())
    assert amendment['original_protocol_unchanged'] and amendment['original_author_functions_unchanged']
    for f,h in amendment['source_hashes'].items():assert sha(ROOT/f)==h,f
    continuation_path=output/'continuation-amendment.json'
    if continuation_path.exists():
        continuation=json.loads(continuation_path.read_text())
        assert continuation['generation_body_identical'] and continuation['preserves_existing_response_bytes']
        for f,h in continuation['source_hashes'].items():assert sha(ROOT/f)==h,f
        assert sha(output/'PAUSED_CHECKPOINT.json')==continuation['checkpoint_sha256']
        assert sha(output/'pre-continuation-jobs.json')==continuation['pre_continuation_jobs_sha256']
        if hardware:assert continuation['hardware_amendment_sha256']==sha(hardware_path)
        paused=json.loads((output/'PAUSED_CHECKPOINT.json').read_text())
        assert paused['status']=='paused_processes_terminated' and not paused['invalid_partial_lines']
        for record in paused['prediction_files']:
            prefix=(output/record['path']).read_bytes()[:record['bytes']]
            assert hashlib.sha256(prefix).hexdigest()==record['sha256'],record['path']
        old=(ROOT/'scripts/run_author_clone_worker.py').read_text()
        new=(ROOT/'scripts/continue_author_clone_worker.py').read_text()
        first,last='                    key=(task','                    sink.write'
        assert old[old.index(first):old.index(last)]==new[new.index(first):new.index(last)]
    assert verify_clone_sources()==protocol['author_sources']
    for author,info in protocol['author_sources'].items():
        for f,h in info['files'].items():assert sha(output/'author-snapshot'/author/f)==h
    for f,h in protocol['manifest_hashes'].items():assert sha(output/'data'/f)==h
    assert sha(output/'cache-provenance.json')==protocol['cache_provenance_sha256']
    for f,h in protocol['cache_hashes'].items():assert sha(output/f)==h
    cache_provenance=json.loads((output/'cache-provenance.json').read_text())
    ancestors={}
    for folder,record in cache_provenance['sources'].items():
        source=Path(folder)/'protocol.json';assert sha(source)==record['protocol_sha256']
        p=record['protocol'];pbody={k:v for k,v in p.items() if k!='sha256'}
        assert hashlib.sha256(json.dumps(pbody,sort_keys=True,separators=(',',':')).encode()).hexdigest()==p['sha256']
        ancestors[p['sha256']]=p
    for job in json.loads((output/'jobs.json').read_text()):
        assert not live(job['pid']),f'Actual worker is still live: {job}'
        state=json.loads((output/('extraction-status.json' if job['extract'] else f"worker-{job['worker']}.json")).read_text())
        assert state['stage']=='complete',state
        if not job['extract']:
            revision=json.loads((output/f"{job['worker']}-revision-validation.json").read_text())
            assert revision['configured_revision']==protocol['model']['revision']==revision['restored_revision_metadata']
            assert not revision['model_weights_or_computation_changed']
            if hardware:
                device=json.loads((output/f"{job['worker']}-device-map.json").read_text())
                assert device['visible_devices']==job['visible_devices']
                assert device['dtype']==job['dtype'] and not device['quantization'] and not device['cpu_or_disk_offload']
                if job['dtype']=='float32':
                    assert device['visible_devices']==hardware['fp32_visible_devices']
                    assert device['parameter_dtypes']==['torch.float32'] and device['parameter_devices']==['cuda:0','cuda:1']
                else:
                    assert device['visible_devices'] in hardware['bf16_visible_devices'] and device['parameter_devices']==['cuda:0']
                worker_devices[job['worker']]=device['visible_devices']
    fit=read(output/'data/fit50.jsonl');assert len(fit)==50 and len({r['group_id'] for r in fit})==50
    fit_hashes={sha(r['image']) for r in fit}
    datasets={b:read(output/'data'/f'{b}.jsonl') for b in ['pope','chair']}
    for bench,rows in datasets.items():
        assert len(rows)==(3000 if bench=='pope' else 500)
        assert len({r['id'] for r in rows})==len(rows)
        assert not fit_hashes & {r['image_sha256'] for r in rows}
        assert all(sha(r['image'])==r['image_sha256'] for r in rows)
    authors=load_authors();journal=read(output/'extraction.jsonl');assert len(journal)==50
    reconstructed={}
    for index,(entry,record) in enumerate(zip(journal,fit)):
        assert entry['index']==index and entry['record']==record
        assert sha(entry['artifact'])==entry['artifact_sha256'] and sha(entry['raw_ig'])==entry['raw_ig_sha256']
        raw=torch.load(entry['raw_ig'],map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in raw.values())
        with np.load(entry['artifact'],allow_pickle=False) as arrays:
            for key in arrays.files:
                assert np.isfinite(arrays[key]).all()
                if key.startswith('hidden_'):reconstructed.setdefault(key,[]).append(arrays[key].copy())
    reconstructed={k:np.stack(v) for k,v in reconstructed.items()}
    with np.load(output/'original-hidden-states.npz',allow_pickle=False) as combined:
        assert set(combined.files)==set(reconstructed)
        assert all(np.array_equal(combined[k],v) for k,v in reconstructed.items())
    artifact_seal=json.loads((output/'grains-artifact.json').read_text())
    for file,key in [('grains-author.pt','artifact_sha256'),('original-hidden-states.npz','hidden_states_sha256'),('extraction.jsonl','journal_sha256')]:
        assert sha(output/file)==artifact_seal[key]
    np.random.seed(42);expected=authors.steering.compute_steering_vector(reconstructed,27,'pca')
    artifact=torch.load(output/'grains-author.pt',map_location='cpu',weights_only=True)
    torch.testing.assert_close(expected,artifact['vector'],rtol=1e-5,atol=1e-6)
    assert torch.isfinite(artifact['vector']).all() and artifact['vector'].norm()>0
    assert json.loads((output/'gemma-ple-api-validation.json').read_text())['status']=='passed'
    grouped={b:{m:{} for m in protocol['methods']} for b in datasets}
    tasks=json.loads((output/'tasks.json').read_text())
    assert len(list((output/'predictions').glob('*.jsonl')))==len(tasks)
    for task in tasks:
        path=output/'predictions'/f"{task['id']}.jsonl";marker=json.loads((output/'claims'/task['id']/'complete.json').read_text())
        assert sha(path)==marker['sha256']
        rows=read(path);references=datasets[task['benchmark']][task['start']:task['stop']]
        assert len(rows)==len(references)==marker['records']
        for row_index,(row,ref) in enumerate(zip(rows,references)):
            assert all(row.get(k)==v for k,v in ref.items())
            assert row['method']==task['method'] and row['dtype']==task['dtype']
            assert row['protocol_sha256']==protocol['sha256']
            preserved=historical_records.get(str(path.relative_to(output)),0)
            if hardware and row_index>=preserved:
                assert row.get('hardware_execution')=='gpu23-resume-20261028'
                assert row['gpu_uuid']==worker_devices[row['worker']]
                expected_devices=hardware['bf16_visible_devices'] if row['dtype']=='bfloat16' else [hardware['fp32_visible_devices']]
                assert row['gpu_uuid'] in expected_devices
            else:assert row['gpu_uuid']==protocol['gpu_uuid']
            assert isinstance(row['prediction'],str) and row['generated_tokens']>0
            if 'adopted_cache_sha256' in row:
                assert row['method'] in ['base','revis','vti']
                cached=output/'cache'/row['method']/f"{row['id'].replace(':','_')}.json"
                assert sha(cached)==row['adopted_cache_sha256']
                original=json.loads(cached.read_text());assert original['prediction']==row['prediction'] and original['generated_tokens']==row['generated_tokens']
                assert row['adopted_from_protocol']==original['protocol_sha256']
                source=ancestors[original['protocol_sha256']]
                for f in ['src/vlm_grounding/backend.py','src/vlm_grounding/baselines.py']:
                    assert source['source_hashes'][f]==protocol['source_hashes'][f]
                assert source['method_parameters'][row['method']]==protocol['method_parameters']['retained'][row['method']]
                if row['method'] in ['revis','vti']:
                    assert source['artifact_hashes'][row['method']]==protocol['retained_artifact_hashes'][row['method']]
            elif row['method'] in ['rudder_clone','grains_clone']:
                d=row['author_diagnostics'];assert d['calls']>0
                if row['method']=='rudder_clone':
                    assert d['active_positions']>0 and d['card_norm']>0
                    assert d['hook_source'].endswith('upstream-author-20261026/RUDDER/Qwen/qwen_methods.py')
                else:assert d['hook_source'].endswith('upstream-author-20261026/GrAInS/src/utils/steering.py')
            bucket=grouped[task['benchmark']][task['method']];assert row['id'] not in bucket;bucket[row['id']]=row
    for bench,methods in grouped.items():
        for method,rows in methods.items():
            assert set(rows)=={r['id'] for r in datasets[bench]}
            methods[method]=[rows[r['id']] for r in datasets[bench]]
    helper_spec=importlib.util.spec_from_file_location('author_rerun_metric_helper',ROOT/'scripts/audit_hallucination_suite.py')
    helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
    data=ROOT/'data/hallucination-suite';cache=json.loads((data/'chair-scorer-state.json').read_text())
    assert cache['built_with_unmodified_constructor']
    scorer_source=ROOT/'upstream/chair/chair.py';assert sha(scorer_source)==cache['scorer_sha256']
    assert sha(data/'sources/annotations_trainval2014.zip')==protocol['dataset_provenance']['sources']['coco_annotations']['sha256']
    with zipfile.ZipFile(data/'sources/annotations_trainval2014.zip') as zipped:
        for name,digest in cache['annotation_hashes'].items():
            assert sha(data/'annotations'/name)==digest
            with zipped.open('annotations/'+name) as member:
                hasher=hashlib.sha256()
                while chunk:=member.read(1024*1024):hasher.update(chunk)
                assert hasher.hexdigest()==digest
    module=_load_external_module(scorer_source,'author_rerun_chair')
    scorer=module.CHAIR.__new__(module.CHAIR);scorer.__dict__.update(cache['state'])
    scorer.imid_to_objects={int(k):set(v) for k,v in cache['state']['imid_to_objects'].items()}
    metrics={};uncertainty={};csv_rows=[]
    for bench,methods in grouped.items():
        folder=output/bench;folder.mkdir(exist_ok=True);metrics[bench]={}
        for method,rows in methods.items():
            if bench=='pope':
                official=read(data/'sources/coco_pope_adversarial.json')
                assert all(r['metadata']['original_question']==q['text'] and r['label']==q['label'] for r,q in zip(rows,official))
                assert sha(data/'sources/coco_pope_adversarial.json')==protocol['dataset_provenance']['sources']['coco_pope_adversarial.json']['sha256']
                value=helper.pope_metrics(rows);helper.verify_upstream_pope(rows,folder,method,'adversarial',value)
                value['strict']=evaluate_predictions(rows)
            else:
                captions=[{'image_id':r['metadata']['image_id'],'caption':r['prediction']} for r in rows]
                path=folder/f'{method}.captions.json';save(path,captions)
                scored=scorer.compute_chair(str(path),'image_id','caption');save(folder/f'{method}.chair.official.json',scored)
                rows=attach_chair_scores(rows,scored);methods[method]=rows;value=evaluate_predictions(rows)
                assert math.isclose(value['CHAIR_s'],scored['overall_metrics']['CHAIRs'])
                assert math.isclose(value['CHAIR_i'],scored['overall_metrics']['CHAIRi'])
            value.update(at_token_cap=sum(r['generated_tokens']>=r['max_new_tokens'] for r in rows),
                         empty_outputs=sum(not r['prediction'].strip() for r in rows),
                         mean_generated_tokens=sum(r['generated_tokens'] for r in rows)/len(rows))
            metrics[bench][method]=value;helper.write_jsonl(folder/f'{method}.jsonl',rows)
            csv_rows.append({'benchmark':bench,'method':method,'matched_base':protocol['comparisons'].get(method),
                             **{k:v for k,v in value.items() if not isinstance(v,dict)}})
        uncertainty[bench]={
            'bf16':helper.intervals({m:methods[m] for m in ['base','revis','vti','rudder_clone']},bench),
            'fp32':helper.intervals({'base':methods['base_fp32'],'grains_clone':methods['grains_clone']},bench)}
    report={'status':'verified_complete','scope':protocol['scope'],'protocol':protocol,'metrics':metrics,'uncertainty':uncertainty,
        'audit':{'actual_workers_exited':True,'all_21000_responses_verified':True,'official_scorers_used':True,
                 'original_author_sources_unmodified':True,'source_input_artifact_hashes_verified':True,
                 'fit50_raw_ig_and_hidden_and_pca_independently_verified':True,'no_old_rudder_or_grains_responses_reused':True,
                 'development_test_image_hash_overlap':0,'matched_precision_baselines':True},
        'execution_amendment':amendment,'execution_amendment_sha256':sha(output/'execution-amendment.json'),
        'analysis_hashes':{'auditor':sha(__file__)},'output':str(output)}
    if hardware:
        report['hardware_amendment']=hardware
        report['hardware_amendment_sha256']=sha(hardware_path)
        report['audit']['historical_response_bytes_preserved']=True
        report['audit']['only_user_selected_gpu2_gpu3_used_after_resume']=True
        report['hardware_note']='Historical GPU0 outputs retained; remaining CHAIR generated on GPU2/GPU3. Precision and model settings retained; hardware differs across the resumed cohort.'
    save(output/'report.json',report)
    columns=list(dict.fromkeys(k for row in csv_rows for k in row))
    with (output/'metrics.csv').open('w',newline='') as sink:
        w=csv.DictWriter(sink,fieldnames=columns);w.writeheader();w.writerows(csv_rows)
    md=['# Gemma E4B: RUDDER·GrAInS 원본 clone 재평가','',
        'RUDDER·GrAInS는 고정 커밋의 원저자 함수를 호출했다. REVIS·VTI는 이전 서버 구현과 벡터를 유지했다. '+('10/27 GPU0의 저장 응답을 보존하고, 사용자 요청으로 10/28 남은 CHAIR를 GPU2·3에서 재개했다. BF16은 각 GPU의 복제 모델, FP32는 두 GPU로 분산한 동일 FP32 모델이다. 같은 정밀도·설정을 유지했지만 저장 분량과 재개 분량의 하드웨어는 다르다.' if hardware else 'GPU0 한 장에서 실행했다.'),'',
        '| 방법 | 정밀도 | POPE Acc ↑ | POPE F1 ↑ | CHAIR_s ↓ | CHAIR_i ↓ | 객체 recall ↑ |',
        '|---|---|---:|---:|---:|---:|---:|']
    for method in protocol['methods']:
        p,c=metrics['pope'][method],metrics['chair'][method]
        dtype='FP32' if method in ['base_fp32','grains_clone'] else 'BF16'
        md.append(f"| {method} | {dtype} | {p['accuracy']*100:.2f} | {p['f1']*100:.2f} | {c['CHAIR_s']*100:.2f} | {c['CHAIR_i']*100:.2f} | {c['object_recall']*100:.2f} |")
    md+=['','표 수치는 %. GrAInS는 Base FP32와, 나머지는 Base BF16과 비교한다.',
         '', 'POPE는 COCO adversarial 전체 3,000문항, CHAIR는 COCO val2014 고정 500장이다. greedy·배치1·문항별 seed42, POPE 최대128토큰·CHAIR 최대512토큰을 유지했다.',
         '', 'Gemma 입력·모듈·반환형 연결은 별도 adapter다. 원저자 checkpoint/환경 전체 재현을 뜻하지 않는다. GrAInS의 공개 token-only IG/ablation과 sequence축 norm, 10배 계수를 보존했다. Gemma PLE 토큰 식별 입력은 IG 경로에서 고정했다. 전체 시각 경로의 IG라고 표현하지 않는다.',
         '', 'GrAInS 개발50장으로 벡터를 새로 추출했고 테스트와 이미지 해시 중복0개다. POPE 공식 evaluate.py와 CHAIR 원본 Python3 포트를 실행해 채점했다. 10,000회 이미지 단위 paired bootstrap은 report.json에 보관했다.',
         '', '다른 POPE 유형과 MMHal 추가 생성·공식 채점은 보류했다. 이전 RUDDER·GrAInS 결과는 새 집계에 사용하지 않았다.', '']
    (output/'REPORT_KO.md').write_text('\n'.join(md))
    print(json.dumps({'status':'verified_complete','metrics':metrics,'output':str(output)},allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
