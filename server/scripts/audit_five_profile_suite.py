"""Audit five complete profiles and official scorers, with FP16 GrAInS."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import zipfile

import numpy as np
import torch
from author_clone_bridge import ROOT,load_authors,verify_clone_sources
from grains_fp16_worker import verified_protocol,read,save,GPU2,GPU3
from resume_gpu23_suite_20261029 import same_instance
from vlm_grounding.evaluation import _load_external_module,attach_chair_scores,evaluate_predictions

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    protocol=verified_protocol(output);source=Path(protocol['source_output'])
    assert sha(source/'protocol.json')==protocol['source_protocol_file_sha256']
    original=json.loads((source/'protocol.json').read_text())
    assert original['sha256']==protocol['source_protocol_sha256']
    assert protocol['method_dtypes']=={m:'float16' if m=='grains_clone' else 'bfloat16' for m in protocol['methods']}
    for name,digest in protocol['source_hashes'].items():assert sha(output/'source-snapshot'/name)==digest,name
    for name,digest in original['source_hashes'].items():assert sha(ROOT/name)==digest,name
    for method,digest in protocol['retained_artifact_hashes'].items():assert sha(source/'artifacts'/(method+'.pt'))==digest
    assert verify_clone_sources()==protocol['author_sources']
    assert json.loads((output/'adoption.json').read_text())['records']==14000
    datasets={b:read(output/'data'/(b+'.jsonl')) for b in ['pope','chair']}
    fit=read(output/'data/fit50.jsonl');assert len(fit)==50
    fit_hashes={sha(r['image']) for r in fit};assert len(fit_hashes)==50
    for bench,rows in datasets.items():
        assert len(rows)==(3000 if bench=='pope' else 500)
        assert len({r['id'] for r in rows})==len(rows)
        assert not fit_hashes & {r['image_sha256'] for r in rows}
        assert all(sha(r['image'])==r['image_sha256'] for r in rows)
    # Reconstruct the exact existing original-author fit50/PCA, independently.
    authors=load_authors();journal=read(source/'extraction.jsonl');assert len(journal)==50
    hidden={}
    for index,(entry,ref) in enumerate(zip(journal,fit)):
        assert entry['index']==index and entry['record']==ref
        assert sha(entry['artifact'])==entry['artifact_sha256'] and sha(entry['raw_ig'])==entry['raw_ig_sha256']
        raw=torch.load(entry['raw_ig'],map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in raw.values())
        with np.load(entry['artifact'],allow_pickle=False) as arrays:
            for key in arrays.files:
                assert np.isfinite(arrays[key]).all()
                if key.startswith('hidden_'):hidden.setdefault(key,[]).append(arrays[key].copy())
    hidden={k:np.stack(v) for k,v in hidden.items()}
    with np.load(source/'original-hidden-states.npz',allow_pickle=False) as arrays:
        assert set(arrays.files)==set(hidden) and all(np.array_equal(arrays[k],v) for k,v in hidden.items())
    seal=json.loads((source/'grains-artifact.json').read_text())
    assert seal['status']=='verified_complete' and sha(source/'grains-author.pt')==seal['artifact_sha256']
    assert sha(source/'extraction.jsonl')==seal['journal_sha256'] and sha(source/'original-hidden-states.npz')==seal['hidden_states_sha256']
    np.random.seed(42);expected=authors.steering.compute_steering_vector(hidden,27,'pca')
    vector=torch.load(protocol['grains_artifact'],map_location='cpu',weights_only=True)['vector']
    torch.testing.assert_close(expected,vector,rtol=1e-5,atol=1e-6)
    preflight=json.loads((output/'preflight.json').read_text())
    assert preflight['status']=='passed' and preflight['native_logits_finite'] and preflight['activation_dtypes']==['torch.float16']
    jobs={j['worker']:j for j in json.loads((output/'jobs.json').read_text())}
    for name,job in jobs.items():
        assert not same_instance(dict(job,output=str(output))),job
        ws=json.loads((output/('preflight.json' if job['preflight'] else 'worker-'+name+'.json')).read_text());assert ws['stage']=='complete'
        device=json.loads((output/(name+'-device-map.json')).read_text())
        assert device['dtype']=='float16' and device['visible_devices']==job['visible_devices'] in [GPU2,GPU3]
        assert not device['quantization'] and not device['cpu_or_disk_offload']
        assert 'torch.float16' in device['parameter_dtypes'] and 'torch.bfloat16' not in device['parameter_dtypes']
    grouped={b:{m:{} for m in protocol['methods']} for b in datasets}
    tasks=json.loads((output/'tasks.json').read_text());assert all(t['method']!='base_fp32' for t in tasks)
    assert len(list((output/'predictions').glob('*.jsonl')))==len(tasks)
    cached_sources={};adopted=0;fresh=0
    for task in tasks:
        file=output/'predictions'/(task['id']+'.jsonl');rows=read(file)
        marker=json.loads((output/'claims'/task['id']/'complete.json').read_text())
        refs=datasets[task['benchmark']][task['start']:task['stop']]
        assert sha(file)==marker['sha256'] and len(rows)==len(refs)==marker['records']
        for row,ref in zip(rows,refs):
            assert all(row.get(k)==v for k,v in ref.items())
            assert row['protocol_sha256']==protocol['sha256'] and row['dtype']==task['dtype']==protocol['method_dtypes'][task['method']]
            assert row['method']==task['method'] and row['prediction'].strip()
            assert 0<row['generated_tokens']<=ref['max_new_tokens']
            if task['method']=='grains_clone':
                fresh+=1;assert row['worker'] in jobs
                assert row['gpu_uuid']==jobs[row['worker']]['visible_devices'] and row['artifact_sha256']==protocol['grains_artifact_sha256']
                assert row['author_diagnostics']['calls']>0 and row['author_diagnostics']['hook_source'].endswith('upstream-author-20261026/GrAInS/src/utils/steering.py')
            else:
                adopted+=1;assert row['adopted_from_protocol']==original['sha256']
                source_file=row['source_prediction_file']
                assert Path(source_file).parent==source/'predictions'
                if source_file not in cached_sources:
                    assert sha(source_file)==row['source_prediction_file_sha256']
                    cached_sources[source_file]=Path(source_file).read_bytes().splitlines()
                line=cached_sources[source_file][row['source_row_index']]
                assert hashlib.sha256(line).hexdigest()==row['source_row_sha256']
                prior=json.loads(line)
                assert all(row.get(k)==v for k,v in prior.items() if k!='protocol_sha256')
            if task['method']=='rudder_clone':assert row['author_diagnostics']['calls']>0
            bucket=grouped[task['benchmark']][task['method']];assert row['id'] not in bucket;bucket[row['id']]=row
    assert adopted==14000 and fresh==3500
    for bench,methods in grouped.items():
        for method,rows in methods.items():
            assert set(rows)=={r['id'] for r in datasets[bench]}
            methods[method]=[rows[ref['id']] for ref in datasets[bench]]
    spec=importlib.util.spec_from_file_location('five_profile_helpers',ROOT/'scripts/audit_hallucination_suite.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    data=ROOT/'data/hallucination-suite';cache=json.loads((data/'chair-scorer-state.json').read_text())
    assert cache['built_with_unmodified_constructor'] and sha(ROOT/'upstream/chair/chair.py')==cache['scorer_sha256']
    assert sha(data/'sources/annotations_trainval2014.zip')==protocol['dataset_provenance']['sources']['coco_annotations']['sha256']
    with zipfile.ZipFile(data/'sources/annotations_trainval2014.zip') as zipped:
        for name,digest in cache['annotation_hashes'].items():
            assert sha(data/'annotations'/name)==digest
            h=hashlib.sha256()
            with zipped.open('annotations/'+name) as stream:
                while chunk:=stream.read(1024*1024):h.update(chunk)
            assert h.hexdigest()==digest
    module=_load_external_module(ROOT/'upstream/chair/chair.py','five_profile_official_chair')
    scorer=module.CHAIR.__new__(module.CHAIR);scorer.__dict__.update(cache['state'])
    scorer.imid_to_objects={int(k):set(v) for k,v in cache['state']['imid_to_objects'].items()}
    metrics={};intervals={};csv_rows=[]
    for bench,methods in grouped.items():
        folder=output/bench;folder.mkdir(exist_ok=True);metrics[bench]={}
        for method,rows in methods.items():
            if bench=='pope':
                value=helper.pope_metrics(rows);helper.verify_upstream_pope(rows,folder,method,'adversarial',value)
            else:
                captions=folder/(method+'.captions.json');save(captions,[{'image_id':r['metadata']['image_id'],'caption':r['prediction']} for r in rows])
                official=scorer.compute_chair(str(captions),'image_id','caption');save(folder/(method+'.chair.official.json'),official)
                rows=attach_chair_scores(rows,official);methods[method]=rows;value=evaluate_predictions(rows)
                assert math.isclose(value['CHAIR_s'],official['overall_metrics']['CHAIRs']) and math.isclose(value['CHAIR_i'],official['overall_metrics']['CHAIRi'])
            value.update(dtype=protocol['method_dtypes'][method],at_token_cap=sum(r['generated_tokens']>=r['max_new_tokens'] for r in rows),empty_outputs=sum(not r['prediction'].strip() for r in rows))
            metrics[bench][method]=value;helper.write_jsonl(folder/(method+'.jsonl'),rows)
            csv_rows.append({'benchmark':bench,'method':method,**{k:v for k,v in value.items() if not isinstance(v,dict)}})
        intervals[bench]=helper.intervals(methods,bench)
    report={'status':'verified_complete','protocol':protocol,'metrics':metrics,'uncertainty':intervals,
        'audit':{'all_17500_responses_verified':True,'official_scorers_used':True,'original_author_sources_unmodified':True,
            'retained_14000_responses_byte_provenance_verified':True,'all_3500_grains_responses_fresh_fp16':True,
            'original_fit50_ig_hidden_and_pca_verified':True,'no_base_fp32_generated_or_included':True,
            'actual_worker_instances_exited':True,'only_gpu2_gpu3_for_new_generation':True,'development_test_image_hash_overlap':0},
        'precision_note':'GrAInS FP16; other methods BF16. GrAInS versus Base is mixed precision and does not isolate method effect.',
        'output':str(output),'analysis_sha256':sha(__file__)}
    save(output/'report.json',report)
    columns=list(dict.fromkeys(k for row in csv_rows for k in row))
    with (output/'metrics.csv').open('w',newline='') as sink:
        writer=csv.DictWriter(sink,fieldnames=columns);writer.writeheader();writer.writerows(csv_rows)
    md=['# Gemma E4B 전체 평가 — GrAInS FP16','',
        'POPE adversarial 각3,000문항, CHAIR 동일 고정500장. GrAInS는 새 FP16 응답이고 다른 네 방법은 기존 BF16 응답을 검증해 보존했다. Base FP32는 제외했다.','',
        '| 방법 | 정밀도 | POPE Acc ↑ | POPE F1 ↑ | CHAIR_s ↓ | CHAIR_i ↓ | 객체 recall ↑ |',
        '|---|---|---:|---:|---:|---:|---:|']
    for method in protocol['methods']:
        p=metrics['pope'][method];c=metrics['chair'][method]
        label='VTI 자체 구현' if method=='vti' else method
        md.append(f"| {label} | {protocol['method_dtypes'][method]} | {p['accuracy']*100:.2f} | {p['f1']*100:.2f} | {c['CHAIR_s']*100:.2f} | {c['CHAIR_i']*100:.2f} | {c['object_recall']*100:.2f} |")
    md+=['','수치는 %. GrAInS와 Base는 FP16/BF16 정밀도가 달라 방법 효과만 분리한 비교가 아니다.',
         'GrAInS의 기존 원본 IG20·fit50·PCA27 벡터는 FP32로 추출했고 추론 시 FP16으로 변환했다. 추론 dtype 변경에 따른 새 POPE·CHAIR 응답만 사용했다.',
         'RUDDER·GrAInS는 고정 원저자 함수와 Gemma 연결 adapter를 사용한다. REVIS·VTI는 선택한 자체 구현이며 원저자 전체 환경 재현을 뜻하지 않는다.',
         '배치1·greedy·문항별 seed42·이미지448·POPE128/CHAIR512토큰, 공식 scorer 및10,000회 이미지 단위 paired bootstrap을 유지했다.',
         '기존 FP32 결과와 원본 raw 파일은 역사 기록으로 보존하고 새 점수 집계에서 제외했다.','']
    (output/'REPORT_KO.md').write_text('\n'.join(md));assert not torch.cuda.is_initialized()
    print(json.dumps({'status':'verified_complete','metrics':metrics}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
