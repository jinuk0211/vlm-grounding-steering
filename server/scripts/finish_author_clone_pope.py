"""Score the complete POPE cohort as soon as its immutable task markers exist.

The CHAIR model workers keep running. This process is CPU-only and does not
change any model, runtime source, task, prediction or generation configuration.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES']=''
from author_clone_bridge import ROOT, load_authors, verify_clone_sources
import numpy as np
import torch
from audit_hallucination_suite import pope_metrics, verify_upstream_pope, intervals
from vlm_grounding.evaluation import evaluate_predictions


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def save(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)


def verify_extraction(output,protocol):
    authors=load_authors();fit=read(output/'data/fit50.jsonl');journal=read(output/'extraction.jsonl')
    assert len(fit)==len(journal)==50
    hidden={}
    for index,(row,record) in enumerate(zip(journal,fit)):
        assert row['index']==index and row['record']==record
        assert sha(row['artifact'])==row['artifact_sha256'] and sha(row['raw_ig'])==row['raw_ig_sha256']
        ig=torch.load(row['raw_ig'],map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in ig.values())
        with np.load(row['artifact'],allow_pickle=False) as data:
            for name in data.files:
                assert np.isfinite(data[name]).all()
                if name.startswith('hidden_'):hidden.setdefault(name,[]).append(data[name].copy())
    hidden={k:np.stack(v) for k,v in hidden.items()}
    with np.load(output/'original-hidden-states.npz',allow_pickle=False) as data:
        assert set(data.files)==set(hidden) and all(np.array_equal(data[k],v) for k,v in hidden.items())
    seal=json.loads((output/'grains-artifact.json').read_text())
    for file,key in [('grains-author.pt','artifact_sha256'),('original-hidden-states.npz','hidden_states_sha256'),('extraction.jsonl','journal_sha256')]:
        assert sha(output/file)==seal[key]
    np.random.seed(42);expected=authors.steering.compute_steering_vector(hidden,27,'pca')
    artifact=torch.load(output/'grains-author.pt',map_location='cpu',weights_only=True)
    assert torch.isfinite(artifact['vector']).all() and artifact['vector'].norm()>0
    torch.testing.assert_close(artifact['vector'],expected,rtol=1e-5,atol=1e-6)
    assert json.loads((output/'gemma-ple-api-validation.json').read_text())['status']=='passed'
    result={'status':'verified_complete','fit_images':50,'raw_ig_and_hidden_states_finite':True,
            'all_file_hashes_verified':True,'pca_vector_independently_recomputed':True,
            'protocol_sha256':protocol['sha256'],'analysis_sha256':sha(__file__),'checked_at':time.time()}
    save(output/'grains-extraction-independent-verification.json',result)


def audit(output):
    protocol=json.loads((output/'protocol.json').read_text());body={k:v for k,v in protocol.items() if k!='sha256'}
    assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==protocol['sha256']
    for file,digest in protocol['source_hashes'].items():assert sha(ROOT/file)==digest,file
    assert verify_clone_sources()==protocol['author_sources']
    amendment=json.loads((output/'execution-amendment.json').read_text())
    for file,digest in amendment['source_hashes'].items():assert sha(ROOT/file)==digest,file
    manifest=read(output/'data/pope.jsonl');assert len(manifest)==3000 and len({r['id'] for r in manifest})==3000
    assert sha(output/'data/pope.jsonl')==protocol['manifest_hashes']['pope.jsonl']
    data=ROOT/'data/hallucination-suite';official=read(data/'sources/coco_pope_adversarial.json')
    assert sha(data/'sources/coco_pope_adversarial.json')==protocol['dataset_provenance']['sources']['coco_pope_adversarial.json']['sha256']
    assert all(r['label']==q['label'] and r['metadata']['original_question']==q['text'] for r,q in zip(manifest,official))
    fit=read(output/'data/fit50.jsonl');fit_hashes={sha(r['image']) for r in fit}
    assert not fit_hashes & {r['image_sha256'] for r in manifest}
    assert all(sha(r['image'])==r['image_sha256'] for r in manifest)
    methods={m:{} for m in protocol['methods']}
    tasks=[t for t in json.loads((output/'tasks.json').read_text()) if t['benchmark']=='pope']
    assert len(tasks)==300
    for task in tasks:
        path=output/'predictions'/f"{task['id']}.jsonl";marker=json.loads((output/'claims'/task['id']/'complete.json').read_text())
        assert sha(path)==marker['sha256']
        rows=read(path);refs=manifest[task['start']:task['stop']];assert len(rows)==len(refs)==marker['records']
        for row,ref in zip(rows,refs):
            assert all(row.get(k)==v for k,v in ref.items())
            assert row['protocol_sha256']==protocol['sha256'] and row['method']==task['method'] and row['dtype']==task['dtype']
            assert row['gpu_uuid']==protocol['gpu_uuid']=='GPU-4a6e1799-a39c-86ed-abdf-afad03791c12'
            if row['method'] in ['rudder_clone','grains_clone']:
                assert 'adopted_cache_sha256' not in row
                d=row['author_diagnostics'];assert d['calls']>0
                expected='RUDDER/Qwen/qwen_methods.py' if row['method']=='rudder_clone' else 'GrAInS/src/utils/steering.py'
                assert d['hook_source'].endswith(expected)
            elif 'adopted_cache_sha256' in row:
                assert row['method'] in ['base','revis','vti']
                path=output/'cache'/row['method']/f"{row['id'].replace(':','_')}.json"
                assert sha(path)==row['adopted_cache_sha256']==protocol['cache_hashes'][str(path.relative_to(output))]
                cached=json.loads(path.read_text());assert cached['prediction']==row['prediction'] and cached['generated_tokens']==row['generated_tokens']
            assert row['id'] not in methods[task['method']];methods[task['method']][row['id']]=row
    for method,rows in methods.items():
        assert set(rows)=={r['id'] for r in manifest}
        methods[method]=[rows[r['id']] for r in manifest]
    folder=output/'pope-complete';folder.mkdir(exist_ok=True);metrics={}
    for method,rows in methods.items():
        metrics[method]=pope_metrics(rows);verify_upstream_pope(rows,folder,method,'adversarial',metrics[method])
        metrics[method]['strict']=evaluate_predictions(rows)
        (folder/f'{method}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    uncertainty={'bf16':intervals({m:methods[m] for m in ['base','revis','vti','rudder_clone']},'pope'),
                 'fp32':intervals({'base':methods['base_fp32'],'grains_clone':methods['grains_clone']},'pope')}
    report={'status':'verified_complete','benchmark':'POPE COCO adversarial','cases_per_method':3000,'total_responses':18000,
       'metrics':metrics,'uncertainty':uncertainty,'comparisons':protocol['comparisons'],'protocol_sha256':protocol['sha256'],
       'audit':{'complete_id_and_source_and_task_hash_coverage':True,'official_pope_scorer_used':True,
                'author_hook_calls_verified':True,'matched_precision_baselines':True,'development_test_image_hash_overlap':0},
       'chair_status':'continues independently; this report covers only POPE','output':str(folder),'checked_at':time.time(),
       'analysis_sha256':sha(__file__)}
    save(folder/'report.json',report)
    with (folder/'metrics.csv').open('w',newline='') as sink:
        writer=csv.DictWriter(sink,fieldnames=['method','dtype','accuracy','f1','precision','recall','invalid']);writer.writeheader()
        for m,v in metrics.items():writer.writerow({'method':m,'dtype':'FP32' if m in ['base_fp32','grains_clone'] else 'BF16',**{k:v[k] for k in ['accuracy','f1','precision','recall','invalid']}})
    names={'base':'Base BF16','base_fp32':'Base FP32','revis':'REVIS 기존','vti':'VTI 기존','rudder_clone':'RUDDER 원본 clone','grains_clone':'GrAInS 원본 clone'}
    text=['# Gemma E4B · POPE adversarial 전체 평가 완료','','각 방법3,000문항. 단위 %. 원본 clone 함수에 Gemma 호환 연결을 적용했다.','',
          '| 방법 | Accuracy ↑ | F1 ↑ | Precision ↑ | Recall ↑ |','|---|---:|---:|---:|---:|']
    for m in ['base','revis','vti','rudder_clone','base_fp32','grains_clone']:
        v=metrics[m];text.append('| '+names[m]+' | '+' | '.join(f'{v[k]*100:.2f}' for k in ['accuracy','f1','precision','recall'])+' |')
    text+=['','GrAInS는 Base FP32, 나머지 방법은 Base BF16과 비교한다. 공식 evaluate.py와 독립 재채점이 일치했다. 테스트 이미지와 개발 이미지 해시 중복0개다.','',
           'CHAIR 고정500장·최대512토큰 평가는 GPU0에서 계속 진행한다. POPE의 부분 집계가 아니라 전체3,000문항 집계이며 전체 벤치마크 완료를 뜻하지 않는다.','']
    (folder/'REPORT_KO.md').write_text('\n'.join(text))
    return report


def main(output):
    state={'pid':os.getpid(),'started_at':time.time(),'scope':'Complete POPE first while CHAIR keeps generating'}
    def update(**kw):state.update(kw,updated_at=time.time());save(output/'pope-only-status.json',state)
    try:
        protocol=json.loads((output/'protocol.json').read_text())
        update(stage='verifying_completed_extraction');verify_extraction(output,protocol)
        tasks=[t for t in json.loads((output/'tasks.json').read_text()) if t['benchmark']=='pope']
        while True:
            completed=sum((output/'claims'/t['id']/'complete.json').exists() for t in tasks)
            update(stage='waiting_for_complete_pope',completed_tasks=completed,total_tasks=len(tasks))
            if completed==len(tasks):break
            suite=json.loads((output/'status.json').read_text())
            if suite['stage']=='failed':raise RuntimeError(suite.get('error','Suite failed'))
            time.sleep(15)
        update(stage='official_pope_scoring');report=audit(output)
        update(stage='complete',report=report['output'],completed_at=time.time())
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
