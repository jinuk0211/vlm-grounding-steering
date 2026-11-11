"""Officially score the completed original-function FP16 GrAInS POPE cohort."""
import json,math,os,time
from pathlib import Path
from qwen_gpu0_migration_api import ROOT,GPU0,sha,save,verify_protocol
from qwen_five_profile_api import CHECKPOINT,REVISION
from audit_qwen_vti_replacement import obj,read,sealed,device_proof,validate_attention
from audit_hallucination_suite import pope_metrics,verify_upstream_pope,write_jsonl,intervals
from author_clone_bridge import verify_clone_sources

PARENT=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386'

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    p=PARENT;protocol=verify_protocol(p);sealed(protocol)
    assert verify_clone_sources()==protocol['author_sources']
    assert sha(p/'tasks.json')==protocol['tasks_sha256']
    refs=read(p/'data/pope.jsonl');assert len(refs)==len({r['id'] for r in refs})==3000
    assert sha(p/'data/pope.jsonl')==protocol['manifest_hashes']['pope.jsonl']
    source=ROOT/'data/hallucination-suite/sources/coco_pope_adversarial.json'
    assert sha(source)==protocol['dataset_provenance']['sources']['coco_pope_adversarial.json']['sha256']
    official=read(source);assert len(official)==3000
    assert all(r['metadata']['original_question']==q['text'] and r['label']==q['label'] for r,q in zip(refs,official))
    seals=obj(p/'artifacts-ready.json');artifact_hash=sha(p/'grains-author.pt')
    assert seals['methods']['grains_clone']['sha256']==artifact_hash
    attention=validate_attention(p)
    jobs={j['worker']:j for j in obj(p/'fp16-stable-jobs.json') if j['kind']=='generation'}
    assert len(jobs)==1
    for j in jobs.values():
        proof=device_proof(p/j['device_proof'],'float16')
        assert j['visible_devices']==GPU0 and proof['attention_score_dtype']=='torch.float32'
        assert proof['fp16_attention_amendment_sha256']==attention['sha256']
    tasks=sorted([t for t in obj(p/'tasks.json') if t['method']=='grains_clone' and t['benchmark']=='pope'],key=lambda t:t['start'])
    assert tasks and tasks[0]['start']==0 and tasks[-1]['stop']==3000
    rows=[];hashes={}
    for task in tasks:
        assert task['dtype']=='float16' and task['start']==len(rows)
        f=p/'predictions'/(task['id']+'.jsonl');marker=p/'claims'/task['id']/'complete.json'
        done=obj(marker);part=read(f);assert sha(f)==done['sha256'] and len(part)==done['records']==task['stop']-task['start']
        for row,ref in zip(part,refs[task['start']:task['stop']]):
            assert all(row.get(k)==v for k,v in ref.items())
            assert row['method']=='grains_clone' and row['dtype']=='float16' and row['gpu_uuid']==GPU0
            assert row['checkpoint']==CHECKPOINT and row['revision']==REVISION and row['model_family']=='qwen2_5_vl'
            assert row['protocol_sha256']==protocol['sha256'] and row['worker'] in jobs
            assert row['artifact_sha256']==artifact_hash and row['attention_score_dtype']=='torch.float32'
            assert row['fp16_attention_amendment_sha256']==attention['sha256']
            assert row['max_new_tokens']==128 and type(row['generated_tokens']) is int and 0<row['generated_tokens']<=128
            assert isinstance(row['prediction'],str) and math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds']>0
            d=row['author_diagnostics'];assert d['calls']>0 and d['layer']==27 and d['alpha']==10
            assert d['hook_source'].endswith('GrAInS/src/utils/steering.py')
        rows+=part;hashes[task['id']]=sha(f)
    assert len(rows)==3000 and [r['id'] for r in rows]==[r['id'] for r in refs]
    # The verified BF16 base is shown as a precision-confounded reference.
    three=p/'bf16-three-complete';summary=obj(three/'summary.json')
    assert summary['status']=='interim_three_bf16_officially_scored'
    assert summary['protocol']['sha256']==protocol['sha256']
    base=read(three/'pope/base.jsonl');assert len(base)==3000 and [r['id'] for r in base]==[r['id'] for r in rows]
    folder=p/'grains-pope-complete';folder.mkdir()
    score=pope_metrics(rows);verify_upstream_pope(rows,folder,'grains_clone','adversarial',score)
    write_jsonl(folder/'grains_clone.jsonl',rows)
    uncertainty=intervals({'base':base,'grains_clone':rows},'pope')
    result={'status':'grains_pope_full_officially_scored','scope':'POPE COCO adversarial 3000 questions',
        'method':'grains_clone','model_dtype':'float16','attention_score_dtype':'float32','gpu_indices':[0],
        'metrics':score,'uncertainty':uncertainty,'protocol_sha256':protocol['sha256'],'artifact_sha256':artifact_hash,
        'input_prediction_sha256':hashes,'source_sha256':sha(__file__),'created_at':time.time(),
        'full_five_method_suite_complete':False,'grains_chair_complete':False,
        'base_reference_dtype':'bfloat16','method_effect_not_isolated_from_precision':True}
    (folder/'REPORT_KO.md').write_text('# GrAInS POPE 완료 코호트\n\n'
        f"COCO adversarial 3,000문항. Accuracy **{100*score['accuracy']:.2f}%**, F1 **{100*score['f1']:.2f}%**.\n\n"
        '원본 GrAInS 함수를 호출했다. 모델 가중치는 FP16, attention 점수는 overflow 방지를 위해 FP32다. '
        'BF16 Base와의 차이에는 정밀도 영향도 포함된다. CHAIR와 원저자 VTI는 아직 진행 중이다.\n')
    save(folder/'summary.json',result)
    print(json.dumps({'output':str(folder),'summary_sha256':sha(folder/'summary.json'),'metrics':score}))

if __name__=='__main__':main()
