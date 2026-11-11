"""Official full POPE scoring for the five-profile FP16-GrAInS protocol."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch
from grains_fp16_worker_v2 import ROOT,verified_protocol,read,save,GPU2,GPU3
from audit_hallucination_suite import pope_metrics,verify_upstream_pope,intervals
from vlm_grounding.evaluation import evaluate_predictions

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    folder=output/'pope-complete';folder.mkdir()
    state={'pid':os.getpid(),'stage':'waiting_for_pope','gpu_compute_started':False,'started_at':time.time()}
    def update(**kw):state.update(kw,updated_at=time.time());save(folder/'status.json',state)
    try:
        protocol=verified_protocol(output)
        tasks=[t for t in json.loads((output/'tasks.json').read_text()) if t['benchmark']=='pope']
        while True:
            done=sum((output/'claims'/t['id']/'complete.json').exists() for t in tasks)
            update(stage='waiting_for_pope',complete_tasks=done,total_tasks=len(tasks))
            if done==len(tasks):break
            current=json.loads((output/'status.json').read_text());assert current['stage']!='failed',current
            time.sleep(15)
        update(stage='official_scoring')
        refs=read(output/'data/pope.jsonl');assert len(refs)==3000
        official=read(ROOT/'data/hallucination-suite/sources/coco_pope_adversarial.json')
        assert sha(ROOT/'data/hallucination-suite/sources/coco_pope_adversarial.json')==protocol['dataset_provenance']['sources']['coco_pope_adversarial.json']['sha256']
        assert all(r['label']==q['label'] and r['metadata']['original_question']==q['text'] for r,q in zip(refs,official))
        fit={sha(r['image']) for r in read(output/'data/fit50.jsonl')}
        assert not fit & {r['image_sha256'] for r in refs}
        assert all(sha(r['image'])==r['image_sha256'] for r in refs)
        grouped={m:{} for m in protocol['methods']};source_files={}
        jobs={j['worker']:j for j in json.loads((output/'jobs.json').read_text())}
        for task in tasks:
            file=output/'predictions'/(task['id']+'.jsonl');rows=read(file)
            marker=json.loads((output/'claims'/task['id']/'complete.json').read_text())
            selected=refs[task['start']:task['stop']]
            assert sha(file)==marker['sha256'] and len(rows)==len(selected)==marker['records']
            for row,ref in zip(rows,selected):
                assert all(row.get(k)==v for k,v in ref.items())
                assert row['protocol_sha256']==protocol['sha256'] and row['method']==task['method']
                assert row['dtype']==task['dtype']==protocol['method_dtypes'][task['method']]
                assert row['prediction'].strip() and 0<row['generated_tokens']<=128
                if task['method']=='grains_clone':
                    job=jobs[row['worker']];assert row['gpu_uuid']==job['visible_devices'] in [GPU2,GPU3]
                    proof=json.loads((output/(row['worker']+'-device-map.json')).read_text())
                    assert proof['parameter_dtypes']==['torch.float16'] and proof['steering_vector_dtype']=='torch.float32'
                    assert row['artifact_sha256']==protocol['grains_artifact_sha256'] and row['author_diagnostics']['calls']>0
                else:
                    name=row['source_prediction_file']
                    if name not in source_files:
                        assert sha(name)==row['source_prediction_file_sha256'];source_files[name]=Path(name).read_bytes().splitlines()
                    line=source_files[name][row['source_row_index']];assert hashlib.sha256(line).hexdigest()==row['source_row_sha256']
                    original=json.loads(line);assert original['prediction']==row['prediction'] and original['dtype']==row['dtype']
                assert row['id'] not in grouped[task['method']];grouped[task['method']][row['id']]=row
        metrics={}
        for method,rows in grouped.items():
            assert set(rows)=={r['id'] for r in refs};rows=[rows[r['id']] for r in refs];grouped[method]=rows
            value=pope_metrics(rows);verify_upstream_pope(rows,folder,method,'adversarial',value)
            value['strict']=evaluate_predictions(rows);metrics[method]=value
            (folder/(method+'.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
        report={'status':'verified_complete','benchmark':'POPE COCO adversarial','methods':protocol['methods'],
            'cases_per_method':3000,'total_responses':15000,'metrics':metrics,'uncertainty':intervals(grouped,'pope'),
            'protocol_sha256':protocol['sha256'],'method_dtypes':protocol['method_dtypes'],'base_fp32_enabled':False,
            'precision_note':'GrAInS model FP16, others BF16; original FP32 steering vector retained; mixed precision comparison',
            'audit':{'official_pope_scorer_used':True,'all_15000_responses_verified':True,'development_test_image_hash_overlap':0},
            'chair_complete':False,'analysis_sha256':sha(__file__),'checked_at':time.time()}
        save(folder/'report.json',report)
        with (folder/'metrics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['method','dtype','accuracy','f1','precision','recall','invalid']);writer.writeheader()
            for m,v in metrics.items():writer.writerow({'method':m,'dtype':protocol['method_dtypes'][m],**{k:v[k] for k in ['accuracy','f1','precision','recall','invalid']}})
        md=['# Gemma E4B · POPE 전체 완료 (GrAInS FP16)','',
            '각 방법 COCO adversarial3,000문항. 단위 %. Base FP32 제외.','',
            '| 방법 | 정밀도 | Accuracy ↑ | F1 ↑ | Precision ↑ | Recall ↑ |','|---|---|---:|---:|---:|---:|']
        for m,v in metrics.items():md.append('| '+m+' | '+protocol['method_dtypes'][m]+' | '+' | '.join(f'{v[k]*100:.2f}' for k in ['accuracy','f1','precision','recall'])+' |')
        md+=['','GrAInS 모델은 FP16, 다른 방법은 BF16이므로 방법 효과만 분리한 비교가 아니다. steering 벡터는 원본 FP32를 유지했다.',
            'RUDDER·GrAInS는 원저자 함수와 Gemma 연결 adapter를 사용한다. REVIS·VTI는 선택한 자체 구현이다.',
            'CHAIR는 별도로 계속 생성 중이며 전체 모델 평가 완료를 뜻하지 않는다.','']
        (folder/'REPORT_KO.md').write_text('\n'.join(md));assert not torch.cuda.is_initialized()
        update(stage='complete',report_sha256=sha(folder/'report.json'),cuda_initialized=False,completed_at=time.time())
        print(json.dumps({'status':'verified_complete','metrics':metrics}))
    except BaseException:update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
