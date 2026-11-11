"""Score four completed Qwen POPE profiles with the original official scorer."""
import json,os,time
from pathlib import Path
import audit_hallucination_suite as official
from audit_qwen_five_profile_suite import CHECKPOINT,REVISION,sha,read,save
from qwen_gpu0_migration_api import ROOT,GPU0,verify_protocol

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    output=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386'
    protocol=verify_protocol(output)
    assert not (output/'report.json').exists(),'Full report already exists; use it instead'
    references=read(output/'data/pope.jsonl');assert len(references)==3000
    assert sha(output/'data/pope.jsonl')==protocol['manifest_hashes']['pope.jsonl']
    assert sha(ROOT/'data/hallucination-suite/sources/coco_pope_adversarial.json')==protocol['dataset_provenance']['sources']['coco_pope_adversarial.json']['sha256']
    source=read(ROOT/'data/hallucination-suite/sources/coco_pope_adversarial.json')
    assert all(r['metadata']['original_question']==q['text'] and r['label']==q['label'] for r,q in zip(references,source))
    methods=['base','revis','vti','rudder_clone']
    tasks=json.loads((output/'tasks.json').read_text())
    workers={j['worker']:j for file in ['jobs.json','supplemental-jobs.json']
             for j in json.loads((output/file).read_text()) if j['kind']=='generation'}
    results={};source_hashes={}
    folder=output/('pope-bf16-full-interim-'+str(int(time.time())))
    folder.mkdir()
    for method in methods:
        selected=sorted((t for t in tasks if t['benchmark']=='pope' and t['method']==method),key=lambda t:t['start'])
        assert selected and selected[0]['start']==0 and selected[-1]['stop']==3000
        rows=[]
        for task in selected:
            path=output/'predictions'/(task['id']+'.jsonl')
            marker=output/'claims'/task['id']/'complete.json'
            assert marker.is_file() and path.is_file()
            m=json.loads(marker.read_text());assert sha(path)==m['sha256']
            part=read(path);assert len(part)==m['records']==task['stop']-task['start']
            for row,ref in zip(part,references[task['start']:task['stop']]):
                assert all(row.get(k)==v for k,v in ref.items())
                assert row['method']==method and row['dtype']=='bfloat16'
                assert row['protocol_sha256']==protocol['sha256'] and row['gpu_uuid']==GPU0
                assert row['checkpoint']==CHECKPOINT and row['revision']==REVISION
                assert row['worker'] in workers and workers[row['worker']]['visible_devices']==GPU0
                assert 0<row['generated_tokens']<=128
            rows.extend(part);source_hashes[task['id']]=sha(path)
        assert len(rows)==3000 and [r['id'] for r in rows]==[r['id'] for r in references]
        score=official.pope_metrics(rows)
        official.verify_upstream_pope(rows,folder,method,'adversarial',score)
        official.write_jsonl(folder/(method+'.jsonl'),rows)
        results[method]=score
    report={'status':'four_bf16_pope_full_verified','scope':'Qwen POPE adversarial 3000 questions for Base/REVIS/VTI/RUDDER only',
        'remaining':'GrAInS POPE and all CHAIR still generating; full five-method suite audit pending',
        'protocol_sha256':protocol['sha256'],'gpu_indices':[0],'source_sha256':sha(__file__),
        'pope_manifest_sha256':sha(output/'data/pope.jsonl'),'prediction_file_sha256':source_hashes,
        'scores':results,'created_at':time.time()}
    save(folder/'report.json',report)
    (folder/'source.py').write_bytes(Path(__file__).read_bytes())
    save(output/'pope-bf16-interim-current.json',{'folder':str(folder),'report_sha256':sha(folder/'report.json')})
    print(json.dumps({'folder':str(folder),'report_sha256':sha(folder/'report.json'),'scores':results}))

if __name__=='__main__':main()
