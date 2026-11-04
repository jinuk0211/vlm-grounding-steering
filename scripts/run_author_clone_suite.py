"""Prepare/resume the requested hybrid suite on physical GPU0 only.

RUDDER and GrAInS are rerun through pinned author functions. Existing Base,
REVIS and VTI responses are reused only when their exact inputs/code match.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from zoneinfo import ZoneInfo

from author_clone_bridge import ROOT, verify_clone_sources

GPU='GPU-4a6e1799-a39c-86ed-abdf-afad03791c12'
CURRENT=ROOT/'outputs/author-clone-rerun-current.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def alive(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1][0] not in ['Z','X']
    except FileNotFoundError:return False


def prepare():
    assert int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1000
    proof_pointer=json.loads((ROOT/'outputs/author-clone-preflight-current.json').read_text())
    assert not alive(proof_pointer['pid'])
    proof=Path(proof_pointer['output'])
    state=json.loads((proof/'status.json').read_text());assert state['stage']=='complete'
    assert (ROOT/'scripts/author_clone_bridge.py').stat().st_mtime<=state['started_at']
    assert json.loads((ROOT/'outputs/author-clone-bridge-validation.json').read_text())['status']=='passed'
    stamp=datetime.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d-%H%M%S')
    output=ROOT/'outputs'/f'author-clone-rerun-{stamp}';output.mkdir()
    for name in ['data','claims','predictions','artifacts','cache','source-snapshot','author-snapshot','proofs']:(output/name).mkdir()
    parent=Path(json.loads((ROOT/'outputs/pope-selected-current.json').read_text())['output'])
    parent_protocol=json.loads((parent/'protocol.json').read_text())
    assert json.loads((parent/'report.json').read_text())['status']=='verified_complete'
    for f in ['src/vlm_grounding/baselines.py','src/vlm_grounding/backend.py']:
        assert sha(ROOT/f)==parent_protocol['source_hashes'][f]
    config=json.loads((parent/'config.json').read_text());save(output/'config.json',config)
    data=ROOT/'data/hallucination-suite'
    datasets={'pope':[r for r in read(data/'pope.jsonl') if r['subset']=='adversarial'],'chair':read(data/'chair.jsonl')}
    assert len(datasets['pope'])==3000 and len(datasets['chair'])==500
    for bench,rows in datasets.items():
        rows=[{**r,'max_new_tokens':128 if bench=='pope' else 512} for r in rows];datasets[bench]=rows
        (output/'data'/f'{bench}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    development=read(ROOT/'data/mmvp-full/fit.jsonl')+read(ROOT/'data/mmvp-full/calibration.jsonl')
    fit=development[:50];assert len({r['group_id'] for r in fit})==50
    test_hashes={r['image_sha256'] for rows in datasets.values() for r in rows}
    assert not test_hashes & {sha(r['image']) for r in development}
    (output/'data/fit50.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in fit))
    retained_hashes={}
    for method in ['revis','vti']:
        source=parent/'artifacts'/f'{method}.pt';assert sha(source)==parent_protocol['artifact_hashes'][method]
        shutil.copy2(source,output/'artifacts'/source.name);retained_hashes[method]=sha(source)
    sources={};adopted={method:{} for method in ['base','revis','vti']}
    chair_parent=Path(json.loads((ROOT/'outputs/chair-selected-current.json').read_text())['output'])
    for source_output in [parent,chair_parent]:
        source_protocol=json.loads((source_output/'protocol.json').read_text())
        for f in ['src/vlm_grounding/baselines.py','src/vlm_grounding/backend.py']:
            assert source_protocol['source_hashes'][f]==parent_protocol['source_hashes'][f]
        assert source_protocol['model']==parent_protocol['model']
        for method in ['revis','vti']:
            assert source_protocol['method_parameters'][method]==parent_protocol['method_parameters'][method]
            assert source_protocol['artifact_hashes'][method]==retained_hashes[method]
        sources[str(source_output)]={'protocol_sha256':sha(source_output/'protocol.json'),'protocol':source_protocol}
        for folder in ['adopted','predictions']:
            for file in (source_output/folder).glob('*.jsonl'):
                for line in file.read_text().splitlines():
                    try:row=json.loads(line)
                    except json.JSONDecodeError:continue
                    method=row['method']
                    if method not in adopted:continue
                    bench=row['benchmark'];ref=next((r for r in datasets[bench] if r['id']==row['id']),None)
                    assert ref is not None
                    assert all(row[k]==ref[k] for k in ['image_sha256','image','prompt','max_new_tokens','inference_key','label'] if k in ref)
                    key=(bench,row['id'])
                    if key in adopted[method]:
                        assert adopted[method][key]['prediction']==row['prediction'] and adopted[method][key]['generated_tokens']==row['generated_tokens']
                    else:adopted[method][key]=row
    # Preserve the complete protocol ancestry of reused rows, including earlier
    # cache adoption. Reuse is confined to unchanged methods and inputs.
    pending=list(sources)
    while pending:
        source_output=Path(pending.pop());source_protocol=sources[str(source_output)]['protocol']
        previous=source_protocol.get('parent_output')
        if previous and previous not in sources:
            ancestor=Path(previous);assert ancestor.is_relative_to(ROOT/'outputs')
            ancestor_protocol=json.loads((ancestor/'protocol.json').read_text())
            for f in ['src/vlm_grounding/backend.py','src/vlm_grounding/baselines.py']:
                assert ancestor_protocol['source_hashes'][f]==parent_protocol['source_hashes'][f]
            sources[previous]={'protocol_sha256':sha(ancestor/'protocol.json'),'protocol':ancestor_protocol}
            pending.append(previous)
    assert {row['protocol_sha256'] for rows in adopted.values() for row in rows.values()} <= {s['protocol']['sha256'] for s in sources.values()}
    for method,rows in adopted.items():
        folder=output/'cache'/method;folder.mkdir()
        for (_,record_id),row in rows.items():save(folder/f"{record_id.replace(':','_')}.json",row)
    save(output/'cache-provenance.json',{'sources':sources,'counts':{m:len(rows) for m,rows in adopted.items()},
         'scope':'Only unchanged BF16 Base/REVIS/VTI; zero old RUDDER/GrAInS responses adopted'})
    methods=['base','revis','vti','rudder_clone','base_fp32','grains_clone']
    files=['src/vlm_grounding/backend.py','src/vlm_grounding/baselines.py','src/vlm_grounding/evaluation.py',
           'scripts/author_clone_bridge.py','scripts/extract_author_grains.py','scripts/run_author_clone_worker.py',
           'scripts/run_author_clone_suite.py','scripts/audit_hallucination_suite.py']
    clones=verify_clone_sources()
    for name,source in clones.items():
        for filename in source['files']:
            dest=output/'author-snapshot'/name/filename;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/'upstream-author-20261026'/name/filename,dest)
    protocol={'scope':'User requested original cloned RUDDER and GrAInS, retaining existing REVIS/VTI',
        'model':config['model'],'methods':methods,'gpu_index':0,'gpu_uuid':GPU,'other_gpu_fallback':False,
        'records_per_method':{'pope_adversarial':3000,'chair_coco_val2014':500},
        'generation':{'do_sample':False,'num_beams':1,'use_cache':True,'seed_per_example':42,'batch_size':1,
                      'pope_max_new_tokens':128,'chair_max_new_tokens':512},
        'source_selection_branch':'codex/baseline-selection-20261026','source_cli_main_commit':parent_protocol['source_commit'],
        'method_parameters':{'retained':{m:config['methods'][m] for m in ['base','revis','vti']},
            'rudder_clone':{'pope':{'layer':1,'alpha':8.0},'chair':{'layer':25,'alpha':5.0},'k':5.0,'c':1.0,'clamp':[0.05,1.0],'dtype':'bfloat16'},
            'grains_clone':{'fit':50,'ig_steps':20,'top_k':50,'top':'pos','pca_layer':27,'alpha':10.0,'original_multiplier':10,'dtype':'float32'}},
        'comparisons':{'rudder_clone':'base','revis':'base','vti':'base','grains_clone':'base_fp32'},
        'architecture_compatibility':['Native Gemma chat/image processing; original CARD and gate functions',
             'Original GrAInS token-only attribution/ablation and PCA functions; native Gemma multimodal generation',
             'Gemma PLE token identity fixed along IG; its context projection differentiates interpolated embeddings',
             'Original tuple output contract via Tensor-to-tuple-to-Tensor bridge; original sequence-axis norm retained',
             'Captum internal batch size one; no change to 20 quadrature nodes or objective'],
        'protocol_overrides':['Common POPE cap128 and user-selected CHAIR cap512 retained rather than Qwen snapshot caps',
                              'Gemma-specific fit50 re-extracted from disjoint annotation-derived COCO development images; no Qwen vectors reused'],
        'author_reproduction_complete':False,'author_core_functions_called':True,'author_sources':clones,
        'source_hashes':{f:sha(ROOT/f) for f in files},'retained_artifact_hashes':retained_hashes,
        'manifest_hashes':{p.name:sha(p) for p in (output/'data').glob('*.jsonl')},
        'cache_provenance_sha256':sha(output/'cache-provenance.json'),
        'cache_hashes':{str(p.relative_to(output)):sha(p) for p in (output/'cache').rglob('*.json')},
        'dataset_provenance':parent_protocol['dataset_provenance'],'chair_scoring':parent_protocol['chair_scoring'],
        'development_test_hash_overlap':0,'deferred':['POPE random/popular','MMHal additional generation and official judging'],
        'dependency_versions':{n:__import__('importlib.metadata',fromlist=['version']).version(n) for n in ['torch','transformers','captum','scikit-learn','numpy']},
        'preflight_output':str(proof),'preflight_result_sha256':sha(proof/'result.json')}
    protocol['sha256']=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    save(output/'protocol.json',protocol)
    for f in files:
        dest=output/'source-snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    for f in ['result.json','status.json','author-sources.json']:shutil.copy2(proof/f,output/'proofs'/f)
    shutil.copy2(ROOT/'outputs/author-clone-bridge-validation.json',output/'proofs/bridge-validation.json')
    tasks=[]
    for bench,block in [('pope',60),('chair',8)]:
        for start in range(0,len(datasets[bench]),block):
            for method in methods:
                tasks.append({'id':f'{bench}-{method}-{start:05d}','benchmark':bench,'method':method,
                     'dtype':'float32' if method in ['base_fp32','grains_clone'] else 'bfloat16',
                     'start':start,'stop':min(start+block,len(datasets[bench]))})
    save(output/'tasks.json',tasks)
    return output


def controller(output):
    jobs=[];started=time.time()
    def status(stage,**kw):save(output/'status.json',{'stage':stage,'controller_pid':os.getpid(),'started_at':started,'updated_at':time.time(),**kw})
    def launch(worker,dtype=None,extract=False):
        script='extract_author_grains.py' if extract else 'run_author_clone_worker.py'
        command=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts'/script),'--output',str(output)]
        if not extract:command+=['--worker',worker,'--dtype',dtype]
        log=output/f'{worker}.log'
        with log.open('xb') as sink:
            p=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU),stdout=sink,stderr=subprocess.STDOUT,start_new_session=True)
        job={'worker':worker,'pid':p.pid,'dtype':dtype,'extract':extract,'log':str(log)};jobs.append(job);save(output/'jobs.json',jobs)
        return job
    try:
        status('starting_author_clone_rerun')
        extraction=launch('grains-extraction',extract=True)
        bf16=[launch(f'bf16-{i}','bfloat16') for i in range(3)]
        fp32=[];extraction_complete=False
        while True:
            states={}
            for job in jobs:
                file=output/('extraction-status.json' if job['extract'] else f"worker-{job['worker']}.json")
                state=json.loads(file.read_text()) if file.exists() else {'stage':'starting'}
                live=alive(job['pid']);states[job['worker']]={**state,'actual_live':live}
                if state['stage']=='failed' or (not live and state['stage']!='complete'):
                    raise RuntimeError(json.dumps({'failed_job':job,'state':state}))
            if not extraction_complete and not alive(extraction['pid']):
                assert states['grains-extraction']['stage']=='complete'
                extraction_complete=True;fp32.append(launch('fp32-0','float32'))
            if extraction_complete and len(fp32)==1 and sum(alive(j['pid']) for j in bf16)<=1 and alive(fp32[0]['pid']):
                fp32.append(launch('fp32-1','float32'))
            completed=sum(x.get('completed_records',0) for name,x in states.items() if name!='grains-extraction')
            status('generating' if extraction_complete else 'extracting_and_generating',workers=states,completed_records=completed,
                   total_records=21000,extracted_fit_records=states['grains-extraction'].get('completed',0))
            if extraction_complete and all(not alive(j['pid']) for j in jobs):break
            time.sleep(15)
        status('auditing')
        with (output/'audit.log').open('ab') as sink:
            subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/audit_author_clone_suite.py'),'--output',str(output)],
                           cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=sink,stderr=subprocess.STDOUT,check=True)
        assert json.loads((output/'report.json').read_text())['status']=='verified_complete'
        status('complete',completed_at=time.time())
    except BaseException:
        status('failed',error=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);args=p.parse_args()
    if args.output:controller(args.output)
    else:
        output=prepare();log=output/'controller.log'
        with log.open('xb') as sink:
            proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--output',str(output)],cwd=ROOT,
                                  env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=sink,stderr=subprocess.STDOUT,start_new_session=True)
        pointer={'output':str(output),'controller_pid':proc.pid,'log':str(log),'scope':'POPE adversarial3000 + CHAIR500; cloned RUDDER/GrAInS; retained REVIS/VTI; GPU0 only'}
        save(CURRENT,pointer);print(json.dumps(pointer,indent=2))
