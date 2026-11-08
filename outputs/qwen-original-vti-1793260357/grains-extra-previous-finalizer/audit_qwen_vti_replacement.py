"""Audit the complete GPU0 suite with original-author VTI replacing local VTI.

The parent protocol is historical and retained verbatim. VTI has a separate
sealed subrun protocol. Only canonical rows for the five final methods enter
the official metrics. Interrupted jobs are audited as interrupted, not passed.
"""
import argparse,csv,hashlib,inspect,json,math,os,random,subprocess,time,zipfile
from pathlib import Path

import numpy as np
import torch
from qwen_gpu0_migration_api import ROOT,GPU0,sha,save,verify_protocol
from qwen_five_profile_api import CHECKPOINT,REVISION,require_gemma_complete
from qwen_fp16_stable_stage import verify_attention_amendment
from qwen_fp16_attention_guard import verify_arithmetic
from author_clone_bridge import load_authors,verify_clone_sources
from vlm_grounding.cli import _validate_artifact
from vlm_grounding.evaluation import _load_external_module,attach_chair_scores,evaluate_predictions

METHODS=['base','revis','rudder_clone','grains_clone','vti_author']
DONORS=METHODS[:-1]
VTI_COMMIT='bb5dfee33a9e555f22af9faa7b3495c0ac3df69e'


def obj(path):return json.loads(Path(path).read_text())


def read(path):
    raw=Path(path).read_bytes()
    assert not raw or raw.endswith(b'\n'),f'Incomplete JSONL line: {path}'
    def reject(value):raise ValueError('Non-finite JSON constant '+value)
    return [json.loads(line,parse_constant=reject) for line in raw.splitlines() if line.strip()]


def sealed(value):
    body={k:v for k,v in value.items() if k!='sha256'}
    assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==value['sha256']


def live(pid,start):
    path=Path('/proc')/str(pid)/'stat'
    try:
        fields=path.read_text().split(') ',1)[1].split()
        return fields[0] not in ('Z','X') and int(fields[19])==start
    except FileNotFoundError:return False


def dead(job):
    assert isinstance(job['pid'],int) and isinstance(job['start_ticks'],int)
    assert not live(job['pid'],job['start_ticks']),f'Process still live: {job}'


def device_proof(path,dtype,gpu=GPU0):
    d=obj(path)
    assert d['checkpoint']==CHECKPOINT and d['revision']==REVISION
    assert d['dtype']==dtype and d['visible_devices']==gpu
    assert not d['quantization'] and not d['cpu_or_disk_offload'] and not d['model_weights_changed']
    assert d['parameter_devices']==['cuda:0']
    assert d['parameter_dtypes']==['torch.'+dtype]
    return d


def parent_assets(parent,protocol):
    """Recheck original inputs, development split, fitted REVIS and raw GrAInS."""
    from audit_qwen_gpu0_final import verify_final_sources
    verify_final_sources(parent)
    for name,digest in protocol['source_hashes'].items():
        assert sha(ROOT/name)==digest==sha(parent/'source-snapshot'/name)
    assert verify_clone_sources()==protocol['author_sources']
    for name,info in protocol['author_sources'].items():
        for path,digest in info['files'].items():assert sha(parent/'author-snapshot'/name/path)==digest
    config=obj(parent/'config.json');seals=obj(parent/'artifacts-ready.json')
    assert seals['status']=='verified_complete' and seals['source_protocol_sha256']==protocol['sha256']
    datasets={b:read(parent/'data'/(b+'.jsonl')) for b in ['pope','chair']}
    fit,calibration,fit50=(read(parent/'data'/n) for n in ['fit.jsonl','calibration.jsonl','fit50.jsonl'])
    assert len(fit)==len(calibration)==32 and fit50==(fit+calibration)[:50]
    assert len({r['group_id'] for r in fit+calibration})==64
    development_hashes={sha(r['image']) for r in fit+calibration};assert len(development_hashes)==64
    for b,rows in datasets.items():
        assert len(rows)==len({r['id'] for r in rows})==(3000 if b=='pope' else 500)
        assert not development_hashes&{r['image_sha256'] for r in rows}
        for path,digest in {(r['image'],r['image_sha256']) for r in rows}:assert sha(path)==digest
    for phase in ['bf16','fp32']:
        path=parent/'proofs'/(phase+'-preflight.json');assert sha(path)==protocol['preflight_hashes'][phase]
        proof=obj(path);assert proof['status']=='passed'
        for n,h in proof['source_hashes'].items():assert protocol['source_hashes'][n]==h
        for n,h in proof['development_manifest_hashes'].items():assert sha(n)==h==sha(parent/'data'/Path(n).name)
    weight=parent/'proofs/weight-cache-verification.json'
    assert sha(weight)==protocol['cached_weight_verification_sha256'] and obj(weight)['all_hf_lfs_sha256_verified']
    path=parent/'artifacts/revis.pt';assert sha(path)==seals['methods']['revis']['sha256']
    side=obj(parent/'revis-artifact.json')
    assert side['status']=='verified_complete' and side['model']==config['model']
    assert side['source_protocol_sha256']==protocol['sha256'] and not side['gemma_vectors_reused']
    assert side['records']==[r['id'] for r in fit+calibration]
    for n,h in side['input_hashes'].items():assert sha(n)==h
    artifact=torch.load(path,map_location='cpu',weights_only=True);_validate_artifact(artifact,config['model'])
    assert all(torch.isfinite(v).all() for v in artifact['vectors'].values())
    assert artifact['vectors'][artifact['layer']].norm()>0
    journal=read(parent/'extraction.jsonl');assert len(journal)==50
    reconstructed={}
    for i,(entry,record) in enumerate(zip(journal,fit50)):
        assert entry['index']==i and entry['record']==record
        assert sha(entry['artifact'])==entry['artifact_sha256'] and sha(entry['raw_ig'])==entry['raw_ig_sha256']
        assert entry['ig']['quadrature_nodes']==20 and entry['ig']['effective_internal_batch_size']==1
        raw=torch.load(entry['raw_ig'],map_location='cpu',weights_only=True)
        assert raw and all(torch.isfinite(v).all() for v in raw.values())
        with np.load(entry['artifact'],allow_pickle=False) as data:
            for name in data.files:
                assert np.isfinite(data[name]).all()
                if name.startswith('hidden_'):reconstructed.setdefault(name,[]).append(data[name].copy())
    reconstructed={k:np.stack(v) for k,v in reconstructed.items()};assert len(reconstructed)==4
    with np.load(parent/'original-hidden-states.npz',allow_pickle=False) as stored:
        assert set(stored.files)==set(reconstructed)
        assert all(np.array_equal(stored[k],v) for k,v in reconstructed.items())
    gs=obj(parent/'grains-artifact.json')
    for n,key in [('grains-author.pt','artifact_sha256'),('original-hidden-states.npz','hidden_states_sha256'),('extraction.jsonl','journal_sha256')]:assert sha(parent/n)==gs[key]
    assert gs['actual_ig_calls']==50 and gs['actual_ig_quadrature_nodes']==20
    grains=torch.load(parent/'grains-author.pt',map_location='cpu',weights_only=True)
    assert grains['model']=={**config['model'],'dtype':'float32'} and not grains['gemma_vectors_reused']
    assert grains['fit_records']==[r['id'] for r in fit50]
    np.random.seed(42);expected=load_authors().steering.compute_steering_vector(reconstructed,27,'pca')
    torch.testing.assert_close(expected,grains['vector'],rtol=1e-5,atol=1e-6)
    assert grains['vector'].dtype==torch.float32 and torch.isfinite(grains['vector']).all() and grains['vector'].norm()>0
    assert sha(parent/'grains-author.pt')==seals['methods']['grains_clone']['sha256']
    hardware=obj(parent/'gpu0-hardware-amendment.json');stop=obj(Path(hardware['stop_proof']))
    assert len(journal)==50 and all(e['gpu_uuid']==GPU0 for e in journal[stop['completed_ig_records']:])
    return datasets,seals,development_hashes


def validate_attention(parent):
    amendment=verify_attention_amendment(parent);arithmetic=verify_arithmetic()
    assert arithmetic==amendment['cpu_arithmetic_check']
    stable=obj(parent/'fp16-stable-preflight.json')
    assert stable['status']=='passed' and stable['attention_score_dtype']=='torch.float32'
    assert stable['attention_amendment_sha256']==sha(parent/'fp16-attention-amendment.json')
    assert any(r['layer']==27 and r['elements']>0 for r in stable['attention_diagnostics']['native_half_score_overflows'])
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as model
    assert sha(model.__file__)==stable['attention_diagnostics']['transformers_model_source_sha256']
    assert hashlib.sha256(inspect.getsource(model.eager_attention_forward).encode()).hexdigest()==stable['attention_diagnostics']['original_attention_source_sha256']
    return {'sha256':sha(parent/'fp16-attention-amendment.json'),'weights':'float16','QK_mask_softmax_scores':'float32',
        'attention_outputs':'float16','steering_vector':'float32','causal_mask_and_fp32_reference_check':arithmetic}


def lifecycle(parent,vti):
    """Refuse live identities and distinguish success, supersession and failure."""
    transition=obj(vti/'transition.json');assert transition['all_selected_instances_exited']
    stopped={(j['pid'],j['start_ticks']):j for j in transition['stopped_jobs']}
    assert len(stopped)==2
    for j in stopped.values():dead(j)
    sc=transition['stopped_controller'];dead({'pid':sc['controller_pid'],'start_ticks':sc['controller_start_ticks']})
    continuation=obj(parent/'gpu0-continuation-amendment.json')
    failed_pre={(j['pid'],j['start_ticks']):j for j in continuation['failed_pre_model_jobs']}
    hardware=obj(parent/'gpu0-hardware-amendment.json');stop=obj(Path(hardware['stop_proof']))
    migrated={(j['pid'],j['start_ticks']) for j in stop['stopped_instances']}
    workers={};jobs=[];seen=set();history=[]
    for name in ['jobs.json','supplemental-jobs.json','fp16-stable-jobs.json']:
        for j in obj(parent/name):
            identity=(j['pid'],j['start_ticks']);assert identity not in seen;seen.add(identity);jobs.append(j);dead(j)
            state=obj(parent/j['state_file'])
            if identity in stopped:
                assert j==stopped[identity] and j['kind']=='generation'
                history.append({'pid':j['pid'],'outcome':'explicitly_superseded_supplemental_replica'})
            elif j.get('gpu0_pre_model_failure'):
                assert j==failed_pre[identity] and state['stage']=='failed' and state['completed']==38
                assert sha(parent/j['state_file'])==j['failed_state_sha256']
                history.append({'pid':j['pid'],'outcome':'preserved_pre_model_failure'});continue
            elif j.get('numerical_preflight_failed'):
                numerical=obj(parent/'gpu0-bf16-priority-amendment.json')
                assert state['stage']==state['status']=='failed' and state['pid']==j['pid']
                assert sha(parent/j['state_file'])==j['failed_state_sha256']==numerical['failed_preflight_state_sha256']
                history.append({'pid':j['pid'],'outcome':'preserved_native_fp16_preflight_failure'});continue
            elif j.get('gpu0_migration_stopped'):
                assert identity in migrated and j['kind']=='extraction' and state['completed']==stop['completed_ig_records']
                history.append({'pid':j['pid'],'outcome':'historical_extraction_stopped_for_gpu0_migration'});continue
            else:
                assert state['pid']==j['pid'] and state['stage']=='complete'
                assert sha(parent/j['state_file'])==j['state_sha256']
                if j.get('direct_launch'):
                    assert {'pid':j['pid'],'start_ticks':j['start_ticks']}==continuation['adopted_extraction_identity']
                    assert j['adopted_completion_observed'] and not j['exit_code_observed'] and state['completed']==50
                    history.append({'pid':j['pid'],'outcome':'adopted_extraction_complete_exit_code_unavailable'})
                else:assert j['exit_code']==0
            if j['kind']=='generation':
                d=device_proof(parent/j['device_proof'],j['dtype']);assert j['visible_devices']==GPU0
                if identity not in stopped:assert sha(parent/j['device_proof'])==j['device_proof_sha256']
                workers[j['worker']]=j
    for j in obj(parent/'preflight-jobs.json'):
        dead(j);folder=Path(j['output']);assert obj(folder/j['state_file'])['status']=='passed' and j['exit_code']==0
        assert sha(folder/j['state_file'])==j['state_sha256'] and sha(folder/j['device_proof'])==j['device_proof_sha256']
    resumed=obj(vti/'resumed-jobs.json');ctrl=obj(vti/'resume-controller-status.json')
    assert ctrl['stage']=='complete';dead(ctrl)
    author=resumed['author_job'];dead(author)
    state=obj(vti/'status.json');assert state['pid']==author['pid'] and state['stage']=='complete'
    assert author['exit_code']==0 and sha(vti/'status.json')==author['state_sha256']
    assert author['source_sha256']==sha(ROOT/'scripts/run_qwen_original_vti.py')
    recovery=resumed['adopted_recovery_job'];dead(recovery)
    rs=obj(parent/recovery['state_file']);assert rs['pid']==recovery['pid'] and rs['stage']=='complete'
    assert ctrl['recovery_exit_observed'] and ctrl['recovery_exit_code_unavailable']
    assert sha(parent/recovery['state_file'])==ctrl['recovery_state_sha256']
    assert recovery['source_sha256']==sha(ROOT/recovery['args'][0])
    name=recovery['args'][recovery['args'].index('--worker')+1]
    assert recovery['args'][recovery['args'].index('--dtype')+1]=='bfloat16'
    assert rs['dtype']=='bfloat16' and rs['gpu_visible_devices']==GPU0
    device_proof(parent/(name+'-device-map.json'),'bfloat16')
    workers[name]={**recovery,'dtype':'bfloat16','visible_devices':GPU0,'worker':name}
    previous=resumed['failed_author_job'];dead(previous)
    failed=obj(vti/('failed-start-'+str(previous['pid']))/'status.json')
    assert failed['pid']==previous['pid'] and failed['stage']=='failed' and failed['completed_records']==0
    assert 'get_demos' in failed['error'] and 'FileNotFoundError' in failed['error']
    for folder,name in [(parent,'status.json'),(parent,'fp16-stable-status.json')]:
        s=obj(folder/name);dead({'pid':s['controller_pid'],'start_ticks':s['controller_start_ticks']})
        assert s['stage'] in ['complete','bf16_generation_complete_fp16_pending']
    # The obsolete five-local-profile finalizer and exporter must remain stopped.
    superseded=obj(parent/'vti-author-old-watchers-superseded.json')
    for j in superseded['stopped']:dead(j)
    history.extend([{'pid':previous['pid'],'outcome':'failed_vti_start_before_predictions'},
        {'pid':recovery['pid'],'outcome':'adopted_recovery_complete_exit_code_unavailable'}])
    return workers,history


def vti_assets(parent,vti,protocol,datasets,development_hashes):
    from run_qwen_original_vti import verify_author,AUTHOR
    sealed(protocol);assert protocol['author_commit']==VTI_COMMIT
    parent_protocol=obj(parent/'protocol.json')
    assert protocol['parent_protocol_sha256']==parent_protocol['sha256']
    assert protocol['model']=={**parent_protocol['model'],'dtype':'float16'}
    assert protocol['method']=='vti_author' and protocol['gpu_indices']==[0]
    assert protocol['attention_amendment_sha256']==sha(parent/'fp16-attention-amendment.json')
    assert protocol['author_sources']==verify_author()
    assert sha(ROOT/'scripts/run_qwen_original_vti.py')==protocol['worker_source_sha256']==sha(vti/'worker-source.py')
    for b in ['pope','chair']:assert sha(parent/'data'/(b+'.jsonl'))==protocol['data_manifest_hashes'][b]
    assert sha(vti/'demo-manifest.json')==protocol['demo_manifest_sha256']
    demo=obj(vti/'demo-manifest.json');source=AUTHOR/'experiments/data/hallucination_vti_demos.jsonl'
    assert sha(source)==demo['author_demos_sha256']
    selected=random.Random(42).sample(read(source),70)
    assert [r['id'] for r in demo['images']]==[r['id'] for r in selected]
    assert [r['image'] for r in demo['images']]==[r['image'] for r in selected]
    eval_ids={int(Path(r['image']).stem.rsplit('_',1)[-1]) for rows in datasets.values() for r in rows}
    assert not eval_ids&{int(r['id']) for r in selected}
    hashes=set()
    for r in demo['images']:
        path=vti/'data/train2014'/r['image'];assert sha(path)==r['sha256'];hashes.add(r['sha256'])
    assert not hashes&{r['image_sha256'] for rows in datasets.values() for r in rows}
    assert not hashes&development_hashes
    check=obj(vti/'cpu-adapter-check.json');assert check['status']=='passed'
    assert check['normalized_patch_packing_max_abs_delta']<1e-5 and check['original_vti_shape_adapter_passed']
    if 'cpu_adapter_check_sha256' in protocol:assert sha(vti/'cpu-adapter-check.json')==protocol['cpu_adapter_check_sha256']
    ready=obj(vti/'artifact-ready.json');digest=sha(vti/'vti-author.pt')
    assert ready['sha256']==digest and ready['author_commit']==VTI_COMMIT and ready['source_hashes']==verify_author()
    artifact=torch.load(vti/'vti-author.pt',map_location='cpu',weights_only=True)
    assert artifact['method']=='vti_author' and artifact['author_commit']==VTI_COMMIT
    assert artifact['author_source_hashes']==verify_author() and artifact['demo_manifest_sha256']==sha(vti/'demo-manifest.json')
    for n,value in [('alpha_image',.9),('alpha_text',.9),('num_demos',70),('num_trials',50),('mask_ratio',.99),('rank',1)]:
        assert artifact[n]==protocol[n]==value
    assert artifact['text_extraction_visual_vti_active'] and artifact['extraction_dtype']=='float16'
    assert set(artifact['original_functions'])=={'get_demos','mask_patches','obtain_visual_vti','obtain_textual_vti','PCA','add_vti_layers','VTILayer.forward'}
    vision,text=artifact['visual_direction'],artifact['textual_direction']
    from transformers import AutoConfig
    architecture=AutoConfig.from_pretrained(CHECKPOINT,revision=REVISION,local_files_only=True,trust_remote_code=False)
    text_config=getattr(architecture,'text_config',architecture);visual_config=architecture.vision_config
    width=getattr(visual_config,'hidden_size',getattr(visual_config,'embed_dim',None))
    grid=448//visual_config.patch_size
    assert vision.shape==(visual_config.depth+1,grid*grid,width)
    assert text.shape==(text_config.num_hidden_layers+1,text_config.hidden_size)
    assert vision.dtype==text.dtype==torch.float32
    assert torch.isfinite(vision).all() and torch.isfinite(text).all()
    assert all(v.float().norm()>0 for v in vision[1:]) and all(v.float().norm()>0 for v in text[1:])
    torch.testing.assert_close(torch.load(vti/'original-visual-direction.pt',map_location='cpu',weights_only=True),vision,rtol=0,atol=0)
    state=obj(vti/'status.json');assert state['demos']==70 and state['trials']==50 and state['vision_forward_calls']==3570
    assert state['source_hashes_after']==verify_author()
    device_proof(vti/'device-map.json','float16')
    preflight=obj(vti/'preflight.json');assert preflight['status']=='passed' and preflight['gpu_uuid']==GPU0
    assert preflight['artifact_sha256']==digest and preflight['model_dtype']=='float16' and preflight['attention_score_dtype']=='float32'
    assert preflight['original_hook_file'].endswith('VTI/vti_utils/llm_layers.py')
    assert preflight['original_pca_file'].endswith('VTI/vti_utils/pca.py')
    for kind in ['vision','text']:
        d=preflight['diagnostics'][kind];assert d['calls']>0 and d['changed_elements']>0 and d['max_abs_delta']>0
    return digest


def rows(parent,vti,protocol,vti_protocol,datasets,seals,workers,vti_digest):
    grouped={b:{m:{} for m in METHODS} for b in datasets};input_hashes={}
    tasks=obj(parent/'tasks.json');assert sha(parent/'tasks.json')==protocol['tasks_sha256']
    assert all(t['method']!='base_fp32' for t in tasks)
    def validate(row,ref,method,bench):
        assert all(row.get(k)==v for k,v in ref.items())
        assert row['method']==method and row['checkpoint']==CHECKPOINT and row['revision']==REVISION
        assert row['gpu_uuid']==GPU0 and isinstance(row['prediction'],str)
        assert type(row['generated_tokens']) is int and 0<row['generated_tokens']<=ref['max_new_tokens']
        assert ref['max_new_tokens']==(128 if bench=='pope' else 512)
        assert math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds']>0
        assert row['dtype']==('float16' if method in ['grains_clone','vti_author'] else 'bfloat16')
        if method=='vti_author':
            assert row['protocol_sha256']==vti_protocol['sha256'] and row['parent_protocol_sha256']==protocol['sha256']
            assert row['author_commit']==VTI_COMMIT and row['artifact_sha256']==vti_digest
            assert row['attention_score_dtype']=='float32' and row['attention_amendment_sha256']==sha(parent/'fp16-attention-amendment.json')
            assert row['original_hook_file'].endswith('VTI/vti_utils/llm_layers.py')
            for kind in ['vision','text']:
                d=row['original_vti_diagnostics'][kind]
                assert d['calls']>0 and d['changed_elements']>0 and math.isfinite(d['max_abs_delta']) and d['max_abs_delta']>0
        else:
            assert row['protocol_sha256']==protocol['sha256'] and row['model_family']=='qwen2_5_vl'
            assert row['worker'] in workers and row['artifact_sha256']==seals['methods'].get(method,{}).get('sha256')
            worker=workers[row['worker']]
            assert row['dtype']==worker['dtype'] and row['gpu_uuid']==worker['visible_devices']
            assert 'adopted_cache_sha256' not in row
            if method=='grains_clone':
                assert row['attention_score_dtype']=='torch.float32'
                assert row['fp16_attention_amendment_sha256']==sha(parent/'fp16-attention-amendment.json')
            if method in ['rudder_clone','grains_clone']:
                d=row['author_diagnostics'];assert d['calls']>0
                if method=='rudder_clone':
                    assert d['active_positions']>0 and d['card_norm']>0
                    assert d['layer']==(1 if bench=='pope' else 25) and d['alpha']==(8.0 if bench=='pope' else 5.0)
                    assert d['hook_source'].endswith('RUDDER/Qwen/qwen_methods.py')
                else:
                    assert d['layer']==27 and d['alpha']==10 and d['hook_source'].endswith('GrAInS/src/utils/steering.py')
        bucket=grouped[bench][method];assert row['id'] not in bucket;bucket[row['id']]=row
    for task in tasks:
        if task['method']=='vti':continue
        assert task['method'] in DONORS
        path=parent/'predictions'/(task['id']+'.jsonl');complete=obj(parent/'claims'/task['id']/'complete.json')
        assert sha(path)==complete['sha256'];part=read(path);refs=datasets[task['benchmark']][task['start']:task['stop']]
        assert len(part)==len(refs)==complete['records']
        for row,ref in zip(part,refs):validate(row,ref,task['method'],task['benchmark'])
        input_hashes[str(path)]=sha(path)
    for bench in ['pope','chair']:
        path=vti/(bench+'.jsonl');complete=obj(vti/(bench+'-complete.json'));part=read(path)
        assert sha(path)==complete['sha256'] and len(part)==complete['records']==len(datasets[bench])
        for row,ref in zip(part,datasets[bench]):validate(row,ref,'vti_author',bench)
        input_hashes[str(path)]=sha(path)
    for bench,methods in grouped.items():
        expected=[r['id'] for r in datasets[bench]]
        for method,bucket in methods.items():
            assert set(bucket)==set(expected);methods[method]=[bucket[i] for i in expected]
    assert sum(len(r) for methods in grouped.values() for r in methods.values())==17500
    return grouped,input_hashes


def score(output,protocol,grouped):
    """Use official original scorers and the established 10,000-image bootstrap."""
    import audit_hallucination_suite as helper
    data=ROOT/'data/hallucination-suite';cache=obj(data/'chair-scorer-state.json');source=ROOT/'upstream/chair/chair.py'
    assert cache['built_with_unmodified_constructor'] and sha(source)==cache['scorer_sha256']
    assert sha(data/'sources/annotations_trainval2014.zip')==protocol['dataset_provenance']['sources']['coco_annotations']['sha256']
    with zipfile.ZipFile(data/'sources/annotations_trainval2014.zip') as archive:
        for n,h in cache['annotation_hashes'].items():
            assert sha(data/'annotations'/n)==h
            with archive.open('annotations/'+n) as stream:
                hasher=hashlib.sha256()
                while chunk:=stream.read(1024*1024):hasher.update(chunk)
                assert hasher.hexdigest()==h
    module=_load_external_module(source,'original_chair_vti_replacement')
    scorer=module.CHAIR.__new__(module.CHAIR);scorer.__dict__.update(cache['state'])
    scorer.imid_to_objects={int(k):set(v) for k,v in cache['state']['imid_to_objects'].items()}
    metrics={};uncertainty={};flat=[]
    official=read(data/'sources/coco_pope_adversarial.json')
    assert sha(data/'sources/coco_pope_adversarial.json')==protocol['dataset_provenance']['sources']['coco_pope_adversarial.json']['sha256']
    for bench,methods in grouped.items():
        folder=output/bench;folder.mkdir();metrics[bench]={}
        for method,part in methods.items():
            if bench=='pope':
                assert len(part)==len(official)==3000
                assert all(r['metadata']['original_question']==q['text'] and r['label']==q['label'] for r,q in zip(part,official))
                value=helper.pope_metrics(part);helper.verify_upstream_pope(part,folder,method,'adversarial',value)
                value['strict']=evaluate_predictions(part)
            else:
                path=folder/(method+'.captions.json');save(path,[{'image_id':r['metadata']['image_id'],'caption':r['prediction']} for r in part])
                detail=scorer.compute_chair(str(path),'image_id','caption');save(folder/(method+'.chair.official.json'),detail)
                part=attach_chair_scores(part,detail);methods[method]=part;value=evaluate_predictions(part)
                assert math.isclose(value['CHAIR_s'],detail['overall_metrics']['CHAIRs'])
                assert math.isclose(value['CHAIR_i'],detail['overall_metrics']['CHAIRi'])
            value.update(at_token_cap=sum(r['generated_tokens']>=r['max_new_tokens'] for r in part),
                empty_outputs=sum(not r['prediction'].strip() for r in part),mean_generated_tokens=sum(r['generated_tokens'] for r in part)/len(part))
            metrics[bench][method]=value;helper.write_jsonl(folder/(method+'.jsonl'),part)
            flat.append({'benchmark':bench,'method':method,**{k:v for k,v in value.items() if not isinstance(v,dict)}})
        uncertainty[bench]=helper.intervals(methods,bench)
    columns=list(dict.fromkeys(k for r in flat for k in r))
    with (output/'metrics.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=columns);w.writeheader();w.writerows(flat)
    return metrics,uncertainty


def audit(parent,vti,output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')=='','Auditor is CPU-only'
    parent=parent.resolve();vti=vti.resolve();output=output.resolve()
    assert parent.parent==vti.parent==ROOT/'outputs' and output.parent==parent
    assert not output.exists(),'Use a new combined-report folder; do not overwrite a report'
    require_gemma_complete();protocol=verify_protocol(parent);sealed(protocol)
    vti_protocol=obj(vti/'protocol.json');sealed(vti_protocol)
    assert obj(parent/'original-vti-current.json')['protocol_sha256']==vti_protocol['sha256']
    assert Path(obj(parent/'original-vti-current.json')['output'])==vti
    workers,history=lifecycle(parent,vti)
    datasets,seals,development_hashes=parent_assets(parent,protocol)
    attention=validate_attention(parent)
    digest=vti_assets(parent,vti,vti_protocol,datasets,development_hashes)
    grouped,input_hashes=rows(parent,vti,protocol,vti_protocol,datasets,seals,workers,digest)
    output.mkdir();metrics,uncertainty=score(output,protocol,grouped)
    report={'status':'verified_complete','scope':{'methods':METHODS,'pope':'COCO adversarial 3000 questions','chair':'fixed 500 COCO val2014 images','total_responses':17500},
        'methods':METHODS,'protocol':protocol,'vti_author_protocol':vti_protocol,
        'vti_author':{'author_commit':VTI_COMMIT,'artifact_sha256':digest,'source_output':str(vti),
            'original_function_calls_and_finite_nonzero_directions_verified':True,'independent_pca_reconstruction_available':False,
            'pca_reconstruction_limitation':'VTI extraction did not persist all raw fit hidden states; original sources, directions, calls and provenance are verified.'},
        'metrics':metrics,'uncertainty':uncertainty,'fp16_attention_amendment':attention,
        'source_output':str(parent),'output':str(output),'created_at':time.time(),
        'input_prediction_sha256':input_hashes,'execution_history':history,
        'analysis_hashes':{'auditor':sha(__file__),'official_metric_helper':sha(ROOT/'scripts/audit_hallucination_suite.py')},
        'audit':{'all_17500_responses_verified':True,'official_scorers_used':True,'actual_workers_exited':True,
            'original_author_sources_unmodified':True,'source_input_artifact_hashes_verified':True,
            'fit50_raw_ig_hidden_and_pca_independently_verified':True,'only_gpu0_used_for_benchmark_responses':True,
            'no_base_fp32_generated_or_included':True,'no_gemma_responses_or_vectors_reused':True,
            'development_test_image_hash_overlap':0,'vti_demo_test_image_hash_overlap':0,
            'matched_precision_baselines':False,'mixed_precision_comparisons':['grains_clone','vti_author'],
            'excluded_historical_local_vti':True,'historical_parent_protocol_and_gpu23_ig_provenance_retained':True}}
    labels={'base':'Base','revis':'REVIS 자체 구현','rudder_clone':'RUDDER 원본 함수 연결','grains_clone':'GrAInS 원본 함수 연결','vti_author':'VTI 원저자 구현 연결'}
    md=['# Qwen2.5-VL-7B · POPE adversarial / CHAIR','','각 방법 POPE 3,000문항·CHAIR 500장, 전체 17,500응답. 평가 응답은 GPU 0에서 생성했다. 수치는 %.','',
        '| 방법 | 모델 정밀도 | POPE Acc ↑ | POPE F1 ↑ | CHAIR_s ↓ | CHAIR_i ↓ | 객체 recall ↑ |','|---|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        p,c=metrics['pope'][m],metrics['chair'][m];dtype='FP16' if m in ['grains_clone','vti_author'] else 'BF16'
        md.append(f"| {labels[m]} | {dtype} | {100*p['accuracy']:.2f} | {100*p['f1']:.2f} | {100*c['CHAIR_s']:.2f} | {100*c['CHAIR_i']:.2f} | {100*c['object_recall']:.2f} |")
    md.extend(['','VTI 자체 구현의 이전 점수는 제외했다. VTI는 고정 저자 commit의 PCA·방향 생성·MLP 개입·정규화 함수를 직접 호출하고 Qwen2.5 입출력과 ViT 배치 차원을 연결했다. 저자 데모 70개·seed42·마스크99%·각50회·rank1·시각/텍스트 강도0.9를 사용했다. 저자 저장소는 Qwen2.5 POPE/CHAIR 실행 경로를 제공하지 않으므로 논문 전체 환경 재현으로 표현하지 않는다.',
        '', 'GrAInS·VTI 모델 가중치와 attention 출력은 FP16, QK·mask·softmax 점수는 overflow 방지를 위해 FP32다. 다른 세 방법은 BF16이므로 방법 비교에는 정밀도 영향도 포함된다. REVIS는 자체 구현이다. RUDDER·GrAInS는 고정 저자 원본 함수를 호출한다.',
        '', '공식 POPE evaluate.py 및 원본 CHAIR Python3 포트로 채점했다. 전체 ID·입력·모델 revision·GPU0·생성 길이·원본 호출·artifact hash를 검증했다. 이미지 단위 paired bootstrap 10,000회의95% 구간은 report.json에 있다.',
        '', 'GrAInS의 50개 원시 IG·hidden-state 기록에서 PCA를 독립 재계산했다. VTI는 원시 fit hidden states를 모두 보존하지 않아 PCA 자체의 독립 재구성은 할 수 없다. 대신 원본 소스·3570회 시각 추출 호출·각 층의 유한한 비영 방향·원본 훅·모든 응답의 provenance를 검증했다.',
        '', '과거 GPU2·3에서 추출한 IG 일부의 이력과 명시적으로 종료한 작업·재시작 실패 기록을 보존했다. 종료 코드를 직접 관측하지 못한 인계 작업은 그 한계를 execution_history에 표시했다.'])
    (output/'REPORT_KO.md').write_text('\n'.join(md)+'\n')
    save(output/'report.json',report)
    print(json.dumps({'status':'verified_complete','output':str(output),'report_sha256':sha(output/'report.json'),'metrics':metrics},allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--vti',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit(args.parent,args.vti,args.output)
