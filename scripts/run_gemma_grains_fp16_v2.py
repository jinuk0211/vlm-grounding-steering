"""Continue four BF16 profiles; replace GrAInS inference with fresh FP16."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from continue_gpu23_suite import GPU2,GPU3
from resume_gpu23_suite_20261029 import ticks,same_instance

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/author-clone-rerun-20261027-165446'
METHODS=['base','revis','vti','rudder_clone','grains_clone']
SOURCES=['src/vlm_grounding/backend.py','src/vlm_grounding/baselines.py','src/vlm_grounding/cli.py','src/vlm_grounding/evaluation.py',
         'scripts/author_clone_bridge.py','scripts/grains_fp16_worker_v2.py','scripts/run_gemma_grains_fp16_v2.py',
         'scripts/audit_five_profile_suite_v2.py','scripts/audit_hallucination_suite.py','upstream/chair/chair.py',
         'data/hallucination-suite/sources/pope_evaluate.py']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def free_selected_gpus():
    lines=subprocess.check_output(['nvidia-smi','-i','2,3','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    rows=[tuple(s.strip() for s in line.split(',')) for line in lines]
    assert {(i,u) for i,u,m in rows}=={('2',GPU2),('3',GPU3)}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines()
    occupied={s.strip() for s in apps}
    available=[('gpu'+i,u) for i,u,m in rows if u not in occupied and int(m)>=21*1024]
    assert available,'Both authorized GPUs occupied; do not interfere or use another GPU'
    return available

def freeze(output):
    from author_clone_bridge import verify_clone_sources
    old=json.loads((OLD/'protocol.json').read_text())
    for file,digest in old['source_hashes'].items():assert sha(ROOT/file)==digest,file
    for name in ['data','predictions','claims','leases','source-snapshot']: (output/name).mkdir()
    for file in (OLD/'data').glob('*.jsonl'):shutil.copy2(file,output/'data'/file.name)
    shutil.copy2(OLD/'config.json',output/'config.json')
    tasks=[dict(t,dtype='float16' if t['method']=='grains_clone' else 'bfloat16') for t in json.loads((OLD/'tasks.json').read_text()) if t['method'] in METHODS]
    save(output/'tasks.json',tasks)
    sources={f:sha(ROOT/f) for f in SOURCES}
    for name in SOURCES:
        dest=output/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    protocol={'scope':'Gemma five-profile POPE adversarial3000 and CHAIR fixed500; user-requested FP16 GrAInS; no Base FP32',
        'methods':METHODS,'model':old['model'],'method_dtypes':{m:'float16' if m=='grains_clone' else 'bfloat16' for m in METHODS},
        'method_parameters':old['method_parameters'],'generation':old['generation'],'source_hashes':sources,
        'author_sources':verify_clone_sources(),'source_protocol_sha256':old['sha256'],'source_output':str(OLD),
        'source_protocol_file_sha256':sha(OLD/'protocol.json'),'gpu_indices':[2,3],'other_gpu_fallback':False,
        'grains_artifact':str(OLD/'grains-author.pt'),'grains_artifact_sha256':sha(OLD/'grains-author.pt'),
        'grains_extraction_dtype':'float32','grains_inference_dtype':'float16','steering_vector_dtype':'float32','precision_note':'Original FP32 vector retained; model weights and decoder activations FP16','excluded_profiles':['base_fp32'],
        'comparisons':{m:'base' for m in METHODS if m!='base'},'mixed_precision_comparisons':['grains_clone'],
        'dataset_provenance':old['dataset_provenance'],'chair_scoring':old['chair_scoring'],'deferred':old['deferred'],
        'manifest_hashes':{p.name:sha(p) for p in (output/'data').glob('*.jsonl')},
        'config_sha256':sha(output/'config.json'),'tasks_sha256':sha(output/'tasks.json'),
        'retained_artifact_hashes':old['retained_artifact_hashes'],
        'request':'2026-10-29 user: GrAInS FP16, Base FP32 unnecessary',
        'limitations':['REVIS and VTI are retained local adaptations','GrAInS FP16 versus Base BF16 differs in precision; cannot isolate only method effect'],
        'created_at':time.time()}
    protocol['sha256']=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest();save(output/'protocol.json',protocol)
    # Preserve the original rows/files. New copies explicitly identify every adopted line.
    for task in tasks:
        if task['method']=='grains_clone':continue
        source=OLD/'predictions'/(task['id']+'.jsonl');data=source.read_bytes();assert data.endswith(b'\n')
        marker=json.loads((OLD/'claims'/task['id']/'complete.json').read_text());assert sha(source)==marker['sha256']
        lines=[line for line in data.splitlines() if line.strip()];assert len(lines)==marker['records']==task['stop']-task['start']
        dest=output/'predictions'/source.name
        with dest.open('x') as sink:
            for index,line in enumerate(lines):
                row=json.loads(line);assert row['method']==task['method'] and row['dtype']=='bfloat16' and row['protocol_sha256']==old['sha256']
                new={**row,'protocol_sha256':protocol['sha256'],'adopted_from_protocol':old['sha256'],
                    'source_prediction_file':str(source),'source_prediction_file_sha256':sha(source),
                    'source_row_index':index,'source_row_sha256':hashlib.sha256(line).hexdigest()}
                sink.write(json.dumps(new,allow_nan=False)+'\n')
        claim=output/'claims'/task['id'];claim.mkdir();save(claim/'complete.json',{'worker':'adopted-verified-bf16','records':len(lines),'sha256':sha(dest)})
    save(output/'adoption.json',{'status':'verified_complete','records':14000,'methods':METHODS[:4],
        'original_raw_bytes_unchanged':True,'source_protocol_sha256':old['sha256'],'created_at':time.time()})
    return protocol

def run(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and output.resolve().parent==ROOT/'outputs'
    lock=(ROOT/'outputs/.five-profile-gemma-controller.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    output.mkdir();state={'controller_pid':os.getpid(),'controller_start_ticks':ticks(os.getpid()),'started_at':time.time(),
        'gpu_indices':[2,3],'base_fp32_enabled':False,'grains_inference_dtype':'float16'}
    def update(**kw):state.update(kw,updated_at=time.time());save(output/'status.json',state)
    jobs=[];children=[]
    def launch(worker,visible,preflight=False):
        script='scripts/grains_fp16_worker_v2.py'
        args=[str(ROOT/'.venv/bin/python'),script,'--output',str(output),'--worker',worker]
        if preflight:args.append('--preflight')
        log=output/(worker+'.log')
        with log.open('xb') as stream:p=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=visible),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        job={'pid':p.pid,'start_ticks':ticks(p.pid),'worker':worker,'visible_devices':visible,'dtype':'float16','preflight':preflight,'log':str(log)}
        jobs.append(job);children.append(p);save(output/'jobs.json',jobs);return p,job
    def wait(group,stage):
        while True:
            progress=[];done=True
            for p,j in group:
                code=p.poll();done &= code is not None
                f=output/('preflight.json' if j['preflight'] else 'worker-'+j['worker']+'.json')
                ws=json.loads(f.read_text()) if f.exists() else {}
                assert code in [None,0] and ws.get('stage')!='failed',{'job':j,'code':code,'state':ws}
                if code==0:assert ws['stage']=='complete',ws
                progress.append({'pid':j['pid'],'worker':j['worker'],'returncode':code,'state':ws})
            update(stage=stage,workers=progress)
            if done:return
            time.sleep(10)
    try:
        update(stage='waiting_for_existing_bf16')
        save(ROOT/'outputs/five-profile-gemma-current.json',{'output':str(output),'controller_pid':os.getpid(),'controller_start_ticks':state['controller_start_ticks']})
        original_jobs=[j for j in json.loads((OLD/'jobs.json').read_text()) if j.get('resume_epoch')=='resume-20261029-1793244374']
        while True:
            active=[]
            for j in original_jobs:
                ws=json.loads((OLD/('worker-'+j['worker']+'.json')).read_text());alive=same_instance(dict(j,output=str(OLD)))
                assert ws['stage']!='failed' and (alive or ws['stage']=='complete'),ws
                active.append(alive)
            if not any(active):break
            update(stage='waiting_for_existing_bf16',retained_workers=original_jobs);time.sleep(10)
        available=free_selected_gpus();protocol=freeze(output);update(stage='preflight_fp16',protocol_sha256=protocol['sha256'])
        wait([launch('fp16-preflight',available[0][1],True)],'preflight_fp16');available=free_selected_gpus()
        tag=str(int(time.time()))
        group=[launch(label+'-grains-fp16-'+tag,visible) for label,visible in available]
        wait(group,'generating_grains_fp16');update(stage='auditing')
        with (output/'audit.log').open('xb') as stream:subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/audit_five_profile_suite_v2.py','--output',str(output)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT,check=True)
        assert json.loads((output/'report.json').read_text())['status']=='verified_complete'
        update(stage='complete',completed_at=time.time(),report_sha256=sha(output/'report.json'))
    except BaseException:
        for p in children:
            if p.poll() is None:p.terminate()
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
