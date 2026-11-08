"""Seal the GPU 1 amendment, snapshot sources, then start three shard workers."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent))
import run_qwen_original_vti as original
from qwen_gpu0_migration_api import ROOT,save,sha,verify_protocol
from qwen_vti_gpu1 import GPU1,SHARDS

V=ROOT/'outputs/qwen-original-vti-1793260357'
SOURCES=['scripts/qwen_vti_gpu1.py','scripts/run_qwen_original_vti.py','scripts/launch_qwen_vti_gpu1.py']

def main():
    os.chdir(ROOT)
    used=subprocess.check_output(['nvidia-smi','--id='+GPU1,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--id='+GPU1,'--query-compute-apps=pid','--format=csv,noheader'],text=True)
    assert int(used)<100 and not apps.strip(),('GPU 1 busy',used,apps)
    stop=json.loads((V/'user-stop-proof-20261029.json').read_text())
    assert stop['all_selected_instances_exited']
    for name,info in stop['files'].items():assert sha(V/name)==info['sha256'],name
    covered={'pope':set(range(2851))}
    chair=set(range(0,45))|set(range(60,115))|set(range(167,211))|set(range(227,281))|set(range(334,378))
    covered['chair']=chair
    new={'pope':set(),'chair':set()}
    for tasks in SHARDS.values():
        for b,s,e in tasks:
            assert not new[b]&set(range(s,e));new[b]|=set(range(s,e))
    assert not covered['pope']&new['pope'] and covered['pope']|new['pope']==set(range(3000))
    assert not covered['chair']&new['chair'] and covered['chair']|new['chair']==set(range(500))
    assert not (V/'gpu1-amendment.json').exists() and not (V/'gpu1').exists()
    snapshot=V/'gpu1-source-snapshot'
    for s in SOURCES:
        (snapshot/s).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/s,snapshot/s)
    amendment={'user_request':'Continue the user-stopped author-VTI run on GPU 1 and finish merge, scoring and verification',
        'parent_protocol_sha256':verify_protocol(original.PARENT)['sha256'],
        'vti_protocol_sha256':json.loads((V/'protocol.json').read_text())['sha256'],
        'artifact_sha256':sha(V/'vti-author.pt'),
        'gpu_index':1,'gpu_uuid':GPU1,'shards':{str(k):[list(t) for t in v] for k,v in SHARDS.items()},
        'stop_proof_sha256':sha(V/'user-stop-proof-20261029.json'),
        'preserved_prefix_files':{n:i['sha256'] for n,i in stop['files'].items()},
        'preserved_chair_indices':{'gpu0':[[0,45],[167,211],[334,378]],'gpu2':[[60,115]],'gpu3':[[227,281]]},
        'preserved_pope_indices':[[0,2851]],
        'source_hashes':{s:sha(ROOT/s) for s in SOURCES},'source_snapshot_dir':'gpu1-source-snapshot',
        'model_math_data_vector_precision_unchanged':True,
        'note':'GPU 1 is a different GPU model from GPU 0 and GPUs 2/3; fp16 numerics may differ slightly. Each row records gpu_uuid.',
        'created_at':time.time()}
    for s,d in amendment['source_hashes'].items():assert sha(snapshot/s)==d
    save(V/'gpu1-amendment.json',amendment)
    (V/'gpu1').mkdir()
    jobs=[]
    for shard in SHARDS:
        log=(V/'gpu1'/f'shard{shard}.log').open('ab')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU1,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','scripts/qwen_vti_gpu1.py','--output',str(V),'--shard',str(shard)],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        jobs.append({'pid':child.pid,'start_ticks':int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]),
            'shard':shard,'tasks':SHARDS[shard],'log':str(V/'gpu1'/f'shard{shard}.log'),'visible_devices':GPU1,
            'source_sha256':amendment['source_hashes']['scripts/qwen_vti_gpu1.py']})
    save(V/'gpu1'/'jobs.json',jobs)
    print(json.dumps({'amendment_sha256':sha(V/'gpu1-amendment.json'),'jobs':jobs}))

if __name__=='__main__':main()
