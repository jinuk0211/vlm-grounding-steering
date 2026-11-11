"""Record a GPU 1 shard interruption, cut only its unterminated tail line, and resume the same worker.

The worker source and gpu1-amendment.json are unchanged; qwen_vti_gpu1.py resumes from
an exact newline-terminated prefix of its own shard rows.
"""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent))
from qwen_gpu0_migration_api import ROOT,save,sha
from qwen_vti_gpu1 import GPU1,SHARDS

V=ROOT/'outputs/qwen-original-vti-1793260357'

def live(pid,start):
    try:fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
    except FileNotFoundError:return False
    return fields[0] not in ('Z','X') and int(fields[19])==start

def main(label,reason):
    os.chdir(ROOT);folder=V/'gpu1';amendment=json.loads((V/'gpu1-amendment.json').read_text())
    assert amendment['source_hashes']['scripts/qwen_vti_gpu1.py']==sha(ROOT/'scripts/qwen_vti_gpu1.py')
    jobs=json.loads((folder/'jobs.json').read_text())
    for j in jobs:assert not live(j['pid'],j['start_ticks']),j
    apps=subprocess.check_output(['nvidia-smi','--id='+GPU1,'--query-compute-apps=pid','--format=csv,noheader'],text=True)
    assert not apps.strip(),('GPU 1 busy',apps)
    incident=folder/('incident-'+label);incident.mkdir()
    record={'label':label,'reason':reason,'recorded_at':time.time(),'previous_jobs':jobs,'shards':{}}
    for k in SHARDS:
        rows=folder/f'shard{k}.jsonl';raw=rows.read_bytes();keep=raw[:raw.rfind(b'\n')+1]
        for name in [f'shard{k}-status.json',f'shard{k}.log']:shutil.copy2(folder/name,incident/name)
        (incident/f'shard{k}.unterminated-tail.bin').write_bytes(raw[len(keep):])
        complete=[json.loads(l) for l in keep.splitlines() if l.strip()]
        record['shards'][str(k)]={'bytes_before':len(raw),'sha256_before':sha(rows),'bytes_kept':len(keep),
            'complete_rows_kept':len(complete),'unterminated_tail_bytes':len(raw)-len(keep)}
        if len(keep)!=len(raw):rows.write_bytes(keep)
        record['shards'][str(k)]['sha256_after']=sha(rows)
        tmp=folder/f'shard{k}-status.json.tmp'
        if tmp.exists():assert tmp.stat().st_size==0;tmp.unlink()
    save(incident/'incident.json',record)
    shutil.copy2(folder/'jobs.json',incident/'jobs.json')
    hf=str(ROOT.parent/'.cache/huggingface');new=[]
    for k in SHARDS:
        log=(folder/f'shard{k}.log').open('ab')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU1,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 HF_HOME=hf,HF_HUB_CACHE=hf+'/hub')
        child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','scripts/qwen_vti_gpu1.py','--output',str(V),'--shard',str(k)],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        new.append({'pid':child.pid,'start_ticks':int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]),
            'shard':k,'tasks':SHARDS[k],'log':str(folder/f'shard{k}.log'),'visible_devices':GPU1,
            'source_sha256':amendment['source_hashes']['scripts/qwen_vti_gpu1.py'],
            'resumed_after_incident':label,'resume_source_sha256':sha(__file__)})
    save(folder/'jobs.json',new)
    print(json.dumps({'incident':record,'jobs':new}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--reason',required=True)
    a=p.parse_args();main(a.label,a.reason)
