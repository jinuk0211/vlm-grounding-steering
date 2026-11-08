"""Wait on actual GPU0 workers, audit the replacement suite, then export bytes."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path

ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
PARENT=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386'
GPU0='GPU-4a6e1799-a39c-86ed-abdf-afad03791c12'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');temp.replace(p)
def process(pid,start):
    p=Path('/proc')/str(pid)
    try:
        fields=(p/'stat').read_text().split(') ',1)[1].split()
        uid=p.stat().st_uid
    except FileNotFoundError:return {'live':False,'reason':'missing'}
    if int(fields[19])!=start:return {'live':False,'reason':'pid_reused'}
    assert uid==os.getuid()
    return {'live':fields[0] not in ['Z','X'],'state':fields[0],'start_ticks':int(fields[19])}

def snapshot(parent,vti):
    original=read(parent/'jobs.json');half=read(parent/'fp16-stable-jobs.json');new=read(vti/'resumed-jobs.json')
    targets=[]
    for pid in [2700448,2700449]:
        j=next(x for x in original if x['pid']==pid)
        targets.append({'pid':pid,'start_ticks':j['start_ticks'],'kind':'parent_bf16','state_path':parent/j['state_file']})
    j=next(x for x in half if x['kind']=='generation')
    targets.append({'pid':j['pid'],'start_ticks':j['start_ticks'],'kind':'grains_fp16','state_path':parent/j['state_file']})
    for name in ['author_job','adopted_recovery_job']:
        j=new[name];folder=vti if name=='author_job' else parent
        targets.append({'pid':j['pid'],'start_ticks':j['start_ticks'],'kind':name,
                        'state_path':vti/'status.json' if name=='author_job' else folder/j['state_file']})
    extra_jobs=vti/'grains-extra-jobs.json'
    extra_status=vti/'grains-extra-status.json'
    if extra_jobs.exists():
        for j in read(extra_jobs):
            assert j['kind']=='generation' and j['dtype']=='float16' and j['visible_devices']==GPU0,j
            assert j.get('output',str(parent))==str(parent),j
            targets.append({'pid':j['pid'],'start_ticks':j['start_ticks'],'kind':'grains_extra_fp16',
                            'state_path':parent/j['state_file']})
        assert extra_status.is_file(),'Extra GrAInS jobs exist without controller state'
    assert len({(t['pid'],t['start_ticks']) for t in targets})==len(targets),'Duplicate worker identity'
    states=[]
    for t in targets:
        p=process(t['pid'],t['start_ticks']);s=read(t['state_path'])
        assert s['pid']==t['pid']
        assert s['stage']!='failed',{'target':t,'state':s}
        if p['live']:
            try:
                args=(Path('/proc')/str(t['pid'])/'cmdline').read_bytes().split(b'\0')
                env=(Path('/proc')/str(t['pid'])/'environ').read_bytes().split(b'\0')
            except FileNotFoundError:
                p=process(t['pid'],t['start_ticks']);args=env=[]
            if any(args) and any(env):
                assert b'--output' in args and (str(parent).encode() in args or str(vti).encode() in args)
                assert ('CUDA_VISIBLE_DEVICES='+GPU0).encode() in env
            else:
                # Linux can clear cmdline/environ while a process is exiting.
                # Re-poll its identity; a live transitional handle stays pending.
                p=process(t['pid'],t['start_ticks'])
        if not p['live']:
            s=read(t['state_path'])
            assert s['pid']==t['pid'] and s['stage']=='complete',{'target':t,'state':s,'process':p}
        states.append({k:str(v) if isinstance(v,Path) else v for k,v in t.items()}|p|
            {'stage':s['stage'],'completed_records':s.get('completed_records'),'vision_forward_calls':s.get('vision_forward_calls')})
    if extra_status.exists():
        s=read(extra_status)
        pid=s.get('controller_pid',s.get('pid'))
        start=s.get('controller_start_ticks',s.get('start_ticks'))
        assert isinstance(pid,int) and isinstance(start,int),s
        p=process(pid,start)
        assert s['stage']!='failed',s
        if p['live']:
            try:
                args=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
                env=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
            except FileNotFoundError:
                p=process(pid,start);args=env=[]
            if any(args) and any(env):
                assert any(a in args for a in [b'scripts/run_qwen_grains_extra_workers.py',
                    str(ROOT/'scripts/run_qwen_grains_extra_workers.py').encode()])
                assert b'CUDA_VISIBLE_DEVICES=' in env,'Extra controller must not load a GPU model'
            else:p=process(pid,start)
        if not p['live']:
            s=read(extra_status)
            assert s.get('controller_pid',s.get('pid'))==pid and s['stage']=='complete',{'controller':s,'process':p}
            assert s.get('controller_start_ticks',s.get('start_ticks'))==start,s
        states.append({'pid':pid,'start_ticks':start,'kind':'grains_extra_controller',
            'state_path':str(extra_status),'stage':s['stage']}|p)
    return states

def export(parent,vti,output):
    report=read(output/'report.json');assert report['status']=='verified_complete'
    assert report['audit']['all_17500_responses_verified']
    # Add provenance without duplicating large direction tensors or image data.
    def copy(src,name):
        dst=output/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert sha(dst)==sha(src),str(dst)
        else:shutil.copy2(src,dst)
    for name in ['protocol.json','tasks.json','config.json','artifacts-ready.json','grains-author.pt',
                 'original-hidden-states.npz','extraction.jsonl']:
        assert (parent/name).is_file(),name;copy(parent/name,name)
    for path in (parent/'data').glob('*.jsonl'):copy(path,'data/'+path.name)
    for path in (parent/'artifacts').glob('*.pt'):copy(path,'artifacts/'+path.name)
    for path in parent.glob('*.json'):copy(path,'parent-provenance/'+path.name)
    for folder in ['source-snapshot','finalization-source-snapshot','author-snapshot']:
        source=parent/folder
        if source.is_dir():
            for path in source.rglob('*.py'):copy(path,'parent-source/'+folder+'/'+str(path.relative_to(source)))
    for name in ['protocol.json','vti-author.pt','demo-manifest.json','artifact-ready.json','preflight.json',
                 'device-map.json','status.json','resumed-jobs.json','resume-controller-status.json',
                 'replacement-jobs.json','transition.json','cpu-adapter-check.json','worker-source.py']:
        assert (vti/name).is_file(),name;copy(vti/name,'vti-author/'+name)
    if (vti/'grains-extra-jobs.json').exists() or (vti/'grains-extra-status.json').exists():
        for name in ['grains-extra-jobs.json','grains-extra-status.json']:
            assert (vti/name).is_file(),name
        for path in sorted(vti.glob('grains-extra-*')):
            if path.is_file() and path.suffix in ['.json','.py']:
                copy(path,'vti-author/metadata/'+path.name)
            elif path.is_dir():
                for item in path.rglob('*'):
                    if item.is_file() and item.suffix in ['.json','.py']:
                        copy(item,'vti-author/metadata/'+path.name+'/'+item.relative_to(path).as_posix())
        copy(ROOT/'scripts/run_qwen_grains_extra_workers.py','analysis-source/run_qwen_grains_extra_workers.py')
    for script in ['audit_qwen_vti_replacement.py','finalize_qwen_vti_replacement.py',
                   'run_qwen_original_vti.py','resume_original_vti.py','launch_original_vti_replacement.py']:
        copy(ROOT/'scripts'/script,'analysis-source/'+script)
    author=ROOT/'upstream-author-20261026/VTI'
    for path in author.rglob('*.py'):
        if '.git' not in path.parts:copy(path,'vti-author/author-source/'+str(path.relative_to(author)))
    for name in ['README.md','experiments/data/hallucination_vti_demos.jsonl']:
        copy(author/name,'vti-author/author-source/'+name)
    paths=sorted(p for p in output.rglob('*') if p.is_file() and p.name not in
        ['verified-results.zip','verified-results.zip.partial','export-manifest.json'])
    files={p.relative_to(output).as_posix():sha(p) for p in paths}
    manifest={'status':'verified_complete_export','report_sha256':sha(output/'report.json'),'files':files,
        'total_responses':17500,'methods':['base','revis','rudder_clone','grains_clone','vti_author'],
        'exporter_source_sha256':sha(__file__),'created_at':time.time()}
    save(output/'export-manifest.json',manifest)
    partial=output/'verified-results.zip.partial';target=output/'verified-results.zip'
    with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for path in paths+[output/'export-manifest.json']:archive.write(path,path.relative_to(output).as_posix())
    partial.replace(target)
    return {'archive':str(target),'archive_sha256':sha(target),'archive_bytes':target.stat().st_size,
        'report_sha256':sha(output/'report.json'),'export_manifest_sha256':sha(output/'export-manifest.json'),
        'file_count':len(paths)+1}

def main(snapshot_only=False):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    parent=PARENT;vti=Path(read(parent/'original-vti-current.json')['output'])
    output=parent/'original-vti-full-final'
    if snapshot_only:
        print(json.dumps(snapshot(parent,vti)));return
    lock=(vti/'replacement-finalizer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'pid':os.getpid(),'start_ticks':int((Path('/proc')/str(os.getpid())/'stat').read_text().split(') ',1)[1].split()[19]),
        'created_at':time.time(),'gpu_indices':[0],'output':str(output),'vti_output':str(vti),
        'source_sha256':sha(__file__),'auditor_sha256':sha(ROOT/'scripts/audit_qwen_vti_replacement.py')}
    def update(**kw):state.update(kw,updated_at=time.time());save(vti/'replacement-finalization-status.json',state)
    try:
        while True:
            states=snapshot(parent,vti);update(stage='waiting_for_actual_gpu0_workers',workers=states)
            if not any(s['live'] for s in states):
                # The resumed controller captures the author job's actual exit
                # code and explicitly records the adopted recovery completion.
                ctrl=read(vti/'resume-controller-status.json')
                assert ctrl['stage']!='failed',ctrl
                if ctrl['stage']=='complete':break
            time.sleep(30)
        update(stage='auditing_full_replacement_suite')
        log=vti/'replacement-audit.log'
        with log.open('ab') as stream:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','scripts/audit_qwen_vti_replacement.py',
                '--parent',str(parent),'--vti',str(vti),'--output',str(output)],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdout=stream,stderr=subprocess.STDOUT)
            update(auditor_pid=child.pid)
            code=child.wait()
        assert code==0,{'auditor_exit_code':code,'log':str(log)}
        assert sha(ROOT/'scripts/audit_qwen_vti_replacement.py')==state['auditor_sha256']
        update(stage='exporting_verified_replacement_suite')
        result=export(parent,vti,output);update(stage='verified_complete_export',**result,completed_at=time.time())
        print(json.dumps(result))
    except BaseException:update(stage='failed',error=traceback.format_exc());raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot-only',action='store_true');main(p.parse_args().snapshot_only)
