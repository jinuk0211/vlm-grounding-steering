"""Frozen author VTI extraction and full Gemma POPE comparison, GPU 2/3 only."""
import argparse, concurrent.futures, contextlib, hashlib, json, os, random, shutil
import subprocess, sys, time, traceback, urllib.request, zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
RUN=Path(__file__).resolve().parents[1]
GPU={2:'GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde',3:'GPU-4c593ff4-fcc4-39c9-d7b4-981dabee1cb3'}
GEMMA=ROOT/'outputs/gemma-grains-fp16-v2-20261029-1793246507'
sys.path[:0]=[str(RUN/'deps'),str(ROOT/'src'),str(ROOT/'scripts')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def state(name,**kw):
 p=RUN/(name+'.json');x=json.loads(p.read_text()) if p.exists() else {'pid':os.getpid(),'started_at':time.time()}
 x.update(kw,updated_at=time.time());save(p,x)
def verify():
 p=json.loads((RUN/'protocol.json').read_text());body={k:v for k,v in p.items() if k!='sha256'}
 assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==p['sha256']
 for name,h in p['source_hashes'].items():assert sha(RUN/name)==h,name
 for name,h in p['runtime_source_hashes'].items():assert sha(ROOT/name)==h,name
 assert sha(RUN/'pope.jsonl')==p['pope_sha256']
 assert sha(RUN/'config.json')==p['config_sha256']
 assert sha(RUN/'development.json')==p['development_sha256']
 assert sha(GEMMA/'pope/base.jsonl')==p['base_prediction_sha256']
 assert sha(GEMMA/'pope/vti.jsonl')==p['local_vti_prediction_sha256']
 return p
def gpu_free(index):
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 return not any(line.startswith(GPU[index]) for line in apps.splitlines())
def wait_gpu(index,name):
 while not gpu_free(index):state(name,stage='waiting_for_free_gpu',gpu=index);time.sleep(10)
def audited_loader(config,name):
 import torch
 import vti_author_gemma as b
 from huggingface_hub import try_to_load_from_cache
 a=ORIGINAL_LOADER(config)
 c=config['model'];cached=try_to_load_from_cache(c['checkpoint'],'config.json',revision=c['revision'])
 assert isinstance(cached,str) and Path(cached).parent.name==c['revision']
 observed=getattr(a.model.config,'_commit_hash',None);assert observed in [None,c['revision']]
 a.model.config._commit_hash=c['revision']
 devices=sorted({str(p.device) for p in a.model.parameters()});dtypes=sorted({str(p.dtype) for p in a.model.parameters() if p.is_floating_point()})
 assert devices==['cuda:0'] and dtypes==['torch.bfloat16']
 save(RUN/(name+'-device.json'),{'visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'devices':devices,'parameter_dtypes':dtypes,'checkpoint':c['checkpoint'],'revision':c['revision'],'observed_revision_metadata':observed,'cached_config':cached,'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21]})
 return a
ORIGINAL_LOADER=None
def configure(name):
 global ORIGINAL_LOADER
 import torch, vti_author_gemma as b
 assert os.environ['CUDA_VISIBLE_DEVICES'] in GPU.values() and torch.cuda.device_count()==1
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ORIGINAL_LOADER=b.load_gemma;b.load_gemma=lambda c:audited_loader(c,name)
 return b
def extraction():
 p=verify();b=configure('extraction');from vti_author import verify_source
 original=b.VisionInputs.__getitem__
 def observed(self,index):
  if self.index!=index:state('extraction',stage='visual_extraction',completed_images=index,total_images=len(self))
  return original(self,index)
 b.VisionInputs.__getitem__=observed
 original_text=b.TextInputs.__getitem__
 def text_observed(self,index):
  if self.index!=index:state('extraction',stage='text_extraction_with_visual_vti',completed_images=index,total_images=len(self))
  return original_text(self,index)
 b.TextInputs.__getitem__=text_observed
 args=SimpleNamespace(output=RUN/'vti-author.pt',overwrite=False,repo=RUN/'upstream/vti-author',image_root=RUN/'images',config=RUN/'config.json',num_demos=70,num_trials=50,mask_ratio=.99,alpha_text=.9,alpha_image=.9,seed=42)
 state('extraction',stage='loading_model');result=b.extract(args,verify_source(args.repo))
 state('extraction',stage='complete',artifact_sha256=sha(args.output),result=result)
def worker(index):
 import torch
 p=verify();name=f'worker-gpu{index}';b=configure(name)
 from vlm_grounding.cli import _seed,_validate_artifact
 from vti_author import verify_source
 config=json.loads((RUN/'config.json').read_text());records=read(RUN/'pope.jsonl')[index-2::2]
 artifact_digest=sha(RUN/'vti-author.pt');artifact=torch.load(RUN/'vti-author.pt',map_location='cpu',weights_only=True)
 assert artifact['author_source']==verify_source(RUN/'upstream/vti-author')
 _validate_artifact(artifact,config['model'],records)
 assert set(x['image_sha256'] for x in artifact['training_images']).isdisjoint(x['image_sha256'] for x in records)
 state(name,stage='loading_model',completed_records=0,total_records=len(records));_seed(config);adapter=b.load_gemma(config)
 _validate_artifact(artifact,config['model'],records,adapter);author=b.load_author(RUN/'upstream/vti-author')
 counter={'calls':0};orig=b.DtypeBridge.forward
 def counted(self,x):counter['calls']+=1;return orig(self,x)
 b.DtypeBridge.forward=counted
 # Real held-out development preflight, before any test output is committed.
 from vti_author import demos
 d=demos(RUN/'upstream/vti-author',RUN/'images',70)[0]
 batch=adapter.prepare(d['image_path'],'Please describe this image in detail.');seen={'calls':0,'changed_elements':0};handles=[]
 with b.intervention(adapter,author,text=artifact['text_direction'],vision=artifact['vision_direction']):
  def check(module,args,out):
   assert torch.isfinite(out).all() and out.dtype==args[0].dtype
   seen['calls']+=1;seen['changed_elements']+=int((out!=args[0]).sum())
  for layer in [*adapter.layers,*adapter.vision_layers]:handles.append(layer.mlp[1].register_forward_hook(check))
  try:response=adapter.generate(batch,max_new_tokens=8)
  finally:
   for h in handles:h.remove()
 assert response.strip() and seen['calls'] and seen['changed_elements']
 save(RUN/(name+'-preflight.json'),{'status':'passed','native_dtype':'bfloat16','response':response,**seen,'benchmark_rows_written':0})
 out=RUN/(name+'.jsonl')
 with out.open('x') as sink:
  for n,r in enumerate(records):
   assert sha(r['image'])==r['image_sha256'];_seed(config);batch=adapter.prepare(r['image'],r['prompt']);counter['calls']=0
   torch.cuda.synchronize();start=time.perf_counter()
   with b.intervention(adapter,author,text=artifact['text_direction'],vision=artifact['vision_direction'],alpha_text=.9,alpha_image=.9):prediction=adapter.generate(batch,max_new_tokens=128)
   torch.cuda.synchronize();assert prediction.strip() and 0<adapter.last_generated_tokens<=128 and counter['calls']>0
   row={**r,'prediction':prediction,'method':'vti_author','dtype':'bfloat16','protocol_sha256':p['sha256'],'artifact_sha256':artifact_digest,'gpu_uuid':GPU[index],'generated_tokens':adapter.last_generated_tokens,'elapsed_seconds':time.perf_counter()-start,'author_mlp_calls':counter['calls']}
   sink.write(json.dumps(row,allow_nan=False)+'\n');sink.flush();state(name,stage='generating',completed_records=n+1,total_records=len(records),last_seconds=row['elapsed_seconds'])
 assert verify_source(RUN/'upstream/vti-author')==artifact['author_source'];state(name,stage='complete',completed_records=len(records),prediction_sha256=sha(out))
def run_child(mode,index):
 name=f'{mode}-gpu{index}';wait_gpu(index,'status');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=GPU[index]
 log=(RUN/(name+'.log')).open('x');proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--mode',mode,'--gpu',str(index)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 job={'pid':proc.pid,'start_ticks':Path(f'/proc/{proc.pid}/stat').read_text().split()[21],'mode':mode,'gpu':index,'visible_devices':GPU[index]};save(RUN/(name+'-job.json'),job)
 return proc,log
def score():
 from audit_hallucination_suite import pope_metrics,verify_upstream_pope,intervals
 p=verify();manifest=read(RUN/'pope.jsonl');allrows=read(RUN/'worker-gpu2.jsonl')+read(RUN/'worker-gpu3.jsonl');by={x['id']:x for x in allrows}
 assert len(allrows)==len(by)==3000;rows=[by[r['id']] for r in manifest]
 for x,r in zip(rows,manifest):
  assert all(x[k]==v for k,v in r.items());assert x['protocol_sha256']==p['sha256'] and x['artifact_sha256']==sha(RUN/'vti-author.pt')
 (RUN/'vti-author.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
 base=read(GEMMA/'pope/base.jsonl');old=read(GEMMA/'pope/vti.jsonl');assert [x['id'] for x in rows]==[x['id'] for x in base]==[x['id'] for x in old]
 for r,a,b in zip(manifest,base,old):assert all(a[k]==v==b[k] for k,v in r.items())
 collection={'base':base,'vti_local':old,'vti_author':rows};metrics={k:pope_metrics(v) for k,v in collection.items()}
 for k,v in collection.items():verify_upstream_pope(v,RUN,k,'adversarial',metrics[k])
 report={'status':'verified_complete','protocol':p,'metrics':metrics,'uncertainty':intervals(collection,'pope'),'all_3000_original_vti_outputs_verified':True,'base_sha256':sha(GEMMA/'pope/base.jsonl'),'vti_local_sha256':sha(GEMMA/'pope/vti.jsonl'),'artifact_sha256':sha(RUN/'vti-author.pt'),'scope':'Gemma architecture adaptation of unmodified VTI functions; not original LLaVA benchmark reproduction'}
 save(RUN/'report.json',report)
 lines=['# Gemma E4B — 원본 VTI POPE adversarial','','동일3,000문항·BF16·448px·greedy·128토큰·문항별 seed42. 수치는 %.','','| 방법 | Acc | Precision | Recall | F1 |','|---|---:|---:|---:|---:|']
 for k,label in [('base','Base'),('vti_local','VTI 자체 구현'),('vti_author','VTI 원본 함수 + Gemma 연결')]:lines.append('| '+label+' | '+' | '.join(f'{metrics[k][v]*100:.4f}' for v in ['accuracy','precision','recall','f1'])+' |')
 delta=100*(metrics['vti_author']['f1']-metrics['base']['f1']);ci=[100*x for x in report['uncertainty']['vti_author']['f1']['delta_ci95']]
 lines+=['',f'원본 연결 VTI의 Base 대비 F1 차이 {delta:+.4f}%p, 이미지 단위 paired bootstrap 10,000회 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]%p.','','저자 VTI commit bb5dfee33a9e555f22af9faa7b3495c0ac3df69e의 전체 원본 모듈을 직접 import했다. 공식 random.sample 방식으로 seed42에서100개 중70개 demo를 골랐고, 50회99% 패치 가림·rank1·text/image alpha0.9를 고정했다. visual VTI 활성화 상태에서 textual 방향을 추출했다. 테스트로 파라미터를 선택하지 않았다.','Gemma processor·패치 위치·MLP 경로 및 원본 FP16 출력의 BF16 복원이 모델 연결부다. 원논문의 LLaVA 환경·전체 프로토콜을 재현했다는 의미는 아니다.']
 (RUN/'REPORT_KO.md').write_text('\n'.join(lines)+'\n');state('status',stage='complete',report_sha256=sha(RUN/'report.json'),metrics=metrics)
 files=[p for p in RUN.rglob('*') if p.is_file() and p.name!='snapshot.zip' and '.git' not in p.parts and 'deps' not in p.relative_to(RUN).parts and 'images' not in p.relative_to(RUN).parts]
 save(RUN/'EXPORT_MANIFEST.json',{p.relative_to(RUN).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in files})
 with zipfile.ZipFile(RUN/'snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in files+[RUN/'EXPORT_MANIFEST.json']:z.write(p,p.relative_to(RUN).as_posix())
def controller():
 state('status',stage='verifying_cpu');verify()
 testenv=os.environ.copy();testenv['CUDA_VISIBLE_DEVICES']=''
 testenv['PYTHONPATH']=os.pathsep.join([str(RUN/'deps'),str(ROOT/'src'),testenv.get('PYTHONPATH','')])
 result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(RUN/'tests'),'-p','test_vti_author_bridge.py','-v'],env=testenv,capture_output=True,text=True)
 (RUN/'cpu-tests.txt').write_text(result.stdout+result.stderr);assert result.returncode==0 and 'skipped' not in result.stderr
 state('status',stage='extracting_author_directions');proc,log=run_child('extract',2);assert proc.wait()==0;log.close()
 state('status',stage='generating_pope');workers=[run_child('worker',i) for i in [2,3]]
 for proc,log in workers:assert proc.wait()==0;log.close()
 state('status',stage='official_scoring');score()
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['controller','extract','worker'],required=True);p.add_argument('--gpu',type=int,choices=[2,3],default=2);a=p.parse_args()
 name='status' if a.mode=='controller' else ('extraction' if a.mode=='extract' else f'worker-gpu{a.gpu}')
 try:
  if a.mode=='controller':controller()
  elif a.mode=='extract':extraction()
  else:worker(a.gpu)
 except BaseException:state(name,stage='failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
