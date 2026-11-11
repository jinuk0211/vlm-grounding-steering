"""One predeclared VTI inference-strength comparison on physical GPU3 only."""
import argparse, collections, hashlib, json, math, os, re, subprocess, sys, time, traceback, zipfile
from pathlib import Path

ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
PARENT=ROOT/'outputs/gemma-vti-author-pope-20261029-v2'
RUN=ROOT/'outputs/gemma-vti-author-pope-alpha0225-20261029-v1'
PRIOR045=ROOT/'outputs/gemma-vti-author-pope-alpha045-20261029-v1'
GEMMA=ROOT/'outputs/gemma-grains-fp16-v2-20261029-1793246507'
GPU='GPU-4c593ff4-fcc4-39c9-d7b4-981dabee1cb3'
sys.path[:0]=[str(PARENT/'scripts'),str(PARENT/'deps'),str(ROOT/'src'),str(ROOT/'scripts')]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def save(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def update(name,**kw):
 path=RUN/(name+'.json');s=json.loads(path.read_text()) if path.exists() else {'started_at':time.time()}
 s.update(kw,pid=os.getpid(),updated_at=time.time());save(path,s)
def gpu_apps():
 text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 return [line.strip() for line in text.splitlines() if line.startswith(GPU)]
def verify():
 p=json.loads((RUN/'protocol.json').read_text());body={k:v for k,v in p.items() if k!='sha256'}
 assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==p['sha256']
 assert p['inference_alpha_text']==p['inference_alpha_image']==.225
 assert p['artifact_text_extraction_vision_alpha']==.9 and p['allowed_gpu_indices']==[3]
 for name,digest in p['source_hashes'].items():assert sha(ROOT/name)==digest,name
 for path,digest in p['input_hashes'].items():assert sha(Path(path))==digest,path
 assert sha(RUN/'config.json')==p['config_sha256']
 return p

def worker():
 import torch
 import vti_author_gemma as bridge
 from vti_author import verify_source,demos
 from vlm_grounding.cli import _seed,_validate_artifact
 from huggingface_hub import try_to_load_from_cache
 p=verify();assert os.environ['CUDA_VISIBLE_DEVICES']==GPU and torch.cuda.device_count()==1
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 config=json.loads((RUN/'config.json').read_text());records=read(PARENT/'pope.jsonl')
 assert len(records)==3000 and len({r['id'] for r in records})==3000
 artifact_digest=sha(PARENT/'vti-author.pt');artifact=torch.load(PARENT/'vti-author.pt',map_location='cpu',weights_only=True)
 assert artifact['author_source']==verify_source(PARENT/'upstream/vti-author')
 assert artifact['alpha_text']==artifact['alpha_image']==.9 and artifact['text_extraction_vision_active']
 _validate_artifact(artifact,config['model'],records)
 assert {r['image_sha256'] for r in records}.isdisjoint(r['image_sha256'] for r in artifact['training_images'])
 update('worker',stage='loading_model',completed_records=0,total_records=3000)
 _seed(config);adapter=bridge.load_gemma(config)
 model=config['model'];cached=try_to_load_from_cache(model['checkpoint'],'config.json',revision=model['revision'])
 assert isinstance(cached,str) and Path(cached).parent.name==model['revision']
 observed=getattr(adapter.model.config,'_commit_hash',None);assert observed in [None,model['revision']]
 adapter.model.config._commit_hash=model['revision'];_validate_artifact(artifact,model,records,adapter)
 devices=sorted({str(x.device) for x in adapter.model.parameters()});dtypes=sorted({str(x.dtype) for x in adapter.model.parameters() if x.is_floating_point()})
 assert devices==['cuda:0'] and dtypes==['torch.bfloat16']
 save(RUN/'device-proof.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'physical_gpu_index':3,'gpu_uuid':GPU,'visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'parameter_devices':devices,'parameter_dtypes':dtypes,'checkpoint_revision':model['revision'],'observed_revision_metadata':observed,'cached_config':cached,'quantization':False,'offload':False})
 author=bridge.load_author(PARENT/'upstream/vti-author');counter={'calls':0};original=bridge.DtypeBridge.forward
 def counted(self,x):counter['calls']+=1;return original(self,x)
 bridge.DtypeBridge.forward=counted
 d=demos(PARENT/'upstream/vti-author',PARENT/'images',70)[0]
 batch=adapter.prepare(d['image_path'],'Please describe this image in detail.');seen={'calls':0,'changed_elements':0};handles=[]
 with bridge.intervention(adapter,author,text=artifact['text_direction'],vision=artifact['vision_direction'],alpha_text=.225,alpha_image=.225):
  def check(module,args,out):
   assert torch.isfinite(out).all() and out.dtype==args[0].dtype
   seen['calls']+=1;seen['changed_elements']+=int((out!=args[0]).sum())
  for layer in [*adapter.layers,*adapter.vision_layers]:
   assert layer.mlp[1].operator.lam==[.225]
   handles.append(layer.mlp[1].register_forward_hook(check))
  try:response=adapter.generate(batch,max_new_tokens=8)
  finally:
   for h in handles:h.remove()
 assert response.strip() and seen['calls'] and seen['changed_elements']
 save(RUN/'preflight.json',{'status':'passed','inference_alpha_text':.225,'inference_alpha_image':.225,'artifact_text_extraction_vision_alpha':.9,'response':response,'native_dtype':'bfloat16','benchmark_rows_written':0,**seen})
 update('worker',stage='generating',completed_records=0,total_records=3000)
 with (RUN/'vti-alpha0225.jsonl').open('x') as sink:
  for i,r in enumerate(records):
   assert sha(r['image'])==r['image_sha256'];_seed(config);batch=adapter.prepare(r['image'],r['prompt']);counter['calls']=0
   torch.cuda.synchronize();started=time.perf_counter()
   with bridge.intervention(adapter,author,text=artifact['text_direction'],vision=artifact['vision_direction'],alpha_text=.225,alpha_image=.225):raw=adapter.generate(batch,max_new_tokens=128)
   torch.cuda.synchronize();prediction=raw.strip()
   assert prediction and 0<adapter.last_generated_tokens<=128 and counter['calls']>0
   row={**r,'method':'vti_author_alpha0225','prediction':prediction,'raw_prediction':raw,'output_normalization':'str.strip() as official author runner','protocol_sha256':p['sha256'],'artifact_sha256':artifact_digest,'dtype':'bfloat16','gpu_uuid':GPU,'alpha_text':.225,'alpha_image':.225,'artifact_text_extraction_vision_alpha':.9,'generated_tokens':adapter.last_generated_tokens,'elapsed_seconds':time.perf_counter()-started,'author_mlp_calls':counter['calls']}
   sink.write(json.dumps(row,allow_nan=False)+'\n');sink.flush()
   update('worker',stage='generating',completed_records=i+1,total_records=3000,last_seconds=row['elapsed_seconds'])
 assert verify_source(PARENT/'upstream/vti-author')==artifact['author_source']
 update('worker',stage='complete',completed_records=3000,predictions_sha256=sha(RUN/'vti-alpha0225.jsonl'))

def all_derived(c):
 tp,tn,fp,fn=c.T;n=tp+tn+fp+fn
 return {'accuracy':(tp+tn)/n,'precision':tp/(tp+fp),'recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn),'yes_ratio':(tp+fp)/n}
def score():
 import numpy as np
 from audit_hallucination_suite import pope_metrics,verify_upstream_pope,matrix,official_choice
 p=verify();cohort=read(PARENT/'pope.jsonl');new=read(RUN/'vti-alpha0225.jsonl')
 records={'base':read(GEMMA/'pope/base.jsonl'),'vti_alpha09':read(PARENT/'trimmed/vti_author.jsonl'),'vti_alpha045':read(PRIOR045/'vti-alpha045.jsonl'),'vti_alpha0225':new}
 assert len(new)==3000 and len({x['id'] for x in new})==3000
 artifact_digest=sha(PARENT/'vti-author.pt')
 for method,rows in records.items():
  assert len(rows)==3000 and [x['id'] for x in rows]==[x['id'] for x in cohort]
  for r,x in zip(cohort,rows):assert all(x[k]==v for k,v in r.items())
  if method=='vti_alpha0225':
   for x in rows:
    assert x['prediction']==x['raw_prediction'].strip() and x['protocol_sha256']==p['sha256']
    assert x['gpu_uuid']==GPU and x['artifact_sha256']==artifact_digest and x['alpha_text']==x['alpha_image']==.225
  else:
   records[method]=[{**x,'prediction':x['prediction'].strip()} for x in rows]
 metrics={k:pope_metrics(v) for k,v in records.items()}
 for k,rows in records.items():verify_upstream_pope(rows,RUN,k,'adversarial',metrics[k])
 groups,base=matrix(records['base'],'pope');n=len(groups);weights=np.random.default_rng(42).multinomial(n,np.full(n,1/n),size=10000)
 samples={};points={}
 for k,rows in records.items():
  gs,m=matrix(rows,'pope');assert gs==groups
  samples[k]=all_derived(weights@m);points[k]=all_derived(m.sum(0,keepdims=True))
 uncertainty={}
 for k in records:
  uncertainty[k]={}
  for m,value in samples[k].items():
   uncertainty[k][m]={'value':float(points[k][m][0]),'ci95':np.quantile(value,[.025,.975]).tolist(),'delta_vs_base':float(points[k][m][0]-points['base'][m][0]),'delta_ci95_vs_base':np.quantile(value-samples['base'][m],[.025,.975]).tolist(),'delta_vs_alpha09':float(points[k][m][0]-points['vti_alpha09'][m][0]),'delta_ci95_vs_alpha09':np.quantile(value-samples['vti_alpha09'][m],[.025,.975]).tolist(),'delta_vs_alpha045':float(points[k][m][0]-points['vti_alpha045'][m][0]),'delta_ci95_vs_alpha045':np.quantile(value-samples['vti_alpha045'][m],[.025,.975]).tolist()}
 changes={};changed_rows=[]
 for reference in ['base','vti_alpha09','vti_alpha045']:
  counts=collections.Counter()
  for a,b in zip(records[reference],new):
   before,after=official_choice(a['prediction']),official_choice(b['prediction']);bc=before==a['label'];ac=after==b['label']
   counts['text_changed']+=a['prediction']!=b['prediction'];counts['official_answer_changed']+=before!=after
   counts['wrong_to_right']+=not bc and ac;counts['right_to_wrong']+=bc and not ac
   counts['yes_to_no']+=before=='yes' and after=='no';counts['no_to_yes']+=before=='no' and after=='yes'
   if before!=after:changed_rows.append({'reference':reference,'id':a['id'],'label':a['label'],'before':before,'after':after,'before_text':a['prediction'],'after_text':b['prediction']})
  changes[reference]=dict(counts)
 save(RUN/'changed-answers.json',changed_rows)
 diagnostics={k:{'non_yes_no':sum(not bool(re.match(r'^(yes|no)\b',r['prediction'],re.I)) for r in v),'at128':sum(r['generated_tokens']==128 for r in v),'mean_generated_tokens':float(np.mean([r['generated_tokens'] for r in v]))} for k,v in records.items()}
 report={'status':'verified_complete_single_alpha0225_candidate','protocol':p,'metrics':metrics,'uncertainty':uncertainty,'response_changes':changes,'diagnostics':diagnostics,'comparability_note':'Same precomputed vectors. Text direction was extracted with visual alpha0.9; inference alpha_text and alpha_image are0.225. Not a fresh alpha0.225 extraction.','exploratory_note':'One additional user-requested candidate after earlier0.9 and0.45 benchmark results. Test-informed exploratory comparison, not an untouched-test confirmation; no further setting changes in this run.','all_3000_rows_verified':True,'official_scorers_used':True,'prediction_sha256':sha(RUN/'vti-alpha0225.jsonl')}
 save(RUN/'report.json',report)
 lines=['# Gemma E4B — VTI 추론 강도0.225 단일 비교','','POPE adversarial 동일3,000문항·BF16·448px·greedy·128토큰·문항별seed42·물리GPU3 단독. 수치는 %.','','| 방법 | Accuracy | F1 | Precision | Recall | Yes 비율 |','|---|---:|---:|---:|---:|---:|']
 for k,label in [('base','Base'),('vti_alpha09','원저자 VTI 추론 alpha0.9'),('vti_alpha045','원저자 VTI 추론 alpha0.45'),('vti_alpha0225','원저자 VTI 추론 alpha0.225')]:lines.append('| '+label+' | '+' | '.join(f'{100*metrics[k][m]:.4f}' for m in ['accuracy','f1','precision','recall','yes_ratio'])+' |')
 lines+=['','기존 벡터를 그대로 사용했다. 텍스트 방향은 vision alpha0.9를 활성화한 상태에서 추출됐으며, 이번에는 추론 시 text/image alpha만0.225로 낮췄다. 새 방향 추출 실험이 아니다.','사용자가0.9 및0.45 결과를 보고 요청한 추가 탐색적1회다.0.225를 사전 지정했고 다른 후보를 추가하거나 테스트 결과로 설정을 조정하지 않았다. 향상을 보장하지 않는다.','']
 for ref,label in [('base','Base'),('vti_alpha09','이전0.9'),('vti_alpha045','이전0.45')]:
  u=uncertainty['vti_alpha0225']['f1'];suffix={'base':'base','vti_alpha09':'alpha09','vti_alpha045':'alpha045'}[ref];delta=u['delta_vs_'+suffix];ci=u['delta_ci95_vs_'+suffix]
  lines.append(f'{label} 대비 F1 차이 {100*delta:+.4f}%p, 이미지 단위 paired bootstrap10,000회95% CI [{100*ci[0]:+.4f},{100*ci[1]:+.4f}]%p. 응답 변화: '+json.dumps(changes[ref],ensure_ascii=False))
 lines+=['','모든 방법에 동일한 str.strip() 후 원본 공식POPE 파서를 사용했다. 새 출력에는 raw_prediction도 보존했다. 답변 형식 이탈 및128토큰 도달 수는 report.json에 명시한다.','원저자 VTI 함수를 Gemma에 연결한 비교이며 원논문 모델·설정 전체 재현을 뜻하지 않는다.']
 (RUN/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
 update('status',stage='complete',metrics=metrics,report_sha256=sha(RUN/'report.json'))
 files=[x for x in RUN.rglob('*') if x.is_file() and x.suffix!='.zip' and x.name!='EXPORT_MANIFEST.json']
 save(RUN/'EXPORT_MANIFEST.json',{x.relative_to(RUN).as_posix():{'sha256':sha(x),'bytes':x.stat().st_size} for x in files})
 with zipfile.ZipFile(RUN/'delivery.zip','x',zipfile.ZIP_DEFLATED) as z:
  for x in files+[RUN/'EXPORT_MANIFEST.json']:z.write(x,x.relative_to(RUN).as_posix())

def controller():
 import fcntl
 lock=(ROOT/'outputs/.gemma-vti-alpha0225.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 verify();update('status',stage='waiting_for_free_gpu3',allowed_gpu_indices=[3])
 while True:
  apps=gpu_apps();memory=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()
  if not apps and int(memory)>20000:break
  update('status',stage='waiting_for_free_gpu3',other_gpu3_apps=apps,free_mib=int(memory));time.sleep(10)
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=GPU;log=(RUN/'worker.log').open('x')
 child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--mode','worker'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 job={'pid':child.pid,'start_ticks':Path(f'/proc/{child.pid}/stat').read_text().split()[21],'script':str(Path(__file__).resolve()),'gpu_uuid':GPU,'physical_gpu_index':3};save(RUN/'worker-job.json',job);update('status',stage='generating',worker=job)
 assert child.wait()==0;log.close();assert json.loads((RUN/'worker.json').read_text())['stage']=='complete'
 save(RUN/'worker-exit.json',{**job,'exit_code':0,'completed_at':time.time(),'predictions_sha256':sha(RUN/'vti-alpha0225.jsonl')})
 update('status',stage='official_scoring');score()

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['controller','worker'],required=True);args=parser.parse_args()
 try:controller() if args.mode=='controller' else worker()
 except BaseException:
  if RUN.exists():update('status' if args.mode=='controller' else 'worker',stage='failed',error=traceback.format_exc())
  raise
