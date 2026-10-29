"""Author runner output.strip() applied uniformly; preserve unmodified raw run."""
import json,hashlib,re,time,collections,zipfile,sys,os
from pathlib import Path
ROOT=Path('/home/project-b/LJW/vlm-grounding-steering');RUN=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from audit_hallucination_suite import pope_metrics,verify_upstream_pope,intervals
assert os.environ['CUDA_VISIBLE_DEVICES']==''
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
out=RUN/'trimmed';out.mkdir(exist_ok=False)
parent=json.loads((RUN/'report.json').read_text());cohort=read(RUN/'pope.jsonl');source=ROOT/'outputs/gemma-grains-fp16-v2-20261029-1793246507/pope'
paths={'base':source/'base.jsonl','vti_local':source/'vti.jsonl','vti_author':RUN/'vti-author.jsonl'}
raw={k:read(p) for k,p in paths.items()};normalized={};strict={}
for k,rows in raw.items():
 assert len(rows)==3000 and [x['id'] for x in rows]==[x['id'] for x in cohort]
 normalized[k]=[{**x,'prediction':x['prediction'].strip(),'raw_prediction':x['prediction'],'output_normalization':'str.strip() as official author runner'} for x in rows]
 (out/(k+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in normalized[k]))
 strict[k]={'leading_yes':sum(bool(re.match(r'^yes\b',x['prediction'],re.I)) for x in normalized[k]),'leading_no':sum(bool(re.match(r'^no\b',x['prediction'],re.I)) for x in normalized[k]),'non_yes_no':sum(not bool(re.match(r'^(yes|no)\b',x['prediction'],re.I)) for x in normalized[k]),'at128':sum(x['generated_tokens']==128 for x in rows),'changed_by_trim':sum(x['prediction']!=x['prediction'].strip() for x in rows)}
metrics={k:pope_metrics(rows) for k,rows in normalized.items()}
for k,rows in normalized.items():verify_upstream_pope(rows,out,k,'adversarial',metrics[k])
author_runner=RUN/'upstream/vti-author/experiments/eval/run_mmhal_vti.py';assert 'outputs = outputs.strip()' in author_runner.read_text()
amend={'reason':'Architecture runner omitted author output.strip(); raw No\n failed literal-space official parser. Apply the same leading/trailing whitespace strip uniformly to every method, without changing model responses or decoding.','normalization':'str.strip()','author_runner':str(author_runner),'author_runner_sha256':sha(author_runner),'parent_raw_report_sha256':sha(RUN/'report.json'),'parent_raw_archive_sha256':sha(RUN/'snapshot.zip'),'raw_prediction_hashes':{k:sha(p) for k,p in paths.items()},'scoring_source_sha256':sha(Path(__file__)),'gpu_indices_used':[2],'new_inference_rows':0,'created_at':time.time()};save(out/'normalization-amendment.json',amend)
report={'status':'verified_complete_with_author_output_normalization','metrics':metrics,'raw_untrimmed_metrics':parent['metrics'],'diagnostics':strict,'uncertainty':intervals(normalized,'pope'),'normalization_amendment':amend,'all3000_ids_and_raw_predictions_preserved':True,'precision':'bfloat16','development_demos':70,'limitations':['Author functions adapted to native Gemma, not original LLaVA full benchmark reproduction','Official literal-space parser uses yes for non-matching answers; strict-leading diagnostics reported separately']};save(out/'report.json',report)
lines=['# Gemma E4B — VTI 원본 함수·POPE 최종 정정 채점','','동일 POPE adversarial3,000문항·BF16·448px·greedy·128토큰·문항별 seed42. GPU2에서만 생성했다. 수치는 %.','','| 방법 | Accuracy | Precision | Recall | F1 |','|---|---:|---:|---:|---:|']
for k,label in [('base','Base'),('vti_local','VTI 자체 구현'),('vti_author','VTI 원본 함수 + Gemma 연결')]:lines.append('| '+label+' | '+' | '.join(f'{100*metrics[k][v]:.4f}' for v in ['accuracy','precision','recall','f1'])+' |')
delta=100*(metrics['vti_author']['f1']-metrics['base']['f1']);ci=[100*x for x in report['uncertainty']['vti_author']['f1']['delta_ci95']]
lines+=['',f'Base 대비 원본 연결 VTI F1 차이 {delta:+.4f}%p, 이미지 단위 paired bootstrap10,000회95% CI [{ci[0]:+.4f},{ci[1]:+.4f}]%p.','','## 채점 정정과 보존','','연결부가 원저자 실행기의 `outputs.strip()`을 빠뜨려 `No\n`을 그대로 공식 단순 파서에 보냈다. 이 파서는 literal space로 나눈 `No`만 찾으므로 줄바꿈이 붙은 No를 Yes로 처리했다. 공백 제거 전 raw 공식 F166.6667%, recall100%는 이 형식 오류의 결과이며 모델이 모두 Yes라고 답했다는 의미가 아니다. 원문3,000개·raw report·raw ZIP은 그대로 보존했다.','Base·자체 VTI·원본 VTI 모든 방법에 동일하게 앞뒤 공백 제거만 적용하고 같은 원본 공식 scorer로 다시 채점했다. 모델 생성·벡터·강도·데이터·내부 공백은 변경하지 않았으며 재추론은0회다. 각 정정 행에 raw_prediction을 보존한다.','',f'원본 VTI의 strict 선두 Yes/No 이외 응답 {strict["vti_author"]["non_yes_no"]}개,128토큰 도달 {strict["vti_author"]["at128"]}개. 공식 파서의 기본 Yes 처리와 별개로 이 수를 명시한다.','','## 방향 추출 수와 원본 범위','','공식 README의 MMHal 실행 예시는70 demos·50 trials·99% mask·alpha text/image0.9를 사용한다. 논문 본문은50개 예시를 썼다고 기술한다. 따라서70개라는 숫자만으로 이번 결과의 원인을 판단할 수 없다. Gemma에서 샘플 수 민감도는 별도 개발셋 및 seed 반복 실험이 있어야 검증할 수 있다.','https://github.com/shengliu66/VTI#experiments','https://arxiv.org/html/2410.15778v1','원저자 commit bb5dfee33a9e555f22af9faa7b3495c0ac3df69e의 함수 전체를 직접 import했다. 모델 입력·패치 위치·unpooled encoder state 수집·MLP 경로·원본 FP16 출력의 BF16 복원이 Gemma 연결부다. visual VTI 활성화 상태에서 textual 방향을 추출했다.']
(out/'REPORT_KO.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'metrics':metrics,'diagnostics':strict,'delta_f1_pp':delta,'delta_f1_ci95_pp':ci},indent=2))
