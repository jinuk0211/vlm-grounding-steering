"""Run the requested phases sequentially on GPU2, then finalize scored reports."""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/project-b/LJW/vlm-grounding-steering');GPU='GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde'
def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def live(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1][0]!='Z'
    except FileNotFoundError:return False
def main(output,prepare_only=False):
    output=Path(output);status=output/'sequence-status.json'
    try:
        train=json.loads((output/'training-job.json').read_text())
        while live(train['training_pid']):
            save(status,{'stage':'waiting_for_ephs_training','training_pid':train['training_pid'],'updated_at':time.time()});time.sleep(10)
        assert json.loads((output/'training-status.json').read_text())['stage']=='complete','EPHS training failed; inspect training logs'
        q=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True);used={s.split(',')[0].strip():int(s.split(',')[1]) for s in q.splitlines()};assert used[GPU]<1000,'GPU2 was taken by another process; no fallback or interference'
        command=[os.sys.executable,'scripts/run_gpu2_requested_eval.py','--output',str(output)]
        if prepare_only:command.append('--prepare-only')
        with (output/'evaluation.log').open('ab') as sink:
            proc=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU),stdout=sink,stderr=subprocess.STDOUT,start_new_session=True)
            save(output/'evaluation-job.json',{'pid':proc.pid,'gpu_index':2,'gpu_uuid':GPU,'log':str(output/'evaluation.log')})
            save(status,{'stage':'running_preparation' if prepare_only else 'running_evaluation','pid':proc.pid,'updated_at':time.time()});code=proc.wait()
        assert code==0,f'Evaluation worker exited with status {code}'
        if prepare_only:save(status,{'stage':'prepared','completed_at':time.time()});return
        state=json.loads((output/'evaluation-status.json').read_text());assert state['stage']=='complete'
        reports={}
        for benchmark in ['pope','chair']:
            path=output/f'{benchmark}-report.json';report=json.loads(path.read_text());assert report['status']=='scored_complete';report.update(status='verified_complete',actual_worker_exited=True);save(path,report);reports[benchmark]=report
        rows=['# Gemma E4B · GPU2 POPE / CHAIR','','REVIS와 VTI는 고정한 저자 코드의 연산을 Gemma에 연결했다. EPHS는 별도 개발 자료에서 head를 선택하고 gain을 학습했다. 저자 논문 환경의 완전 재현 결과는 아니다.','','| 방법 | POPE Acc ↑ | POPE F1 ↑ | CHAIR_s ↓ | CHAIR_i ↓ | 객체 recall ↑ |','|---|---:|---:|---:|---:|---:|']
        for method in ['base','ephs','revis','vti']:
            p=reports['pope']['metrics'].get(method);c=reports['chair']['metrics'][method]
            rows.append('| '+method+' | '+(' | '.join(f'{p[k]*100:.2f}' for k in ['accuracy','f1']) if p else '평가 범위 밖 | 평가 범위 밖')+' | '+' | '.join(f'{c[k]*100:.2f}' for k in ['CHAIR_s','CHAIR_i','object_recall'])+' |')
        rows+=['','모든 수치는 %다. POPE adversarial 3,000문항, CHAIR COCO val2014 고정 500장·최대 512토큰이다. 모두 GPU2 한 장에서 실행했다. 입력·코드·벡터 해시, 개발/평가 분리, 공식 채점 결과와 불확실성 구간은 각 report.json에 기록했다.','']
        (output/'REPORT_KO.md').write_text('\n'.join(rows));save(status,{'stage':'complete','output':str(output),'completed_at':time.time()})
    except BaseException:
        save(status,{'stage':'failed','error':traceback.format_exc(),'updated_at':time.time()});raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();main(a.output,a.prepare_only)
