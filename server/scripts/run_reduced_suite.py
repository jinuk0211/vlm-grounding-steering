"""Finish POPE adversarial, then full COCO500 CHAIR with the selected token cap."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')
PLAN = ROOT / 'outputs/reduced-suite-plan.json'
STATUS = ROOT / 'outputs/reduced-suite-status.json'


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def wait_jobs(pointer, stage):
    output = Path(pointer['output'])
    while True:
        states, live = [], []
        for job in pointer['jobs']:
            file = output / f"worker-{job['worker']}.json"
            states.append(json.loads(file.read_text()) if file.exists() else {'stage': 'starting', 'completed_records': 0})
            try:
                state = (Path('/proc') / str(job['pid']) / 'stat').read_text().split(') ', 1)[1][0]
            except FileNotFoundError:
                state = 'exited'
            live.append(state not in ['exited', 'Z'])
        for state, active in zip(states, live):
            if state['stage'] == 'failed' or (not active and state['stage'] != 'complete'):
                raise RuntimeError(json.dumps({'worker_failure': state, 'live': active}))
        save(STATUS, {'stage': stage, 'output': str(output), 'workers_live': sum(live),
                      'completed_records': sum(s.get('completed_records', 0) for s in states),
                      'completed_tasks': len(list((output / 'claims').glob('*/complete.json'))),
                      'total_tasks': len(json.loads((output / 'tasks.json').read_text())), 'updated_at': time.time()})
        if not any(live) and all(s['stage'] == 'complete' for s in states):
            return
        time.sleep(15)


def audit(output):
    with (Path(output) / 'audit.log').open('ab') as sink:
        subprocess.run([os.sys.executable, 'scripts/audit_selected_benchmark.py', '--output', output],
                       cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT, check=True)
    report = json.loads((Path(output) / 'report.json').read_text())
    assert report['status'] == 'verified_complete'
    return report


def main():
    plan = json.loads(PLAN.read_text())
    pope_pointer = json.loads((ROOT / 'outputs/pope-selected-current.json').read_text())
    wait_jobs(pope_pointer, 'pope_generating')
    save(STATUS, {'stage': 'pope_auditing', 'updated_at': time.time()})
    pope = audit(pope_pointer['output'])
    plan = json.loads(PLAN.read_text())
    caption_tokens = plan['caption_tokens']
    assert caption_tokens in [128, 512]
    save(STATUS, {'stage': 'chair_starting', 'pope_output': pope_pointer['output'], 'caption_tokens': caption_tokens, 'updated_at': time.time()})
    result = subprocess.run([os.sys.executable, 'scripts/launch_selected_benchmark.py', '--benchmark', 'chair', '--caption-tokens', str(caption_tokens)],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    print(result.stdout, flush=True)
    chair_pointer = json.loads((ROOT / 'outputs/chair-selected-current.json').read_text())
    wait_jobs(chair_pointer, 'chair_generating')
    save(STATUS, {'stage': 'chair_auditing', 'updated_at': time.time()})
    chair = audit(chair_pointer['output'])
    assert pope['cases_per_method'] == 3000 and chair['cases_per_method'] == 500
    combined = Path(plan['combined_output'])
    combined.mkdir(exist_ok=True)
    report = {'status': 'verified_complete', 'scope': 'User-prioritized POPE adversarial + CHAIR COCO500',
              'methods': ['base', 'revis', 'rudder', 'grains', 'vti'], 'pope': pope, 'chair': chair,
              'deferred': ['POPE random', 'POPE popular', 'MMHal-Bench additional generation and GPT-4 scoring'],
              'original_partial_output_preserved': plan['parent_output'], 'completed_at': time.time()}
    save(combined / 'report.json', report)
    rows = ['# Gemma E4B · POPE adversarial / CHAIR', '',
            '사용자 요청에 따라 POPE 한 유형과 CHAIR를 우선 완료했다. Base와 baseline 4개를 같은 이미지·생성 조건으로 비교했다.', '',
            '| 방법 | POPE Acc ↑ | POPE F1 ↑ | CHAIR_s ↓ | CHAIR_i ↓ | 객체 recall ↑ |',
            '|---|---:|---:|---:|---:|---:|']
    import csv
    csv_rows = []
    for method in report['methods']:
        p, c = pope['metrics'][method], chair['metrics'][method]
        rows.append(f"| {method} | {p['accuracy']*100:.2f} | {p['f1']*100:.2f} | {c['CHAIR_s']*100:.2f} | {c['CHAIR_i']*100:.2f} | {c['object_recall']*100:.2f} |")
        csv_rows.append({'method': method, 'pope_accuracy': p['accuracy'], 'pope_f1': p['f1'], 'CHAIR_s': c['CHAIR_s'], 'CHAIR_i': c['CHAIR_i'], 'object_recall': c['object_recall'],
                         'caption_mean_words': c['mean_words'], 'caption_at_token_cap': c['at_token_cap']})
    rows += ['', '표의 수치는 %다. CHAIR_s/i는 낮을수록 좋다.', '',
             f"- POPE adversarial 전체 3,000문항 × 다섯 방법. CHAIR는 COCO val2014 고정 500장 × 다섯 방법, 최대 {caption_tokens}토큰이다.",
             '- CHAIR는 지표이며, 500장 평가가 전체 COCO validation 평가를 의미하지 않는다.',
             '- POPE는 공식 evaluate.py 실행 결과와 독립 재채점이 일치했다. 엄격한 yes/no 형식 검사도 별도 기록했다.',
             '- CHAIR는 고정된 원 CHAIR Python3 포트를 수정 없이 실행했다. 전체 원본 COCO annotation에서 만든 정답을 사용했다.',
             '- Gemma에 적용한 저장소의 논문 기반 구현으로, 논문 저자 환경의 재현 결과는 아니다. 추출 32장·보정 32장과 방법별 강도는 고정했다.',
             '- 테스트와 개발 이미지 중복 0개, 전체 ID·응답 수·코드/입력/벡터 해시 검증 완료. 이미지 단위 paired bootstrap 10,000회 구간은 report.json에 보관했다.',
             '- random/popular 추가 평가와 MMHal 추가 생성·공식 채점은 보류했다. 앞서 생성한 원본 응답은 보존했다.',
             '', f"POPE 상세 결과: `{pope['output']}`", f"CHAIR 상세 결과: `{chair['output']}`", '',
             '공식 출처: [POPE](https://github.com/RUCAIBox/POPE), [CHAIR](https://github.com/LisaAnne/Hallucination), [Python3 포트](https://github.com/Maxlinn/CHAIR-metric-standalone)', '']
    (combined / 'REPORT_KO.md').write_text('\n'.join(rows))
    with (combined / 'metrics.csv').open('w', newline='') as sink:
        writer = csv.DictWriter(sink, fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    save(STATUS, {'stage': 'complete', 'combined_output': str(combined), 'pope_output': pope['output'], 'chair_output': chair['output'], 'completed_at': time.time()})


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        save(STATUS, {'stage': 'failed', 'error': traceback.format_exc(), 'updated_at': time.time()})
        raise
