"""Official CPU-only interim scoring on immutable, matched-image snapshots."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import traceback
import zipfile

import torch
import vlm_grounding.baselines as baselines
from vlm_grounding.evaluation import _load_external_module, attach_chair_scores, evaluate_predictions

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')
RUN = ROOT / 'outputs/author-clone-rerun-20261027-165446'
BF16 = ['base', 'revis', 'vti', 'rudder_clone']
FP32 = ['base_fp32', 'grains_clone']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def main(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert output.resolve().parent == RUN
    output.mkdir()
    state = {'pid': os.getpid(), 'started_at': time.time(), 'stage': 'snapshotting',
        'full_evaluation_complete': False, 'gpu_compute_started': False}
    def update(**values):
        state.update(values, updated_at=time.time()); save(output / 'status.json', state)
    try:
        protocol = json.loads((RUN / 'protocol.json').read_text())
        body = {k: v for k, v in protocol.items() if k != 'sha256'}
        assert hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == protocol['sha256']
        for file, digest in protocol['source_hashes'].items():
            assert sha(ROOT / file) == digest, file
        assert sha(baselines.__file__) == protocol['source_hashes']['src/vlm_grounding/baselines.py']
        for method in ('revis', 'vti'):
            assert sha(RUN / 'artifacts' / (method + '.pt')) == protocol['retained_artifact_hashes'][method]
        refs = lines(RUN / 'data/chair.jsonl'); by_id = {r['id']: r for r in refs}
        assert len(refs) == len(by_id) == 500
        tasks = json.loads((RUN / 'tasks.json').read_text())
        collected = {m: {} for m in BF16 + FP32}; files = {}; tails = {}
        for task in tasks:
            if task['benchmark'] != 'chair':
                continue
            path = RUN / 'predictions' / (task['id'] + '.jsonl')
            if not path.exists():
                continue
            raw = path.read_bytes(); boundary = raw.rfind(b'\n') + 1
            committed = raw[:boundary]
            if boundary < len(raw):
                tails[str(path.relative_to(RUN))] = len(raw) - boundary
            files[str(path.relative_to(RUN))] = {'committed_bytes': len(committed),
                'committed_prefix_sha256': hashlib.sha256(committed).hexdigest()}
            rows = [json.loads(s) for s in committed.splitlines() if s.strip()]
            references = refs[task['start']:task['stop']]
            assert len(rows) <= len(references)
            for row, ref in zip(rows, references):
                assert all(row.get(k) == v for k, v in ref.items())
                assert row['method'] == task['method'] and row['dtype'] == task['dtype']
                assert row['protocol_sha256'] == protocol['sha256']
                assert isinstance(row['prediction'], str) and 0 < row['generated_tokens'] <= 512
                assert row['id'] not in collected[task['method']]
                if 'adopted_cache_sha256' in row:
                    path = RUN / 'cache' / task['method'] / (row['id'].replace(':', '_') + '.json')
                    assert sha(path) == row['adopted_cache_sha256']
                    original = json.loads(path.read_text())
                    assert row['prediction'] == original['prediction'] and row['generated_tokens'] == original['generated_tokens']
                if task['method'] in ('rudder_clone', 'grains_clone'):
                    assert row['author_diagnostics']['calls'] > 0
                collected[task['method']][row['id']] = row
        counts = {m: len(rows) for m, rows in collected.items()}
        groups = {'bf16_common': BF16, 'fp32_small_common': FP32}
        snapshots = {}; cohorts = {}
        for name, methods in groups.items():
            common = set.intersection(*(set(collected[m]) for m in methods))
            ids = [r['id'] for r in refs if r['id'] in common]
            assert ids
            cohorts[name] = {'n': len(ids), 'record_ids': ids,
                'image_ids': [by_id[k]['metadata']['image_id'] for k in ids],
                'selection': 'All committed IDs common to these methods, in original fixed500 order; no metric-based selection'}
            snapshots[name] = {m: [collected[m][k] for k in ids] for m in methods}
            folder = output / name; folder.mkdir()
            for method, rows in snapshots[name].items():
                (folder / (method + '.snapshot.jsonl')).write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
        used = {r['id']: r for methods in snapshots.values() for rows in methods.values() for r in rows}
        assert all(sha(r['image']) == r['image_sha256'] for r in used.values())
        save(output / 'snapshot.json', {'captured_at': time.time(), 'source_protocol_sha256': protocol['sha256'],
            'completed_counts': counts, 'source_prefixes': files, 'inflight_tail_bytes_excluded': tails,
            'cohorts': cohorts, 'full_evaluation_complete': False})
        update(stage='official_scoring', completed_counts=counts, cohort_sizes={k: v['n'] for k, v in cohorts.items()})
        data = ROOT / 'data/hallucination-suite'
        cache = json.loads((data / 'chair-scorer-state.json').read_text())
        source = ROOT / 'upstream/chair/chair.py'
        assert cache['built_with_unmodified_constructor'] and sha(source) == cache['scorer_sha256']
        assert sha(data / 'sources/annotations_trainval2014.zip') == protocol['dataset_provenance']['sources']['coco_annotations']['sha256']
        with zipfile.ZipFile(data / 'sources/annotations_trainval2014.zip') as zipped:
            for name, digest in cache['annotation_hashes'].items():
                assert sha(data / 'annotations' / name) == digest
                with zipped.open('annotations/' + name) as stream:
                    h = hashlib.sha256()
                    while chunk := stream.read(1024 * 1024):
                        h.update(chunk)
                    assert h.hexdigest() == digest
        module = _load_external_module(source, 'interim_original_chair')
        scorer = module.CHAIR.__new__(module.CHAIR); scorer.__dict__.update(cache['state'])
        scorer.imid_to_objects = {int(k): set(v) for k, v in cache['state']['imid_to_objects'].items()}
        spec = importlib.util.spec_from_file_location('interim_metric_helper', ROOT / 'scripts/audit_hallucination_suite.py')
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        metrics = {}; csv_rows = []
        for group, methods in snapshots.items():
            metrics[group] = {}; folder = output / group
            for method, rows in methods.items():
                caption_file = folder / (method + '.captions.json')
                save(caption_file, [{'image_id': r['metadata']['image_id'], 'caption': r['prediction']} for r in rows])
                scored = scorer.compute_chair(str(caption_file), 'image_id', 'caption')
                save(folder / (method + '.chair.official.json'), scored)
                enriched = attach_chair_scores(rows, scored); methods[method] = enriched
                value = evaluate_predictions(enriched)
                assert math.isclose(value['CHAIR_s'], scored['overall_metrics']['CHAIRs'])
                assert math.isclose(value['CHAIR_i'], scored['overall_metrics']['CHAIRi'])
                value.update(at_token_cap=sum(r['generated_tokens'] >= 512 for r in rows),
                    empty_outputs=sum(not r['prediction'].strip() for r in rows),
                    mean_generated_tokens=sum(r['generated_tokens'] for r in rows) / len(rows))
                metrics[group][method] = value; helper.write_jsonl(folder / (method + '.jsonl'), enriched)
                csv_rows.append({'benchmark': 'chair', 'cohort': group, 'method': method,
                    **{k: v for k, v in value.items() if not isinstance(v, dict)}})
                update(stage='official_scoring', last_scored_method=method, last_scored_cohort=group)
        uncertainty = helper.intervals(snapshots['bf16_common'], 'chair')
        pope_path = RUN / 'pope-complete/report.json'; pope = json.loads(pope_path.read_text())
        assert pope['status'] == 'verified_complete' and pope['cases_per_method'] == 3000
        assert pope['protocol_sha256'] == protocol['sha256']
        save(output / 'pope-complete-report.json', pope)
        report = {'status': 'verified_interim_snapshot', 'full_evaluation_complete': False,
            'model': protocol['model'], 'source_output': str(RUN), 'source_protocol_sha256': protocol['sha256'],
            'snapshot_sha256': sha(output / 'snapshot.json'), 'completed_counts': counts,
            'cohorts': cohorts, 'chair': metrics, 'chair_bf16_uncertainty': uncertainty,
            'pope_complete': pope['metrics'], 'pope_report_sha256': sha(pope_path),
            'analysis_sha256': sha(__file__), 'gpu_compute_started': False,
            'vti_label': 'VTI local adaptation; not original-author VTI reproduction',
            'limitations': ['CHAIR is an interim common-image cohort; final fixed500 results may change',
                'FP32 small cohort is separate and too small for a stable performance conclusion',
                'Historical GPU0 responses and resumed GPU2/3 responses coexist under the recorded hardware amendment'],
            'completed_at': time.time()}
        save(output / 'report.json', report)
        for method, value in pope['metrics'].items():
            csv_rows.append({'benchmark': 'pope_adversarial', 'cohort': 'all3000', 'method': method,
                **{k: v for k, v in value.items() if not isinstance(v, dict)}})
        columns = list(dict.fromkeys(k for r in csv_rows for k in r))
        with (output / 'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(csv_rows)
        n = cohorts['bf16_common']['n']
        cohort_note = ('BF16 네 방법의 고정500장 결과다. FP32 두 방법까지 포함한 전체 실험의 최종 결과는 아니다.' if n == 500
            else '전체500장 최종 결과가 아니다.')
        md = ['# Gemma E4B 중간 평가', '',
            f'POPE adversarial은 각 3,000문항 완료. CHAIR는 BF16 네 방법에 공통으로 생성 완료된 동일 {n}장으로 공식 채점했다. ' + cohort_note, '',
            '| 방법 | POPE Acc ↑ | POPE F1 ↑ | CHAIR_s ↓ | CHAIR_i ↓ | 객체 recall ↑ |',
            '|---|---:|---:|---:|---:|---:|']
        for method in BF16 + FP32:
            p = pope['metrics'][method]; c = metrics['bf16_common'].get(method)
            label = 'VTI (local adaptation)' if method == 'vti' else method
            chair_values = f"{100*c['CHAIR_s']:.2f} | {100*c['CHAIR_i']:.2f} | {100*c['object_recall']:.2f}" if c else '대기 | 대기 | 대기'
            md.append(f"| {label} | {100*p['accuracy']:.2f} | {100*p['f1']:.2f} | {chair_values} |")
        md += ['', '수치는 %. CHAIR_s는 환각 객체가 한 개 이상 있는 캡션 비율, CHAIR_i는 생성한 객체 언급 중 환각 비율이다. 둘 다 낮을수록 좋다.',
            '', 'GrAInS와 Base FP32의 소수 공통 이미지 점수는 별도 report.json에만 보관했다. BF16 표와 표본 수가 달라 같은 CHAIR 열로 비교하지 않았다.',
            '', 'VTI는 현재 선택한 독립 구현이다. 원본과 개입 위치·정규화·PCA·mask 처리·prefill 범위가 다르므로 원본 VTI 성능으로 인용하면 안 된다. REVIS도 선택한 서버 구현이며 RUDDER는 고정 원저자 함수와 Gemma 연결 adapter를 사용한다.',
            '', '진행 중인 생성·저장 파일과 실행 설정은 변경하지 않았다. 원본 CHAIR Python3 포트·COCO annotations와 검증된 scorer state를 사용했다. 공통 cohort 선택에는 점수나 정답을 사용하지 않았다.', '']
        (output / 'REPORT_KO.md').write_text('\n'.join(md))
        assert not torch.cuda.is_initialized()
        update(stage='complete', report_sha256=sha(output / 'report.json'), cuda_initialized=False)
        print(json.dumps({'status': report['status'], 'output': str(output), 'cohort_sizes': {k: v['n'] for k, v in cohorts.items()}, 'chair': metrics}))
    except BaseException:
        update(stage='failed', error=traceback.format_exc()); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
