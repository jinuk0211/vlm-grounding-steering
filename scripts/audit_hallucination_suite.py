"""Independently audit full coverage and score real outputs with official tools."""
import argparse
import ast
import collections
import contextlib
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from vlm_grounding.evaluation import (
    _load_external_module, attach_chair_scores, evaluate_predictions, mmhal_responses,
    parse_yes_no,
)

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')
DATA = ROOT / 'data/hallucination-suite'
METHODS = ['base', 'revis', 'rudder', 'grains', 'vti']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def official_choice(text):
    # Exact RUCAIBox/POPE evaluate.py algorithm, including its literal-space split.
    words = text.split('.')[0].replace(',', '').split(' ')
    return 'no' if any(word in words for word in ('No', 'not', 'no')) else 'yes'


def pope_metrics(rows, parser=official_choice):
    counts = collections.Counter((parser(r['prediction']), r['label']) for r in rows)
    tp, tn = counts[('yes', 'yes')], counts[('no', 'no')]
    fp = counts[('yes', 'no')]
    # An invalid answer to a positive question is a miss in strict sensitivity.
    fn = sum(r['label'] == 'yes' for r in rows) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {'n': len(rows), 'accuracy': (tp + tn) / len(rows), 'precision': precision,
            'recall': recall, 'f1': f1, 'yes_ratio': (tp + fp) / len(rows),
            'invalid': sum(parser(r['prediction']) is None for r in rows),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def verify_upstream_pope(rows, output, method, subset, expected):
    directory = output / 'pope-official'
    directory.mkdir(exist_ok=True)
    answers = directory / f'{method}-{subset}.answers.jsonl'
    labels = DATA / 'sources' / f'coco_pope_{subset}.json'
    write_jsonl(answers, [{'question': r['metadata']['original_question'], 'answer': r['prediction']} for r in rows])
    tree = ast.parse((DATA / 'sources/pope_evaluate.py').read_text())
    replacements = {'ans_file': str(answers), 'label_file': str(labels)}
    replaced = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in replacements:
                node.value = ast.Constant(replacements[name])
                replaced.add(name)
    assert replaced == replacements.keys()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(ast.fix_missing_locations(tree), 'official_POPE_evaluate.py', 'exec'), {})
    text = buffer.getvalue()
    (directory / f'{method}-{subset}.metrics.txt').write_text(text)
    for label, key in [('Accuracy', 'accuracy'), ('Precision', 'precision'), ('Recall', 'recall'), ('F1 score', 'f1'), ('Yes ratio', 'yes_ratio')]:
        match = re.search(re.escape(label) + r':\s*([0-9.eE+-]+)', text)
        assert match and math.isclose(float(match.group(1)), expected[key], abs_tol=1e-12)


def matrix(rows, kind):
    by_group = collections.defaultdict(list)
    for row in rows:
        by_group[row['group_id']].append(row)
    groups = sorted(by_group)
    result = []
    for group in groups:
        values = by_group[group]
        if kind == 'pope':
            m = pope_metrics(values)
            result.append([m[k] for k in ['tp', 'tn', 'fp', 'fn']])
        else:
            assert len(values) == 1
            c = values[0]['chair']
            result.append([int(c['hallucinated_mentions'] > 0), 1, c['hallucinated_mentions'],
                           c['generated_mentions'], c['correct_distinct_objects'], c['ground_truth_objects']])
    return groups, np.asarray(result, dtype=float)


def derived(values, kind):
    if kind == 'pope':
        tp, tn, fp, fn = values.T
        return {'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'f1': 2 * tp / (2 * tp + fp + fn)}
    hs, n, hi, ni, correct, gt = values.T
    return {'CHAIR_s': hs / n, 'CHAIR_i': hi / ni, 'object_recall': correct / gt}


def intervals(records, kind):
    groups, base = matrix(records['base'], kind)
    n = len(groups)
    weights = np.random.default_rng(42).multinomial(n, np.full(n, 1 / n), size=10000)
    base_scores = derived(weights @ base, kind)
    result = {}
    for method, rows in records.items():
        method_groups, values = matrix(rows, kind)
        assert groups == method_groups
        samples = derived(weights @ values, kind)
        point = derived(values.sum(axis=0, keepdims=True), kind)
        baseline = derived(base.sum(axis=0, keepdims=True), kind)
        result[method] = {}
        for name in samples:
            assert np.isfinite(samples[name]).all()
            result[method][name] = {'value': float(point[name][0]),
                                  'ci95': np.quantile(samples[name], [.025, .975]).tolist(),
                                  'delta_vs_base': float(point[name][0] - baseline[name][0]),
                                  'delta_ci95': np.quantile(samples[name] - base_scores[name], [.025, .975]).tolist()}
    return result


def main(args):
    current = json.loads((ROOT / 'outputs/hallucination-suite-current.json').read_text())
    output = Path(args.output or current['output']).resolve()
    assert output.is_relative_to(ROOT / 'outputs')
    jobs = json.loads((output / 'jobs.json').read_text())
    protocol = json.loads((output / 'protocol.json').read_text())
    content = {k: v for k, v in protocol.items() if k != 'sha256'}
    assert hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == protocol['sha256']
    if str(output) == current['output']:
        assert protocol['sha256'] == current['protocol_sha256']
    for name, digest in protocol['source_hashes'].items():
        assert sha(ROOT / name) == digest
    for name, digest in protocol['manifest_hashes'].items():
        assert sha(DATA / name) == digest
    data_sources = protocol['dataset_provenance']['sources']
    assert sha(DATA / 'sources/annotations_trainval2014.zip') == data_sources['coco_annotations']['sha256']
    assert sha(DATA / 'sources/response_template.json') == data_sources['mmhal_template_sha256']
    for subset in ['random', 'popular', 'adversarial']:
        name = f'coco_pope_{subset}.json'
        assert sha(DATA / 'sources' / name) == data_sources[name]['sha256']
    for job in jobs:
        state = json.loads((output / f"worker-{job['worker']}.json").read_text())
        assert state['stage'] == 'complete', state
        process = Path('/proc') / str(job['pid'])
        assert not process.exists() or process.joinpath('stat').read_text().split(') ', 1)[1][0] == 'Z', 'Wait for actual worker exit'
    manifests = {'pope': read(DATA / 'pope-unique.jsonl'), 'chair': read(DATA / 'chair.jsonl'), 'mmhal': read(DATA / 'mmhal.jsonl')}
    expected = {'pope': 5127, 'chair': 500, 'mmhal': 96}
    assert {k: len(v) for k, v in manifests.items()} == expected
    hashes = json.loads((DATA / 'image-hashes.json').read_text())
    assert all(sha(path) == digest for path, digest in hashes.items())
    dev = [json.loads(line) for filename in ['fit.jsonl', 'calibration.jsonl'] for line in (ROOT / 'data/mmvp-full' / filename).read_text().splitlines()]
    dev_hashes = {sha(row['image']) for row in dev}
    dev_ids = {int(Path(row['image']).stem.rsplit('_', 1)[-1]) for row in dev}
    assert not set(hashes.values()) & dev_hashes
    assert not {int(Path(row['image']).stem.rsplit('_', 1)[-1]) for row in manifests['pope'] + manifests['chair']} & dev_ids
    unique = {method: {bench: {} for bench in manifests} for method in METHODS}
    tasks = json.loads((output / 'tasks.json').read_text())
    assert len({t['id'] for t in tasks}) == len(tasks)
    for task in tasks:
        path = output / 'predictions' / f"{task['id']}.jsonl"
        marker = json.loads((output / 'claims' / task['id'] / 'complete.json').read_text())
        assert sha(path) == marker['prediction_sha256']
        rows = read(path)
        source = manifests[task['benchmark']][task['start']:task['stop']]
        assert len(rows) == len(source) == marker['records']
        for row, reference in zip(rows, source):
            assert all(row[k] == v for k, v in reference.items()), f'Changed manifest {row["id"]}'
            assert row['method'] == task['method'] and row['protocol_sha256'] == protocol['sha256']
            assert row['gpu_uuid'] in protocol['gpu_uuids'].values()
            assert row['generated_tokens'] > 0 and isinstance(row['prediction'], str)
            key = row['inference_key']
            target = unique[task['method']][task['benchmark']]
            assert key not in target, 'Duplicate generation'
            target[key] = row
    assert len(list((output / 'predictions').glob('*.jsonl'))) == len(tasks)
    for method in METHODS:
        assert {k: len(v) for k, v in unique[method].items()} == expected
    for method, digest in protocol['artifact_hashes'].items():
        path = output / 'artifacts' / f'{method}.pt'
        assert sha(path) == digest
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        assert all(torch.isfinite(v).all() for v in artifact['vectors'].values())
        assert any(float(v.norm()) > 0 for v in artifact['vectors'].values())
    merged = output / 'merged'
    merged.mkdir(exist_ok=True)
    metric_rows, scores, enriched_chair, question_sets = [], {}, {}, {s: {} for s in ['random', 'popular', 'adversarial']}
    pope_full = read(DATA / 'pope.jsonl')
    assert len(pope_full) == 9000
    cache = json.loads((DATA / 'chair-scorer-state.json').read_text())
    chair_path = ROOT / 'upstream/chair/chair.py'
    assert sha(chair_path) == cache['scorer_sha256']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=chair_path.parent, text=True).strip() == protocol['chair_scoring']['port_commit']
    assert cache['built_with_unmodified_constructor']
    for name, digest in cache['annotation_hashes'].items():
        assert sha(DATA / 'annotations' / name) == digest
    with zipfile.ZipFile(DATA / 'sources/annotations_trainval2014.zip') as archive:
        for name, digest in cache['annotation_hashes'].items():
            with archive.open('annotations/' + name) as source:
                hasher = hashlib.sha256()
                while chunk := source.read(1024 * 1024):
                    hasher.update(chunk)
                assert hasher.hexdigest() == digest
    module = _load_external_module(chair_path, '_suite_chair_scorer')
    scorer = module.CHAIR.__new__(module.CHAIR)
    scorer.__dict__.update(cache['state'])
    scorer.imid_to_objects = {int(k): set(v) for k, v in cache['state']['imid_to_objects'].items()}
    assert set(scorer.imid_to_objects) == {r['metadata']['image_id'] for r in manifests['chair']}
    for method in METHODS:
        scores[method] = {}
        pope_rows = []
        for reference in pope_full:
            prediction = unique[method]['pope'][reference['inference_key']]
            pope_rows.append({**prediction, **reference, 'method': method,
                              'generation_source_id': prediction['id']})
        write_jsonl(merged / f'{method}.pope.jsonl', pope_rows)
        for subset in ['random', 'popular', 'adversarial']:
            rows = [r for r in pope_rows if r['subset'] == subset]
            assert len(rows) == 3000 and len({r['id'] for r in rows}) == 3000
            original = read(DATA / 'sources' / f'coco_pope_{subset}.json')
            assert all(row['label'] == source['label'] and row['metadata']['original_question'] == source['text'] for row, source in zip(rows, original))
            metrics = pope_metrics(rows)
            verify_upstream_pope(rows, output, method, subset, metrics)
            strict = pope_metrics(rows, parse_yes_no)
            metrics['strict_leading_yes_no'] = strict
            scores[method]['pope_' + subset] = metrics
            metric_rows.append({'method': method, 'benchmark': 'POPE-' + subset,
                                **{k: v for k, v in metrics.items() if not isinstance(v, dict)}, 'strict_invalid': strict['invalid']})
            question_sets[subset][method] = rows
        chair_rows = [unique[method]['chair'][r['inference_key']] for r in manifests['chair']]
        payload = [{'image_id': r['metadata']['image_id'], 'caption': r['prediction']} for r in chair_rows]
        path = merged / f'{method}.chair.captions.json'
        save(path, payload)
        result = scorer.compute_chair(str(path), 'image_id', 'caption')
        result['scorer_provenance'] = {'implementation': 'unmodified Maxlinn Python3 port of original CHAIR',
            'sha256': cache['scorer_sha256'], 'commit': protocol['chair_scoring']['port_commit'],
            'ground_truth': 'unmodified CHAIR constructor on all original COCO2014 annotations; JSON cached selected cohort state',
            'cache_sha256': sha(DATA / 'chair-scorer-state.json'), 'annotation_hashes': cache['annotation_hashes']}
        save(merged / f'{method}.chair.official.json', result)
        enriched = attach_chair_scores(chair_rows, result)
        enriched_chair[method] = enriched
        write_jsonl(merged / f'{method}.chair.jsonl', enriched)
        metrics = evaluate_predictions(enriched)
        assert math.isclose(metrics['CHAIR_s'], result['overall_metrics']['CHAIRs'], abs_tol=1e-12)
        assert math.isclose(metrics['CHAIR_i'], result['overall_metrics']['CHAIRi'], abs_tol=1e-12)
        metrics['at_token_cap'] = sum(r['generated_tokens'] >= r['max_new_tokens'] for r in enriched)
        scores[method]['chair'] = metrics
        metric_rows.append({'method': method, 'benchmark': 'CHAIR-COCO500', **metrics})
        mmhal_rows = [unique[method]['mmhal'][r['inference_key']] for r in manifests['mmhal']]
        write_jsonl(merged / f'{method}.mmhal.jsonl', mmhal_rows)
        responses = mmhal_responses(mmhal_rows, DATA / 'sources/response_template.json')
        save(merged / f'{method}.mmhal.responses.json', responses)
        metrics = evaluate_predictions(mmhal_rows)
        assert metrics['mmhal_status'] == 'not_scored'
        metrics.update(reason='Official GPT-4 scoring deferred by explicit user request',
                       at_token_cap=sum(r['generated_tokens'] >= r['max_new_tokens'] for r in mmhal_rows))
        scores[method]['mmhal'] = metrics
        metric_rows.append({'method': method, 'benchmark': 'MMHal-Bench', **metrics})
    uncertainty = {f'pope_{subset}': intervals(rows, 'pope') for subset, rows in question_sets.items()}
    uncertainty['chair'] = intervals(enriched_chair, 'chair')
    report = {'status': 'verified_complete_with_user_deferred_mmhal_scoring', 'protocol': protocol,
              'analysis_source_hashes': {name: sha(ROOT / 'scripts' / name) for name in
                  ['audit_hallucination_suite.py', 'build_chair_cache.py', 'finish_hallucination_suite.py']},
              'metrics': scores, 'uncertainty': uncertainty,
              'audit': {'all_methods': METHODS, 'pope_questions_per_method': 9000, 'pope_unique_generations_per_method': expected['pope'],
                        'chair_images_per_method': 500, 'mmhal_questions_per_method': 96,
                        'total_benchmark_responses': 5 * (9000 + 500 + 96),
                        'total_unique_generations': 5 * sum(expected.values()),
                        'identical_cohorts': True, 'source_and_input_hashes_match': True,
                        'artifact_hashes_finite_nonzero': True, 'development_test_image_hash_overlap': 0,
                        'official_POPE_source_execution_matches': True, 'official_CHAIR_source_used': True,
                        'worker_processes_finished': True, 'mmhal_official_schema_exported': True,
                        'mmhal_judge_deferred_by_user': True},
              'validation': {'cpu_suite': '86 passed, 2 skipped before WordNet repair',
                             'actual_CHAIR_integration_after_repair': '15 passed, 1 skipped (unrequested Causal-HalBench)'},
              'completed_at': time.time(), 'output': str(output)}
    save(output / 'report.json', report)
    fields = list(dict.fromkeys(k for row in metric_rows for k in row))
    with (output / 'metrics.csv').open('w', newline='') as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader(); writer.writerows(metric_rows)
    lines = ['# Gemma E4B · POPE / CHAIR / MMHal-Bench', '',
             'Base와 baseline 4개를 동일한 데이터·설정으로 평가했다. Gemma에 적용한 저장소의 논문 기반 구현이며, 논문 저자 환경의 재현 실험은 아니다.', '',
             '| 방법 | POPE random Acc / F1 | POPE popular Acc / F1 | POPE adversarial Acc / F1 | CHAIR_s ↓ | CHAIR_i ↓ |',
             '|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        m = scores[method]
        values = [f"{m['pope_'+s]['accuracy']*100:.2f} / {m['pope_'+s]['f1']*100:.2f}" for s in ['random', 'popular', 'adversarial']]
        lines.append(f"| {method} | {' | '.join(values)} | {m['chair']['CHAIR_s']*100:.2f} | {m['chair']['CHAIR_i']*100:.2f} |")
    lines += ['', '모든 수치는 % 단위다. POPE는 각 유형 전체 3,000문항(총 9,000문항)을 공식 스크립트로 채점했다. CHAIR는 seed 42로 고정한 COCO val2014 500장 평가이며, 전체 COCO validation 평가를 뜻하지 않는다.', '',
              'MMHal-Bench는 각 방법의 전체 96문항 답변과 공식 response JSON을 생성했다. 사용자의 요청에 따라 GPT-4 공식 채점은 보류했으며, MMHal 점수·환각률은 산출하지 않았다.', '',
              '## 프로토콜과 검증', '',
              '- Gemma google/gemma-4-E4B-it, 고정 revision, BF16, 448px 전처리, greedy, 매 문항 seed 42.',
              '- POPE max_new_tokens 128, CHAIR·MMHal 512. CHAIR prompt: Please describe this image in detail. 토큰 한도 도달 건수와 평균 출력 길이를 report.json에 포함했다.',
              '- COCO train2017의 기존 32장 벡터와 REVIS 보정 32장을 고정 재사용했다. 테스트 이미지 ID 및 정확한 이미지 해시 중복은 0개다.',
              '- 테스트 정답으로 강도나 하이퍼파라미터를 선택하지 않았다. GrAInS는 검증된 Gauss–Legendre IG 16단계 벡터를 사용했다.',
              '- POPE의 완전히 동일한 이미지·질문은 한 번 생성 후 동일 응답을 재사용했다. 세 유형의 모든 문항은 보존했으며, 유형별 3,000문항을 각각 채점했다.',
              '- CHAIR는 원 CHAIR의 고정된 Python3 포트를 수정 없이 실행했다. 전체 원본 COCO annotation에서 만든 정답 집합을 캐시했다.',
              '- 이미지 단위 paired bootstrap 10,000회로 점수 및 Base 대비 차이의 95% 구간을 report.json에 보관했다.',
              '- 모든 원본 응답과 채점 근거는 merged/와 pope-official/에 보관했다.', '',
              '## 공식 출처', '',
              '- [POPE 데이터·채점 코드](https://github.com/RUCAIBox/POPE)',
              '- [원 CHAIR](https://github.com/LisaAnne/Hallucination)',
              '- [CHAIR Python3 포트](https://github.com/Maxlinn/CHAIR-metric-standalone)',
              '- [MMHal-Bench 데이터·공식 평가](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench)',
              '', f'서버 결과: `{output}`', '']
    (output / 'REPORT_KO.md').write_text('\n'.join(lines))
    print(json.dumps({'status': report['status'], 'metrics': scores, 'output': str(output)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='Specific run directory; defaults to current run pointer')
    main(parser.parse_args())
