"""Independently audit all 1,500 responses and write the final MMVP report."""

import ast
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

REPO = Path('/home/project-b/LJW/vlm-grounding-steering')
DATA = REPO / 'data/mmvp-full'
METHODS = ('base', 'revis', 'rudder', 'grains', 'vti')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def answer(text):
    value = text.strip().upper().strip('(). \r\n\t')
    return value if value in ('A', 'B') else None


def main():
    state = json.loads((REPO / 'outputs/mmvp-full-current.json').read_text())
    output = Path(state['output'])
    protocol = json.loads((output / 'protocol.json').read_text())
    protocol_copy = {k: v for k, v in protocol.items() if k != 'sha256'}
    assert hashlib.sha256(json.dumps(protocol_copy, sort_keys=True).encode()).hexdigest() == protocol['sha256']
    assert protocol['evaluation_records'] == 300 and protocol['evaluation_pairs'] == 150
    previous_source = Path(protocol['supersedes']) / 'source-snapshot/baselines.before-ig-fix.py'
    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
    before, after = functions(previous_source), functions(REPO / 'src/vlm_grounding/baselines.py')
    changed_functions = [name for name in before if before[name] != after[name]]
    assert set(changed_functions) == {'_preference_attributions', '_grains_extract'}
    for name, expected in protocol['source_hashes'].items():
        assert digest(REPO / name) == expected, f'Source changed during the experiment: {name}'
    assert digest(output / 'config.json') == protocol['config_sha256']
    assert digest(DATA / 'evaluation.jsonl') == protocol['evaluation_sha256']
    assert digest(DATA / 'fit.jsonl') == protocol['fit_sha256']
    assert digest(DATA / 'calibration.jsonl') == protocol['calibration_sha256']
    assert digest(DATA / 'source/Questions.csv') == protocol['questions_sha256']
    source = list(csv.DictReader((DATA / 'source/Questions.csv').open()))
    assert len(source) == 300 and {int(r['Index']) for r in source} == set(range(1, 301))
    source = {int(r['Index']): r for r in source}
    manifest = {r['index']: r for r in jsonl(DATA / 'evaluation.jsonl')}
    fit, calibration = jsonl(DATA / 'fit.jsonl'), jsonl(DATA / 'calibration.jsonl')
    assert len(fit) == len(calibration) == 32
    assert not {r['group_id'] for r in fit} & {r['group_id'] for r in calibration}
    test_hashes = {digest(r['image']) for r in manifest.values()}
    development_hashes = {digest(r['image']) for r in fit + calibration}
    assert not test_hashes & development_hashes
    correctness = {}
    corrected_correctness = {}
    results = {}
    provenance = {}
    corrections = {99: 'B', 100: 'A', 279: 'B', 280: 'A'}
    for method in METHODS:
        rows = jsonl(output / f'{method}.jsonl')
        metadata = json.loads((output / f'{method}.metadata.json').read_text())
        assert len(rows) == 300 and [r['index'] for r in rows] == list(range(1, 301))
        assert metadata['status'] == 'complete' and metadata['cohort_complete'] is True
        assert metadata['requested_limit'] is None and metadata['completed_records'] == 300
        assert metadata['protocol']['sha256'] == protocol['sha256']
        assert metadata['checkpoint_revision'] == protocol['config']['model']['revision']
        flags, corrected, invalid = [], [], 0
        for row in rows:
            index = row['index']
            truth = answer(source[index]['Correct Answer'])
            parsed = answer(row['response'])
            assert truth in ('A', 'B') and row['target'] == truth
            assert row['method'] == method and row['protocol_sha256'] == protocol['sha256']
            assert row['prompt'] == manifest[index]['prompt'] and row['image'] == manifest[index]['image']
            assert row['image_sha256'] == digest(row['image'])
            assert row['choice'] == parsed and row['correct'] == (parsed == truth)
            assert 1 <= row['generated_tokens'] <= protocol['config']['generation']['max_new_tokens']
            assert row['elapsed_seconds'] > 0
            flags.append(parsed == truth)
            corrected.append(parsed == corrections.get(index, truth))
            invalid += int(parsed is None)
        correctness[method] = np.array(flags).reshape(150, 2)
        corrected_correctness[method] = np.array(corrected).reshape(150, 2)
        results[method] = {
            'questions': 300, 'pairs': 150, 'correct_questions': sum(flags),
            'question_accuracy': float(np.mean(flags)),
            'correct_pairs': int(correctness[method].all(axis=1).sum()),
            'pair_accuracy': float(correctness[method].all(axis=1).mean()),
            'invalid_responses': invalid,
            'corrected_label_question_accuracy': float(np.mean(corrected)),
            'corrected_label_pair_accuracy': float(corrected_correctness[method].all(axis=1).mean()),
            'inference_seconds': sum(r['elapsed_seconds'] for r in rows),
            'generated_tokens': sum(r['generated_tokens'] for r in rows),
            'peak_gpu_allocated_gb': metadata['peak_gpu_allocated_gb'],
        }
        for key in ('correct_questions', 'correct_pairs', 'invalid_responses'):
            assert results[method][key] == metadata['metrics'][key]
        if method not in ('base', 'rudder'):
            artifact_path = output / 'artifacts' / f'{method}.pt'
            assert digest(artifact_path) == metadata['artifact_sha256']
            artifact = torch.load(artifact_path, map_location='cpu', weights_only=True)
            assert artifact['name'] == method and artifact['gpu_verified']
            expected_ids = {r['id'] for r in fit + (calibration if method == 'revis' else [])}
            assert set(artifact['_provenance']['record_ids']) == expected_ids
            assert all(torch.isfinite(v).all() for v in artifact['vectors'].values())
            assert any(float(v.norm()) > 0 for v in artifact['vectors'].values())
            if method == 'grains':
                assert artifact['ig_quadrature'] == 'gauss-legendre'
                assert len(artifact['attribution_token_counts']) == 32
                assert any(r['positive'] + r['negative'] > 0
                           for r in artifact['attribution_token_counts'])
            provenance[method] = {'record_count': len(expected_ids),
                                  'artifact_sha256': digest(artifact_path),
                                  'selected_layer': artifact.get('layer'),
                                  'threshold': artifact.get('threshold')}
    rng = np.random.default_rng(42)
    samples = rng.integers(0, 150, size=(10000, 150))
    differences = {}
    base_pairs = correctness['base'].all(axis=1).astype(float)
    base_questions = correctness['base'].mean(axis=1)
    for method in METHODS[1:]:
        pair_delta = correctness[method].all(axis=1).astype(float) - base_pairs
        question_delta = correctness[method].mean(axis=1) - base_questions
        differences[method] = {'pair_accuracy_delta': float(pair_delta.mean()),
                               'pair_accuracy_delta_ci95': np.quantile(pair_delta[samples].mean(axis=1), [.025, .975]).tolist(),
                               'question_accuracy_delta': float(question_delta.mean()),
                               'question_accuracy_delta_ci95': np.quantile(question_delta[samples].mean(axis=1), [.025, .975]).tolist()}
    for job in state['jobs']:
        status = json.loads((output / f"status-gpu{job['gpu']}.json").read_text())
        assert status['stage'] == 'complete'
        assert status['completed_methods'] == job['methods']
    report = {'status': 'verified_complete', 'benchmark': 'MMVP',
              'evaluation_questions_per_method': 300, 'evaluation_pairs_per_method': 150,
              'methods': list(METHODS), 'total_responses': 1500,
              'protocol': protocol, 'metrics': results, 'paired_differences': differences,
              'artifact_provenance': provenance,
              'validation_tests': {
                  'cpu_regression': (REPO / 'outputs/setup/pytest-after-ig-fix.txt').read_text().strip(),
                  'model_class_integration': (REPO / 'outputs/setup/ig-model-integration.txt').read_text().strip(),
              },
              'audit': {'full_coverage': True, 'identical_cohorts': True,
                        'common_generation_protocol': True, 'no_test_limit': True,
                        'source_and_input_hashes_verified': True,
                        'development_test_image_hash_overlap': 0,
                        'baseline_artifacts_finite_nonzero': True,
                        'independent_rescoring_matches': True,
                        'numerical_fix_changed_functions': changed_functions,
                        'other_baseline_functions_unchanged': True},
              'wall_seconds_since_gpu_launch': time.time() - protocol['started_at'],
              'output': str(output)}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', *results['base']])
        writer.writeheader()
        for method in METHODS:
            writer.writerow({'method': method, **results[method]})
    lines = ['# Gemma 4 E4B: full MMVP baseline comparison', '',
             'All 300 original MMVP questions (150 image pairs) were completed for each method.',
             'The four steering methods use this repository’s paper-derived local adaptations; author reproduction is false.',
             '', '| Method | Question accuracy | Pair accuracy | Invalid | Inference seconds |',
             '|---|---:|---:|---:|---:|']
    for method in METHODS:
        m = results[method]
        lines.append(f"| {method} | {m['correct_questions']}/300 ({100*m['question_accuracy']:.2f}%) | {m['correct_pairs']}/150 ({100*m['pair_accuracy']:.2f}%) | {m['invalid_responses']} | {m['inference_seconds']:.1f} |")
    lines += ['', '## Protocol and data', '',
              '- Model: google/gemma-4-E4B-it, revision ' + protocol['config']['model']['revision'],
              '- BF16, 448px preprocessing, greedy generation, max_new_tokens=128, seed 42.',
              '- Prompt: original question and options, followed by “Answer with the option letter A or B only.”',
              '- Scoring: strict exact A/B parsing; invalid responses remain incorrect. Pair accuracy requires both consecutive answers correct.',
              '- Development: 32 COCO train2017 images for vector extraction and 32 disjoint images for REVIS calibration, derived from Ultralytics COCO128.',
              '- Test labels were not used for extraction, calibration or hyperparameter selection. Configured alpha/mask strengths and 16 IG steps were retained.',
              '- GrAInS uses Gauss-Legendre IG, matching Captum’s default: the earlier trapezoidal zero endpoint produced NaN gradients on the real BF16 checkpoint. Explicit finite checks and two regression tests were added; all 74 CPU tests passed (2 optional tests skipped). All methods were rerun under the repaired common code snapshot.',
              '- Primary scores use original author labels. Separate corrected-label sensitivity is included in metrics.csv and report.json for indices 99, 100, 279, 280.',
              '- Inference times exclude model loading, preprocessing and vector extraction; GPU synchronization is included.',
              '- Paired 95% confidence intervals resample the 150 image pairs, with 10,000 bootstrap replicates.',
              '', '## Verification', '',
              'Verified 1,500 unique responses, exact coverage, common protocol hashes, unmodified source/input hashes, complete metadata, independent raw-response rescoring, finite nonzero vectors, and zero exact image-hash overlap.',
              '', '## Sources', '',
              '- [Official MMVP benchmark and 300-image release](https://github.com/tsb0601/MMVP)',
              '- [Original dataset](https://huggingface.co/datasets/MMVP/MMVP)',
              '- [LMMS-Eval pair scoring and known label corrections](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/mmvp/utils.py)',
              '- [Ultralytics COCO128 source](https://docs.ultralytics.com/datasets/detect/coco128/)',
              '- [Captum IntegratedGradients quadrature](https://captum.ai/api/integrated_gradients.html)',
              '', 'Server output: `' + str(output) + '`', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'status': report['status'], 'metrics': results, 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
