"""Audit one complete user-selected cohort, including adopted response provenance."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import zipfile
from pathlib import Path

import torch

from vlm_grounding.evaluation import _load_external_module, attach_chair_scores, evaluate_predictions

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')
DATA = ROOT / 'data/hallucination-suite'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(args):
    output = Path(args.output).resolve()
    assert output.is_relative_to(ROOT / 'outputs')
    protocol = json.loads((output / 'protocol.json').read_text())
    canonical = {k: v for k, v in protocol.items() if k != 'sha256'}
    assert hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == protocol['sha256']
    benchmark = protocol['selected_benchmark']
    for name, digest in protocol['source_hashes'].items():
        assert sha(ROOT / name) == digest
    selected_data = Path(protocol['data_directory'])
    for name, digest in protocol['manifest_hashes'].items():
        assert sha(selected_data / name) == digest
    for name, digest in protocol['adopted_hashes'].items():
        assert sha(output / 'adopted' / name) == digest
    parent = json.loads((Path(protocol['parent_output']) / 'protocol.json').read_text())
    parent_body = {k: v for k, v in parent.items() if k != 'sha256'}
    assert hashlib.sha256(json.dumps(parent_body, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == parent['sha256'] == protocol['parent_protocol_sha256']
    assert parent['model'] == protocol['model'] and parent['method_parameters'] == protocol['method_parameters']
    for name in ['src/vlm_grounding/backend.py', 'src/vlm_grounding/baselines.py']:
        assert parent['source_hashes'][name] == protocol['source_hashes'][name]
    for job in json.loads((output / 'jobs.json').read_text()):
        status = json.loads((output / f"worker-{job['worker']}.json").read_text())
        assert status['stage'] == 'complete', status
        try:
            state = (Path('/proc') / str(job['pid']) / 'stat').read_text().split(') ', 1)[1][0]
        except FileNotFoundError:
            state = 'exited'
        assert state in ['Z', 'exited'], 'Worker process has not exited'
    filename = 'pope-unique.jsonl' if benchmark == 'pope' else 'chair.jsonl'
    manifest = read(selected_data / filename)
    assert len(manifest) == (3000 if benchmark == 'pope' else 500)
    assert len({r['id'] for r in manifest}) == len(manifest)
    dev = [json.loads(line) for name in ['fit.jsonl', 'calibration.jsonl'] for line in (ROOT / 'data/mmvp-full' / name).read_text().splitlines()]
    dev_hashes = {sha(row['image']) for row in dev}
    images = {row['image']: row['image_sha256'] for row in manifest}
    assert not set(images.values()) & dev_hashes
    assert all(sha(path) == digest for path, digest in images.items())
    adopted = {(row['method'], row['inference_key']): row for file in (output / 'adopted').glob('*.jsonl') for row in read(file)}
    by_method = {m: {} for m in protocol['methods']}
    tasks = json.loads((output / 'tasks.json').read_text())
    for task in tasks:
        path = output / 'predictions' / f"{task['id']}.jsonl"
        marker = json.loads((output / 'claims' / task['id'] / 'complete.json').read_text())
        assert sha(path) == marker['prediction_sha256']
        rows, references = read(path), manifest[task['start']:task['stop']]
        assert len(rows) == len(references) == marker['records']
        for row, ref in zip(rows, references):
            assert all(row.get(k) == v for k, v in ref.items())
            assert row['method'] == task['method'] and row['protocol_sha256'] == protocol['sha256']
            assert row['gpu_uuid'] in protocol['gpu_uuids'].values()
            if 'adopted_from_protocol' in row:
                cached = adopted[(row['method'], row['inference_key'])]
                assert row['adopted_from_protocol'] == protocol['parent_protocol_sha256'] == cached['protocol_sha256']
                assert row['prediction'] == cached['prediction'] and row['generated_tokens'] == cached['generated_tokens']
                assert row['max_new_tokens'] == cached['max_new_tokens']
            assert row['id'] not in by_method[task['method']]
            by_method[task['method']][row['id']] = row
    assert len(list((output / 'predictions').glob('*.jsonl'))) == len(tasks)
    for method, records in by_method.items():
        assert len(records) == len(manifest) and records.keys() == {r['id'] for r in manifest}
        by_method[method] = [records[r['id']] for r in manifest]
    for method, digest in protocol['artifact_hashes'].items():
        path = output / 'artifacts' / f'{method}.pt'
        assert sha(path) == digest
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        assert all(torch.isfinite(v).all() for v in artifact['vectors'].values())
        assert any(float(v.norm()) > 0 for v in artifact['vectors'].values())
    spec = importlib.util.spec_from_file_location('_full_suite_auditor', ROOT / 'scripts/audit_hallucination_suite.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    merged = output / 'merged'
    merged.mkdir(exist_ok=True)
    metrics = {}
    if benchmark == 'chair':
        cache = json.loads((DATA / 'chair-scorer-state.json').read_text())
        assert cache['built_with_unmodified_constructor']
        source = ROOT / 'upstream/chair/chair.py'
        assert sha(source) == cache['scorer_sha256']
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source.parent, text=True).strip() == protocol['chair_scoring']['port_commit']
        subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'chair.py'], cwd=source.parent, check=True, capture_output=True)
        module = _load_external_module(source, '_selected_chair')
        scorer = module.CHAIR.__new__(module.CHAIR)
        scorer.__dict__.update(cache['state'])
        scorer.imid_to_objects = {int(k): set(v) for k, v in cache['state']['imid_to_objects'].items()}
        for name, digest in cache['annotation_hashes'].items():
            assert sha(DATA / 'annotations' / name) == digest
        archive = DATA / 'sources/annotations_trainval2014.zip'
        assert sha(archive) == protocol['dataset_provenance']['sources']['coco_annotations']['sha256']
        with zipfile.ZipFile(archive) as zipped:
            for name, digest in cache['annotation_hashes'].items():
                hasher = hashlib.sha256()
                with zipped.open('annotations/' + name) as source:
                    while chunk := source.read(1024 * 1024):
                        hasher.update(chunk)
                assert hasher.hexdigest() == digest
    for method, rows in by_method.items():
        if benchmark == 'pope':
            assert sha(DATA / 'sources/coco_pope_adversarial.json') == protocol['dataset_provenance']['sources']['coco_pope_adversarial.json']['sha256']
            original = read(DATA / 'sources/coco_pope_adversarial.json')
            assert all(r['subset'] == 'adversarial' for r in rows)
            assert all(r['label'] == q['label'] and r['metadata']['original_question'] == q['text'] for r, q in zip(rows, original))
            metrics[method] = helper.pope_metrics(rows)
            helper.verify_upstream_pope(rows, output, method, 'adversarial', metrics[method])
            metrics[method]['strict'] = evaluate_predictions(rows)
        else:
            payload = [{'image_id': r['metadata']['image_id'], 'caption': r['prediction']} for r in rows]
            path = merged / f'{method}.captions.json'
            save(path, payload)
            result = scorer.compute_chair(str(path), 'image_id', 'caption')
            result['scorer_provenance'] = {'implementation': 'unmodified Maxlinn CHAIR Python3 port',
                'sha256': cache['scorer_sha256'], 'cache_sha256': sha(DATA / 'chair-scorer-state.json'),
                'annotation_hashes': cache['annotation_hashes']}
            save(merged / f'{method}.chair.official.json', result)
            rows = attach_chair_scores(rows, result)
            by_method[method] = rows
            metrics[method] = evaluate_predictions(rows)
            assert math.isclose(metrics[method]['CHAIR_s'], result['overall_metrics']['CHAIRs'])
            assert math.isclose(metrics[method]['CHAIR_i'], result['overall_metrics']['CHAIRi'])
        metrics[method]['at_token_cap'] = sum(r['generated_tokens'] >= r['max_new_tokens'] for r in rows)
        metrics[method]['generated_tokens_mean'] = sum(r['generated_tokens'] for r in rows) / len(rows)
        helper.write_jsonl(merged / f'{method}.jsonl', rows)
    uncertainty = helper.intervals(by_method, benchmark)
    report = {'status': 'verified_complete', 'benchmark': benchmark, 'protocol': protocol, 'metrics': metrics,
              'uncertainty': uncertainty, 'cases_per_method': len(manifest), 'total_responses': 5 * len(manifest),
              'audit': {'all_five_methods_complete': True, 'identical_cohorts': True, 'source_and_input_hashes_verified': True,
                        'adopted_responses_verified': True, 'development_test_exact_hash_overlap': 0,
                        'actual_workers_exited': True, 'official_scorer_used': True},
              'analysis_hashes': {'auditor': sha(__file__), 'helper': sha(ROOT / 'scripts/audit_hallucination_suite.py')},
              'output': str(output)}
    save(output / 'report.json', report)
    columns = list(dict.fromkeys(['method'] + [key for value in metrics.values() for key, val in value.items() if not isinstance(val, dict)]))
    with (output / 'metrics.csv').open('w', newline='') as sink:
        writer = csv.DictWriter(sink, fieldnames=columns);writer.writeheader()
        for method, value in metrics.items():
            writer.writerow({'method': method, **{k: v for k, v in value.items() if not isinstance(v, dict)}})
    print(json.dumps({'status': report['status'], 'benchmark': benchmark, 'metrics': metrics, 'output': str(output)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
