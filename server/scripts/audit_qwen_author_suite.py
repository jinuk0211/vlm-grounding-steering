"""Prepared independent full-coverage Qwen audit and unmodified official scoring."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import zipfile

import numpy as np
import torch
from author_clone_bridge import ROOT, load_authors, verify_clone_sources
from qwen_gpu23_adapter import CHECKPOINT, REVISION, GPU2, GPU3, PAIR, require_gemma_complete, verify_run_protocol
from vlm_grounding.cli import _validate_artifact
from vlm_grounding.evaluation import _load_external_module, attach_chair_scores, evaluate_predictions

METHODS = ['base', 'revis', 'vti', 'rudder_clone', 'base_fp32', 'grains_clone']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def save(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def actual_job_live(job, output):
    process = Path('/proc') / str(job['pid'])
    try:
        fields = (process / 'stat').read_text().split(') ', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return False
        if int(fields[19]) != job['start_ticks'] or process.stat().st_uid != os.getuid():
            return False
        command = [s.decode() for s in (process / 'cmdline').read_bytes().split(b'\0') if s]
        return job['script'] in command and '--output' in command and command[command.index('--output') + 1] == str(output)
    except FileNotFoundError:
        return False


def audit(output):
    assert output.resolve().parent == ROOT / 'outputs'
    require_gemma_complete()
    protocol = verify_run_protocol(output)
    body = {k: v for k, v in protocol.items() if k != 'sha256'}
    assert hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == protocol['sha256']
    assert protocol['model']['checkpoint'] == CHECKPOINT and protocol['model']['revision'] == REVISION
    assert protocol['model']['family'] == 'qwen2_5_vl' and protocol['methods'] == METHODS
    assert protocol['gpu_indices'] == [2, 3] and not protocol['other_gpu_fallback']
    assert sha(output / 'tasks.json') == protocol['tasks_sha256']
    for file, digest in protocol['source_hashes'].items():
        assert sha(ROOT / file) == digest and sha(output / 'source-snapshot' / file) == digest, file
    assert verify_clone_sources() == protocol['author_sources']
    for name, info in protocol['author_sources'].items():
        for file, digest in info['files'].items():
            assert sha(output / 'author-snapshot' / name / file) == digest
    for file, digest in protocol['manifest_hashes'].items():
        assert sha(output / 'data' / file) == digest
    datasets = {b: read(output / 'data' / (b + '.jsonl')) for b in ('pope', 'chair')}
    fit, calibration, fit50 = (read(output / 'data' / file) for file in ('fit.jsonl', 'calibration.jsonl', 'fit50.jsonl'))
    assert len(fit) == len(calibration) == 32 and fit50 == (fit + calibration)[:50]
    assert len({r['group_id'] for r in fit + calibration}) == 64
    development_hashes = {sha(r['image']) for r in fit + calibration}
    assert len(development_hashes) == 64
    for bench, rows in datasets.items():
        assert len(rows) == len({r['id'] for r in rows}) == (3000 if bench == 'pope' else 500)
        assert not development_hashes & {r['image_sha256'] for r in rows}
        assert all(sha(r['image']) == r['image_sha256'] for r in rows)
    for phase in ('bf16', 'fp32'):
        proof_path = output / 'proofs' / (phase + '-preflight.json')
        assert sha(proof_path) == protocol['preflight_hashes'][phase]
        proof = json.loads(proof_path.read_text()); assert proof['status'] == 'passed'
        for file, digest in proof['source_hashes'].items():
            assert protocol['source_hashes'][file] == digest
        for file, digest in proof['development_manifest_hashes'].items():
            assert sha(file) == digest
            assert sha(output / 'data' / Path(file).name) == digest
    assert sha(output / 'proofs/weight-cache-verification.json') == protocol['cached_weight_verification_sha256']
    assert json.loads((output / 'proofs/weight-cache-verification.json').read_text())['all_hf_lfs_sha256_verified']
    jobs = json.loads((output / 'jobs.json').read_text()); workers = {}
    for job in jobs:
        assert not actual_job_live(job, output), job
        assert json.loads((output / job['state_file']).read_text())['stage'] == 'complete'
        device = json.loads((output / job['device_proof']).read_text())
        assert device['checkpoint'] == CHECKPOINT and device['revision'] == REVISION
        assert device['visible_devices'] == job['visible_devices'] and device['dtype'] == job['dtype']
        assert not device['quantization'] and not device['cpu_or_disk_offload']
        if job['dtype'] == 'float32':
            assert device['visible_devices'] == PAIR
            assert device['parameter_dtypes'] == ['torch.float32'] and device['parameter_devices'] == ['cuda:0', 'cuda:1']
        else:
            assert device['visible_devices'] in (GPU2, GPU3) and device['parameter_devices'] == ['cuda:0']
            assert 'torch.bfloat16' in device['parameter_dtypes']
        if job['kind'] == 'generation':
            workers[job['worker']] = job
    config = json.loads((output / 'config.json').read_text())
    seals = json.loads((output / 'artifacts-ready.json').read_text())
    assert seals['status'] == 'verified_complete' and seals['source_protocol_sha256'] == protocol['sha256']
    for method in ('revis', 'vti'):
        path = output / 'artifacts' / (method + '.pt')
        assert sha(path) == seals['methods'][method]['sha256']
        sidecar = json.loads((output / (method + '-artifact.json')).read_text())
        assert sidecar['status'] == 'verified_complete' and sidecar['model'] == config['model']
        assert sidecar['source_protocol_sha256'] == protocol['sha256'] and not sidecar['gemma_vectors_reused']
        expected = fit + calibration if method == 'revis' else fit
        assert sidecar['records'] == [r['id'] for r in expected]
        for path_string, digest in sidecar['input_hashes'].items():
            assert sha(path_string) == digest
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        _validate_artifact(artifact, config['model'])
        for group in ('vectors', 'vision_vectors'):
            if group in artifact:
                assert all(torch.isfinite(v).all() for v in artifact[group].values())
        assert any(v.norm() > 0 for v in artifact['vectors'].values())
    journal = read(output / 'extraction.jsonl'); assert len(journal) == 50
    reconstructed = {}
    for index, (entry, record) in enumerate(zip(journal, fit50)):
        assert entry['index'] == index and entry['record'] == record
        assert sha(entry['artifact']) == entry['artifact_sha256'] and sha(entry['raw_ig']) == entry['raw_ig_sha256']
        assert entry['ig']['quadrature_nodes'] == 20 and entry['ig']['effective_internal_batch_size'] == 1
        raw = torch.load(entry['raw_ig'], map_location='cpu', weights_only=True)
        assert all(torch.isfinite(v).all() for v in raw.values())
        with np.load(entry['artifact'], allow_pickle=False) as arrays:
            for name in arrays.files:
                assert np.isfinite(arrays[name]).all()
                if name.startswith('hidden_'):
                    reconstructed.setdefault(name, []).append(arrays[name].copy())
    reconstructed = {k: np.stack(v) for k, v in reconstructed.items()}; assert len(reconstructed) == 4
    with np.load(output / 'original-hidden-states.npz', allow_pickle=False) as stored:
        assert set(stored.files) == set(reconstructed)
        assert all(np.array_equal(stored[k], v) for k, v in reconstructed.items())
    grains_seal = json.loads((output / 'grains-artifact.json').read_text())
    for name, key in [('grains-author.pt', 'artifact_sha256'), ('original-hidden-states.npz', 'hidden_states_sha256'), ('extraction.jsonl', 'journal_sha256')]:
        assert sha(output / name) == grains_seal[key]
    assert grains_seal['actual_ig_calls'] == 50 and grains_seal['actual_ig_quadrature_nodes'] == 20
    grains = torch.load(output / 'grains-author.pt', map_location='cpu', weights_only=True)
    fp32_model = {**config['model'], 'dtype': 'float32'}
    assert grains['model'] == fp32_model and not grains['gemma_vectors_reused']
    assert grains['fit_records'] == [r['id'] for r in fit50]
    np.random.seed(42); authors = load_authors()
    expected_vector = authors.steering.compute_steering_vector(reconstructed, 27, 'pca')
    torch.testing.assert_close(expected_vector, grains['vector'], rtol=1e-5, atol=1e-6)
    assert torch.isfinite(grains['vector']).all() and grains['vector'].norm() > 0
    assert sha(output / 'grains-author.pt') == seals['methods']['grains_clone']['sha256']
    grouped = {b: {m: {} for m in METHODS} for b in datasets}
    tasks = json.loads((output / 'tasks.json').read_text())
    assert len(list((output / 'predictions').glob('*.jsonl'))) == len(tasks) == 678
    for task in tasks:
        path = output / 'predictions' / (task['id'] + '.jsonl')
        marker = json.loads((output / 'claims' / task['id'] / 'complete.json').read_text())
        assert marker['sha256'] == sha(path)
        rows = read(path); refs = datasets[task['benchmark']][task['start']:task['stop']]
        assert len(rows) == len(refs) == marker['records']
        for row, ref in zip(rows, refs):
            assert all(row.get(k) == v for k, v in ref.items())
            assert row['method'] == task['method'] and row['dtype'] == task['dtype']
            assert row['protocol_sha256'] == protocol['sha256'] and row['model_family'] == 'qwen2_5_vl'
            assert row['checkpoint'] == CHECKPOINT and row['revision'] == REVISION
            assert 'adopted_cache_sha256' not in row
            assert row['worker'] in workers and row['gpu_uuid'] == workers[row['worker']]['visible_devices']
            assert row['gpu_uuid'] in ((GPU2, GPU3) if row['dtype'] == 'bfloat16' else (PAIR,))
            assert row['artifact_sha256'] == seals['methods'].get(row['method'], {}).get('sha256')
            assert isinstance(row['prediction'], str) and 0 < row['generated_tokens'] <= ref['max_new_tokens']
            if row['method'] in ('rudder_clone', 'grains_clone'):
                diagnostics = row['author_diagnostics']; assert diagnostics['calls'] > 0
                if row['method'] == 'rudder_clone':
                    assert diagnostics['active_positions'] > 0 and diagnostics['card_norm'] > 0
                    assert diagnostics['layer'] == (1 if task['benchmark'] == 'pope' else 25)
                    assert diagnostics['alpha'] == (8.0 if task['benchmark'] == 'pope' else 5.0)
                    assert diagnostics['hook_source'].endswith('RUDDER/Qwen/qwen_methods.py')
                else:
                    assert diagnostics['layer'] == 27 and diagnostics['alpha'] == 10.0
                    assert diagnostics['hook_source'].endswith('GrAInS/src/utils/steering.py')
            bucket = grouped[task['benchmark']][task['method']]
            assert row['id'] not in bucket; bucket[row['id']] = row
    for bench, methods in grouped.items():
        for method, rows in methods.items():
            assert set(rows) == {r['id'] for r in datasets[bench]}
            methods[method] = [rows[r['id']] for r in datasets[bench]]
    score_and_report(output, protocol, grouped)


def score_and_report(output, protocol, grouped):
    spec = importlib.util.spec_from_file_location('qwen_official_metric_helper', ROOT / 'scripts/audit_hallucination_suite.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
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
    module = _load_external_module(source, 'qwen_original_chair')
    scorer = module.CHAIR.__new__(module.CHAIR); scorer.__dict__.update(cache['state'])
    scorer.imid_to_objects = {int(k): set(v) for k, v in cache['state']['imid_to_objects'].items()}
    metrics = {}; uncertainty = {}; flat = []
    for bench, methods in grouped.items():
        folder = output / bench; folder.mkdir(exist_ok=True); metrics[bench] = {}
        for method, rows in methods.items():
            if bench == 'pope':
                official = read(data / 'sources/coco_pope_adversarial.json')
                assert all(r['metadata']['original_question'] == q['text'] and r['label'] == q['label'] for r, q in zip(rows, official))
                assert sha(data / 'sources/coco_pope_adversarial.json') == protocol['dataset_provenance']['sources']['coco_pope_adversarial.json']['sha256']
                value = helper.pope_metrics(rows)
                helper.verify_upstream_pope(rows, folder, method, 'adversarial', value)
                value['strict'] = evaluate_predictions(rows)
            else:
                path = folder / (method + '.captions.json')
                save(path, [{'image_id': r['metadata']['image_id'], 'caption': r['prediction']} for r in rows])
                scored = scorer.compute_chair(str(path), 'image_id', 'caption')
                save(folder / (method + '.chair.official.json'), scored)
                rows = attach_chair_scores(rows, scored); methods[method] = rows
                value = evaluate_predictions(rows)
                assert math.isclose(value['CHAIR_s'], scored['overall_metrics']['CHAIRs'])
                assert math.isclose(value['CHAIR_i'], scored['overall_metrics']['CHAIRi'])
            value.update(at_token_cap=sum(r['generated_tokens'] >= r['max_new_tokens'] for r in rows),
                         empty_outputs=sum(not r['prediction'].strip() for r in rows),
                         mean_generated_tokens=sum(r['generated_tokens'] for r in rows) / len(rows))
            metrics[bench][method] = value; helper.write_jsonl(folder / (method + '.jsonl'), rows)
            flat.append({'benchmark': bench, 'method': method, 'matched_base': protocol['comparisons'].get(method),
                         **{k: v for k, v in value.items() if not isinstance(v, dict)}})
        uncertainty[bench] = {
            'bf16': helper.intervals({m: methods[m] for m in ('base', 'revis', 'vti', 'rudder_clone')}, bench),
            'fp32': helper.intervals({'base': methods['base_fp32'], 'grains_clone': methods['grains_clone']}, bench)}
    report = {'status': 'verified_complete', 'scope': protocol['scope'], 'protocol': protocol,
        'metrics': metrics, 'uncertainty': uncertainty, 'output': str(output),
        'analysis_hashes': {'auditor': sha(__file__)},
        'audit': {'all_21000_responses_verified': True, 'official_scorers_used': True,
            'actual_workers_exited': True, 'original_author_sources_unmodified': True,
            'source_input_artifact_hashes_verified': True, 'model_specific_vectors_refit': True,
            'fit50_raw_ig_hidden_and_pca_independently_verified': True,
            'development_test_image_hash_overlap': 0, 'matched_precision_baselines': True,
            'only_gpu2_gpu3_used': True, 'no_gemma_responses_or_vectors_reused': True}}
    save(output / 'report.json', report)
    columns = list(dict.fromkeys(k for row in flat for k in row))
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(flat)
    md = ['# Qwen2.5-VL-7B · POPE adversarial / CHAIR', '',
        '각 프로필 POPE 3,000문항·CHAIR 500장. 수치는 %. GPU2·3만 사용했다.', '',
        '| 방법 | 정밀도 | POPE Acc ↑ | POPE F1 ↑ | CHAIR_s ↓ | CHAIR_i ↓ | 객체 recall ↑ |',
        '|---|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        p, c = metrics['pope'][method], metrics['chair'][method]
        dtype = 'FP32' if method in ('base_fp32', 'grains_clone') else 'BF16'
        label = {'vti': 'VTI 자체 구현', 'revis': 'REVIS 자체 구현'}.get(method, method)
        md.append(f"| {label} | {dtype} | {100*p['accuracy']:.2f} | {100*p['f1']:.2f} | {100*c['CHAIR_s']:.2f} | {100*c['CHAIR_i']:.2f} | {100*c['object_recall']:.2f} |")
    md += ['', 'GrAInS는 Base FP32와, 나머지는 Base BF16과 비교한다. 모델은 고정 revision의 기존 LJW 캐시를 사용했다.',
        '', 'REVIS·VTI는 선택한 서버 자체 구현으로 Qwen용 벡터를 새로 추출했다. VTI는 원본과 개입 위치·정규화·PCA·mask·prefill 범위가 달라 원본 VTI의 재현 점수로 인용하지 않는다. RUDDER·GrAInS는 고정 저자 커밋의 원본 함수를 호출했다. 전체 저자 checkpoint·환경 재현 완료를 주장하지 않는다.',
        '', 'greedy·배치1·문항별 seed42, 이미지448·POPE 최대128토큰·CHAIR 최대512토큰을 유지했다. Qwen native 입력, 원본 token-only IG/ablation·PCA·sequence축 norm·10배 계수를 보존했다. hidden-state 출력의 장치 이동만 연결했으며 전체 시각 경로 IG라고 표현하지 않는다.',
        '', '공식 POPE evaluate.py와 원본 CHAIR Python3 포트로 채점했다. 데이터·원본 함수 호출·벡터·전체 ID coverage·GPU 사용을 독립 검증했고, 이미지 단위 paired bootstrap 10,000회의95% 구간은 report.json에 있다.', '']
    (output / 'REPORT_KO.md').write_text('\n'.join(md))
    print(json.dumps({'status': 'verified_complete', 'metrics': metrics, 'output': str(output)}, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args().output)
