"""Prepared real-Qwen preflight; loader refuses to run before Gemma completes."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from author_clone_bridge import ROOT, load_authors, verify_clone_sources, rudder_context
from qwen_author_compat import extract_one_qwen, grains_on_qwen_layer_device
from qwen_gpu23_adapter import GPU2, PAIR, load_qwen, require_gemma_complete
from vlm_grounding.baselines import build_baseline, extract_baseline
from vlm_grounding.cli import _seed, _provenance, _validate_artifact


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main(output, phase):
    assert output.resolve().parent == ROOT / 'outputs'
    output.mkdir(parents=True, exist_ok=True)
    result = {'status': 'running', 'phase': phase, 'pid': os.getpid(), 'started_at': time.time(),
              'scope': 'Development-only compatibility preflight, not benchmark predictions'}
    try:
        require_gemma_complete()
        source_files = ['src/vlm_grounding/backend.py', 'src/vlm_grounding/baselines.py',
            'src/vlm_grounding/cli.py', 'src/vlm_grounding/evaluation.py',
            'scripts/author_clone_bridge.py', 'scripts/qwen_author_compat.py',
            'scripts/qwen_gpu23_adapter.py', 'scripts/preflight_qwen_authors.py']
        result['source_hashes'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}
        import vlm_grounding.backend as actual_backend
        import vlm_grounding.baselines as actual_baselines
        import vlm_grounding.cli as actual_cli
        import vlm_grounding.evaluation as actual_evaluation
        result['installed_source_hashes'] = {}
        for name, module in [('backend.py', actual_backend), ('baselines.py', actual_baselines),
                             ('cli.py', actual_cli), ('evaluation.py', actual_evaluation)]:
            digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            assert digest == result['source_hashes']['src/vlm_grounding/' + name]
            result['installed_source_hashes'][name] = digest
        result['dependency_versions'] = {name: importlib.metadata.version(name) for name in
            ['torch', 'transformers', 'accelerate', 'captum', 'scikit-learn', 'numpy']}
        config = json.loads((ROOT / 'configs/qwen2_5_vl_7b.json').read_text())
        config['model'].update(local_files_only=True, dtype='bfloat16' if phase == 'bf16' else 'float32')
        assert os.environ['CUDA_VISIBLE_DEVICES'] == (GPU2 if phase == 'bf16' else PAIR)
        _seed(config)
        adapter = load_qwen(config['model'], output, phase)
        authors = load_authors()
        result['author_sources'] = verify_clone_sources()
        fit_path = ROOT / 'data/mmvp-full/fit.jsonl'
        cal_path = ROOT / 'data/mmvp-full/calibration.jsonl'
        fit = read(fit_path); calibration = read(cal_path)
        result['development_manifest_hashes'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (fit_path, cal_path)}
        result['development_record_ids'] = [r['id'] for r in fit[:2] + calibration[:2]]
        record = fit[0]
        results = {}
        if phase == 'bf16':
            batch = adapter.prepare(record['image'], 'Please describe this image in detail.')
            results['base'] = adapter.generate(batch, max_new_tokens=8)
            diagnostics = {}
            _seed(config)
            with rudder_context(adapter, batch, 'chair', authors, diagnostics):
                results['rudder_chair'] = adapter.generate(batch, max_new_tokens=32)
            assert diagnostics['calls'] > 0 and diagnostics['active_positions'] > 0
            results['rudder_chair_diagnostics'] = diagnostics
            pope_batch = adapter.prepare(record['image'], record['prompt'] + '\nPlease answer yes or no.')
            diagnostics = {}
            with rudder_context(adapter, pope_batch, 'pope', authors, diagnostics):
                results['rudder_pope'] = adapter.generate(pope_batch, max_new_tokens=8)
            assert diagnostics['calls'] > 0 and diagnostics['active_positions'] > 0
            results['rudder_pope_diagnostics'] = diagnostics
            for method in ('revis', 'vti'):
                _seed(config)
                records = fit[:2] + ([{**r, 'role': 'calibration'} for r in calibration[:2]] if method == 'revis' else [])
                artifact = extract_baseline(method, adapter, records, config['methods'][method])
                vectors = list(artifact['vectors'].values())
                assert all(torch.isfinite(v).all() for v in vectors) and any(v.norm() > 0 for v in vectors)
                paths = [str(fit_path)] + ([str(cal_path)] if method == 'revis' else [])
                artifact['_provenance'] = _provenance(adapter, config, paths, records)
                artifact['gpu_verified'] = True
                _validate_artifact(artifact, config['model'], adapter=adapter)
                for group in ('vectors', 'vision_vectors'):
                    if group in artifact:
                        artifact[group] = {k: v.to(adapter.device) for k, v in artifact[group].items()}
                _seed(config)
                with build_baseline(method, adapter, artifact, config['methods'][method]):
                    results[method] = adapter.generate(batch, max_new_tokens=32)
                results[method + '_vector_layers'] = len(vectors)
        else:
            for parameter in adapter.model.parameters():
                parameter.requires_grad_(False)
            batch = adapter.prepare(record['image'], record['prompt'], record['chosen'])
            ids, mask = batch['input_ids'], batch['attention_mask']
            with torch.no_grad():
                native = adapter.model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1].float()
                embedded = adapter.model.get_input_embeddings()(ids)
                supplied = adapter.model(inputs_embeds=embedded, attention_mask=mask, use_cache=False).logits[:, -1].float()
            torch.testing.assert_close(native, supplied, rtol=1e-5, atol=1e-5)
            results['input_embedding_endpoint_max_logit_error'] = float((native - supplied).abs().max())
            del native, supplied, embedded, batch
            observed_nodes = []
            original_attribute = authors.ig_utils.IntegratedGradients.attribute
            def observed_attribute(instance, *args, **kwargs):
                observed_nodes.append(int(kwargs.get('n_steps', 50)))
                return original_attribute(instance, *args, **kwargs)
            authors.ig_utils.IntegratedGradients.attribute = observed_attribute
            try:
                _seed(config)
                values, transport = extract_one_qwen(adapter, record, authors)
            finally:
                authors.ig_utils.IntegratedGradients.attribute = original_attribute
            assert observed_nodes and set(observed_nodes) == {20}
            arrays = {k: np.stack([v]) for k, v in values.items() if k.startswith('hidden_')}
            assert len(arrays) == 4 and all(np.isfinite(v).all() for v in arrays.values())
            _seed(config)
            vector = authors.steering.compute_steering_vector(arrays, 27, 'pca')
            assert torch.isfinite(vector).all() and vector.norm() > 0
            batch = adapter.prepare(record['image'], 'Please describe this image in detail.')
            results['base_fp32'] = adapter.generate(batch, max_new_tokens=8)
            diagnostics = {}
            _seed(config)
            with grains_on_qwen_layer_device(adapter, vector, authors, diagnostics):
                results['grains_clone'] = adapter.generate(batch, max_new_tokens=32)
            assert diagnostics['calls'] > 0 and diagnostics['changed_elements'] > 0
            results.update(grains_diagnostics=diagnostics, actual_ig_quadrature_nodes=observed_nodes,
                           attribution_transport=transport, pca_vector_norm=float(vector.norm()),
                           attribution_scope='Released original token-only IG/ablation; not full visual-path IG')
        assert verify_clone_sources() == result['author_sources']
        result.update(status='passed', results=results, device_proof=adapter.qwen_gpu23_device_proof,
                      max_memory_allocated_gb=[torch.cuda.max_memory_allocated(i)/1e9 for i in range(torch.cuda.device_count())],
                      model=config['model'], completed_at=time.time())
    except BaseException:
        result.update(status='failed', error=traceback.format_exc(), completed_at=time.time())
        raise
    finally:
        destination = output / (phase + '-preflight.json')
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(destination)
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('bf16', 'fp32'), required=True)
    args = parser.parse_args()
    main(args.output, args.phase)

