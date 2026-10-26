"""Connect the cloned public methods to the existing sealed paired benchmark.

All CARD/gating and GrAInS attribution/PCA/intervention operations are imported
from the cloned repositories. This file handles inputs, generation and scoring.
"""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from grains_memory import checkpoint_ig

ROOT = Path('/workspace/original-baselines-20261025/grains_gpu_retry_20261026')
OLD = Path('/workspace/qwen-author-check-20261024')
DATA = Path('/workspace/vlm-baselines-20261024/experiment_data')
MODEL = '/workspace/vlm-validation-1683e86/hf-cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5'
SOURCE_SEAL = hashlib.sha256((ROOT / 'execution_manifest_compat_20261026.json').read_bytes()).hexdigest()
EXTERNAL_SEAL = hashlib.sha256((ROOT / 'external_manifest_compat_20261026.json').read_bytes()).hexdigest()
sys.path.insert(0, str(OLD / 'scripts'))
import run_qwen_revis_author as common  # noqa: E402
from vlm_grounding.data import read_manifest, validate_disjoint  # noqa: E402
from vlm_grounding.generation_health import failure_reasons  # noqa: E402


def verify():
    if common.digest(ROOT / 'execution_manifest_compat_20261026.json') != SOURCE_SEAL:
        raise RuntimeError('Execution manifest changed after startup')
    if common.digest(ROOT / 'external_manifest_compat_20261026.json') != EXTERNAL_SEAL:
        raise RuntimeError('Shared manifest changed after startup')
    manifest = json.loads((ROOT / 'execution_manifest_compat_20261026.json').read_text())
    for relative, expected in manifest.items():
        if common.digest(ROOT / relative) != expected:
            raise RuntimeError('Sealed source changed: ' + relative)
    external = json.loads((ROOT / 'external_manifest_compat_20261026.json').read_text())
    for path, expected in external.items():
        if common.digest(path) != expected:
            raise RuntimeError('Shared input/scorer/runtime changed: ' + path)
    return manifest


def tensor_digest(tensor):
    values = tensor.detach().contiguous().cpu()
    return hashlib.sha256(values.view(torch.uint8).numpy().tobytes()).hexdigest()


@contextmanager
def capture_generation(model, processor, destination):
    original = model.generate

    def observed(*args, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        tail = result[0, kwargs['input_ids'].shape[1]:].detach().cpu()
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, (tuple, list)) else [eos]
        raw = processor.decode(tail, skip_special_tokens=True)
        destination.update(
            prediction=raw.strip(), raw_decoded=raw,
            generated_token_ids=tail.tolist(), generated_tokens=tail.numel(),
            ended_with_eos=bool(tail.numel() and int(tail[-1]) in eos),
            elapsed_seconds=time.perf_counter() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            prepared_tensor_sha256={k: tensor_digest(v) for k, v in kwargs.items() if torch.is_tensor(v)},
            input_token_ids=kwargs['input_ids'][0].detach().cpu().tolist(),
        )
        return result

    model.generate = observed
    try:
        yield
    finally:
        model.generate = original


def rudder_setup():
    sys.path.insert(0, str(ROOT / 'RUDDER'))
    from Qwen import qwen_chair, qwen_pope
    arguments = SimpleNamespace(
        model_id=MODEL, cache_dir=None, dtype='bf16', device_map='auto',
        attn_implementation='sdpa', decoding='greedy', max_new_tokens=8,
        repetition_penalty=1.0, force_yesno_first_token=False,
        pooling='attn', local=False, gate_min=0.05, gate_max=1.0, rms_match=False,
    )
    model, processor = qwen_pope.load_model_and_processor(arguments)
    return model, processor, (qwen_pope, qwen_chair, arguments)


def rudder_generate(model, processor, modules, row, condition):
    pope, chair, arguments = modules
    with Image.open(row['image']) as handle:
        image = handle.convert('RGB')
    result = {}
    with capture_generation(model, processor, result):
        if row['task'] == 'pope':
            loader = [([image], [row['prompt']], [row['label']], [row['id']])]
            kwargs = dict(do_sample=False, num_beams=1, use_cache=True)
            if condition == 'base':
                pope.run_once_baseline(model, processor, loader, arguments, kwargs)
            else:
                pope.run_once_bayes(model, processor, loader, 1, 8.0, 5.0, 1.0, arguments, kwargs)
        else:
            loader = [([image], [Path(row['image']).name])]
            kwargs = dict(do_sample=False, num_beams=1, use_cache=True, max_new_tokens=512,
                          repetition_penalty=1.0)
            if condition == 'base':
                chair.run_once_caption(model, processor, loader, kwargs)
            else:
                chair.run_once_beta(model, processor, loader, 25, 5.0, 5.0, 1.0, 'attn', kwargs)
    if not result:
        raise RuntimeError('Original RUDDER function produced no observed generation')
    return result


def grains_setup(args, fit):
    sys.path.insert(0, str(ROOT / 'GrAInS/src'))
    from utils.model import load_vlm_model_and_processor
    from utils.steering import compute_steering_vector, add_steering_hook
    from token_ablation_vlm import run_token_ablation_contrastive
    from attribution.gradient import vlm_grad

    model, processor = load_vlm_model_and_processor(MODEL)
    common.write_json(args.output / 'loaded_model.json', {
        'device': str(model.device), 'dtype': str(next(model.parameters()).dtype),
        'hf_device_map': getattr(model, 'hf_device_map', None),
        'cuda_available': torch.cuda.is_available(),
        'trainable_parameter_tensors': sum(p.requires_grad for p in model.parameters()),
    })
    if model.device.type != 'cuda':
        raise RuntimeError('GPU execution requested, but input embeddings are not on CUDA')
    cpu_layers = [name for name, device in model.hf_device_map.items()
                  if name.startswith('model.language_model.layers.') and device == 'cpu']
    target_name = cpu_layers[0] + '.mlp.up_proj'
    target = model.get_submodule(target_name)
    original_forward = target._old_forward
    witness_recorded = False

    def witnessed_forward(*positional, **keywords):
        nonlocal witness_recorded
        parameters = list(target.parameters(recurse=False))
        if any(parameter.requires_grad for parameter in parameters):
            raise RuntimeError('Offload recreated a trainable parameter during input-only IG')
        if not witness_recorded:
            common.write_json(args.output / 'gpu_weight_gradient_witness.json', {
                'module': target_name, 'grad_enabled': torch.is_grad_enabled(),
                'input_requires_grad': bool(positional and positional[0].requires_grad),
                'parameters': [{'device': str(parameter.device), 'dtype': str(parameter.dtype),
                                'requires_grad': parameter.requires_grad} for parameter in parameters],
            })
            witness_recorded = True
        return original_forward(*positional, **keywords)

    target._old_forward = witnessed_forward
    hidden = {}
    extraction = args.output / 'extraction'
    extraction.mkdir(exist_ok=True)
    sealed_images = json.loads((args.output / 'header.json').read_text())['image_sha256']
    for index, row in enumerate(fit):
        if common.digest(row['image']) != sealed_images[row['id']]:
            raise RuntimeError('Fit image changed: ' + row['id'])
        with Image.open(row['image']) as handle:
            image = handle.convert('RGB')
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': row['prompt'].strip()}, {'type': 'image'},
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        started = time.perf_counter()
        original_ig = vlm_grad.apply_integrated_gradients_contrastive
        raw_ig = extraction / f'example_{index:03}_ig.pt'

        def observed_ig(*positional, **keywords):
            with checkpoint_ig(model) as replay_calls:
                attributed = original_ig(*positional, **keywords)
            torch.save({'positive_ig': attributed[0].detach().cpu(),
                        'negative_ig': attributed[1].detach().cpu(),
                        'pos_embeddings_sha256': tensor_digest(positional[1]),
                        'neg_embeddings_sha256': tensor_digest(positional[2]),
                        'steps': keywords.get('steps', 20)}, raw_ig)
            common.write_json(extraction / f'example_{index:03}_checkpoint.json', {
                'decoder_forward_and_replay_calls': replay_calls,
                'eval_preserved': not model.training, 'ig_cache': False,
                'generation_cache_restored': model.config.text_config.use_cache,
            })
            return attributed

        vlm_grad.apply_integrated_gradients_contrastive = observed_ig
        try:
            values = run_token_ablation_contrastive(
                model, processor, image, prompt, row['chosen'].strip(), row['rejected'].strip(),
                top_k=50, top='pos', attribution_method='integrated_gradients',
            )
        finally:
            vlm_grad.apply_integrated_gradients_contrastive = original_ig
        artifact = extraction / f'example_{index:03}.pt'
        torch.save(values, artifact)
        for key, value in values.items():
            if key.startswith('hidden_'):
                hidden.setdefault(key, []).append(value)
        common.append(args.output / 'extraction.jsonl', {
            **row, 'artifact': str(artifact), 'sha256': common.digest(artifact),
            'raw_ig': str(raw_ig), 'raw_ig_sha256': common.digest(raw_ig),
            'elapsed_seconds': time.perf_counter() - started,
        })
        common.write_json(args.output / 'extraction_progress.json', {
            'completed': index + 1, 'total': len(fit), 'scope': 'fit extraction, not test scores',
        })
        print(f'ORIGINAL_GRAINS_EXTRACTION {index + 1}/{len(fit)}', flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    hidden = {key: np.stack(values) for key, values in hidden.items()}
    np.savez(args.output / 'original_hidden_states.npz', **hidden)
    vector = compute_steering_vector(hidden, 27, 'pca')
    torch.save(vector, args.output / 'original_pca_vector.pt')
    return model, processor, (vector, add_steering_hook)


def grains_generate(model, processor, method, row, condition):
    vector, add_steering_hook = method
    text = row['prompt'] if row['task'] == 'caption' else 'Answer yes or no based on the image.\nQuestion: ' + row['prompt']
    prompt = processor.apply_chat_template([{'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': text},
    ]}], tokenize=False, add_generation_prompt=True)
    with Image.open(row['image']) as handle:
        batch = processor(images=handle.convert('RGB'), text=prompt, return_tensors='pt').to(model.device)
    handles = add_steering_hook(model, 27, vector, alpha=10.0) if condition != 'base' else []
    if condition != 'base' and len(handles) != 1:
        raise RuntimeError('Original GrAInS hook did not attach to exactly one layer')
    result = {}
    try:
        with torch.no_grad(), capture_generation(model, processor, result):
            model.generate(**batch, do_sample=False, num_beams=1, use_cache=True,
                           max_new_tokens=256, repetition_penalty=1.0)
    finally:
        for handle in handles:
            handle.remove()
    return result


def score(rows, args, count):
    verify()
    report = common.score_progress(rows, args, count)
    for condition, values in report['conditions'].items():
        values['output_flags']['repeated'] = sum(
            row['output_checks']['repeated'] for row in rows if row['method'] == condition)
    common.write_json(args.output / f'progress_{count:03}.json', report)
    common.write_json(args.output / 'latest_progress.json', report)
    verify()


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    sources = verify()
    records = common.interleaved_pilot(read_manifest(DATA / 'test.jsonl'))
    fit = read_manifest(DATA / 'grains_fit.jsonl')[:50] if args.method == 'grains' else []
    validate_disjoint(records + fit)
    if len(records) != 100 or sum(row['task'] == 'caption' for row in records) != 50:
        raise RuntimeError('Expected the existing 50 caption + 50 POPE cohort')
    if args.method == 'grains' and len(fit) != 50:
        raise RuntimeError('Expected the existing 50 fit inputs')
    image_hashes = {row['id']: common.digest(row['image']) for row in records + fit}
    if {image_hashes[row['id']] for row in fit} & {image_hashes[row['id']] for row in records}:
        raise RuntimeError('Fit/test image bytes overlap')
    common.write_json(args.output / 'header.json', {
        'method': args.method, 'source_sha256': sources, 'test_records': records, 'fit_records': fit,
        'test_manifest_sha256': common.digest(DATA / 'test.jsonl'),
        'fit_manifest_sha256': common.digest(DATA / 'grains_fit.jsonl') if fit else None,
        'image_sha256': image_hashes,
        'model': MODEL, 'seed': 42,
        'rudder': {'caption': {'layer': 25, 'alpha': 5, 'k': 5, 'c': 1, 'cap': 512},
                   'pope': {'layer': 1, 'alpha': 8, 'k': 5, 'c': 1, 'cap': 8}},
        'grains': {'dtype': 'float32', 'device_map': 'auto', 'gpu_weight_budget': '14GiB',
                   'weights_require_grad': False,
                   'ig_steps': 20, 'ig_internal_batch_size': 1, 'saved_backward_tensors': 'cpu_nonpinned',
                   'ig_decoder_checkpoint': 'nonreentrant_full_recompute', 'ig_cache': False,
                   'top_k': 50, 'mode': 'pos', 'pca_layer': 27, 'alpha': 10, 'public_scale': 10, 'cap': 256},
        'scope': 'Direct cloned method functions; existing benchmark/scorer and explicit native generation adapter',
        'generation_adapter_limit': 'Public GrAInS VLM generation helpers are missing; native model.generate is connected to its unchanged original steering hook.',
        'grains_public_input_limit': 'Original attribution/ablation helpers forward token embeddings or token IDs without pixel_values; this execution preserves that released behavior and does not certify native visual-path equivalence.',
    })
    rows = []
    try:
        model, processor, method = rudder_setup() if args.method == 'rudder' else grains_setup(args, fit)
        common.write_json(args.output / 'loaded_model.json', {
            'device': str(model.device), 'dtype': str(next(model.parameters()).dtype),
            'hf_device_map': getattr(model, 'hf_device_map', None), 'cuda_available': torch.cuda.is_available(),
            'trainable_parameter_tensors': sum(p.requires_grad for p in model.parameters()),
        })
        generate = rudder_generate if args.method == 'rudder' else grains_generate
        for index, record in enumerate(records):
            if common.digest(record['image']) != image_hashes[record['id']]:
                raise RuntimeError('Test image changed: ' + record['id'])
            for condition in ('base', args.method):
                result = generate(model, processor, method, record, condition)
                row = {**record, **result, 'method': condition}
                cap = 256 if args.method == 'grains' else (512 if record['task'] == 'caption' else 8)
                reasons = failure_reasons(row, cap)
                row['output_checks'] = {**common.output_checks(row, cap),
                                        'repeated': any('repetition' in reason for reason in reasons)}
                row['health_failures'] = reasons
                common.append(args.output / 'samples.jsonl', row)
                rows.append(row)
            common.write_json(args.output / 'generation_progress.json', {'completed_pairs': index + 1, 'of': 100})
            if (index + 1) % 10 == 0:
                score(rows, args, index + 1)
        verify()
        common.write_json(args.output / 'complete.json', {'completed_pairs': 100, 'raw_rows': len(rows),
                                                         'full_reproduction_complete': False})
    except Exception as error:
        common.write_json(args.output / 'failure.json', {'error': repr(error), 'raw_rows': len(rows),
                                                        'completed_pairs': len(rows) // 2})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['rudder', 'grains'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    arguments.chair = Path('/workspace/vlm-baselines-20261024/upstream/chair/chair.py')
    arguments.annotations = OLD / 'annotations'
    main(arguments)
