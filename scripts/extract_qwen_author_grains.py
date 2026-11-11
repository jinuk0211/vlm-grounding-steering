"""Prepared fit50 extraction through the original GrAInS code on native Qwen."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from author_clone_bridge import ROOT, load_authors, verify_clone_sources
from qwen_author_compat import extract_one_qwen
from qwen_gpu23_adapter import PAIR, load_qwen, require_gemma_complete, verify_run_protocol
from vlm_grounding.cli import _seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main(output):
    assert output.resolve().parent == ROOT / 'outputs'
    state = {'pid': os.getpid(), 'started_at': time.time(), 'completed': 0, 'total': 50}
    def update(**values):
        state.update(values, updated_at=time.time()); save(output / 'extraction-status.json', state)
    try:
        require_gemma_complete()
        assert os.environ['CUDA_VISIBLE_DEVICES'] == PAIR
        protocol = verify_run_protocol(output)
        assert verify_clone_sources() == protocol['author_sources']
        config = json.loads((output / 'config.json').read_text())
        config['model']['dtype'] = 'float32'
        _seed(config); update(stage='loading_fp32_model')
        adapter = load_qwen(config['model'], output, 'grains-extraction')
        for parameter in adapter.model.parameters():
            parameter.requires_grad_(False)
        authors = load_authors()
        fit = [json.loads(s) for s in (output / 'data/fit50.jsonl').read_text().splitlines() if s.strip()]
        assert len(fit) == len({r['group_id'] for r in fit}) == 50
        extraction = output / 'extraction'; extraction.mkdir()
        hidden = {}; ig_calls = []
        original_ig = authors.grad.apply_integrated_gradients_contrastive
        original_attribute = authors.ig_utils.IntegratedGradients.attribute
        def observed_attribute(instance, *args, **kwargs):
            steps = int(kwargs.get('n_steps', 50)); assert steps == 20
            ig_calls.append({'quadrature_nodes': steps, 'effective_internal_batch_size': 1})
            return original_attribute(instance, *args, **kwargs)
        authors.ig_utils.IntegratedGradients.attribute = observed_attribute
        with (output / 'extraction.jsonl').open('x') as journal:
            for index, record in enumerate(fit):
                assert sha(record['image']) == record['metadata']['image_sha256']
                _seed(config); update(stage='extracting', index=index, record_id=record['id'])
                raw_path = extraction / f'{index:03d}.ig.pt'
                def observed(*args, **kwargs):
                    values = original_ig(*args, **kwargs)
                    assert all(torch.isfinite(v).all() for v in values)
                    torch.save({'positive': values[0].detach().cpu(), 'negative': values[1].detach().cpu()}, raw_path)
                    return values
                authors.grad.apply_integrated_gradients_contrastive = observed
                before = len(ig_calls); start = time.perf_counter()
                try:
                    values, transport = extract_one_qwen(adapter, record, authors)
                finally:
                    authors.grad.apply_integrated_gradients_contrastive = original_ig
                assert len(ig_calls) == before + 1 and raw_path.is_file()
                path = extraction / f'{index:03d}.npz'
                np.savez(path, **values)
                for name, value in values.items():
                    assert np.isfinite(value).all()
                    if name.startswith('hidden_'):
                        hidden.setdefault(name, []).append(value)
                entry = {'index': index, 'record': record, 'artifact': str(path), 'artifact_sha256': sha(path),
                         'raw_ig': str(raw_path), 'raw_ig_sha256': sha(raw_path), 'transport': transport,
                         'ig': ig_calls[-1], 'seconds': time.perf_counter() - start}
                journal.write(json.dumps(entry, allow_nan=False) + '\n'); journal.flush()
                update(completed=index + 1, last_seconds=entry['seconds'],
                       max_memory_gb=[torch.cuda.max_memory_allocated(i)/1e9 for i in range(2)])
                print(json.dumps({'extracted': index + 1, 'of': 50, 'seconds': entry['seconds']}), flush=True)
        authors.ig_utils.IntegratedGradients.attribute = original_attribute
        assert len(ig_calls) == 50 and len(hidden) == 4
        arrays = {k: np.stack(v) for k, v in hidden.items()}
        np.savez(output / 'original-hidden-states.npz', **arrays)
        _seed(config)
        vector = authors.steering.compute_steering_vector(arrays, 27, 'pca')
        assert torch.isfinite(vector).all() and vector.norm() > 0
        artifact = {'vector': vector.cpu(), 'layer': 27, 'alpha': 10.0,
            'fit_records': [r['id'] for r in fit], 'model': config['model'],
            'author_commit': protocol['author_sources']['GrAInS']['commit'],
            'ig_steps': 20, 'top_k': 50, 'mode': 'pos', 'vector_method': 'pca',
            'architecture_bridge': 'Native Qwen; hidden-state output device coalescing only; original token-only IG/ablation retained',
            'gemma_vectors_reused': False}
        torch.save(artifact, output / 'grains-author.pt')
        assert verify_clone_sources() == protocol['author_sources']
        save(output / 'grains-artifact.json', {
            'status': 'verified_complete', 'artifact_sha256': sha(output / 'grains-author.pt'),
            'hidden_states_sha256': sha(output / 'original-hidden-states.npz'),
            'journal_sha256': sha(output / 'extraction.jsonl'), 'fit_manifest_sha256': sha(output / 'data/fit50.jsonl'),
            'fit_records': 50, 'actual_ig_quadrature_nodes': 20, 'actual_ig_calls': len(ig_calls),
            'vector_norm': float(vector.norm()), 'model': config['model'],
            'source_protocol_sha256': protocol['sha256'], 'gpu_visible_devices': PAIR})
        update(stage='complete', completed_at=time.time())
    except BaseException:
        update(stage='failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
