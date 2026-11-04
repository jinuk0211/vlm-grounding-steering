"""Precompute disjoint CHAIR ranges with the pinned author VTI artifact on GPU 0."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

import run_qwen_original_vti as original
from qwen_gpu0_migration_api import GPU0, load_gpu0, save, sha, verify_protocol
from qwen_fp16_attention_guard import stable_fp16_attention


def ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().split()[21])


def worker(out, start, end):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == GPU0 and torch.cuda.device_count() == 1
    assert sha(original.__file__) == '1901e8aaab25b149c102f40d1afb5349e71a4f6afe6de15fdfaf6ed40cd92553'
    manifest = json.loads((out / 'protocol.json').read_text())
    assert manifest['parent_protocol_sha256'] == verify_protocol(original.PARENT)['sha256']
    artifact_path = out / 'vti-author.pt'
    artifact_digest = sha(artifact_path)
    ready = json.loads((out / 'artifact-ready.json').read_text())
    assert ready['sha256'] == artifact_digest and ready['author_commit'] == original.COMMIT
    refs = original.read(original.PARENT / 'data/chair.jsonl')
    assert len(refs) == 500 and 0 <= start < end <= len(refs)
    shard_dir = out / 'chair-parallel'
    shard_dir.mkdir(exist_ok=True)
    label = f'{start:03d}-{end:03d}'
    state_path = shard_dir / f'{label}-status.json'
    row_path = shard_dir / f'{label}.jsonl'
    assert not row_path.exists()
    state = dict(pid=os.getpid(), start_ticks=ticks(os.getpid()), stage='loading',
                 range=[start, end], completed_records=0, gpu_uuid=GPU0,
                 source_sha256=sha(__file__), original_worker_sha256=sha(original.__file__),
                 artifact_sha256=artifact_digest)

    def update(**kw):
        state.update(kw, updated_at=time.time())
        save(state_path, state)

    update()
    try:
        utils, layers = original.imports()
        attention = {}
        with stable_fp16_attention(attention):
            config = json.loads((original.PARENT / 'config.json').read_text())['model']
            config['dtype'] = 'float16'
            adapter = load_gpu0(config, original.PARENT, 'vti-author-chair-' + label)
            proof = original.PARENT / ('vti-author-chair-' + label + '-device-map.json')
            (shard_dir / (label + '-device-map.json')).write_bytes(proof.read_bytes())
            artifact = torch.load(artifact_path, map_location='cpu', weights_only=True)
            assert artifact['author_commit'] == original.COMMIT
            assert artifact['author_source_hashes'] == original.verify_author()
            original.install(adapter.vision_layers, artifact['visual_direction'][1:], .9, layers, vision=True)
            original.install(adapter.layers, artifact['textual_direction'][1:], .9, layers)
            diagnostics = {}
            handles = []
            for name, module in adapter.model.named_modules():
                if isinstance(module, layers.VTILayer):
                    kind = 'vision' if '.visual.' in name else 'text'

                    def observe(_module, args, value, kind=kind):
                        assert torch.isfinite(value).all()
                        d = diagnostics.setdefault(kind, {'calls': 0, 'changed_elements': 0, 'max_abs_delta': 0.0})
                        diff = (value.float() - args[0].float()).abs()
                        d['calls'] += 1
                        d['changed_elements'] += int((diff > 0).sum())
                        d['max_abs_delta'] = max(d['max_abs_delta'], float(diff.max()))

                    handles.append(module.register_forward_hook(observe))
            assert len(handles) == len(adapter.layers) + len(adapter.vision_layers)
            update(stage='generating')
            with row_path.open('x') as sink:
                for i in range(start, end):
                    ref = refs[i]
                    assert sha(ref['image']) == ref['image_sha256']
                    diagnostics.clear()
                    began = time.perf_counter()
                    response = adapter.generate(adapter.prepare(ref['image'], ref['prompt']),
                                                max_new_tokens=ref['max_new_tokens'])
                    assert all(diagnostics.get(k, {}).get('changed_elements', 0) > 0 for k in ('text', 'vision'))
                    row = {**ref, 'method': 'vti_author', 'prediction': response,
                           'generated_tokens': adapter.last_generated_tokens,
                           'elapsed_seconds': time.perf_counter() - began,
                           'gpu_uuid': GPU0, 'dtype': 'float16', 'attention_score_dtype': 'float32',
                           'checkpoint': config['checkpoint'], 'revision': config['revision'],
                           'protocol_sha256': manifest['sha256'],
                           'parent_protocol_sha256': manifest['parent_protocol_sha256'],
                           'author_commit': original.COMMIT, 'artifact_sha256': artifact_digest,
                           'original_hook_file': layers.VTILayer.forward.__code__.co_filename,
                           'original_vti_diagnostics': dict(diagnostics),
                           'attention_amendment_sha256': sha(original.PARENT / 'fp16-attention-amendment.json')}
                    sink.write(json.dumps(row, allow_nan=False) + '\n')
                    sink.flush()
                    update(completed_records=i-start+1, last_index=i)
            for handle in handles:
                handle.remove()
        assert len(original.read(row_path)) == end-start
        update(stage='complete', completed_at=time.time(), output_sha256=sha(row_path),
               device_proof_sha256=sha(shard_dir / (label + '-device-map.json')))
    except BaseException:
        update(stage='failed', error=traceback.format_exc())
        raise


def controller(out):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == GPU0
    shard_dir = out / 'chair-parallel'
    shard_dir.mkdir(exist_ok=True)
    state_path = shard_dir / 'controller-status.json'
    ranges = [(0, 167), (167, 334), (334, 500)]
    state = dict(pid=os.getpid(), start_ticks=ticks(os.getpid()), stage='launching',
                 ranges=ranges, source_sha256=sha(__file__), gpu_uuid=GPU0,
                 original_worker_sha256=sha(original.__file__),
                 artifact_sha256=sha(out / 'vti-author.pt'), jobs=[])

    def update(**kw):
        state.update(kw, updated_at=time.time())
        save(state_path, state)

    update()
    jobs = []
    stopped_original = False
    original_pid = 4107032
    original_ticks = 331187361
    try:
        assert not (out / 'chair.jsonl').exists()
        for start, end in ranges:
            label = f'{start:03d}-{end:03d}'
            log = (shard_dir / (label + '.log')).open('w')
            command = [sys.executable, str(Path(__file__).resolve()), '--output', str(out),
                       '--start', str(start), '--end', str(end)]
            process = subprocess.Popen(command, cwd=str(original.ROOT), env=os.environ.copy(),
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            jobs.append((process, log, start, end))
            state['jobs'].append(dict(pid=process.pid, start_ticks=ticks(process.pid), range=[start,end],
                                      log=str(shard_dir / (label + '.log'))))
            update(stage='generating')
        pending = set(range(len(jobs)))
        while pending:
            for index in list(pending):
                process, log, start, end = jobs[index]
                code = process.poll()
                if code is not None:
                    log.close()
                    state['jobs'][index]['exit_code'] = code
                    pending.remove(index)
                    update()
                    assert code == 0, (start,end,code)
            if pending and not stopped_original:
                progress=json.loads((out/'status.json').read_text())
                if (progress.get('benchmark')=='pope' and
                    progress.get('benchmark_completed',0)>=2900 and
                    not (out/'chair.jsonl').exists()):
                    assert progress['pid']==original_pid and ticks(original_pid)==original_ticks
                    os.kill(original_pid,signal.SIGSTOP)
                    stopped_original=True
                    update(original_paused_at_pope=progress['benchmark_completed'])
            if pending:time.sleep(5)
        refs = original.read(original.PARENT / 'data/chair.jsonl')
        rows = []
        for start,end in ranges:
            label=f'{start:03d}-{end:03d}'
            status=json.loads((shard_dir / (label+'-status.json')).read_text())
            assert status['stage']=='complete' and status['range']==[start,end]
            assert status['output_sha256']==sha(shard_dir/(label+'.jsonl'))
            rows.extend(original.read(shard_dir/(label+'.jsonl')))
        assert len(rows)==500
        for row,ref in zip(rows,refs):
            assert all(row.get(k)==v for k,v in ref.items())
            assert row['method']=='vti_author' and row['artifact_sha256']==state['artifact_sha256']
        assert not (out/'chair.jsonl').exists(), 'original worker already started CHAIR'
        temporary=out/'chair-parallel-merge.tmp'
        with temporary.open('x') as sink:
            for row in rows:sink.write(json.dumps(row,allow_nan=False)+'\n')
        temporary.replace(out/'chair.jsonl')
        if stopped_original:
            assert ticks(original_pid)==original_ticks
            os.kill(original_pid,signal.SIGCONT)
            stopped_original=False
        update(stage='complete', chair_sha256=sha(out/'chair.jsonl'), completed_records=500,
               completed_at=time.time())
    except BaseException:
        update(stage='failed',error=traceback.format_exc())
        raise
    finally:
        if stopped_original and ticks(original_pid)==original_ticks:
            os.kill(original_pid,signal.SIGCONT)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--controller',action='store_true')
    parser.add_argument('--start',type=int)
    parser.add_argument('--end',type=int)
    args=parser.parse_args()
    controller(args.output) if args.controller else worker(args.output,args.start,args.end)
