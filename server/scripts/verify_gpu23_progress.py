"""CPU-only independent checks of preserved bytes and newly generated rows."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(output):
    assert output.resolve().parent == ROOT / 'outputs'
    hardware = json.loads((output / 'hardware-amendment.json').read_text())
    checkpoint = json.loads((output / 'PAUSED_CHECKPOINT.json').read_text())
    continuation = json.loads((output / 'continuation-amendment.json').read_text())
    protocol = json.loads((output / 'protocol.json').read_text())
    assert sha(output / 'PAUSED_CHECKPOINT.json') == continuation['checkpoint_sha256']
    assert sha(output / 'hardware-amendment.json') == continuation['hardware_amendment_sha256']
    assert sha(output / 'protocol.json') == checkpoint['protocol_sha256']
    assert hardware['current_gpu_indices'] == [2, 3] and not hardware['automatic_gpu_fallback']
    for file, digest in {**protocol['source_hashes'], **hardware['source_hashes']}.items():
        assert sha(ROOT / file) == digest, file
    for file, digest in protocol['manifest_hashes'].items():
        assert sha(output / 'data' / file) == digest
    for method, digest in protocol['retained_artifact_hashes'].items():
        assert sha(output / 'artifacts' / (method + '.pt')) == digest
    grains = json.loads((output / 'grains-artifact.json').read_text())
    assert sha(output / 'grains-author.pt') == grains['artifact_sha256']
    original = {r['path']: r for r in checkpoint['prediction_files']}
    for file, seal in original.items():
        prefix = (output / file).read_bytes()[:seal['bytes']]
        assert hashlib.sha256(prefix).hexdigest() == seal['sha256'], file
    datasets = {b: [json.loads(s) for s in (output / 'data' / (b + '.jsonl')).read_text().splitlines() if s.strip()]
                for b in ('pope', 'chair')}
    tasks = json.loads((output / 'tasks.json').read_text())
    jobs = {j['worker']: j for j in json.loads((output / 'jobs.json').read_text()) if not j['extract']}
    device_maps = {}
    live = []
    for name, job in jobs.items():
        process = Path('/proc') / str(job['pid'])
        if process.exists():
            state = (process / 'stat').read_text().split(') ', 1)[1][0]
            if state not in ('Z', 'X'):
                visible = next(v.split(b'=', 1)[1].decode() for v in (process / 'environ').read_bytes().split(b'\0') if v.startswith(b'CUDA_VISIBLE_DEVICES='))
                assert visible == job['visible_devices']
                assert set(visible.split(',')) <= set(hardware['bf16_visible_devices'])
                assert process.stat().st_uid == os.getuid() and os.readlink(str(process / 'cwd')) == str(ROOT)
                live.append({'worker': name, 'pid': job['pid'], 'state': state, 'visible_devices': visible})
        proof = output / (name + '-device-map.json')
        if proof.exists():
            value = json.loads(proof.read_text())
            assert value['visible_devices'] == job['visible_devices'] and value['dtype'] == job['dtype']
            assert not value['quantization'] and not value['cpu_or_disk_offload']
            if job['dtype'] == 'float32':
                assert value['parameter_dtypes'] == ['torch.float32'] and value['parameter_devices'] == ['cuda:0', 'cuda:1']
            else:
                assert 'torch.bfloat16' in value['parameter_dtypes'] and value['parameter_devices'] == ['cuda:0']
            device_maps[name] = value
    counts = collections.Counter(); new_counts = collections.Counter(); checked_images = set(); incomplete_tail_bytes = 0
    for task in tasks:
        path = output / 'predictions' / (task['id'] + '.jsonl')
        if not path.exists():
            continue
        data = path.read_bytes()
        if data and not data.endswith(b'\n'):
            boundary = data.rfind(b'\n') + 1
            incomplete_tail_bytes += len(data) - boundary
            data = data[:boundary]
        rows = [json.loads(s) for s in data.splitlines() if s.strip()]
        refs = datasets[task['benchmark']][task['start']:task['stop']]
        assert len(rows) <= len(refs)
        old_count = original.get(str(path.relative_to(output)), {}).get('records', 0)
        for index, (row, ref) in enumerate(zip(rows, refs)):
            assert all(row.get(k) == v for k, v in ref.items())
            assert row['method'] == task['method'] and row['dtype'] == task['dtype']
            assert row['protocol_sha256'] == protocol['sha256']
            counts[task['benchmark'] + ':' + task['method']] += 1
            if index < old_count:
                continue
            assert row['hardware_execution'] == 'gpu23-resume-20261028'
            assert row['worker'] in jobs and row['worker'] in device_maps
            assert row['gpu_uuid'] == jobs[row['worker']]['visible_devices']
            expected = hardware['bf16_visible_devices'] if row['dtype'] == 'bfloat16' else [hardware['fp32_visible_devices']]
            assert row['gpu_uuid'] in expected
            assert row['generated_tokens'] > 0 and isinstance(row['prediction'], str)
            if row['image'] not in checked_images:
                assert sha(Path(row['image'])) == row['image_sha256']
                checked_images.add(row['image'])
            if row['method'] in ('rudder_clone', 'grains_clone'):
                diagnostics = row['author_diagnostics']
                assert diagnostics['calls'] > 0
                expected_hook = 'RUDDER/Qwen/qwen_methods.py' if row['method'] == 'rudder_clone' else 'GrAInS/src/utils/steering.py'
                assert diagnostics['hook_source'].endswith(expected_hook)
                if row['method'] == 'rudder_clone':
                    assert diagnostics['card_norm'] > 0 and diagnostics['active_positions'] > 0
            new_counts[task['benchmark'] + ':' + task['method']] += 1
    result = {'status': 'verified_partial_snapshot', 'checked_at': time.time(), 'counts': dict(counts),
              'new_counts_since_pause': dict(new_counts), 'preserved_records': sum(r['records'] for r in original.values()),
              'live_processes_verified': live, 'all_historical_bytes_unchanged': True,
              'new_source_input_artifact_author_call_and_gpu_records_verified': True,
              'inflight_incomplete_tail_bytes_excluded': incomplete_tail_bytes,
              'final_evaluation_complete': False, 'analysis_sha256': sha(Path(__file__))}
    (output / 'gpu23-partial-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    verify(parser.parse_args().output)
