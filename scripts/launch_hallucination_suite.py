"""Freeze provenance and launch isolated batch-1 workers on an idle GPU."""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')
DATA = ROOT / 'data/hallucination-suite'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    prep = json.loads((DATA / 'preparation-status.json').read_text())
    assert prep['stage'] == 'complete', prep
    names = {'0': 'GPU-4a6e1799-a39c-86ed-abdf-afad03791c12'}
    workers_per_gpu = 4
    gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    gpu_memory = {}
    for line in gpu_lines.splitlines():
        index, uuid, used = [x.strip() for x in line.split(',')]
        gpu_memory[index] = (uuid, int(used))
    for index, uuid in names.items():
        assert gpu_memory[index][0] == uuid and gpu_memory[index][1] < 1000, f'GPU {index} no longer idle'
    output = ROOT / 'outputs' / ('hallucination-suite-' + datetime.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d-%H%M%S'))
    output.mkdir()
    for name in ('claims', 'predictions', 'artifacts', 'source-snapshot'):
        (output / name).mkdir()
    previous = json.loads((ROOT / 'outputs/mmvp-full-current.json').read_text())
    previous_output = Path(previous['output'])
    old_report = json.loads((previous_output / 'report.json').read_text())
    assert old_report['status'] == 'verified_complete'
    config = json.loads((ROOT / 'configs/gemma4_e4b.local.json').read_text())
    assert config['model'] == json.loads((previous_output / 'config.json').read_text())['model']
    artifact_hashes = {}
    for method in ('revis', 'grains', 'vti'):
        source = previous_output / 'artifacts' / f'{method}.pt'
        target = output / 'artifacts' / source.name
        shutil.copy2(source, target)
        artifact_hashes[method] = sha(target)
    source_files = ['src/vlm_grounding/backend.py', 'src/vlm_grounding/baselines.py', 'src/vlm_grounding/evaluation.py', 'src/vlm_grounding/data.py',
                    'scripts/prepare_hallucination_suite.py', 'scripts/run_hallucination_worker.py', 'scripts/launch_hallucination_suite.py']
    for name in source_files:
        path = output / 'source-snapshot' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    manifest_files = ['pope.jsonl', 'pope-unique.jsonl', 'chair.jsonl', 'mmhal.jsonl', 'provenance.json', 'image-hashes.json']
    for name in manifest_files:
        shutil.copy2(DATA / name, output / 'source-snapshot' / name)
    protocol = {'model': config['model'], 'methods': ['base', 'revis', 'rudder', 'grains', 'vti'],
                'method_parameters': config['methods'], 'generation': {'do_sample': False, 'num_beams': 1, 'use_cache': True, 'seed_per_example': 42,
                'batch_size_per_worker': 1, 'pope_max_new_tokens': 128, 'chair_mmhal_max_new_tokens': 512},
                'evaluation_records_per_method': {'pope': 9000, 'chair': 500, 'mmhal': 96},
                'dataset_provenance': json.loads((DATA / 'provenance.json').read_text()),
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'source_hashes': {name: sha(ROOT / name) for name in source_files},
                'manifest_hashes': {name: sha(DATA / name) for name in manifest_files},
                'artifact_hashes': artifact_hashes, 'artifact_source': str(previous_output),
                'adaptation': 'repository paper-derived Gemma local adaptations; not author reproductions',
                'development_budget': {'fit_images': 32, 'revis_calibration_images': 32, 'source': 'COCO128 train2017', 'artifact_reuse_across_benchmarks': True},
                'pope_scoring': 'official first-sentence No/not/no parser, plus separately reported strict leading yes/no metrics',
                'chair_scoring': {'port_commit': '4087a26211aa2339b9a76307cb8f0321ef691d0a', 'cohort_images': 500, 'seed': 42},
                'mmhal_scoring': 'deferred by explicit user request; all96 responses per method exported in official schema',
                'gpu_uuids': names, 'workers_per_gpu': workers_per_gpu}
    protocol['sha256'] = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    save(output / 'protocol.json', protocol)
    save(output / 'config.json', config)
    lengths = {'pope': prep['records']['pope_unique'], 'chair': 500, 'mmhal': 96}
    block_sizes = {'pope': 60, 'chair': 8, 'mmhal': 8}
    task_lists = {}
    for benchmark in ('pope', 'mmhal', 'chair'):
        tasks = []
        for start in range(0, lengths[benchmark], block_sizes[benchmark]):
            for method in protocol['methods']:
                tasks.append({'id': f'{benchmark}-{method}-{start:05d}', 'benchmark': benchmark, 'method': method,
                              'start': start, 'stop': min(lengths[benchmark], start + block_sizes[benchmark])})
        task_lists[benchmark] = tasks
    # Interleave short QA, captions and POPE so a failure in any task becomes visible early.
    tasks = []
    for i in range(max(map(len, task_lists.values()))):
        for benchmark in ('mmhal', 'chair', 'pope'):
            if i < len(task_lists[benchmark]):
                tasks.append(task_lists[benchmark][i])
    save(output / 'tasks.json', tasks)
    jobs = []
    for gpu, uuid in names.items():
        for slot in range(workers_per_gpu):
            worker = f'gpu{gpu}-{slot}'
            log = output / f'{worker}.log'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
            command = f'cd {ROOT}; source scripts/ljw_env.sh; exec python scripts/run_hallucination_worker.py --output {output} --worker {worker}'
            with log.open('ab') as sink:
                process = subprocess.Popen(['bash', '-lc', command], env=env, stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
            jobs.append({'gpu': int(gpu), 'uuid': uuid, 'worker': worker, 'pid': process.pid, 'log': str(log)})
            save(output / 'jobs.json', jobs)
            save(ROOT / 'outputs/hallucination-suite-current.json', {'output': str(output), 'jobs': jobs, 'protocol_sha256': protocol['sha256']})
    print(json.dumps({'output': str(output), 'jobs': jobs, 'tasks': len(tasks), 'unique_pope_questions': lengths['pope'], 'protocol_sha256': protocol['sha256']}, indent=2))


if __name__ == '__main__':
    main()
