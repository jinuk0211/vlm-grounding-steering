"""Fetch pinned official evaluation inputs; never download COCO image archives."""
import concurrent.futures as cf
import hashlib
import json
import random
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')
DATA = ROOT / 'data/hallucination-suite'
DATA.mkdir(parents=True, exist_ok=True)
STATUS = DATA / 'preparation-status.json'


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'LJW-evaluation'}), timeout=45) as h:
        return json.load(h)


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size:
        return path
    temp = path.with_suffix(path.suffix + '.part')
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=45) as response, temp.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temp.replace(path)
            return path
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def main():
    save(STATUS, {'stage': 'sources', 'started_at': time.time()})
    sources = {}
    pope_commit = get('https://api.github.com/repos/RUCAIBox/POPE/commits/main')['sha']
    mmhal_commit = get('https://huggingface.co/api/datasets/Shengcao1006/MMHal-Bench')['sha']
    sources['pope_commit'] = pope_commit
    sources['mmhal_commit'] = mmhal_commit
    pope = {}
    for subset in ('random', 'popular', 'adversarial'):
        name = f'coco_pope_{subset}.json'
        path = download(f'https://raw.githubusercontent.com/RUCAIBox/POPE/{pope_commit}/output/coco/{name}', DATA / 'sources' / name)
        pope[subset] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(pope[subset]) == 3000
        assert len({r['question_id'] for r in pope[subset]}) == 3000
        assert sum(r['label'] == 'yes' for r in pope[subset]) == 1500
        sources[name] = {'sha256': sha(path), 'url': f'https://raw.githubusercontent.com/RUCAIBox/POPE/{pope_commit}/output/coco/{name}'}
    download(f'https://raw.githubusercontent.com/RUCAIBox/POPE/{pope_commit}/evaluate.py', DATA / 'sources/pope_evaluate.py')
    template_path = download(f'https://huggingface.co/datasets/Shengcao1006/MMHal-Bench/resolve/{mmhal_commit}/response_template.json', DATA / 'sources/response_template.json')
    download(f'https://huggingface.co/datasets/Shengcao1006/MMHal-Bench/resolve/{mmhal_commit}/eval_gpt4.py', DATA / 'sources/mmhal_eval_gpt4.py')
    template = json.loads(template_path.read_text())
    assert len(template) == 96
    sources['mmhal_template_sha256'] = sha(template_path)
    save(STATUS, {'stage': 'coco_annotations', 'updated_at': time.time()})
    annotation_url = 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip'
    archive = download(annotation_url, DATA / 'sources/annotations_trainval2014.zip')
    sources['coco_annotations'] = {'url': annotation_url, 'sha256': sha(archive)}
    annotations = DATA / 'annotations'
    annotations.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for kind in ('captions', 'instances'):
            for split in ('train', 'val'):
                filename = f'{kind}_{split}2014.json'
                destination = annotations / filename
                if not destination.exists():
                    with z.open('annotations/' + filename) as source, destination.open('wb') as output:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk)
    val = json.loads((annotations / 'instances_val2014.json').read_text())
    image_by_name = {row['file_name']: row for row in val['images']}
    fit = [json.loads(line) for filename in ['fit.jsonl', 'calibration.jsonl']
           for line in (ROOT / 'data/mmvp-full' / filename).read_text().splitlines()]
    dev_hashes = {r['metadata']['image_sha256'] for r in fit}
    dev_ids = {int(Path(r['image']).stem.rsplit('_', 1)[-1]) for r in fit}
    candidates = sorted([row for row in val['images'] if row['id'] not in dev_ids], key=lambda r: r['id'])
    chair_images = sorted(random.Random(42).sample(candidates, 500), key=lambda r: r['id'])
    pope_names = {r['image'] for rows in pope.values() for r in rows}
    assert len(pope_names) == 500
    assert not {image_by_name[n]['id'] for n in pope_names} & dev_ids, 'POPE/dev overlap: must regenerate artifact on disjoint dev'
    coco_names = pope_names | {r['file_name'] for r in chair_images}
    todo = [(f'https://s3.amazonaws.com/images.cocodataset.org/val2014/{name}', DATA / 'images/coco' / name) for name in sorted(coco_names)]
    mmhal_files = get(f'https://huggingface.co/api/datasets/Shengcao1006/MMHal-Bench/tree/{mmhal_commit}/images?limit=1000')
    mmhal_names = {row['path'].rsplit('/', 1)[-1]: row['path'] for row in mmhal_files if row['type'] == 'file'}
    for row in template:
        name = row['image_src'].split('/')[-1]
        todo.append((f'https://huggingface.co/datasets/Shengcao1006/MMHal-Bench/resolve/{mmhal_commit}/{mmhal_names[name]}', DATA / 'images/mmhal' / name))
    todo = list(dict.fromkeys(todo))
    save(STATUS, {'stage': 'images', 'completed': 0, 'total': len(todo), 'updated_at': time.time()})
    with cf.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(download, url, path) for url, path in todo]
        for i, future in enumerate(cf.as_completed(futures), 1):
            future.result()
            if i % 25 == 0 or i == len(todo):
                save(STATUS, {'stage': 'images', 'completed': i, 'total': len(todo), 'updated_at': time.time()})
                print(json.dumps({'downloaded': i, 'total': len(todo)}), flush=True)
    from PIL import Image
    hashes = {}
    for _, path in todo:
        with Image.open(path) as image:
            image.verify()
        hashes[str(path)] = sha(path)
    assert not set(hashes.values()) & dev_hashes
    evaluation = []
    unique = {}
    for subset, rows in pope.items():
        for source in rows:
            image = str(DATA / 'images/coco' / source['image'])
            prompt = source['text'] + '\nPlease answer yes or no.'
            key = hashlib.sha256((image + '\n' + prompt).encode()).hexdigest()
            row = {'id': f"pope:{subset}:{source['question_id']}", 'group_id': f"coco:{image_by_name[source['image']]['id']}",
                   'task': 'pope', 'benchmark': 'pope', 'subset': subset, 'split': 'test',
                   'image': image, 'prompt': prompt, 'label': source['label'], 'image_sha256': hashes[image],
                   'inference_key': key, 'max_new_tokens': 128,
                   'metadata': {'question_id': source['question_id'], 'original_question': source['text']}}
            evaluation.append(row)
            if key in unique:
                assert unique[key]['label'] == row['label']
            else:
                unique[key] = row
    pope_unique = list(unique.values())
    captions = []
    for source in chair_images:
        image = str(DATA / 'images/coco' / source['file_name'])
        row = {'id': f"chair:{source['id']}", 'group_id': f"coco:{source['id']}",
               'task': 'caption', 'benchmark': 'chair', 'split': 'test', 'image': image,
               'prompt': 'Please describe this image in detail.', 'max_new_tokens': 512,
               'image_sha256': hashes[image], 'metadata': {'image_id': source['id']}}
        row['inference_key'] = row['id']
        captions.append(row)
    from vlm_grounding.data import prepare_mmhal
    mmhal = prepare_mmhal(template_path, DATA / 'images/mmhal')
    for row in mmhal:
        row.update(benchmark='mmhal', max_new_tokens=512, inference_key=row['id'], image_sha256=hashes[row['image']])
    jsonl(DATA / 'pope.jsonl', evaluation)
    jsonl(DATA / 'pope-unique.jsonl', pope_unique)
    jsonl(DATA / 'chair.jsonl', captions)
    jsonl(DATA / 'mmhal.jsonl', mmhal)
    save(DATA / 'image-hashes.json', hashes)
    save(DATA / 'provenance.json', {'sources': sources, 'records': {'pope': 9000, 'pope_unique': len(pope_unique), 'chair': 500, 'mmhal': 96},
         'chair_selection': {'population': 'COCO val2014 excluding fixed development image IDs', 'count': 500, 'seed': 42, 'algorithm': 'random.Random(42).sample(sorted_by_image_id, 500)', 'fixed_metric_cohort': True, 'entire_COCO_validation': False},
         'development': {'source': str(ROOT / 'data/mmvp-full'), 'fit_images': 32, 'calibration_images': 32, 'image_id_overlap': 0, 'exact_image_hash_overlap': 0},
         'prompt_and_decoding': {'caption': 'Please describe this image in detail.', 'pope_suffix': 'Please answer yes or no.', 'do_sample': False, 'num_beams': 1, 'pope_max_new_tokens': 128, 'caption_mmhal_max_new_tokens': 512},
         'mmhal_scoring': 'official GPT-4 scoring deferred at explicit user request; generation only', 'completed_at': time.time()})
    save(STATUS, {'stage': 'complete', 'records': {'pope': 9000, 'pope_unique': len(pope_unique), 'chair': 500, 'mmhal': 96}, 'updated_at': time.time()})


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        save(STATUS, {'stage': 'failed', 'error': traceback.format_exc(), 'updated_at': time.time()})
        raise
