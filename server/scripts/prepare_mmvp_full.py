"""Fetch the complete author MMVP release and disjoint COCO development data."""

import csv
import hashlib
import json
from pathlib import Path
import random
import time
from urllib.request import urlopen
import zipfile

from huggingface_hub import snapshot_download
from PIL import Image
import yaml

REPO = Path('/home/project-b/LJW/vlm-grounding-steering')
DATA = REPO / 'data/mmvp-full'
MMVP_REVISION = '37eafecab8a3940c50c2ade5b36de69dbc99a8cf'
FIT_COUNT = CALIBRATION_COUNT = 32


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(name, value):
    path = DATA / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def download(url, path):
    if path.exists():
        return path
    temporary = path.with_suffix(path.suffix + '.part')
    for attempt in range(4):
        try:
            with urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
            temporary.replace(path)
            return path
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    save('preparation-status.json', {'stage': 'mmvp_download', 'started_at': time.time()})
    source = DATA / 'source'
    snapshot_download('MMVP/MMVP', repo_type='dataset', revision=MMVP_REVISION,
                      local_dir=source, allow_patterns=['Questions.csv', 'MMVP Images/*.jpg', 'README.md'],
                      max_workers=8)
    questions = list(csv.DictReader((source / 'Questions.csv').open()))
    assert len(questions) == 300 and {int(r['Index']) for r in questions} == set(range(1, 301))
    evaluation = []
    image_hashes = set()
    for row in questions:
        index = int(row['Index'])
        image = source / 'MMVP Images' / f'{index}.jpg'
        with Image.open(image) as opened:
            opened.verify()
        image_hashes.add(sha(image))
        target = row['Correct Answer'].strip().strip('()').upper()
        assert target in {'A', 'B'}
        prompt = (row['Question'] + '\n' + row['Options'] +
                  '\nAnswer with the option letter A or B only.')
        evaluation.append({'id': f'mmvp:{index}', 'index': index,
                           'pair_index': (index - 1) // 2, 'image': str(image),
                           'question': row['Question'], 'options': row['Options'],
                           'prompt': prompt, 'target': target, 'split': 'test',
                           'image_sha256': sha(image)})
    evaluation.sort(key=lambda r: r['index'])
    (DATA / 'evaluation.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in evaluation))
    save('preparation-status.json', {'stage': 'coco_development', 'mmvp_records': len(evaluation)})
    archive = download('https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip',
                       DATA / 'coco128.zip')
    with zipfile.ZipFile(archive) as compressed:
        for member in compressed.infolist():
            target = (DATA / member.filename).resolve()
            assert target.is_relative_to(DATA), 'Archive path leaves the data folder'
        compressed.extractall(DATA)
    names_path = download('https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco128.yaml',
                          DATA / 'coco128.yaml')
    categories = yaml.safe_load(names_path.read_text())['names']
    present = {}
    areas = {}
    image_dir = DATA / 'coco128/images/train2017'
    for path in sorted((DATA / 'coco128/labels/train2017').glob('*.txt')):
        image_id = int(path.stem)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            category, _x, _y, width, height = map(float, line.split())
            category = int(category)
            present.setdefault(image_id, set()).add(category)
            key = (image_id, category)
            areas[key] = areas.get(key, 0) + width * height
    images = [{'id': int(p.stem), 'file_name': p.name} for p in sorted(image_dir.glob('*.jpg'))
              if int(p.stem) in present and len(present[int(p.stem)]) < len(categories)]
    rng = random.Random(42)
    rng.shuffle(images)
    chosen = images[:FIT_COUNT + CALIBRATION_COUNT]
    assert len(chosen) == FIT_COUNT + CALIBRATION_COUNT
    paths = []
    for row in chosen:
        path = image_dir / row['file_name']
        with Image.open(path) as image:
            image.verify()
        assert sha(path) not in image_hashes, 'Development image overlaps MMVP test image'
        paths.append(path)
    fit, calibration = [], []
    for position, (row, path) in enumerate(zip(chosen, paths)):
        split = 'fit' if position < FIT_COUNT else 'tune'
        image_id = row['id']
        positive = position % 2 == 0
        if positive:
            category = max(present[image_id], key=lambda c: (areas[(image_id, c)], -c))
        else:
            category = rng.choice(sorted(set(categories) - present[image_id]))
        label = 'yes' if positive else 'no'
        record = {'id': f'coco-dev:{image_id}', 'group_id': f'coco:{image_id}',
                  'image': str(path), 'prompt': f'Is there a {categories[category]} in this image? Answer yes or no.',
                  'label': label, 'task': 'pope', 'split': split,
                  'chosen': 'Yes.' if positive else 'No.',
                  'rejected': 'No.' if positive else 'Yes.',
                  'neutral': 'I cannot determine this without an image.',
                  'metadata': {'source': 'COCO train2017 via Ultralytics COCO128', 'image_id': image_id,
                               'yolo_class_id': category, 'annotation_derived_preferences': True,
                               'image_sha256': sha(path)}}
        (fit if split == 'fit' else calibration).append(record)
    for name, records in [('fit.jsonl', fit), ('calibration.jsonl', calibration)]:
        (DATA / name).write_text(''.join(json.dumps(r) + '\n' for r in records))
    assert not {r['group_id'] for r in fit} & {r['group_id'] for r in calibration}
    provenance = {'benchmark': 'MMVP', 'dataset_repo': 'MMVP/MMVP',
                  'dataset_revision': MMVP_REVISION, 'questions_sha256': sha(source / 'Questions.csv'),
                  'evaluation_records': 300, 'evaluation_pairs': 150,
                  'fit_records': len(fit), 'calibration_records': len(calibration),
                  'development_source': 'COCO train2017 via Ultralytics COCO128', 'seed': 42,
                  'development_archive_sha256': sha(archive), 'exact_image_hash_overlap': 0,
                  'protocol': 'Original MMVP questions/options, direct A/B answer; greedy max_new_tokens=128',
                  'label_policy': 'Original author answers; corrected-label sensitivity reported separately',
                  'known_corrected_indices': {99: 'B', 100: 'A', 279: 'B', 280: 'A'}}
    save('provenance.json', provenance)
    save('preparation-status.json', {'stage': 'complete', **provenance})
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()
