"""Create disjoint EPHS development masks from cached original COCO annotations."""
import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from vlm_grounding.data import prepare_coco

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def main():
    stamp = datetime.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d-%H%M%S')
    output = ROOT / 'outputs' / f'gpu2-ephs-chair-{stamp}'
    output.mkdir()
    data = output / 'data'
    data.mkdir()
    images = ROOT / 'data/mmvp-full/coco128/images/train2017'
    available = {int(p.stem) for p in images.glob('*.jpg')}
    tests = read(ROOT / 'data/hallucination-suite/pope.jsonl') + read(ROOT / 'data/hallucination-suite/chair.jsonl')
    test_ids = {int(r['group_id'].split(':')[-1]) for r in tests}
    test_hashes = {r['image_sha256'] for r in tests}
    selected_images, selected_annotations, categories, source_hashes = {}, [], None, {}
    for name in ['instances_train2014.json', 'instances_val2014.json']:
        path = ROOT / 'data/hallucination-suite/annotations' / name
        source_hashes[name] = sha(path)
        obj = json.loads(path.read_text())
        categories = obj['categories']
        for img in obj['images']:
            if img['id'] in available and img['id'] not in test_ids:
                path = images / f"{img['id']:012d}.jpg"
                if sha(path) not in test_hashes:
                    selected_images[img['id']] = {**img, 'file_name': path.name}
        selected_annotations.extend(a for a in obj['annotations'] if a['image_id'] in selected_images)
        del obj
    subset = {'images': list(selected_images.values()), 'annotations': selected_annotations, 'categories': categories}
    annotations = data / 'instances_dev.json'
    annotations.write_text(json.dumps(subset))
    rows = prepare_coco(annotations, images, data, seed=42, discovery_count=32, fit_count=32, tune_count=0, test_count=0,
                        exclude_group_ids={f'coco:{i}' for i in test_ids})
    discovery = [r for r in rows if r['split'] == 'discovery']
    fit = [r for r in rows if r['split'] == 'fit' and r['task'] == 'pope']
    assert len(discovery) == 32 and len(fit) == 64
    assert {r['group_id'] for r in discovery}.isdisjoint(r['group_id'] for r in fit)
    for r in discovery + fit:
        r['image_sha256'] = sha(r['image'])
        assert r['image_sha256'] not in test_hashes and r['group_id'] not in {f'coco:{i}' for i in test_ids}
    for name, values in [('discovery.jsonl', discovery), ('fit.jsonl', fit)]:
        (data / name).write_text(''.join(json.dumps(r) + '\n' for r in values))
    protocol = {'model': json.loads((ROOT / 'configs/gemma4_e4b.local.json').read_text()),
        'development': {'discovery_images':32,'fit_images':32,'fit_questions':64,'seed':42,
            'mask_source':'original COCO segmentation, same-shape background controls', 'test_id_overlap':0,'test_hash_overlap':0},
        'source_annotation_hashes':source_hashes,
        'development_input_hashes': {str(Path(r[k])): sha(r[k]) for r in discovery + fit for k in ['image','object_masked_image','control_image'] if k in r},
        'manifest_hashes':{n:sha(data / n) for n in ['discovery.jsonl','fit.jsonl','instances_dev.json']},
        'source_hashes':{n:sha(ROOT / n) for n in ['src/vlm_grounding/backend.py','src/vlm_grounding/method.py','src/vlm_grounding/cli.py','scripts/prepare_gpu2_ephs.py','scripts/train_gpu2_ephs.py']},
        'gpu_index':2,'gpu_uuid':'GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde','automatic_other_gpu_fallback':False}
    (output / 'training-protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    pointer={'output':str(output),'stage':'development_prepared'}
    (ROOT / 'outputs/gpu2-ephs-chair-current.json').write_text(json.dumps(pointer,indent=2)+'\n')
    print(json.dumps({**pointer,'discovery':len(discovery),'fit':len(fit),'candidate_images':len(selected_images)},indent=2))

if __name__ == '__main__':
    main()
