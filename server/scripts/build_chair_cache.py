"""Build genuine CHAIR ground truth once from all original COCO annotations."""
import contextlib
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

from vlm_grounding.evaluation import _load_external_module

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')
DATA = ROOT / 'data/hallucination-suite'
STATUS = DATA / 'chair-cache-status.json'


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    save(STATUS, {'stage': 'building_from_full_annotations', 'pid': os.getpid(), 'started_at': time.time()})
    module_path = ROOT / 'upstream/chair/chair.py'
    module = _load_external_module(module_path, '_chair_cache_builder')
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
        scorer = module.CHAIR(str(DATA / 'annotations'))
    ids = {int(json.loads(line)['metadata']['image_id']) for line in (DATA / 'chair.jsonl').read_text().splitlines()}
    assert len(ids) == 500 and ids <= scorer.imid_to_objects.keys()
    state = {**scorer.__dict__, 'imid_to_objects': {str(i): sorted(scorer.imid_to_objects[i]) for i in sorted(ids)}}
    result = {'state': state, 'cohort_image_ids': sorted(ids), 'built_with_unmodified_constructor': True,
              'scorer_sha256': hashlib.sha256(module_path.read_bytes()).hexdigest(),
              'annotation_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((DATA / 'annotations').glob('*.json'))},
              'built_at': time.time()}
    save(DATA / 'chair-scorer-state.json', result)
    save(STATUS, {'stage': 'complete', 'pid': os.getpid(), 'images': 500, 'completed_at': time.time()})


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        save(STATUS, {'stage': 'failed', 'error': traceback.format_exc(), 'updated_at': time.time()})
        raise
