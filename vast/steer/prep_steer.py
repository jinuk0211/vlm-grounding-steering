"""Inputs vdd_steer.py and score_sets.py expect, on this box (run once):
  sets     /workspace/steer/sets/chair_test500.jsonl (+ chair_test100 = rows 0-99): the paper's CHAIR test rows in the paper's
           order (outputs/rbd2-20261102/runs/chair_test500_all/base-0-500.jsonl), image paths moved here, images fetched
           from COCO val2014 and sha256-checked against the paper's manifest
  config   read_gap_collect_v2.MODELS['qwen']/config.json = configs/qwen2_5_vl_7b.json model block (pinned revision)
  scorer   CHAIR state for the 500 test images from the pilot's unmodified CHAIR (tools/chair, cached constructor
           state out/chair_state.pkl), in score_sets.py's format; the scorer source is copied to upstream/chair/chair.py
"""
import concurrent.futures as cf, hashlib, json, pickle, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

R = Path('/home/project-b/LJW/vlm-grounding-steering'); W = Path('/workspace/steer'); P = Path('/workspace/tether-pilot')
sys.path[:0] = [str(R / 'scripts'), str(R / 'src')]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
step = sys.argv[1]

if step == 'sets':
    rows = [json.loads(l) for l in open(R / 'outputs/rbd2-20261102/runs/chair_test500_all/base-0-500.jsonl') if l.strip()]
    keep = ('id', 'group_id', 'task', 'benchmark', 'split', 'image', 'prompt', 'max_new_tokens', 'image_sha256', 'metadata', 'inference_key')
    (W / 'images').mkdir(parents=True, exist_ok=True); (W / 'sets').mkdir(exist_ok=True)
    refs = []
    for r in rows:
        ref = {k: r[k] for k in keep if k in r}; ref['image'] = str(W / 'images' / Path(r['image']).name); refs.append(ref)
    for f in ('abl_nosteer', 'rbd_full', 'base'):
        ids = [json.loads(l)['id'] for l in open(R / f'outputs/rbd2-20261102/runs/chair_abl500_all/{f}-0-500.jsonl') if l.strip()]
        assert ids == [r['id'] for r in refs], f'row order differs: {f}'
    def get(ref):
        p = Path(ref['image'])
        for k in range(8):
            try:
                if not p.exists():
                    data = urllib.request.urlopen('http://images.cocodataset.org/val2014/' + p.name, timeout=60).read()
                    tmp = p.with_suffix('.part'); tmp.write_bytes(data); tmp.rename(p)
                assert sha(p) == ref['image_sha256'], f'sha256 mismatch {p.name}'
                return p.name
            except AssertionError:
                raise
            except Exception:
                p.unlink(missing_ok=True); time.sleep(2 + k)
        raise RuntimeError(f'failed {p.name}')
    with cf.ThreadPoolExecutor(16) as ex: done = list(ex.map(get, refs))
    (W / 'sets/chair_test500.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in refs))
    (W / 'sets/chair_test100.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in refs[:100]))
    print('sets ok', len(done), 'images sha256-checked')

if step == 'config':
    import read_gap_collect_v2 as C
    model = json.loads((R / 'configs/qwen2_5_vl_7b.json').read_text())['model']
    C.MODELS['qwen'].mkdir(parents=True, exist_ok=True)
    (C.MODELS['qwen'] / 'config.json').write_text(json.dumps({'model': model}, indent=1))
    print('config ok', C.MODELS['qwen'] / 'config.json', model['checkpoint'], model['revision'])

if step == 'scorer':
    src = P / 'tools/chair/chair.py'; dst = R / 'upstream/chair/chair.py'; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(src, dst)
    commit = subprocess.run(['git', '-C', str(P / 'tools/chair'), 'log', '--format=%h %ad %s', '-1'], capture_output=True, text=True).stdout.strip()
    state = pickle.loads((P / 'out/chair_state.pkl').read_bytes())
    ids = {int(json.loads(l)['metadata']['image_id']) for l in open(W / 'sets/chair_test500.jsonl')}
    assert ids <= set(state['imid_to_objects']), 'test ids missing from the cached CHAIR state'
    def plain(v):
        if isinstance(v, set): return sorted(v)
        if isinstance(v, dict): return {str(k): plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [plain(x) for x in v]
        return v
    st = {k: plain(v) for k, v in state.items() if k != 'imid_to_objects'}
    st['imid_to_objects'] = {str(i): sorted(state['imid_to_objects'][i]) for i in sorted(ids)}
    out = {'state': st, 'cohort_image_ids': sorted(ids), 'built_with_unmodified_constructor': True, 'scorer_sha256': sha(dst),
           'source': f'pilot tools/chair ({commit}); state from out/chair_state.pkl', 'built_at': time.time()}
    (W / 'chair-test500-scorer-state.json').write_text(json.dumps(out))
    print('scorer ok; chair.py commit:', commit, '; state keys:', sorted(st))
