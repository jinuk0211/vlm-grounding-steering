"""New lease namespace around the unchanged, sealed generation body."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
output = Path(sys.argv[sys.argv.index('--output') + 1])
epoch = os.environ['LJW_RESUME_EPOCH']
assert epoch.startswith('resume-20261029-') and '/' not in epoch
seal = json.loads((output / 'resumes' / epoch / 'amendment.json').read_text())
for name, digest in seal['new_source_hashes'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
source = (ROOT / 'scripts/continue_gpu23_worker.py').read_text()
old = "output/'continuation-claims'/task['id']"
new = "output/'resumes'/os.environ['LJW_RESUME_EPOCH']/'claims'/task['id']"
assert source.count(old) == 1
source = source.replace(old, new)
line = "                    row['hardware_execution']='gpu23-resume-20261028'"
assert source.count(line) == 1
source = source.replace(line, line + "\n                    row['resume_epoch']=os.environ['LJW_RESUME_EPOCH']")
module = types.ModuleType('continue_gpu23_worker')
module.__file__ = str(ROOT / 'scripts/continue_gpu23_worker.py')
exec(compile(source, module.__file__, 'exec'), module.__dict__)
sys.modules['continue_gpu23_worker'] = module
runpy.run_path(str(ROOT / 'scripts/continue_gpu23_entry.py'), run_name='__main__')
