"""Keep the pinned entry and author math; select explicitly authorized devices."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import continue_gpu23_worker as worker
from gpu23_device_adapter import load_on_selected_devices, grains_on_layer_device

root = Path(__file__).resolve().parents[1]
output = Path(sys.argv[sys.argv.index('--output') + 1])
name = sys.argv[sys.argv.index('--worker') + 1]
seal = json.loads((output / 'hardware-amendment.json').read_text())
for file, digest in seal['source_hashes'].items():
    assert hashlib.sha256((root / file).read_bytes()).hexdigest() == digest
worker.load_adapter = lambda config: load_on_selected_devices(config, output, name)
worker.grains_context = grains_on_layer_device
sys.modules['run_author_clone_worker'] = worker
runpy.run_path(str(root / 'scripts/author_clone_worker_entry.py'), run_name='__main__')

