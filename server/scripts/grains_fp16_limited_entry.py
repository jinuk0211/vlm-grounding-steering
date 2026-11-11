"""Run the sealed FP16 worker with a bounded PyTorch allocator on shared GPU2."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import grains_fp16_worker_v2 as worker

def main(output,name):
    assert os.environ['CUDA_VISIBLE_DEVICES']==worker.GPU2 and torch.cuda.device_count()==1
    amendment=json.loads((output/'shared-gpu2-amendment.json').read_text())
    for file,digest in amendment['source_hashes'].items():
        assert hashlib.sha256((worker.ROOT/file).read_bytes()).hexdigest()==digest
    fraction=amendment['allocator_fraction'];assert fraction==0.70
    torch.cuda.set_per_process_memory_fraction(fraction,0)
    proof={'pid':os.getpid(),'visible_devices':worker.GPU2,'allocator_fraction':fraction,
           'device_total_bytes':torch.cuda.get_device_properties(0).total_memory,
           'worker_source_sha256':hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest(),
           'generation_and_author_math_unchanged':True}
    (output/(name+'-resource-limit.json')).write_text(json.dumps(proof,indent=2)+'\n')
    worker.main(output,name,False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',required=True)
    a=p.parse_args();main(a.output,a.worker)
