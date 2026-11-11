"""Repair only the legacy helper's Gemma completion guard; no math changes."""
import argparse
import json
from pathlib import Path
from qwen_gpu0_migration_api import ROOT,sha,verify_protocol

def verify_repair(output):
    data=json.loads((output/'gpu0-repair-amendment.json').read_text())
    assert sha(output/'gpu0-hardware-amendment.json')==data['hardware_amendment_sha256']
    for name,digest in data['source_hashes'].items():assert sha(ROOT/name)==digest,name
    verify_protocol(output)
    return data

def main(args):
    verify_repair(args.output)
    import extract_qwen_author_grains as legacy
    from qwen_five_profile_api import require_gemma_complete
    legacy.require_gemma_complete=require_gemma_complete
    import qwen_gpu0_stage_entry as entry
    entry.main(args)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',required=True)
    p.add_argument('--worker');p.add_argument('--dtype',choices=['bfloat16','float16']);p.add_argument('--lease-epoch')
    main(p.parse_args())
