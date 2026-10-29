"""Restore missing HF revision metadata after checking the pinned local snapshot."""
import argparse
import hashlib
import json
from pathlib import Path

import run_author_clone_worker as worker
from author_clone_bridge import ROOT


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--worker',required=True);parser.add_argument('--dtype',required=True)
    args=parser.parse_args()
    seal=json.loads((args.output/'execution-amendment.json').read_text())
    for name,digest in seal['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    original=worker.load_adapter
    def load_with_revision(config):
        from huggingface_hub import try_to_load_from_cache
        cached=try_to_load_from_cache(config['checkpoint'],'config.json',revision=config['revision'])
        assert isinstance(cached,str) and Path(cached).parent.name==config['revision']
        adapter=original(config)
        observed=getattr(adapter.model.config,'_commit_hash',None)
        assert observed in [None,config['revision']], observed
        adapter.model.config._commit_hash=config['revision']
        (args.output/f'{args.worker}-revision-validation.json').write_text(json.dumps({
            'configured_revision':config['revision'],'cached_config':cached,
            'observed_config_revision':observed,'restored_revision_metadata':config['revision'],
            'model_weights_or_computation_changed':False},indent=2)+'\n')
        return adapter
    worker.load_adapter=load_with_revision
    worker.main(args)
