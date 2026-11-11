"""Resume IG prefixes and route unchanged author/generation math to GPU0."""
import argparse
import inspect
import json
import os
from pathlib import Path

from qwen_gpu0_migration_api import GPU0,load_gpu0,verify_protocol,sha

def extraction(output):
    import extract_qwen_author_grains as module
    import numpy as np
    protocol=verify_protocol(output);amend=json.loads((output/'gpu0-hardware-amendment.json').read_text())
    stop=json.loads(Path(amend['stop_proof']).read_text());raw=(output/'extraction.jsonl').read_bytes()
    assert len(raw)==stop['journal_bytes']==stop['committed_journal_bytes'] and raw.endswith(b'\n')
    assert sha(output/'extraction.jsonl')==stop['journal_prefix_sha256']
    prior=[json.loads(s) for s in raw.splitlines() if s.strip()];assert len(prior)==stop['completed_ig_records']
    hidden={}
    for entry in prior:
        assert sha(entry['artifact'])==entry['artifact_sha256'] and sha(entry['raw_ig'])==entry['raw_ig_sha256']
        assert entry['ig']['quadrature_nodes']==20
        with np.load(entry['artifact'],allow_pickle=False) as arrays:
            for name in arrays.files:
                assert np.isfinite(arrays[name]).all()
                if name.startswith('hidden_'):hidden.setdefault(name,[]).append(arrays[name].copy())
    for name in ['grains-author.pt','original-hidden-states.npz','grains-artifact.json']:
        assert not (output/name).exists(),'A completed artifact already exists; do not restart extraction'
    module.load_qwen=load_gpu0;module.verify_run_protocol=verify_protocol;module.PAIR=GPU0
    module._resume_prior_hidden=hidden;module._resume_start=len(prior)
    module._gpu0_uuid=GPU0
    module._gpu0_extraction_directory=Path(amend['migration'])/'remaining-extraction'
    source=inspect.getsource(module.main)
    replacements={
        "'completed': 0":"'completed': _resume_start",
        "'extraction-status.json'":"'extraction-gpu0-status.json'",
        "load_qwen(config['model'], output, 'grains-extraction')":"load_qwen(config['model'], output, 'grains-extraction-gpu0')",
        "extraction = output / 'extraction'; extraction.mkdir()":"extraction = _gpu0_extraction_directory; extraction.mkdir()",
        "hidden = {}; ig_calls = []":"hidden = _resume_prior_hidden; ig_calls = []",
        "(output / 'extraction.jsonl').open('x')":"(output / 'extraction.jsonl').open('a')",
        "for index, record in enumerate(fit):":"for index, record in enumerate(fit[_resume_start:], start=_resume_start):",
        "range(2)":"range(torch.cuda.device_count())",
        "assert len(ig_calls) == 50 and len(hidden) == 4":"assert len(ig_calls) + _resume_start == 50 and len(hidden) == 4",
        "'actual_ig_calls': len(ig_calls)":"'actual_ig_calls': len(ig_calls) + _resume_start",
        "'ig': ig_calls[-1], 'seconds'":"'ig': ig_calls[-1], 'gpu_uuid': _gpu0_uuid, 'seconds'",
    }
    for old,new in replacements.items():
        assert old in source,old;source=source.replace(old,new)
    exec(compile(source,str(Path(module.__file__))+'::prefix_preserving_gpu0','exec'),module.__dict__)
    module.main(output)
    seal=json.loads((output/'grains-artifact.json').read_text())
    seal['hardware_history']={'preserved_gpu23_records':len(prior),'new_gpu0_records':50-len(prior),
        'stop_proof_sha256':sha(Path(amend['stop_proof'])),'hardware_amendment_sha256':sha(output/'gpu0-hardware-amendment.json')}
    from qwen_gpu0_migration_api import save
    save(output/'grains-artifact.json',seal)

def main(args):
    assert os.environ['CUDA_VISIBLE_DEVICES']==GPU0
    verify_protocol(args.output)
    if args.stage=='extract':extraction(args.output)
    elif args.stage=='preflight_fp16':
        import qwen_five_profile_stage as module
        module.load_qwen=load_gpu0;module.verify_run_protocol=verify_protocol
        module.fp16_preflight(args.output)
    elif args.stage=='generation':
        import run_qwen_five_profile_worker as module
        module.GPU2=GPU0;module.GPU3=GPU0
        module.load_qwen=load_gpu0;module.verify_run_protocol=verify_protocol
        module.main(args)
    else:raise ValueError(args.stage)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',required=True)
    p.add_argument('--worker');p.add_argument('--dtype',choices=['bfloat16','float16']);p.add_argument('--lease-epoch')
    main(p.parse_args())
