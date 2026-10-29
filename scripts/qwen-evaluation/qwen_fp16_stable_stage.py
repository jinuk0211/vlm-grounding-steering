"""Explicitly amended FP16 eager score arithmetic; frozen author hooks untouched."""
import argparse,hashlib,inspect,json,os,time
from pathlib import Path
from qwen_gpu0_migration_api import ROOT,GPU0,sha,save,verify_protocol,load_gpu0
from qwen_gpu0_stage_v2 import verify_repair
from qwen_fp16_attention_guard import stable_fp16_attention,verify_arithmetic

def verify_attention_amendment(out):
    verify_repair(out);a=json.loads((out/'fp16-attention-amendment.json').read_text())
    assert a['parent_protocol_sha256']==verify_protocol(out)['sha256']
    for path,digest in a['source_hashes'].items():assert sha(ROOT/path)==digest
    assert sha(out/'fp16-cpu-diagnostic.json')==a['native_cpu_diagnostic_sha256']
    return a

def main(args):
    assert os.environ['CUDA_VISIBLE_DEVICES']==GPU0
    out=args.output;verify_attention_amendment(out);diag={}
    with stable_fp16_attention(diag):
        def loader(config,output,worker):
            adapter=load_gpu0(config,output,worker)
            proof=output/(worker+'-device-map.json');value=json.loads(proof.read_text())
            value.update(attention_score_dtype='torch.float32',attention_output_dtype='torch.float16',
                fp16_attention_amendment_sha256=sha(out/'fp16-attention-amendment.json'))
            save(proof,value)
            return adapter
        if args.stage=='preflight_fp16':
            import qwen_five_profile_stage as stage
            stage.load_qwen=loader;stage.verify_run_protocol=verify_protocol;stage._attention_diagnostics=diag
            source=inspect.getsource(stage.fp16_preflight)
            source=source.replace("'fp16-preflight.json'","'fp16-stable-preflight.json'")
            source=source.replace("output,'fp16-preflight'","output,'fp16-stable-preflight'")
            source=source.replace('handle.remove();del result',"handle.remove();del result;_attention_diagnostics['trace_native_half_scores']=False")
            exec(compile(source,str(Path(stage.__file__))+'::fp16_attention_scores','exec'),stage.__dict__)
            diag['trace_native_half_scores']=True
            stage.fp16_preflight(out)
            proof=json.loads((out/'fp16-stable-preflight.json').read_text())
            proof.update(attention_diagnostics=diag,attention_score_dtype='torch.float32',
                attention_amendment_sha256=sha(out/'fp16-attention-amendment.json'))
            save(out/'fp16-stable-preflight.json',proof)
        elif args.stage=='generation':
            assert args.dtype=='float16'
            proof=json.loads((out/'fp16-stable-preflight.json').read_text());assert proof['status']=='passed'
            import run_qwen_five_profile_worker as worker
            worker.GPU2=GPU0;worker.GPU3=GPU0;worker.load_qwen=loader;worker.verify_run_protocol=verify_protocol
            worker._attention_amendment_sha256=sha(out/'fp16-attention-amendment.json')
            source=inspect.getsource(worker.main)
            old="'gpu_uuid': visible, 'model_family'"
            assert old in source
            source=source.replace(old,"'gpu_uuid': visible, 'attention_score_dtype': 'torch.float32', 'fp16_attention_amendment_sha256': _attention_amendment_sha256, 'model_family'")
            exec(compile(source,str(Path(worker.__file__))+'::fp16_attention_metadata','exec'),worker.__dict__)
            worker.main(args)
        else:raise ValueError(args.stage)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',required=True)
    p.add_argument('--worker');p.add_argument('--dtype',choices=['float16']);p.add_argument('--lease-epoch')
    main(p.parse_args())
