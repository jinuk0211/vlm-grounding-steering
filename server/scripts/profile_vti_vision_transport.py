"""Short GPU0-only transport profile; writes no benchmark predictions."""
import json,os,time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_gpu0_migration_api import GPU0,ROOT,save
from qwen_fp16_attention_guard import stable_fp16_attention

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==GPU0 and torch.cuda.device_count()==1
    torch.set_num_threads(4)
    out=ROOT/'outputs/qwen-original-vti-1793260357'
    import subprocess
    free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>8*1024,free
    state={'pid':os.getpid(),'stage':'loading_vision_only','benchmark_predictions_written':0,'gpu_uuid':GPU0,'initial_free_mib':free,'started_at':time.time()}
    save(out/'vision-transport-profile.json',state)
    processor=AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct',revision='cc594898137f460bfe9f0759e9844b3ce807cfb5',local_files_only=True)
    native=Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct',revision='cc594898137f460bfe9f0759e9844b3ce807cfb5',local_files_only=True,dtype=torch.float16,attn_implementation='eager')
    vision=native.model.visual;del native
    vision=vision.to('cuda').eval();assert {str(p.device) for p in vision.parameters()}=={'cuda:0'}
    image=Image.open(next((out/'data/train2014').glob('*.jpg'))).convert('RGB').resize((448,448),Image.Resampling.BICUBIC)
    inputs=processor.image_processor(images=[image],return_tensors='pt')
    pixel=inputs['pixel_values'].to('cuda',dtype=torch.float16);grid=inputs['image_grid_thw'].to('cuda')
    samples=[];reference=None
    with stable_fp16_attention({}),torch.no_grad():
        vision(pixel,grid_thw=grid);torch.cuda.synchronize()
        for repeat in range(3):
            for mode in (['cpu_per_block','gpu_until_author_average'] if repeat%2==0 else ['gpu_until_author_average','cpu_per_block']):
                states=[]
                def keep(x):return (x.detach().cpu() if mode=='cpu_per_block' else x.detach())[None]
                def first(_m,args,kwargs):states.append(keep(args[0] if args else kwargs['hidden_states']))
                def capture(_m,args,output):states.append(keep(output))
                handles=[vision.blocks[0].register_forward_pre_hook(first,with_kwargs=True)]
                handles.extend(b.register_forward_hook(capture) for b in vision.blocks)
                torch.cuda.synchronize();start=time.perf_counter()
                try:vision(pixel,grid_thw=grid)
                finally:
                    for h in handles:h.remove()
                if mode=='cpu_per_block':assert all(torch.isfinite(s).all() for s in states)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                # Check complete captured values after timing, including the
                # GPU-transport variant, before making a performance claim.
                cpu=[s.cpu() for s in states];assert len(cpu)==33
                assert all(torch.isfinite(s).all() for s in cpu)
                if reference is None:reference=cpu
                else:assert all(torch.equal(a,b) for a,b in zip(reference,cpu))
                samples.append({'mode':mode,'repeat':repeat,'seconds':elapsed,'captured_shape':list(cpu[0].shape)})
                state.update(stage='profiling',samples=samples);save(out/'vision-transport-profile.json',state)
                del states,cpu
    import statistics
    medians={m:statistics.median(s['seconds'] for s in samples if s['mode']==m) for m in ['cpu_per_block','gpu_until_author_average']}
    state.update(stage='passed',samples=samples,medians=medians,identical_full_hidden_state_values=True,
        ratio_cpu_over_gpu=medians['cpu_per_block']/medians['gpu_until_author_average'],
        maximum_gpu_allocated_gb=torch.cuda.max_memory_allocated()/1e9,completed_at=time.time())
    save(out/'vision-transport-profile.json',state);print(json.dumps(state))

if __name__=='__main__':main()
