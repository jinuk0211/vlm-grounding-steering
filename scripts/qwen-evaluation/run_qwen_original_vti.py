"""Pinned VTI author implementation with Qwen2.5 input/shape adapters only.

Author PCA, patch masks, directions, MLP placement and VTILayer are invoked
from the unmodified checkout. No vlm_grounding.baselines VTI is imported.
"""
import argparse,collections,contextlib,hashlib,importlib,json,os,random,sys,time,traceback,types
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from qwen_gpu0_migration_api import ROOT,GPU0,sha,save,load_gpu0,verify_protocol
from qwen_fp16_attention_guard import stable_fp16_attention

PARENT=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386'
AUTHOR=ROOT/'upstream-author-20261026/VTI'
COMMIT='bb5dfee33a9e555f22af9faa7b3495c0ac3df69e'

def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def verify_author():
    import subprocess
    assert subprocess.check_output(['git','-C',str(AUTHOR),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(AUTHOR),'status','--porcelain'],text=True).strip()
    names=subprocess.check_output(['git','-C',str(AUTHOR),'ls-files'],text=True).splitlines()
    return {n:sha(AUTHOR/n) for n in names}

def imports():
    sys.path[:0]=[str(ROOT/'.author-deps'),str(AUTHOR)]
    return importlib.import_module('vti_utils.utils'),importlib.import_module('vti_utils.llm_layers')

class LayerView(torch.nn.Module):
    def __init__(self,layers):super().__init__();self.layers=layers

class VisionShape(torch.nn.Module):
    """Qwen ViT uses [tokens,hidden]; original VTI expects a batch axis."""
    def __init__(self,author_layer):super().__init__();self.author_layer=author_layer
    def forward(self,x):
        assert x.ndim==2
        return self.author_layer(x.unsqueeze(0)).squeeze(0)

def install(layers,direction,alpha,author_layers,vision=False):
    originals=[layer.mlp for layer in layers]
    device=next(layers[0].parameters(),direction).device
    author_layers.add_vti_layers(LayerView(layers),direction[:,None].to(device),alpha=[alpha])
    for layer,original in zip(layers,originals):
        assert isinstance(layer.mlp,torch.nn.Sequential) and layer.mlp[0] is original
        assert isinstance(layer.mlp[1],author_layers.VTILayer)
        if vision:layer.mlp[1]=VisionShape(layer.mlp[1])
    return originals

class ImageProcessor:
    """Expose fixed-size normalized CHW pixels to original get_demos/masks."""
    def __init__(self,processor):
        self.mean=torch.tensor(processor.image_mean)[:,None,None]
        self.std=torch.tensor(processor.image_std)[:,None,None]
    def __call__(self,image):
        image=ImageOps.exif_transpose(image).convert('RGB').resize((448,448),Image.Resampling.BICUBIC)
        x=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
        return {'pixel_values':((x-self.mean)/self.std)[None]}

class VisionCapture:
    def __init__(self,adapter,progress):self.adapter=adapter;self.progress=progress;self.calls=0
    def __call__(self,pixels,**kwargs):
        a=self.adapter;states=[];handles=[]
        def first(_m,args,kw):states.append((args[0] if args else kw['hidden_states']).detach().cpu()[None])
        def capture(_m,args,output):states.append(output.detach().cpu()[None])
        handles.append(a.vision_layers[0].register_forward_pre_hook(first,with_kwargs=True))
        handles.extend(layer.register_forward_hook(capture) for layer in a.vision_layers)
        try:
            # Pixels already have Qwen normalization; its processor only packs
            # spatial/temporal patches. This avoids PIL quantization round trips.
            batch=a.processor.image_processor(images=[pixels[0].float().cpu()],do_rescale=False,do_normalize=False,return_tensors='pt')
            visual=a.model.model.visual
            visual(batch['pixel_values'].to('cuda',dtype=torch.float16),grid_thw=batch['image_grid_thw'].to('cuda'))
        finally:
            for h in handles:h.remove()
        assert len(states)==len(a.vision_layers)+1
        assert all(s.shape==states[0].shape and torch.isfinite(s).all() for s in states)
        self.calls+=1
        if self.calls%10==0:self.progress(vision_forward_calls=self.calls)
        return types.SimpleNamespace(hidden_states=tuple(states))

class TextCapture:
    def __init__(self,adapter):self.adapter=adapter
    def __call__(self,**batch):
        inputs={k:v for k,v in batch.items() if k!='labels'};inputs.update(output_hidden_states=True,return_dict=True,use_cache=False)
        result=self.adapter.model(**inputs)
        assert len(result.hidden_states)==len(self.adapter.layers)+1
        return result

def prompt_pairs(adapter,args,model,tokenizer,demos,question,model_is_llaval=True):
    pairs=[]
    for d in demos:
        path=Path(args.data_file)/'train2014'/d['image']
        with Image.open(path) as source:image=ImageOps.exif_transpose(source).convert('RGB').resize((448,448),Image.Resampling.BICUBIC)
        prompt=adapter.processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':question}]}],tokenize=False,add_generation_prompt=True)
        # Original author captions terminate at their final token, without an
        # added assistant EOS. Only multimodal input preparation is adapted.
        styles=[]
        for key in ['h_value','value']:
            b=adapter.processor(text=[prompt+d[key]],images=[image],return_tensors='pt')
            styles.append({k:v.to('cuda',dtype=torch.float16 if v.is_floating_point() else v.dtype) for k,v in b.items() if torch.is_tensor(v)})
        pairs.append(tuple(styles))
    return tuple(pairs)

def fit(adapter,out,utils,layers,update):
    args=types.SimpleNamespace(num_demos=70,num_trials=50,mask_ratio=.99,data_file=str(out/'data'))
    random.seed(42)
    original_prompts=utils.get_prompts
    utils.get_prompts=lambda *a,**k:prompt_pairs(adapter,*a,**k)
    try:
        images,inputs=utils.get_demos(args,ImageProcessor(adapter.processor.image_processor),adapter.model,None,
            file_path=str(AUTHOR/'experiments/data/hallucination_vti_demos.jsonl'))
    finally:utils.get_prompts=original_prompts
    update(stage='extracting_original_visual_direction',demos=70,trials=50)
    api=types.SimpleNamespace(vision_model=VisionCapture(adapter,update))
    visual,_=utils.obtain_visual_vti(api,images,rank=1)
    assert torch.isfinite(visual).all() and visual.shape[0]==len(adapter.vision_layers)+1
    torch.save(visual,out/'original-visual-direction.pt')
    install(adapter.vision_layers,visual[1:],.9,layers,vision=True)
    update(stage='extracting_original_textual_direction',visual_direction_sha256=sha(out/'original-visual-direction.pt'))
    textual,_=utils.obtain_textual_vti(TextCapture(adapter),inputs,None,rank=1)
    assert torch.isfinite(textual).all() and textual.shape==(len(adapter.layers)+1,adapter.hidden_size)
    artifact={'method':'vti_author','author_commit':COMMIT,'author_source_hashes':verify_author(),
        'visual_direction':visual,'textual_direction':textual,'alpha_image':.9,'alpha_text':.9,
        'num_demos':70,'num_trials':50,'mask_ratio':.99,'rank':1,
        'text_extraction_visual_vti_active':True,'extraction_dtype':'float16',
        'demo_manifest_sha256':sha(out/'demo-manifest.json'),'original_functions':[
            'get_demos','mask_patches','obtain_visual_vti','obtain_textual_vti','PCA','add_vti_layers','VTILayer.forward']}
    torch.save(artifact,out/'vti-author.pt');return artifact

def main(out):
    sys.dont_write_bytecode=True
    assert os.environ['CUDA_VISIBLE_DEVICES']==GPU0 and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    manifest=json.loads((out/'protocol.json').read_text());assert manifest['author_commit']==COMMIT
    assert manifest['author_sources']==verify_author();assert manifest['parent_protocol_sha256']==verify_protocol(PARENT)['sha256']
    assert manifest['worker_source_sha256']==sha(__file__)
    state={'pid':os.getpid(),'started_at':time.time(),'gpu_uuid':GPU0,'completed_records':0}
    def update(**kw):state.update(kw,updated_at=time.time());save(out/'status.json',state)
    utils,layers=imports();attention={}
    try:
        with stable_fp16_attention(attention):
            config=json.loads((PARENT/'config.json').read_text())['model'];config['dtype']='float16'
            update(stage='loading_original_vti_model')
            adapter=load_gpu0(config,PARENT,'vti-author-'+out.name.rsplit('-',1)[-1])
            # Keep the immutable parent device proof as well as a subrun copy.
            proof=PARENT/('vti-author-'+out.name.rsplit('-',1)[-1]+'-device-map.json')
            (out/'device-map.json').write_bytes(proof.read_bytes())
            artifact_path=out/'vti-author.pt'
            if artifact_path.exists():
                artifact=torch.load(artifact_path,map_location='cpu',weights_only=True)
                assert artifact['author_commit']==COMMIT and artifact['author_source_hashes']==verify_author()
                install(adapter.vision_layers,artifact['visual_direction'][1:],.9,layers,vision=True)
            else:artifact=fit(adapter,out,utils,layers,update)
            install(adapter.layers,artifact['textual_direction'][1:],.9,layers)
            artifact_digest=sha(artifact_path)
            save(out/'artifact-ready.json',{'sha256':artifact_digest,'author_commit':COMMIT,'source_hashes':verify_author()})
            diagnostics={};handles=[]
            for name,module in adapter.model.named_modules():
                if isinstance(module,layers.VTILayer):
                    kind='vision' if '.visual.' in name else 'text'
                    def observe(_module,args,value,kind=kind):
                        assert torch.isfinite(value).all()
                        d=diagnostics.setdefault(kind,{'calls':0,'changed_elements':0,'max_abs_delta':0.0})
                        diff=(value.float()-args[0].float()).abs();d['calls']+=1
                        d['changed_elements']+=int((diff>0).sum());d['max_abs_delta']=max(d['max_abs_delta'],float(diff.max()))
                    handles.append(module.register_forward_hook(observe))
            assert len(handles)==len(adapter.layers)+len(adapter.vision_layers)
            update(stage='original_vti_preflight')
            record=read(PARENT/'data/pope.jsonl')[0];response=adapter.generate(adapter.prepare(record['image'],record['prompt']),max_new_tokens=8)
            assert all(diagnostics.get(k,{}).get('changed_elements',0)>0 for k in ['text','vision'])
            save(out/'preflight.json',{'status':'passed','gpu_uuid':GPU0,'response':response,'diagnostics':diagnostics,
                'original_hook_file':layers.VTILayer.forward.__code__.co_filename,'original_pca_file':utils.PCA.fit.__code__.co_filename,
                'artifact_sha256':artifact_digest,'model_dtype':'float16','attention_score_dtype':'float32'})
            for bench in ['pope','chair']:
                references=read(PARENT/'data'/(bench+'.jsonl'));path=out/(bench+'.jsonl');prefix=read(path) if path.exists() else []
                for r,ref in zip(prefix,references):assert all(r.get(k)==v for k,v in ref.items()) and r['method']=='vti_author'
                with path.open('a' if path.exists() else 'x') as sink:
                    for i,ref in enumerate(references[len(prefix):],start=len(prefix)):
                        assert sha(ref['image'])==ref['image_sha256'];diagnostics.clear();start=time.perf_counter()
                        response=adapter.generate(adapter.prepare(ref['image'],ref['prompt']),max_new_tokens=ref['max_new_tokens'])
                        assert all(diagnostics.get(k,{}).get('changed_elements',0)>0 for k in ['text','vision'])
                        row={**ref,'method':'vti_author','prediction':response,'generated_tokens':adapter.last_generated_tokens,
                            'elapsed_seconds':time.perf_counter()-start,'gpu_uuid':GPU0,'dtype':'float16','attention_score_dtype':'float32',
                            'checkpoint':config['checkpoint'],'revision':config['revision'],'protocol_sha256':manifest['sha256'],
                            'parent_protocol_sha256':manifest['parent_protocol_sha256'],'author_commit':COMMIT,'artifact_sha256':artifact_digest,
                            'original_hook_file':layers.VTILayer.forward.__code__.co_filename,'original_vti_diagnostics':dict(diagnostics),
                            'attention_amendment_sha256':sha(PARENT/'fp16-attention-amendment.json')}
                        sink.write(json.dumps(row,allow_nan=False)+'\n');sink.flush()
                        state['completed_records']+=1;update(stage='generating_original_vti',benchmark=bench,benchmark_completed=i+1)
                assert len(read(path))==len(references)
                save(out/(bench+'-complete.json'),{'records':len(references),'sha256':sha(path)})
            for h in handles:h.remove()
            update(stage='complete',completed_at=time.time(),source_hashes_after=verify_author())
    except BaseException:update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
