"""Execute pinned author functions; bridge only Gemma I/O, module paths and dtype."""
from __future__ import annotations
import ast
from contextlib import contextmanager
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm

ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
AUTH=ROOT/'upstream-author-20261027-revis-vti'
sys.path.insert(0,str(ROOT/'.author-deps'))

def load_definitions(path,names,environment):
    tree=ast.parse(Path(path).read_text())
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    assert {n.name for n in selected}==set(names)
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future]+selected,type_ignores=[]))
    namespace={'__name__':'author_bridge_extracted',**environment}
    exec(compile(module,str(path),'exec'),namespace)
    return namespace

def sources():
    import subprocess
    result={}
    for name,expected in [('Revis','75f4d29d402d4c1c4dbb2c69e88e3aa50cd0c9cc'),('VTI','bb5dfee33a9e555f22af9faa7b3495c0ac3df69e')]:
        directory=AUTH/name
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=directory,text=True).strip();assert commit==expected
        subprocess.run(['git','diff','--exit-code','HEAD'],cwd=directory,check=True,capture_output=True)
        result[name]={'commit':commit,'files':{p.relative_to(directory).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob('*.py') if '.git' not in p.parts}}
    return result

def libraries():
    from sklearn.decomposition import PCA as SklearnPCA
    env={'torch':torch,'nn':nn,'F':F,'np':np,'Image':Image,'random':random,'os':os,'tqdm':tqdm,'PCA':SklearnPCA}
    revis=load_definitions(AUTH/'Revis/steering_utils/vector_calculator.py',[
        'get_random_unknown_response','get_pure_direction_with_pca','_get_prompt_content','get_hidden_states_retrospective','_calculate_das_vectors_internal'],env)
    revis.update(load_definitions(AUTH/'Revis/steering_utils/llm_layers.py',['MADRSteeringHook_visual_only'],env))
    vti=load_definitions(AUTH/'VTI/vti_utils/pca.py',['svd_flip','PCA'],env)
    vti=load_definitions(AUTH/'VTI/vti_utils/icv_utils.py',['mask_patches','obtain_textual_vti','obtain_visual_vti'],{**env,'PCA':vti['PCA']})
    vti.update(load_definitions(AUTH/'VTI/vti_utils/llm_layers.py',['VTILayer'],env))
    return revis,vti

def tensor_output(value):return value[0] if isinstance(value,(tuple,list)) else value
def replace_output(value,tensor):return (tensor,*value[1:]) if isinstance(value,tuple) else tensor

def extract(adapter,output,notify):
    output=Path(output);data=output/'data';revis,vti=libraries();random.seed(42);np.random.seed(42);torch.manual_seed(42)
    demos=[json.loads(s) for s in (data/'author-preferences.jsonl').read_text().splitlines()]
    source_rows={r['id']:r for r in [json.loads(s) for s in (AUTH/'VTI/experiments/data/hallucination_vti_demos.jsonl').read_text().splitlines()]}
    for row in demos:
        assert all(row[key]==value for key,value in source_rows[row['id']].items())
        assert hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest()==row['image_sha256']
    # Author template/state extraction stays unchanged. The input image is resized
    # exactly as in the shared Gemma adapter before reaching that function.
    original_states=revis['get_hidden_states_retrospective'];memo={};positive=[];negative=[]
    def gemma_states(model,processor,question,answer,pil_image,model_name='Gemma4'):
        if pil_image is not None:pil_image=pil_image.resize((448,448),Image.Resampling.BICUBIC)
        return original_states(model,processor,question,answer,pil_image,model_name='Gemma4')
    def cached_states(model,processor,question,answer,pil_image,model_name='Gemma4'):
        key=(question,answer,hashlib.sha256(pil_image.tobytes()).hexdigest() if pil_image is not None else None)
        if key not in memo:
            memo[key]=gemma_states(adapter.model,adapter.processor,question,answer,pil_image)
        return memo[key]
    revis['get_hidden_states_retrospective']=cached_states
    dataset=[{'image_id':Path(r['image_path']).stem,'question':r['question'],'caption':r['value'],'h_caption':r['h_value']} for r in demos]
    revis['load_steering_data']=lambda _:dataset
    # The original extractor reads this metadata to allocate layer containers.
    metadata_model=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=len(adapter.layers)))
    notify(stage='extracting_author_revis',extraction_images=len(demos))
    vectors=revis['_calculate_das_vectors_internal'](metadata_model,adapter.processor,'author-preferences',str(data/'author-images'),'Gemma4')
    assert set(range(1,len(adapter.layers)+1)).issubset(vectors)
    # VTI's original get_prompts appends the caption to a generation prefix;
    # its last captured token is a caption token, without an appended EOS.
    def vti_text_states(row,answer):
        with Image.open(row['image_path']) as source:image=source.convert('RGB').resize((448,448),Image.Resampling.BICUBIC)
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':row['question']}]}]
        prefix=adapter.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        batch=adapter.processor(text=[prefix+answer],images=[image],return_tensors='pt')
        batch={k:v.to(device=adapter.device,dtype=adapter.dtype if v.is_floating_point() else v.dtype) for k,v in batch.items()}
        with torch.no_grad():result=adapter.model(**batch,output_hidden_states=True,use_cache=False,return_dict=True)
        return torch.cat([h[:,-1].detach().float().cpu() for h in result.hidden_states],dim=0)
    for row in demos:
        positive.append(vti_text_states(row,row['value']))
        negative.append(vti_text_states(row,row['h_value']))
    text_states=[(n,p) for n,p in zip(negative,positive)]
    vti['get_hiddenstates']=lambda model,inputs,image_tensor:text_states
    text_direction,_=vti['obtain_textual_vti'](None,None,None,rank=1)
    del memo,text_states,positive,negative
    notify(stage='calibrating_author_revis')
    records=[json.loads(s) for s in (data/'author-calibration.jsonl').read_text().splitlines()]
    risks={layer:[] for layer in range(len(adapter.layers))};categories=[];calrows=[]
    for index,row in enumerate(records):
        captured={};handles=[]
        for layer,module in enumerate(adapter.layers):
            vector=vectors[layer+1]['visual'].to(adapter.device)
            def capture(_module,_args,value,layer=layer,vector=vector):
                if layer not in captured:
                    h=tensor_output(value)[:,-1,:]
                    captured[layer]=float(-(F.normalize(h,dim=-1)@F.normalize(vector.to(h),dim=-1)).item())
            handles.append(module.register_forward_hook(capture))
        try:pred=adapter.generate(adapter.prepare(row['image'],row['prompt']),max_new_tokens=5).strip().lower()
        finally:
            for handle in handles:handle.remove()
        choice='yes' if 'yes' in pred else ('no' if 'no' in pred else 'unknown')
        category='TP' if row['label']=='yes' and choice=='yes' else 'TN' if row['label']=='no' and choice=='no' else 'FP' if row['label']=='no' and choice=='yes' else 'Other'
        categories.append(category);calrows.append({**row,'prediction':pred,'category':category,'risks':captured})
        for layer,value in captured.items():risks[layer].append(value)
        if index%20==0:notify(calibration_completed=index+1,calibration_total=len(records))
    candidates=[]
    for layer in reversed(range(10,len(adapter.layers))):
        factual=[risk for risk,category in zip(risks[layer],categories) if category in ['TP','TN']]
        hall=[risk for risk,category in zip(risks[layer],categories) if category=='FP']
        if factual and hall:
            separation=float(np.mean(hall)-np.mean(factual));tau=float(np.percentile(factual,90.0))
            candidates.append({'layer':layer,'separation':separation,'threshold':tau})
    valid=[c for c in candidates if c['separation']>0]
    if not valid:raise ValueError('No author REVIS calibration layer has positive separation; never calibrate on test labels')
    selected=valid[0]
    revis_artifact={'vectors':vectors,'layer':selected['layer'],'threshold':selected['threshold'],'alpha':1.6,'gamma':1.0,'calibration_candidates':candidates,'calibration_categories':{k:categories.count(k) for k in ['TP','TN','FP','Other']}}
    (output/'author-revis-calibration.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in calrows))
    torch.save(revis_artifact,output/'author-revis.pt')
    notify(stage='extracting_author_vti_vision',vti_trials_per_image=50)
    vision=adapter.model.model.vision_tower;vision_states=[]
    for index,row in enumerate(demos):
        batch=adapter.prepare(row['image_path'],row['question']);pixels=batch['pixel_values'];positions=batch['image_position_ids']
        valid_positions=(positions[0]>=0).all(-1);pos=positions[0,valid_positions];real=pixels[0,valid_positions]
        gh,gw=(int(pos[:,j].max())+1 for j in [0,1]);assert gh*gw==len(real)
        patch=int(adapter.processor.image_processor.patch_size);assert real.shape[-1]==patch*patch*3
        # Undo the processor's exact patchify, run author's mean-patch mask,
        # then restore its original layout. Padding and positions stay intact.
        order=(pos[:,0]*gw+pos[:,1]).argsort();grid=real[order].reshape(gh,gw,patch,patch,3)
        chw=grid.permute(4,0,2,1,3).reshape(3,gh*patch,gw*patch)
        def forward_vision(pixel):
            with torch.no_grad():result=vision(pixel_values=pixel,pixel_position_ids=positions,output_hidden_states=True,return_dict=True)
            assert result.hidden_states is not None
            return tuple(h.detach().cpu() for h in result.hidden_states)
        original=forward_vision(pixels);total=None
        for trial in range(50):
            ids=torch.randperm(gh*gw)[:int(.99*gh*gw)]
            masked=vti['mask_patches'](chw,ids,patch_size=patch)
            flattened=masked.reshape(3,gh,patch,gw,patch).permute(1,3,2,4,0).reshape(gh*gw,-1)
            changed=pixels.clone();inverse=order.argsort();changed[0,valid_positions]=flattened[inverse]
            states=forward_vision(changed)
            if total is None:total=[s.float() for s in states]
            else:
                for accumulator,state in zip(total,states):accumulator.add_(state.float())
        # Author averages corruptions before concatenating layer states.
        corrupted=torch.cat([(s/50).to(original[j].dtype) for j,s in enumerate(total)],dim=0)
        clean=torch.cat(original,dim=0);vision_states.append((corrupted,clean))
        notify(vision_extraction_completed=index+1,vision_extraction_total=len(demos))
        del batch,pixels,states,total,original
    vti['get_visual_hiddenstates']=lambda model,image_tensor,model_is_llaval=True:vision_states
    visual_direction,_=vti['obtain_visual_vti'](None,None,rank=1)
    vti_artifact={'text_direction':text_direction[1:].contiguous(),'vision_direction':visual_direction[1:].contiguous(),'alpha_text':.9,'alpha_vision':.9,'mask_ratio':.99,'num_trials':50,'vision_padding_note':'Gemma padded patch positions preserved; valid spatial patch masking only'}
    assert vti_artifact['text_direction'].shape[0]==len(adapter.layers)
    assert vti_artifact['vision_direction'].shape[0]==len(adapter.vision_layers)
    for value in [vti_artifact['text_direction'],vti_artifact['vision_direction']]:assert torch.isfinite(value).all() and value.norm()>0
    torch.save(vti_artifact,output/'author-vti.pt')
    return revis_artifact,vti_artifact

@contextmanager
def revis_context(adapter,artifact,diagnostics=None):
    revis,_=libraries();hook=revis['MADRSteeringHook_visual_only']({'visual':artifact['vectors'][artifact['layer']+1]['visual']},artifact['layer'],artifact['alpha'],artifact['threshold'],artifact['gamma'])
    handles=[];module=adapter.layers[artifact['layer']]
    def apply(m,args,out):
        new=hook.hook_fn(m,args,out)
        if diagnostics is not None:
            before=tensor_output(out);after=tensor_output(new);diagnostics['calls']=diagnostics.get('calls',0)+1;diagnostics['changed_elements']=diagnostics.get('changed_elements',0)+int((before!=after).sum())
        return new
    handles.append(module.register_forward_hook(apply))
    try:yield
    finally:
        for handle in handles:handle.remove()

@contextmanager
def vti_context(adapter,artifact,diagnostics=None):
    _,vti=libraries();handles=[]
    for modules,directions,alpha in [(adapter.layers,artifact['text_direction'],artifact['alpha_text']),(adapter.vision_layers,artifact['vision_direction'],artifact['alpha_vision'])]:
        for index,layer in enumerate(modules):
            assert hasattr(layer,'mlp')
            intervention=vti['VTILayer'](directions[index].unsqueeze(0).to(adapter.device),[alpha])
            def apply(_module,_args,value,intervention=intervention):
                tensor=tensor_output(value);changed=intervention(tensor).to(tensor.dtype)
                assert changed.shape==tensor.shape
                if diagnostics is not None:
                    assert torch.isfinite(changed).all()
                    diagnostics['calls']=diagnostics.get('calls',0)+1;diagnostics['changed_elements']=diagnostics.get('changed_elements',0)+int((tensor!=changed).sum())
                return replace_output(value,changed)
            handles.append(layer.mlp.register_forward_hook(apply))
    try:yield
    finally:
        for handle in handles:handle.remove()
