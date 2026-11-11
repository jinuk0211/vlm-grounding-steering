"""Compare the Gemma hook bridge against the unmodified author operators."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from author_revis_vti_bridge import libraries,revis_context,vti_context,sources

class Layer(nn.Module):
    def __init__(self):super().__init__();self.mlp=nn.Identity()
    def forward(self,x):return self.mlp(x)

def main():
    rv,vt=libraries();torch.manual_seed(42);checks=[]
    for dtype in [torch.float32,torch.bfloat16]:
        for tokens in [1,5]:
            x=torch.randn(1,tokens,8).to(dtype);d=torch.randn(8)
            adapter=SimpleNamespace(layers=[Layer()],vision_layers=[Layer()],device=torch.device('cpu'))
            ra={'vectors':{1:{'visual':d}},'layer':0,'threshold':-.2,'alpha':1.6,'gamma':1.0}
            expected=rv['MADRSteeringHook_visual_only']({'visual':d},0,1.6,-.2,1.).hook_fn(None,None,x)
            with revis_context(adapter,ra):actual=adapter.layers[0](x)
            assert torch.equal(expected,actual)
            va={'text_direction':d.unsqueeze(0),'vision_direction':torch.randn(1,tokens,8),'alpha_text':.9,'alpha_vision':.9}
            direct=vt['VTILayer'](d.unsqueeze(0),[.9])(x).to(dtype)
            with vti_context(adapter,va):actual=adapter.layers[0](x)
            assert torch.equal(direct,actual)
            assert torch.equal(adapter.layers[0](x),x),'Hooks leaked after context exit'
            checks.append({'dtype':str(dtype),'tokens':tokens,'revis_bitwise_equal':True,'vti_bitwise_equal_after_dtype_bridge':True,'hook_cleanup':True})
    result={'status':'passed','checks':checks,'author_sources':sources(),'adaptations':['Gemma module paths and BF16 dtype restoration','Author source definitions loaded unchanged with AST to avoid unrelated model imports']}
    out=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/gpu2-author-revis-vti-bridge-validation.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','checks':len(checks),'output':str(out)}))
if __name__=='__main__':main()
