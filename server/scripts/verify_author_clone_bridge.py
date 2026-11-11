"""Check the tuple bridge and bounded Captum schedule against literal originals."""
import json
from types import SimpleNamespace

import torch
from author_clone_bridge import ROOT, NamedLayerView, grains_context, load_authors, verify_clone_sources


def main():
    authors=load_authors()
    results=[]
    torch.manual_seed(42)
    for dtype in [torch.float32,torch.bfloat16]:
        for tokens in [1,5]:
            value=torch.randn(1,tokens,8).to(dtype)
            vector=torch.randn(8)
            direct=torch.nn.Identity()
            handles=authors.steering.add_steering_hook(NamedLayerView(direct,27),27,vector,alpha=10.0)
            expected=direct((value.clone(),))[0]
            for h in handles:h.remove()
            wrapped=torch.nn.Identity();adapter=SimpleNamespace(layers={27:wrapped});diag={}
            with grains_context(adapter,vector,authors,diag):
                actual=wrapped(value.clone())
            assert torch.equal(expected,actual)
            assert diag['calls']==1
            results.append({'check':'literal_tuple_vs_native_bridge','dtype':str(dtype),'tokens':tokens,'bitwise_equal':True})
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.projection=torch.nn.Linear(8,7)
        def forward(self,inputs_embeds,attention_mask=None):
            # Token mixing makes the complete prompt attribution nonzero.
            values=inputs_embeds.cumsum(1).tanh()
            return SimpleNamespace(logits=self.projection(values))
    toy=Toy().eval()
    pos=torch.randn(1,5,8,requires_grad=True);neg=torch.randn(1,4,8,requires_grad=True)
    args=(toy,pos,neg,torch.ones(1,5),torch.ones(1,4),2,3)
    bounded=authors.ig_utils.IntegratedGradients
    original=bounded.__bases__[0]
    authors.ig_utils.IntegratedGradients=original
    full=authors.grad.apply_integrated_gradients_contrastive(*args,steps=20)
    authors.ig_utils.IntegratedGradients=bounded
    chunked=authors.grad.apply_integrated_gradients_contrastive(*args,steps=20)
    for a,b in zip(full,chunked):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)
    results.append({'check':'Captum_20_full_vs_one_sample_schedule','max_abs_error':max(float((a-b).abs().max()) for a,b in zip(full,chunked))})
    report={'status':'passed','sources':verify_clone_sources(),'checks':results}
    (ROOT/'outputs/author-clone-bridge-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','checks':results}))


if __name__=='__main__':main()
