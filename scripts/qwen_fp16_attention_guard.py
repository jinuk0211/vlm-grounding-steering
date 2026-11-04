"""FP16 weights/outputs with FP32 QK scores to prevent eager-attention overflow."""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import torch

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@contextmanager
def stable_fp16_attention(diagnostics):
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen
    original=qwen.eager_attention_forward
    diagnostics.update(role='FP16 weights and outputs; FP32 QK, mask and softmax arithmetic',calls=0,
        author_steering_changed=False,source_sha256=digest(__file__),
        transformers_model_source_sha256=digest(qwen.__file__),
        original_attention_source_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest())
    def forward(module,query,key,value,attention_mask,scaling,dropout=0.0,**kwargs):
        if query.dtype!=torch.float16:
            return original(module,query,key,value,attention_mask,scaling,dropout,**kwargs)
        keys=qwen.repeat_kv(key,module.num_key_value_groups)
        values=qwen.repeat_kv(value,module.num_key_value_groups)
        if diagnostics.get('trace_native_half_scores') and getattr(module,'layer_idx',None) is not None:
            raw_half=torch.matmul(query,keys.transpose(2,3))
            bad=int((~torch.isfinite(raw_half)).sum())
            if bad:diagnostics.setdefault('native_half_score_overflows',[]).append({'layer':module.layer_idx,'elements':bad})
            del raw_half
        weights=torch.matmul(query.float(),keys.float().transpose(2,3))*scaling
        if attention_mask is not None:weights=weights+attention_mask.float()
        probabilities=torch.nn.functional.softmax(weights,dim=-1,dtype=torch.float32).to(query.dtype)
        probabilities=torch.nn.functional.dropout(probabilities,p=dropout,training=module.training)
        output=torch.matmul(probabilities,values).transpose(1,2).contiguous()
        assert torch.isfinite(output).all(),'Nonfinite output despite FP32 attention scores'
        diagnostics['calls']+=1
        return output,probabilities
    qwen.eager_attention_forward=forward
    try:yield
    finally:qwen.eager_attention_forward=original

def verify_arithmetic():
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen
    from types import SimpleNamespace
    module=SimpleNamespace(num_key_value_groups=1,training=False)
    query=torch.full((1,1,3,4),200,dtype=torch.float16)
    key=query.clone();value=torch.tensor([[[[1.,2.,3.,4.],[2.,3.,4.,5.],[3.,4.,5.,6.]]]],dtype=torch.float16)
    mask=torch.full((1,1,3,3),float('-inf'),dtype=torch.float16).triu(1)
    native,_=qwen.eager_attention_forward(module,query,key,value,mask,0.5)
    assert not torch.isfinite(native).all()
    d={}
    with stable_fp16_attention(d):
        actual,prob=qwen.eager_attention_forward(module,query,key,value,mask,0.5)
    ref,_=qwen.eager_attention_forward(module,query.float(),key.float(),value.float(),mask.float(),0.5)
    torch.testing.assert_close(actual.float(),ref,rtol=1e-3,atol=1e-3)
    assert prob[0,0,0,1:].eq(0).all() and actual.dtype==torch.float16
    return {'legacy_overflow_reproduced':True,'fp32_reference_matched':True,'causal_mask_preserved':True,
        'output_dtype':'torch.float16','source_sha256':digest(__file__)}
