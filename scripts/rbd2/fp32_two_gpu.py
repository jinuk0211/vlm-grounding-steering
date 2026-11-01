"""Qwen2.5-VL in float32 over two RTX 2080 Ti (22 GB each).

Turing has no bf16 support: bf16 decoding here produced corrupted tokens (" addCriterion" inside words), and fp16
overflows in Qwen2.5-VL (garbage from the first token). float32 does not fit one card, so the decoder layers are
split over cuda:0 and cuda:1. Vision tower, embeddings and lm_head stay on cuda:0, so the logits come out on
adapter.device, where rbd_decode's Belief keeps its V_obj ids. Only the loader changes; ModelAdapter is the repo's.
"""
import torch
import vlm_grounding.backend as B

_device_count=torch.cuda.device_count
ON_GPU0=12   # of 28 decoder layers; cuda:0 also holds vision (~2.7 GB), embeddings and lm_head (~2.2 GB each)

def device_map(n_layers):
    m={'model.visual':0,'model.language_model.embed_tokens':0,'model.language_model.rotary_emb':0,
       'model.language_model.norm':1,'lm_head':0}
    m.update({f'model.language_model.layers.{i}':0 if i<ON_GPU0 else 1 for i in range(n_layers)})
    return m

def load_adapter(config):
    from transformers import AutoConfig,AutoProcessor,Qwen2_5_VLForConditionalGeneration
    assert config['family']=='qwen2_5_vl' and config['dtype']=='float32'
    torch.cuda.device_count=_device_count
    shared={'revision':str(config.get('revision','main')),'local_files_only':bool(config.get('local_files_only',False)),
            'trust_remote_code':False}
    n=AutoConfig.from_pretrained(config['checkpoint'],**shared).text_config.num_hidden_layers
    processor=AutoProcessor.from_pretrained(config['checkpoint'],**shared)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(config['checkpoint'],dtype=torch.float32,
        attn_implementation=str(config.get('attn_implementation','eager')),device_map=device_map(n),**shared)
    ad=B.ModelAdapter(model,processor,config)
    assert ad.device==model.lm_head.weight.device==torch.device('cuda',0)
    return ad

def install():
    """Swap the loader; callers assert a single visible GPU before loading, so report one until load_adapter runs."""
    B.load_adapter=load_adapter
    torch.cuda.device_count=lambda:1
