"""Native Qwen inputs and device transport around unchanged author functions.

This module is preparation for the run after Gemma completes. No model is
loaded on import. Unlike Gemma, Qwen does not need a token-identity PLE proxy.
"""
import contextlib

import numpy as np
from PIL import Image

from author_clone_bridge import grains_context as original_grains_context


class QwenAttributionAPI:
    """Co-locate returned hidden states for the author's unchanged torch.stack.

    The original helper stacks last-token states across layers before moving
    them to CPU. Model-parallel Qwen can return those states on two devices.
    Copying them to the input device preserves values, dtype, shape and order.
    Native forward, gradients, token IDs and logits are otherwise unchanged.
    """
    def __init__(self, model):
        self.native = model
        self.moved_hidden_state_tensors = 0

    def __getattr__(self, name):
        return getattr(self.native, name)

    def eval(self):
        self.native.eval()
        return self

    def get_input_embeddings(self):
        return self.native.get_input_embeddings()

    def __call__(self, *args, **kwargs):
        outputs = self.native(*args, **kwargs)
        if kwargs.get('output_hidden_states') and outputs.hidden_states is not None:
            device = next(self.native.parameters()).device
            states = outputs.hidden_states
            self.moved_hidden_state_tensors += sum(value.device != device for value in states)
            outputs.hidden_states = tuple(value.to(device) for value in states)
        return outputs


def extract_one_qwen(adapter, record, authors):
    assert adapter.model_type == 'qwen2_5_vl'
    with Image.open(record['image']) as source:
        image = source.convert('RGB').resize((448, 448), Image.Resampling.BICUBIC)
    prompt = adapter.processor.apply_chat_template([{'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': record['prompt']}]}],
        tokenize=False, add_generation_prompt=True)
    api = QwenAttributionAPI(adapter.model)
    values = authors.ablation(api, adapter.processor, image, prompt,
                             record['chosen'], record['rejected'],
                             top_k=50, top='pos', attribution_method='integrated_gradients')
    for name, value in values.items():
        assert np.isfinite(value).all(), name
    return values, {'hidden_state_tensors_moved': api.moved_hidden_state_tensors,
                    'native_qwen_forward': True, 'gemma_ple_proxy_used': False,
                    'original_token_only_ig_and_ablation_preserved': True}


@contextlib.contextmanager
def grains_on_qwen_layer_device(adapter, vector, authors, diagnostics=None):
    assert adapter.model_type == 'qwen2_5_vl' and len(adapter.layers) == 28
    device = next(adapter.layers[27].parameters()).device
    with original_grains_context(adapter, vector.to(device), authors, diagnostics):
        yield
    if diagnostics is not None:
        diagnostics['steering_vector_device'] = str(device)
        diagnostics['architecture'] = 'native_qwen2_5_vl'

