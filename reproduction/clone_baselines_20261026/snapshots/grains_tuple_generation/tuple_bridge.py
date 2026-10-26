"""Preserve the original GrAInS decoder tuple contract on native Qwen 5.x."""

from contextlib import contextmanager

import torch


@contextmanager
def original_tuple_hook(model, layer_index, vector, alpha, add_hook):
    layer = model.model.language_model.layers[layer_index]
    handles = []
    original_hooks = set(layer._forward_hooks)

    def wrap(module, inputs, output):
        if not torch.is_tensor(output):
            raise TypeError('Expected native Qwen Tensor decoder output')
        return (output,)

    def unwrap(module, inputs, output):
        if not isinstance(output, tuple) or len(output) != 1:
            raise TypeError('Original GrAInS hook changed the decoder tuple contract')
        return output[0]

    try:
        handles.append(layer.register_forward_hook(wrap))
        steering = add_hook(model, layer_index, vector, alpha=alpha)
        handles.extend(steering)
        if len(steering) != 1 or steering[0].id not in layer._forward_hooks:
            raise RuntimeError('Original steering hook did not attach to the selected decoder')
        handles.append(layer.register_forward_hook(unwrap))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
        if set(layer._forward_hooks) != original_hooks:
            raise RuntimeError('Decoder hooks were not restored')
