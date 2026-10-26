"""Recompute decoder activations during the cloned, unchanged IG objective."""

from contextlib import contextmanager

import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop


@contextmanager
def checkpoint_ig(model):
    """Bound saved activations without changing eval mode, precision, or IG nodes.

    Wrap the complete dispatched forward, including Accelerate pre/post hooks.
    Full recomputation is required so its post-hook releases offloaded weights.
    KV caches are disabled only in this context to avoid mutating a cache twice.
    """
    layers = model.model.language_model.layers
    if model.training or any(layer.training for layer in layers):
        raise RuntimeError('IG checkpointing requires the original eval mode')
    configurations = {id(config): config for config in
                      (model.config, model.config.text_config, model.model.language_model.config)}
    absent = object()
    cache_settings = [(config, getattr(config, 'use_cache', absent)) for config in configurations.values()]
    originals = [(layer, layer.__dict__.get('forward', absent)) for layer in layers]
    calls = {str(index): 0 for index in range(len(layers))}

    def wrapped(original, index):
        def recomputed(*args, **kwargs):
            calls[index] += 1
            return original(*args, **kwargs)

        def forward(*args, **kwargs):
            if not torch.is_grad_enabled():
                return original(*args, **kwargs)
            return checkpoint(recomputed, *args, use_reentrant=False, **kwargs)

        return forward

    try:
        for config, _ in cache_settings:
            config.use_cache = False
        for index, layer in enumerate(layers):
            layer.forward = wrapped(layer.forward, str(index))
        with set_checkpoint_early_stop(False):
            yield calls
    finally:
        for layer, original in originals:
            if original is absent:
                del layer.forward
            else:
                layer.forward = original
        for config, original in cache_settings:
            if original is absent:
                del config.use_cache
            else:
                config.use_cache = original
