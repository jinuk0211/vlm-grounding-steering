"""Prove input IG is preserved through eval checkpointing and GPU/CPU dispatch."""

import copy
import importlib.util
from pathlib import Path
import unittest

import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from accelerate import dispatch_model

from grains_memory import checkpoint_ig


def native_model():
    config = Qwen2_5_VLConfig(
        text_config={'vocab_size': 64, 'hidden_size': 16, 'intermediate_size': 32,
                     'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 2,
                     'rope_parameters': {'rope_type': 'default', 'mrope_section': [1, 1, 2]},
                     'bos_token_id': 1, 'eos_token_id': 2},
        vision_config={'depth': 1, 'hidden_size': 16, 'intermediate_size': 32, 'num_heads': 2,
                       'patch_size': 2, 'temporal_patch_size': 1, 'spatial_merge_size': 2,
                       'out_hidden_size': 16, 'window_size': 8, 'fullatt_block_indexes': [0]},
        image_token_id=61, video_token_id=63, vision_start_token_id=60, vision_end_token_id=62)
    return Qwen2_5_VLForConditionalGeneration(config).eval().requires_grad_(False)


class MemoryTests(unittest.TestCase):
    def test_native_ig_matches_with_dispatched_recomputation(self):
        self.assertTrue(torch.cuda.is_available())
        torch.manual_seed(77)
        source = native_model()
        reference = copy.deepcopy(source).cuda()
        dispatched = dispatch_model(source, device_map={
            'model.visual': 0, 'model.language_model.embed_tokens': 0,
            'model.language_model.layers.0': 0, 'model.language_model.layers.1': 'cpu',
            'model.language_model.norm': 'cpu', 'model.language_model.rotary_emb': 'cpu',
            'lm_head': 'cpu',
        })
        observations = {}

        def observe(module, name):
            hook = module._hf_hook
            original_pre, original_post = hook.pre_forward, hook.post_forward
            counts = {'pre': 0, 'post': 0}
            observations[name] = counts

            def pre(module, *args, **kwargs):
                result = original_pre(module, *args, **kwargs)
                counts['pre'] += 1
                self.assertTrue(all(p.device.type == 'cuda' and not p.requires_grad
                                    for p in module.parameters()))
                return result

            def post(module, output):
                result = original_post(module, output)
                counts['post'] += 1
                self.assertTrue(all(p.device.type == 'meta' for p in module.parameters()))
                return result

            hook.pre_forward, hook.post_forward = pre, post

        observe(dispatched.model.language_model.layers[1].mlp.up_proj, 'decoder')
        observe(dispatched.lm_head, 'lm_head')
        module_path = Path(__file__).resolve().parents[1] / 'GrAInS/src/attribution/gradient/utils.py'
        spec = importlib.util.spec_from_file_location('clone_ig', module_path)
        ig = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ig)
        pos = torch.randn(1, 72, 16, device='cuda', requires_grad=True)
        neg = torch.randn(1, 65, 16, device='cuda', requires_grad=True)
        arguments = (pos, neg, torch.ones(1, 72, dtype=torch.long, device='cuda'),
                     torch.ones(1, 65, dtype=torch.long, device='cuda'), 7, 9)
        expected = ig.apply_integrated_gradients_contrastive(reference, *arguments)
        original_forward = dispatched.model.language_model.layers[1].forward
        with checkpoint_ig(dispatched) as calls:
            self.assertFalse(dispatched.training)
            self.assertFalse(dispatched.config.text_config.use_cache)
            actual = ig.apply_integrated_gradients_contrastive(dispatched, *arguments)
        self.assertIs(dispatched.model.language_model.layers[1].forward, original_forward)
        self.assertTrue(dispatched.config.text_config.use_cache)
        self.assertEqual(calls, {'0': 80, '1': 80})
        self.assertEqual(observations, {'decoder': {'pre': 80, 'post': 80},
                                        'lm_head': {'pre': 40, 'post': 40}})
        self.assertTrue(all(p.device.type == 'meta' for p in
                            dispatched.model.language_model.layers[1].parameters()))
        for before, after in zip(expected, actual):
            torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
            self.assertTrue(bool(torch.isfinite(after).all()))
            self.assertGreater(after.abs().sum().item(), 0)
            self.assertEqual(torch.argsort(before.sum(-1), descending=True)[:50].tolist(),
                             torch.argsort(after.sum(-1), descending=True)[:50].tolist())
        with torch.no_grad():
            before = reference(inputs_embeds=pos, attention_mask=arguments[2]).logits
            with checkpoint_ig(dispatched):
                after = dispatched(inputs_embeds=pos, attention_mask=arguments[2]).logits
        torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)

    def test_restores_forward_cache_and_eval_on_error(self):
        model = native_model()
        layer = model.model.language_model.layers[0]
        original_function = layer.forward.__func__
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with checkpoint_ig(model):
                raise RuntimeError('deliberate')
        self.assertTrue(model.config.text_config.use_cache)
        self.assertIs(layer.forward.__func__, original_function)
        self.assertFalse(model.training)
        self.assertFalse(layer.training)


if __name__ == '__main__':
    unittest.main(verbosity=2)
