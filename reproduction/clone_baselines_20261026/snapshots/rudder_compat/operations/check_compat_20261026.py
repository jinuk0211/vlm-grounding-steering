"""Check native image markers and unchanged, memory-batched original IG."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image
from transformers import AutoProcessor

ROOT = Path('/workspace/original-baselines-20261025')
SNAPSHOT = Path('/workspace/vlm-validation-1683e86/hf-cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5')


def original(repo, relative):
    return subprocess.check_output(
        ['git', 'show', 'HEAD:' + relative], cwd=ROOT / repo, text=True,
    )


def module_from_source(name, source):
    module = types.ModuleType(name)
    exec(compile(source, name, 'exec'), module.__dict__)
    return module


def definition(source, name):
    return next(node for node in ast.parse(source).body if getattr(node, 'name', None) == name)


class CompatibilityTests(unittest.TestCase):
    def test_native_pope_image_tokens(self):
        sys.path.insert(0, str(ROOT / 'RUDDER'))
        from Qwen.qwen_methods import qwenvl_singleturn_prompt

        processor = AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True)
        image = Image.new('RGB', (56, 56), color=(64, 128, 192))
        question = 'Is there a cat in the image?'
        prompt = qwenvl_singleturn_prompt(question)
        image_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        for count in (1, 2):
            inputs = processor(text=[prompt] * count, images=[image] * count,
                               return_tensors='pt', padding=True)
            tokens = (inputs['input_ids'] == image_id).sum(dim=1)
            expected = inputs['image_grid_thw'].prod(dim=1) // processor.image_processor.merge_size**2
            self.assertTrue(torch.equal(tokens.cpu(), expected.cpu()))
            self.assertTrue(bool((tokens > 0).all()))
        self.assertIn(question, prompt)
        self.assertTrue(prompt.endswith('<|im_start|>assistant\n'))

    def test_original_ig_twenty_nodes_with_single_example_memory(self):
        relative = 'src/attribution/gradient/utils.py'
        old = module_from_source('original_ig', original('GrAInS', relative))
        new = module_from_source('batched_ig', (ROOT / 'GrAInS' / relative).read_text())

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(4, 7)
                self.batches = []

            def forward(self, inputs_embeds, attention_mask):
                self.batches.append(inputs_embeds.shape[0])
                values = torch.sin(inputs_embeds) * attention_mask[..., None]
                return types.SimpleNamespace(logits=self.projection(values))

        torch.manual_seed(1701)
        model = TinyModel().eval()
        pos = torch.randn(1, 3, 4, requires_grad=True)
        neg = torch.randn(1, 2, 4, requires_grad=True)
        arguments = (model, pos, neg, torch.ones(1, 3), torch.ones(1, 2), 2, 5)
        expected = old.apply_integrated_gradients_contrastive(*arguments)
        self.assertEqual(model.batches, [20, 20])
        model.batches.clear()
        actual = new.apply_integrated_gradients_contrastive(*arguments)
        self.assertEqual(model.batches, [1, 1] * 20)
        for before, after in zip(expected, actual):
            torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(after.sum(-1), before.sum(-1), rtol=1e-5, atol=1e-6)

    def test_only_ig_batch_keyword_changes_the_calculation_function(self):
        relative = 'src/attribution/gradient/utils.py'
        name = 'apply_integrated_gradients_contrastive'
        before = definition(original('GrAInS', relative), name)
        after = definition((ROOT / 'GrAInS' / relative).read_text(), name)
        after.body = [child for node in after.body for child in
                      (node.body if isinstance(node, ast.With) else [node])]
        removed = 0
        for node in ast.walk(after):
            if isinstance(node, ast.Call):
                removed += sum(keyword.arg == 'internal_batch_size' for keyword in node.keywords)
                node.keywords = [keyword for keyword in node.keywords if keyword.arg != 'internal_batch_size']
        self.assertEqual(removed, 1)
        self.assertEqual(ast.dump(after), ast.dump(before))

    def test_card_gate_pca_and_normalization_match_public_source(self):
        relative = 'Qwen/qwen_methods.py'
        old = original('RUDDER', relative)
        new = (ROOT / 'RUDDER' / relative).read_text()
        for name in ('compute_card_vector_batch', 'BayesianGatingHookMaskedDynamic', 'get_qwenvl_self_attn'):
            self.assertEqual(ast.dump(definition(old, name)), ast.dump(definition(new, name)))
        relative = 'src/utils/steering.py'
        self.assertEqual((ROOT / 'GrAInS' / relative).read_text(), original('GrAInS', relative))

    def test_native_qwen_ig_chunking_matches_original(self):
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
        import copy

        torch.manual_seed(31)
        config = Qwen2_5_VLConfig(
            text_config={'vocab_size': 64, 'hidden_size': 16, 'intermediate_size': 32,
                         'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 2,
                         'rope_parameters': {'rope_type': 'default', 'mrope_section': [1, 1, 2]},
                         'bos_token_id': 1, 'eos_token_id': 2},
            vision_config={'depth': 1, 'hidden_size': 16, 'intermediate_size': 32, 'num_heads': 2,
                           'patch_size': 2, 'temporal_patch_size': 1, 'spatial_merge_size': 2,
                           'out_hidden_size': 16, 'window_size': 8, 'fullatt_block_indexes': [0]},
            image_token_id=61, video_token_id=63, vision_start_token_id=60, vision_end_token_id=62)
        model = Qwen2_5_VLForConditionalGeneration(config).eval()
        pos = model.get_input_embeddings()(torch.randint(3, 50, (1, 72))).detach().requires_grad_(True)
        neg = model.get_input_embeddings()(torch.randint(3, 50, (1, 65))).detach().requires_grad_(True)
        relative = 'src/attribution/gradient/utils.py'
        old = module_from_source('original_native_ig', original('GrAInS', relative))
        new = module_from_source('batched_native_ig', (ROOT / 'GrAInS' / relative).read_text())
        arguments = (pos, neg, torch.ones(1, 72, dtype=torch.long), torch.ones(1, 65, dtype=torch.long), 7, 9)
        expected = old.apply_integrated_gradients_contrastive(model, *arguments)
        actual = new.apply_integrated_gradients_contrastive(copy.deepcopy(model), *arguments)
        for before, after in zip(expected, actual):
            torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-6)
            self.assertEqual(torch.argsort(before.sum(-1), descending=True)[:50].tolist(),
                             torch.argsort(after.sum(-1), descending=True)[:50].tolist())

    def test_vlm_loader_uses_gpu_offload_and_original_fp32(self):
        spec = importlib.util.spec_from_file_location('original_vlm_loader', ROOT / 'GrAInS/src/utils/model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = MagicMock(device=torch.device('cuda:0'))
        model.parameters.side_effect = lambda: iter([torch.zeros(1, dtype=torch.float32)])
        accelerator = MagicMock()
        accelerator.prepare.side_effect = lambda model, processor: (model, processor)
        with patch.object(module, 'Accelerator', return_value=accelerator) as constructor, \
             patch.object(module.AutoProcessor, 'from_pretrained') as processor, \
             patch.object(module.AutoModelForVision2Seq, 'from_pretrained', return_value=model) as loader:
            module.load_vlm_model_and_processor(str(SNAPSHOT))
            constructor.assert_called_once_with(mixed_precision='no')
            loader.assert_called_once_with(str(SNAPSHOT), torch_dtype=torch.float32,
                                           device_map='auto', max_memory={0: '14GiB', 'cpu': '48GiB'})
            accelerator.prepare.assert_called_once_with(model, processor.return_value)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CompatibilityTests))
    print(json.dumps({'tests': result.testsRun, 'failures': len(result.failures),
                      'errors': len(result.errors), 'skips': len(result.skipped),
                      'actual_7b_inference': False}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
