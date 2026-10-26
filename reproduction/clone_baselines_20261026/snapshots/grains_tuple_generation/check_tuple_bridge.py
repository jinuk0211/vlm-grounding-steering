"""Compare the unchanged author hook on legacy tuples and native Tensors."""

import importlib.util
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys

import torch

from tuple_bridge import original_tuple_hook


ROOT = Path('/workspace/original-baselines-20261025/grains_gpu_retry_20261026')
spec = importlib.util.spec_from_file_location('author_steering', ROOT / 'GrAInS/src/utils/steering.py')
author = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author)


class Layer(torch.nn.Module):
    def __init__(self, legacy):
        super().__init__()
        self.legacy = legacy
        self.calls = 0

    def forward(self, values):
        self.calls += 1
        output = values.clone()
        return (output,) if self.legacy else output


class Model:
    def __init__(self, legacy):
        self.layer = Layer(legacy)
        self.model = SimpleNamespace(language_model=SimpleNamespace(layers=[self.layer]))

    def named_modules(self):
        yield 'model.language_model.layers.0', self.layer


class BridgeTests(unittest.TestCase):
    def test_tuple_oracle_prefill_decode_batch_and_zero_alpha(self):
        torch.manual_seed(51)
        vector = torch.randn(8)
        for shape in [(1, 4, 8), (1, 1, 8), (2, 4, 8)]:
            for alpha in [0.0, 10.0]:
                with self.subTest(shape=shape, alpha=alpha):
                    inputs = torch.randn(*shape)
                    legacy, native = Model(True), Model(False)
                    handles = author.add_steering_hook(legacy, 0, vector, alpha=alpha)
                    expected = legacy.layer(inputs)[0]
                    for handle in handles:
                        handle.remove()
                    with original_tuple_hook(native, 0, vector, alpha, author.add_steering_hook):
                        actual = native.layer(inputs)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(native.layer.calls, 1)
                    self.assertEqual(native.layer._forward_hooks, {})

    def test_removes_all_handles_after_error(self):
        native = Model(False)
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with original_tuple_hook(native, 0, torch.ones(8), 10.0, author.add_steering_hook):
                raise RuntimeError('deliberate')
        self.assertEqual(native.layer._forward_hooks, {})

    def test_native_qwen_cached_decode_matches_original_tuple_oracle(self):
        sys.path.insert(0, str(ROOT / 'operations'))
        from check_grains_memory import native_model

        torch.manual_seed(52)
        reference = native_model()
        native = copy.deepcopy(reference)
        vector = torch.randn(16)
        target = reference.model.language_model.layers[1]
        handle = author.add_steering_hook(reference, 1, vector, alpha=10.0)[0]
        hook = target._forward_hooks[handle.id]
        handle.remove()
        oracle = target.register_forward_hook(lambda module, inputs, output: hook(module, inputs, (output,))[0])
        tokens = torch.randint(3, 50, (1, 6))
        with torch.no_grad(), original_tuple_hook(native, 1, vector, 10.0, author.add_steering_hook):
            left = reference(input_ids=tokens, use_cache=True)
            right = native(input_ids=tokens, use_cache=True)
            torch.testing.assert_close(left.logits, right.logits, rtol=0, atol=0)
            self.assertIsNotNone(right.past_key_values)
            token = torch.tensor([[7]])
            left = reference(input_ids=token, past_key_values=left.past_key_values, use_cache=True)
            right = native(input_ids=token, past_key_values=right.past_key_values, use_cache=True)
            torch.testing.assert_close(left.logits, right.logits, rtol=0, atol=0)
            self.assertEqual(right.past_key_values.get_seq_length(), 7)
        oracle.remove()


if __name__ == '__main__':
    unittest.main(verbosity=2)
