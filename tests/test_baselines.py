import unittest
from types import SimpleNamespace

import torch
from torch import nn

from vlm_grounding.baselines import (
    beta_gate, build_baseline, extract_baseline, norm_preserving_add, orthogonalize,
    principal_direction, revis_update, sequence_log_probability,
)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Identity()

    def forward(self, hidden_states):
        return (hidden_states + self.projection(hidden_states * 0.25), "unchanged-cache")


class CausalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(4))

    def forward(self, hidden_states):
        denominators = torch.arange(1, hidden_states.shape[1] + 1).view(1, -1, 1)
        context = hidden_states.cumsum(1) / denominators
        return hidden_states + self.projection(context)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.vision = nn.Linear(4, 4, bias=False)
        self.blocks = nn.ModuleList([CausalBlock(), CausalBlock()])
        self.readout = nn.Linear(4, 8, bias=False)
        with torch.no_grad():
            self.vision.weight.copy_(torch.eye(4))

    def forward(self, input_ids, pixel_values=None, labels=None):
        hidden = self.embedding(input_ids)
        if pixel_values is not None:
            hidden = torch.cat((self.vision(pixel_values).unsqueeze(0), hidden), dim=1)
        for block in self.blocks:
            hidden = block(hidden)
        return SimpleNamespace(logits=self.readout(hidden))


class TinyAdapter:
    """Small actual differentiable multimodal graph, with no downloaded weights."""

    def __init__(self):
        torch.manual_seed(19)
        self.model = TinyModel()
        self.layers = self.model.blocks
        self.vision_layers = [self.model.vision]
        self.attn_projections = [block.projection for block in self.layers]
        self.attn_updates = self.attn_projections
        self.device = torch.device("cpu")

    def prepare(self, image, prompt, answer=None):
        input_ids = [1, 2]
        if answer is not None:
            input_ids.append(3 if answer == "chosen" else 4 if answer == "rejected" else 5)
        batch = {"input_ids": torch.tensor([input_ids])}
        if image is not None:
            variation = 0.1 if image.endswith("a") else 0.4
            batch["pixel_values"] = torch.tensor([[1., variation, 0., 0.], [0., 0., 1., variation]])
        if answer is not None:
            prefix = (2 if image is not None else 0) + 2
            batch["labels"] = torch.tensor([[-100] * prefix + [input_ids[-1]]])
        return batch

    def forward(self, batch):
        return self.model(**batch)


class BaselineTests(unittest.TestCase):
    def test_purification_removes_prior_without_normalizing(self):
        vector = orthogonalize(torch.tensor([3., 4.]), torch.tensor([2., 0.]))
        torch.testing.assert_close(vector, torch.tensor([0., 4.]))
        torch.testing.assert_close(orthogonalize(vector, torch.zeros(2)), vector)

    def test_revis_gate_targets_only_low_alignment(self):
        hidden = torch.tensor([[[1., 0.], [-1., 1.]]])
        vector = torch.tensor([1., 0.])
        result = revis_update(hidden, vector, alpha=2., threshold=0.)
        torch.testing.assert_close(result[0, 0], hidden[0, 0])
        torch.testing.assert_close(result[0, 1], torch.tensor([1., 1.]))

    def test_rudder_gate_is_increasing_and_bounded(self):
        gate = beta_gate(torch.linspace(-1., 1., 20))
        self.assertTrue(torch.all(gate[1:] >= gate[:-1]))
        self.assertGreaterEqual(float(gate.min()), 0.05)
        self.assertLessEqual(float(gate.max()), 1.)
        self.assertAlmostEqual(float(beta_gate(torch.tensor(0.))), 0.5)

    def test_norm_preservation_and_zero_safety(self):
        hidden = torch.tensor([[3., 4.], [0., 0.], [-1., 0.]])
        result = norm_preserving_add(hidden, torch.tensor([1., 0.]), 1.)
        torch.testing.assert_close(result.norm(dim=-1), hidden.norm(dim=-1))
        self.assertTrue(torch.isfinite(result).all())

    def test_pca_direction_and_position_shape(self):
        samples = torch.tensor([[1., 0.], [2., 0.], [4., 0.]])
        torch.testing.assert_close(principal_direction(samples), torch.tensor([1., 0.]))
        positions = torch.stack((samples, samples * 2), dim=1)
        self.assertEqual(principal_direction(positions).shape, (2, 2))
        torch.testing.assert_close(principal_direction(torch.zeros(3, 4)), torch.zeros(4))

    def test_likelihood_ignores_prompt_labels(self):
        logits = torch.zeros(1, 4, 3, requires_grad=True)
        labels = torch.tensor([[-100, -100, 1, 2]])
        objective = sequence_log_probability(logits, labels)
        self.assertAlmostEqual(float(objective.detach()), -2 * torch.log(torch.tensor(3.)).item())
        objective.backward()
        torch.testing.assert_close(logits.grad[:, 0], torch.zeros(1, 3))
        self.assertGreater(float(logits.grad[:, 1:].abs().sum()), 0)

    def test_hooks_clean_up_on_exception_and_preserve_cache(self):
        block = Block()
        adapter = SimpleNamespace(layers=[block])
        artifact = {"name": "revis", "layer": 0, "threshold": -2.,
                    "vectors": {0: torch.tensor([1., 0.])}}
        hidden = torch.tensor([[[0., 1.], [0., 1.]]])
        expected = block(hidden)[0]
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with build_baseline("revis", adapter, artifact, {"alpha": 1.}):
                output = block(hidden)
                self.assertEqual(output[1], "unchanged-cache")
                torch.testing.assert_close(output[0][:, 0], expected[:, 0])
                torch.testing.assert_close(output[0][:, -1], expected[:, -1] + torch.tensor([1., 0.]))
                raise RuntimeError("intentional")
        self.assertEqual(len(block._forward_hooks), 0)
        torch.testing.assert_close(block(hidden)[0], expected)

    def test_rudder_uses_prompt_direction_then_cached_decode(self):
        block = Block()
        adapter = SimpleNamespace(layers=[block], attn_updates=[block.projection])
        with build_baseline("rudder", adapter, config={"layer": 0, "alpha": 2.}):
            prompt = torch.tensor([[[1., 0.], [1., 0.]]])
            result = block(prompt)[0]
            torch.testing.assert_close(result[:, 0], prompt[:, 0] * 1.25)
            self.assertGreater(float(result[0, -1, 0]), 1.25)
            opposite = block(torch.tensor([[[-1., 0.]]]))[0]
            # Low/negative alignment must produce less reinforcement.
            self.assertLess(float(opposite[0, 0, 0] + 1.25), float(result[0, -1, 0] - 1.25))
        self.assertEqual(len(block._forward_pre_hooks), 0)
        self.assertEqual(len(block.projection._forward_hooks), 0)

    def test_rejects_mismatched_artifact(self):
        with self.assertRaisesRegex(ValueError, "name"):
            with build_baseline("revis", None, {"name": "vti"}):
                pass

    def test_real_extraction_graph_vti_and_grains(self):
        adapter = TinyAdapter()
        records = [{"id": image, "image": image, "prompt": "describe",
                    "chosen": "chosen", "rejected": "rejected"} for image in ("a", "b")]
        before = {name: parameter.detach().clone() for name, parameter in adapter.model.named_parameters()}
        for method, config in (
            ("vti", {"masks_per_image": 2, "mask_fraction": 0.5}),
            ("grains", {"ig_steps": 3, "top_k": 2}),
        ):
            artifact = extract_baseline(method, adapter, records, config)
            self.assertFalse(artifact["author_reproduction"])
            self.assertEqual(set(artifact["vectors"]), {0, 1})
            self.assertTrue(all(torch.isfinite(v).all() for v in artifact["vectors"].values()))
            self.assertTrue(any(v.norm() > 0 for v in artifact["vectors"].values()))
            batch = adapter.prepare("a", "describe")
            with build_baseline(method, adapter, artifact):
                self.assertTrue(torch.isfinite(adapter.forward(batch).logits).all())
            self.assertTrue(all(len(module._forward_hooks) == 0 for module in adapter.model.modules()))
            self.assertTrue(all(len(module._forward_pre_hooks) == 0 for module in adapter.model.modules()))
        for name, parameter in adapter.model.named_parameters():
            torch.testing.assert_close(parameter, before[name])
            self.assertIsNone(parameter.grad)
        self.assertTrue(adapter.model.training)

    def test_revis_extraction_requires_calibration_and_preserves_scale(self):
        adapter = TinyAdapter()
        records = [{"id": "a", "image": "a", "prompt": "describe",
                    "chosen": "chosen", "rejected": "rejected"}]
        with self.assertRaisesRegex(ValueError, "calibration"):
            extract_baseline("revis", adapter, records)
        artifact = extract_baseline("revis", adapter, records, {"layer": 1, "threshold": 0.})
        self.assertEqual(artifact["calibration"], "externally-specified")
        self.assertGreater(float(artifact["vectors"][1].norm()), 0)

    def test_revis_rejects_overlapping_calibration_images(self):
        adapter = TinyAdapter()
        record = {"id": "a", "image": "a", "prompt": "describe",
                  "chosen": "chosen", "rejected": "rejected"}
        with self.assertRaisesRegex(ValueError, "disjoint"):
            extract_baseline("revis", adapter, [record, {**record, "role": "calibration"}])


if __name__ == "__main__":
    unittest.main()
