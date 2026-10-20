"""CPU behavioral tests; these do not constitute VLM performance validation."""

import json
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from vlm_grounding.method import discover, fit, steer


class TinyAdapter:
    """Two query heads, one KV head: o_proj still receives both query heads."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.num_heads = 2
        self.num_key_value_heads = 1
        self.projection = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(4))
        self.model = nn.Sequential(self.projection)
        self.layers = [self.model]
        self.attn_projections = [self.projection]
        self.images = {
            "clean": [2.0, 0.0, 1.0, 0.0],
            "masked": [0.0, 0.0, 1.0, 0.0],
            "control": [2.0, 0.0, 0.0, 0.0],
            "negative": [-2.0, 0.0, 1.0, 0.0],
        }

    def prepare(self, image_path, prompt, answer=None):
        return {"values": torch.tensor([[self.images[image_path], self.images[image_path]]])}

    def forward(self, batch):
        features = self.model(batch["values"])
        margin = features[..., 0] + 0.25 * features[..., 2]
        logits = torch.stack([margin / 2, -margin / 2, margin * 0], dim=-1)
        return SimpleNamespace(logits=logits)

    def answer_margin(self, batch):
        output = self.forward(batch)
        return output.logits[0, -1, 0] - output.logits[0, -1, 1]


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.adapter = TinyAdapter()
        self.records = [{
            "id": "dev-1", "group_id": "image-1", "image": "clean",
            "object_masked_image": "masked", "control_image": "control",
            "prompt": "Is the object visible?", "label": "yes",
        }]

    def selection(self):
        return discover(self.adapter, self.records, {"top_k": 1})

    def test_causal_selection_prefers_object_head_over_background_head(self):
        selection = self.selection()
        self.assertEqual(selection["selected_heads"][0]["head"], 0)
        self.assertAlmostEqual(selection["head_scores"][0]["score"], 1.0)
        self.assertLess(selection["head_scores"][1]["score"], 0.0)
        self.assertEqual(self.adapter.num_key_value_heads, 1)
        self.assertFalse(self.adapter.projection._forward_pre_hooks)

    def test_selection_modes_and_json_round_trip(self):
        for mode in ("causal", "random", "magnitude"):
            result = discover(self.adapter, self.records, {"top_k": 1, "selection_mode": mode, "seed": 7})
            self.assertEqual(json.loads(json.dumps(result)), result)
            self.assertEqual(result["selection_mode"], mode)
        first = discover(self.adapter, self.records, {"top_k": 1, "selection_mode": "random", "seed": 7})
        second = discover(self.adapter, self.records, {"top_k": 1, "selection_mode": "random", "seed": 7})
        self.assertEqual(first["selected_heads"], second["selected_heads"])

    def test_fitting_changes_gains_without_changing_base_weights(self):
        adapter = self.adapter
        selection = self.selection()
        weights = adapter.projection.weight.detach().clone()
        fit_records = [
            {"id": "fit-y", "image": "clean", "prompt": "Object?", "label": "yes"},
            {"id": "fit-n", "image": "negative", "prompt": "Object?", "label": "no"},
        ]
        artifact = fit(adapter, fit_records, selection, {"epochs": 3, "lr": 0.1})
        self.assertTrue(torch.equal(weights, adapter.projection.weight))
        self.assertIsNone(adapter.projection.weight.grad)
        self.assertTrue(adapter.projection.weight.requires_grad)
        self.assertTrue(adapter.model.training)
        self.assertEqual(artifact["fit_steps"], 6)
        self.assertEqual(artifact["positions"], "last")
        self.assertGreater(artifact["selected_heads"][0]["delta"], 0)
        self.assertLessEqual(abs(artifact["selected_heads"][0]["delta"]), artifact["max_gain"])
        artifact = json.loads(json.dumps(artifact))
        batch = adapter.prepare("clean", "Object?")
        base_margin = float(adapter.answer_margin(batch).detach())
        with steer(adapter, artifact):
            self.assertGreater(float(adapter.answer_margin(batch).detach()), base_margin)
        self.assertEqual(float(adapter.answer_margin(batch).detach()), base_margin)
        self.assertFalse(adapter.projection._forward_pre_hooks)

    def test_hook_cleanup_after_inference_exception_and_zero_strength(self):
        artifact = {
            **self.selection(), "stage": "fitted", "max_gain": 0.25,
            "selected_heads": [{"layer": 0, "head": 0, "delta": 0.2}],
        }
        batch = self.adapter.prepare("clean", "Object?")
        base_margin = float(self.adapter.answer_margin(batch).detach())
        with steer(self.adapter, artifact, {"strength": 0}):
            self.assertEqual(float(self.adapter.answer_margin(batch).detach()), base_margin)
        with self.assertRaisesRegex(RuntimeError, "planned"):
            with steer(self.adapter, artifact):
                raise RuntimeError("planned")
        self.assertFalse(self.adapter.projection._forward_pre_hooks)

    def test_default_steering_changes_only_last_prefill_or_decode_position(self):
        artifact = {
            **self.selection(), "stage": "fitted", "max_gain": 0.25,
            "selected_heads": [{"layer": 0, "head": 0, "delta": 0.2}],
        }
        for length in (1, 4):
            batch = {"values": torch.tensor([[[2.0, 0.0, 1.0, 0.0]] * length])}
            original = self.adapter.forward(batch).logits
            with steer(self.adapter, artifact):
                changed = self.adapter.forward(batch).logits
            self.assertGreater(float(changed[0, -1, 0].detach()), float(original[0, -1, 0].detach()))
            self.assertTrue(torch.equal(changed[:, :-1], original[:, :-1]))
            with steer(self.adapter, artifact, {"strength": 0}):
                identity = self.adapter.forward(batch).logits
            self.assertTrue(torch.equal(identity, original))

    def test_all_positions_is_explicit_ablation_and_override_is_rejected(self):
        artifact = {
            **self.selection(), "stage": "fitted", "max_gain": 0.25, "positions": "all",
            "selected_heads": [{"layer": 0, "head": 0, "delta": 0.2}],
        }
        batch = self.adapter.prepare("clean", "Object?")
        original = self.adapter.forward(batch).logits
        with steer(self.adapter, artifact):
            changed = self.adapter.forward(batch).logits
        self.assertTrue(torch.all(changed[..., 0] > original[..., 0]))
        with self.assertRaisesRegex(ValueError, "match the fitted artifact"):
            with steer(self.adapter, artifact, {"positions": "last"}):
                pass

    def test_rejects_unbalanced_fit_and_invalid_artifacts(self):
        selection = self.selection()
        with self.assertRaisesRegex(ValueError, "both yes and no"):
            fit(self.adapter, [{"image": "clean", "prompt": "?", "label": "yes"}], selection, {})
        with self.assertRaisesRegex(ValueError, "fitted"):
            with steer(self.adapter, selection):
                pass
        artifact = {
            **selection, "stage": "fitted", "max_gain": 0.25,
            "selected_heads": [{"layer": 0, "head": 0, "delta": 0.26}],
        }
        with self.assertRaisesRegex(ValueError, "bounded"):
            with steer(self.adapter, artifact):
                pass
        self.assertFalse(self.adapter.projection._forward_pre_hooks)

    def test_discovery_rejects_uninformative_removal(self):
        self.records[0]["object_masked_image"] = "clean"
        with self.assertRaisesRegex(ValueError, "No discovery examples"):
            self.selection()
        self.assertTrue(self.adapter.model.training)
        self.assertFalse(self.adapter.projection._forward_pre_hooks)

    def test_detached_adapter_margin_fails_without_leaking_hooks_or_frozen_weights(self):
        selection = self.selection()
        differentiable_margin = self.adapter.answer_margin
        self.adapter.answer_margin = lambda batch: differentiable_margin(batch).detach()
        records = [
            {"image": "clean", "prompt": "?", "label": "yes"},
            {"image": "negative", "prompt": "?", "label": "no"},
        ]
        with self.assertRaisesRegex(ValueError, "preserve gradients"):
            fit(self.adapter, records, selection, {})
        self.assertTrue(self.adapter.projection.weight.requires_grad)
        self.assertTrue(self.adapter.model.training)
        self.assertFalse(self.adapter.projection._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
