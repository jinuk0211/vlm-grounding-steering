"""Extraction-to-generation smoke checks on actual tiny HF model classes.

The fixture supplies token/pixel tensors directly; this does not validate real
checkpoint processors or benchmark performance.
"""

import pytest
import torch

from test_backend import tiny_adapter
from vlm_grounding.baselines import build_baseline, extract_baseline


@pytest.mark.parametrize("family", ["qwen2_5_vl", "qwen3_vl", "gemma4", "llava"])
@pytest.mark.parametrize("method", ["revis", "grains", "vti"])
def test_extracted_artifact_generates_and_zero_strength_restores(family, method):
    adapter, image_batch = tiny_adapter(family)

    def prepare(image, prompt, answer=None):
        if image is None:
            batch = {"input_ids": torch.tensor([[1, 8]])}
        else:
            batch = {key: value.clone() for key, value in image_batch.items()}
            if image == "b":
                batch["pixel_values"] = batch["pixel_values"] * 0.9 + 0.01
        prefix = batch["input_ids"].shape[1]
        if answer is not None:
            token = {"chosen": 6, "rejected": 7, "neutral": 9}[answer]
            batch["input_ids"] = torch.cat((batch["input_ids"], torch.tensor([[token, 2]])), dim=1)
            if "mm_token_type_ids" in batch:
                batch["mm_token_type_ids"] = torch.cat(
                    (batch["mm_token_type_ids"], torch.zeros(1, 2, dtype=torch.long)), dim=1)
            batch["labels"] = batch["input_ids"].clone()
            batch["labels"][:, :prefix] = -100
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        return batch

    adapter.prepare = prepare
    records = [{"id": name, "image": name, "prompt": "fixture", "chosen": "chosen",
                "rejected": "rejected", "neutral": "neutral"} for name in ("a", "b")]
    configuration = {
        "revis": {"layer": 0, "threshold": -2.},
        "grains": {"layers": [0], "ig_steps": 2, "top_k": 2},
        "vti": {"layers": [0], "vision_layers": [0], "masks_per_image": 2,
                "mask_fraction": 0.5, "mask_patch_size": 2},
    }[method]
    # HF installs persistent vision-output capture hooks lazily on LLaVA.
    # Establish those first, then verify our hooks do not leak or remove them.
    with torch.no_grad():
        adapter.forward(image_batch)
    original_hooks = {name: (dict(module._forward_hooks), dict(module._forward_pre_hooks))
                      for name, module in adapter.model.named_modules()}
    weights = {name: param.detach().clone() for name, param in adapter.model.named_parameters()}
    artifact = extract_baseline(method, adapter, records, configuration)
    assert any(vector.norm() > 0 for vector in artifact["vectors"].values())
    assert all(torch.isfinite(vector).all() for vector in artifact["vectors"].values())
    batch = prepare("a", "fixture")
    with torch.no_grad():
        original = adapter.forward(batch).logits
        with build_baseline(method, adapter, artifact, {"alpha": 0., "vision_alpha": 0.}):
            torch.testing.assert_close(adapter.forward(batch).logits, original)
        with build_baseline(method, adapter, artifact, {"alpha": 0.5, "vision_alpha": 0.5}):
            assert not torch.allclose(adapter.forward(batch).logits[:, -1], original[:, -1])
        with build_baseline(method, adapter, artifact, {"alpha": 0.5, "vision_alpha": 0.5}):
            adapter.generate(batch, max_new_tokens=2)
        torch.testing.assert_close(adapter.forward(batch).logits, original)
    for name, module in adapter.model.named_modules():
        assert (dict(module._forward_hooks), dict(module._forward_pre_hooks)) == original_hooks[name]
    for name, param in adapter.model.named_parameters():
        torch.testing.assert_close(param, weights[name])
        assert param.grad is None
