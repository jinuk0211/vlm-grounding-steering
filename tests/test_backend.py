"""Small real HF architectures on CPU; no downloaded models or tokenizers."""

from types import SimpleNamespace

import pytest
import torch
from transformers import (
    LlavaConfig,
    LlavaForConditionalGeneration,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from vlm_grounding.backend import ModelAdapter
from vlm_grounding.baselines import build_baseline
from vlm_grounding.method import steer


class TinyProcessor:
    """Only tokenizer/decoding plumbing; HF model forward passes below are real."""

    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: {"Yes": [6], "No": [7]}[text],
        eos_token="<eos>",
    )

    def apply_chat_template(self, messages, **kwargs):
        return "prompt:"

    def __call__(self, text, **kwargs):
        ids = [1, 8, 9] if text[0] == "prompt:" else [1, 8, 9, 6, 2]
        tensor = torch.tensor([ids])
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}

    def batch_decode(self, ids, **kwargs):
        return [" ".join(str(int(token)) for token in row) for row in ids]


def tiny_adapter(family):
    text = {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    torch.manual_seed(7)
    if family == "qwen2_5_vl":
        text.update({"rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}})
        config = Qwen2_5_VLConfig(
            text_config=text,
            vision_config={
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 32,
                "patch_size": 2,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "window_size": 8,
                "fullatt_block_indexes": [0, 1],
            },
            image_token_id=4,
            video_token_id=10,
            vision_start_token_id=3,
            vision_end_token_id=5,
        )
        config._attn_implementation = "eager"
        model = Qwen2_5_VLForConditionalGeneration(config).eval()
        batch = {
            "input_ids": torch.tensor([[1, 3, 4, 4, 4, 4, 5, 8]]),
            "pixel_values": torch.randn(16, 24),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }
    else:
        text["model_type"] = "llama"
        config = LlavaConfig(
            text_config=text,
            vision_config={
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "image_size": 8,
                "patch_size": 4,
                "projection_dim": 16,
            },
            image_token_index=4,
            image_seq_length=4,
            vision_feature_layer=-1,
        )
        config._attn_implementation = "eager"
        model = LlavaForConditionalGeneration(config).eval()
        batch = {
            "input_ids": torch.tensor([[1, 4, 4, 4, 4, 8]]),
            "pixel_values": torch.randn(1, 3, 8, 8),
        }
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    adapter = ModelAdapter(model, TinyProcessor(), {"image_size": 8})
    return adapter, batch


@pytest.mark.parametrize("family", ["qwen2_5_vl", "llava"])
def test_real_architecture_head_intervention_and_cached_generation(family):
    adapter, batch = tiny_adapter(family)
    assert adapter.num_heads == 4
    assert len(adapter.layers) == len(adapter.vision_layers) == 2
    with torch.no_grad():
        original = adapter.forward(batch).logits.clone()
    artifact = {
        "stage": "fitted",
        "num_heads": 4,
        "layer_count": 2,
        "max_gain": 0.25,
        "positions": "last",
        "selected_heads": [{"layer": 0, "head": 1, "delta": 0.2}],
    }
    with steer(adapter, artifact):
        changed = adapter.forward(batch).logits
        assert not torch.allclose(changed[:, -1], original[:, -1])
        adapter.generate(batch, max_new_tokens=2)
        assert 1 <= adapter.last_generated_tokens <= 2
    with torch.no_grad():
        assert torch.allclose(adapter.forward(batch).logits, original, atol=1e-6)
    assert not adapter.attn_projections[0]._forward_pre_hooks


@pytest.mark.parametrize("family", ["qwen2_5_vl", "llava"])
@pytest.mark.parametrize("name", ["revis", "rudder", "grains", "vti"])
def test_all_baseline_hooks_on_real_multimodal_architectures(family, name):
    adapter, batch = tiny_adapter(family)
    vector = torch.linspace(-0.2, 0.3, 32)
    artifact = {
        "name": name,
        "vectors": {0: vector},
        "layer": 0,
        "threshold": -2.0,
        "vision_vectors": {0: torch.linspace(-0.1, 0.1, 16)},
    }
    with torch.no_grad():
        original = adapter.forward(batch).logits.clone()
        with build_baseline(name, adapter, artifact, {"layer": 0, "alpha": 0.5}):
            changed = adapter.forward(batch).logits
            assert torch.isfinite(changed).all()
            assert not torch.allclose(changed[:, -1], original[:, -1])
        # Generation gets a fresh context so RUDDER cannot reuse another prompt's CARD.
        with build_baseline(name, adapter, artifact, {"layer": 0, "alpha": 0.5}):
            adapter.generate(batch, 2)
        assert torch.allclose(adapter.forward(batch).logits, original, atol=1e-6)
    for module in [*adapter.layers, *adapter.vision_layers, *adapter.attn_projections]:
        assert not module._forward_hooks
        assert not module._forward_pre_hooks


def test_teacher_forcing_masks_prompt_and_keeps_answer():
    adapter, _ = tiny_adapter("llava")
    batch = adapter.prepare(None, "Is there a cat?", "Yes")
    assert batch["labels"].tolist() == [[-100, -100, -100, 6, 2]]
    assert batch["input_ids"].dtype == torch.int64
    assert adapter.answer_margin(adapter.prepare(None, "test")).requires_grad
