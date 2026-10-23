"""Small real HF architectures on CPU; no downloaded models or tokenizers."""

from types import SimpleNamespace

import pytest
import torch
from transformers import (
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    LlavaConfig,
    LlavaForConditionalGeneration,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

from vlm_grounding.backend import ModelAdapter, load_adapter
from vlm_grounding.baselines import _masked_pixels, build_baseline
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
    elif family == "qwen3_vl":
        text.update({"head_dim": 8, "rope_scaling": {
            "rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True,
        }})
        config = Qwen3VLConfig(
            text_config=text,
            vision_config={
                "depth": 2, "hidden_size": 16, "intermediate_size": 32,
                "num_heads": 2, "out_hidden_size": 32, "patch_size": 2,
                "temporal_patch_size": 2, "spatial_merge_size": 2,
                "num_position_embeddings": 16, "deepstack_visual_indexes": [0],
            },
            image_token_id=4,
            video_token_id=10,
            vision_start_token_id=3,
            vision_end_token_id=5,
        )
        config._attn_implementation = "eager"
        model = Qwen3VLForConditionalGeneration(config).eval()
        batch = {
            "input_ids": torch.tensor([[1, 3, 4, 4, 4, 4, 5, 8]]),
            "pixel_values": torch.randn(16, 24),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
            "mm_token_type_ids": torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]]),
        }
    elif family == "gemma4":
        config = Gemma4Config(
            text_config=Gemma4TextConfig(
                vocab_size=32, vocab_size_per_layer_input=32,
                hidden_size_per_layer_input=16, hidden_size=32,
                intermediate_size=48, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                layer_types=["sliding_attention", "full_attention"], sliding_window=32,
            ),
            vision_config=Gemma4VisionConfig(
                hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                patch_size=2, pooling_kernel_size=2, position_embedding_size=64,
            ),
            image_token_id=4, boi_token_id=3, eoi_token_id=5,
        )
        config._attn_implementation = "eager"
        model = Gemma4ForConditionalGeneration(config).eval()
        batch = {
            "input_ids": torch.tensor([[1, 3, 4, 5, 8]]),
            "pixel_values": torch.rand(1, 4, 12),
            "image_position_ids": torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]]),
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


@pytest.mark.parametrize("family", ["qwen2_5_vl", "qwen3_vl", "gemma4", "llava"])
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


@pytest.mark.parametrize("family", ["qwen2_5_vl", "qwen3_vl", "gemma4", "llava"])
@pytest.mark.parametrize("name", ["revis", "rudder", "grains", "vti"])
def test_all_baseline_hooks_on_real_multimodal_architectures(family, name):
    adapter, batch = tiny_adapter(family)
    modules = [*adapter.layers, *adapter.vision_layers,
               *adapter.attn_projections, *adapter.attn_updates]
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
        original_hooks = [(set(module._forward_hooks), set(module._forward_pre_hooks))
                          for module in modules]
        with build_baseline(name, adapter, artifact, {"layer": 0, "alpha": 0.5}):
            changed = adapter.forward(batch).logits
            assert torch.isfinite(changed).all()
            assert not torch.allclose(changed[:, -1], original[:, -1])
        # Generation gets a fresh context so RUDDER cannot reuse another prompt's CARD.
        with build_baseline(name, adapter, artifact, {"layer": 0, "alpha": 0.5}):
            adapter.generate(batch, 2)
        assert torch.allclose(adapter.forward(batch).logits, original, atol=1e-6)
    for module, (forward_hooks, pre_hooks) in zip(modules, original_hooks):
        assert set(module._forward_hooks) == forward_hooks
        assert set(module._forward_pre_hooks) == pre_hooks


def test_gemma4_rudder_adds_card_after_attention_normalization():
    adapter, batch = tiny_adapter("gemma4")
    layer = adapter.layers[0]
    captured = {}

    def capture_update(_module, _args, output):
        captured["update"] = output.detach().clone()

    def capture_residual(_module, args):
        captured["residual"] = args[0].detach().clone()

    handles = [layer.post_attention_layernorm.register_forward_hook(capture_update),
               layer.pre_feedforward_layernorm.register_forward_pre_hook(capture_residual)]
    try:
        with torch.no_grad():
            adapter.forward(batch)
            update = captured["update"].clone()
            original = captured["residual"].clone()
            pooled = update.mean(1, keepdim=True)
            card = pooled / pooled.norm(dim=-1, keepdim=True)
            with build_baseline("rudder", adapter, config={
                "layer": 0, "alpha": 0.5, "pooling": "mean",
                "gate_min": 1.0, "gate_max": 1.0,
            }):
                adapter.forward(batch)
            # The attention norm sees the original update; only its output is steered.
            torch.testing.assert_close(captured["update"], update)
            torch.testing.assert_close(captured["residual"][:, :-1], original[:, :-1])
            torch.testing.assert_close(captured["residual"][:, -1:],
                                       original[:, -1:] + 0.5 * card)
    finally:
        for handle in handles:
            handle.remove()


def test_teacher_forcing_masks_prompt_and_keeps_answer():
    adapter, _ = tiny_adapter("llava")
    batch = adapter.prepare(None, "Is there a cat?", "Yes")
    assert batch["labels"].tolist() == [[-100, -100, -100, 6, 2]]
    assert batch["input_ids"].dtype == torch.int64
    assert adapter.answer_margin(adapter.prepare(None, "test")).requires_grad


def test_qwen3vl_loader_uses_pinned_transformers_class_without_weights(monkeypatch):
    adapter, _ = tiny_adapter("qwen3_vl")
    monkeypatch.setattr("transformers.AutoProcessor.from_pretrained",
                        lambda *args, **kwargs: TinyProcessor())
    monkeypatch.setattr("transformers.Qwen3VLForConditionalGeneration.from_pretrained",
                        lambda *args, **kwargs: adapter.model)
    loaded = load_adapter({"family": "qwen3_vl", "checkpoint": "fixture",
                           "device": "cpu", "dtype": "float32"})
    assert loaded.model_type == "qwen3_vl"
    assert loaded.model is adapter.model


def test_gemma4_loader_uses_local_tiny_checkpoint_without_downloads(monkeypatch, tmp_path):
    adapter, _ = tiny_adapter("gemma4")
    adapter.model.save_pretrained(tmp_path)
    monkeypatch.setattr("transformers.AutoProcessor.from_pretrained",
                        lambda *args, **kwargs: TinyProcessor())
    loaded = load_adapter({"family": "gemma4", "checkpoint": str(tmp_path),
                           "device": "cpu", "dtype": "float32", "local_files_only": True})
    assert loaded.model_type == "gemma4"
    assert len(loaded.vision_layers) == 2
    assert loaded.attn_projections[0].in_features == 32


def test_gemma4_patch_mask_preserves_image_positions():
    _, batch = tiny_adapter("gemma4")
    batch["pixel_values"] = torch.cat([batch["pixel_values"], torch.rand(1, 1, 12)], dim=1)
    batch["image_position_ids"] = torch.cat(
        [batch["image_position_ids"], torch.tensor([[[-1, -1]]])], dim=1
    )
    original = batch["pixel_values"].clone()
    masked = _masked_pixels(batch, torch.Generator().manual_seed(42), 0.25, 2)
    assert torch.equal(masked["image_position_ids"], batch["image_position_ids"])
    assert torch.equal(batch["pixel_values"], original)
    assert (masked["pixel_values"] == 0).all(dim=-1).sum().item() == 1
    assert torch.equal(masked["pixel_values"][:, -1], original[:, -1])
