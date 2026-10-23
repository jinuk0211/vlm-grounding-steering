"""Head steering across Gemma's mixed attention widths and shared KV layers."""

import torch
from test_backend import TinyProcessor
from transformers import (
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)

from vlm_grounding.backend import ModelAdapter
from vlm_grounding.method import steer


def test_gemma4_head_steering_with_mixed_widths_and_shared_kv():
    torch.manual_seed(7)
    config = Gemma4Config(
        text_config=Gemma4TextConfig(
            vocab_size=32, vocab_size_per_layer_input=32,
            hidden_size_per_layer_input=16, hidden_size=32,
            intermediate_size=48, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, global_head_dim=16, num_kv_shared_layers=2,
            layer_types=["sliding_attention", "full_attention"] * 2, sliding_window=32,
        ),
        vision_config=Gemma4VisionConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            patch_size=2, pooling_kernel_size=2, position_embedding_size=64,
        ),
        image_token_id=4, boi_token_id=3, eoi_token_id=5,
    )
    config._attn_implementation = "eager"
    adapter = ModelAdapter(
        Gemma4ForConditionalGeneration(config).eval(), TinyProcessor(), {"image_size": 8}
    )
    batch = {
        "input_ids": torch.tensor([[1, 3, 4, 5, 8]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "pixel_values": torch.rand(1, 4, 12),
        "image_position_ids": torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]]),
    }
    assert [projection.in_features // adapter.num_heads
            for projection in adapter.attn_projections] == [8, 16, 8, 16]
    assert [layer.self_attn.is_kv_shared_layer
            for layer in adapter.layers] == [False, False, True, True]

    with torch.no_grad():
        original = adapter.forward(batch).logits.clone()
        for layer in range(4):
            artifact = {
                "stage": "fitted", "num_heads": 4, "layer_count": 4,
                "max_gain": 0.25, "positions": "last",
                "selected_heads": [{"layer": layer, "head": 1, "delta": 0.2}],
            }
            with steer(adapter, artifact):
                changed = adapter.forward(batch).logits
                assert torch.isfinite(changed).all()
                assert not torch.allclose(changed[:, -1], original[:, -1])
                torch.testing.assert_close(changed[:, :-1], original[:, :-1])
                adapter.generate(batch, max_new_tokens=2)
                assert 1 <= adapter.last_generated_tokens <= 2
            torch.testing.assert_close(adapter.forward(batch).logits, original)
            assert not adapter.attn_projections[layer]._forward_pre_hooks
