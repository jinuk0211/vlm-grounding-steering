"""The Jacobian-lens probe on a tiny random Qwen3.5 (transformers 5 only; jlens on the path)."""
import numpy as np
import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5")
jlens = pytest.importorskip("jlens")

from common import MODELS, gray_image  # noqa: E402
from jlens_probe import TOP_K, Readout, model_inputs, object_token_ids  # noqa: E402
from tiny import tiny_model  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    model, processor = tiny_model(MODELS["qwen35"], layers=4)
    lm = jlens.from_hf(model, processor.tokenizer)
    node_ids = {"dog": [100, 101], "cat": [102], "car": [103, 104, 105]}
    identity = {layer: torch.eye(lm.d_model) for layer in range(lm.n_layers)}
    return model, processor, lm, node_ids, identity


def test_with_an_identity_jacobian_both_lenses_agree_and_the_last_layer_is_the_output(setup) -> None:
    model, processor, lm, node_ids, identity = setup
    readout = Readout(model, lm, identity, node_ids, model.config.image_token_id)
    out = readout(model_inputs(processor, gray_image(), "cpu"))
    np.testing.assert_allclose(out["score"][:, 0], out["score"][:, 1], atol=1e-5)
    assert (out["top_ids"][:, 0] == out["top_ids"][:, 1]).mean() > 0.99
    last = len(readout.layers) - 1
    np.testing.assert_allclose(out["score"][last, 1, 0], out["final_score"], atol=1e-3)
    assert out["span_argmax"].max() < 196 and out["span_hits"].max() <= 196


def test_rank_below_top_k_means_the_object_is_in_the_top_list(setup) -> None:
    model, processor, lm, node_ids, identity = setup
    readout = Readout(model, lm, identity, node_ids, model.config.image_token_id)
    out = readout(model_inputs(processor, gray_image(), "cpu"))
    for layer in range(len(readout.layers)):
        for place in range(2):
            listed = set(out["top_ids"][layer, 1, place].tolist())
            for j, node in enumerate(readout.nodes):
                in_list = bool(listed & set(node_ids[node]))
                assert in_list == (out["rank"][layer, 1, place, j] < TOP_K)


def test_a_jacobian_changes_the_readout(setup) -> None:
    model, processor, lm, node_ids, _ = setup
    torch.manual_seed(1)
    random = {layer: torch.randn(lm.d_model, lm.d_model) / lm.d_model ** 0.5 for layer in range(lm.n_layers)}
    readout = Readout(model, lm, random, node_ids, model.config.image_token_id)
    out = readout(model_inputs(processor, gray_image(), "cpu"))
    assert not np.allclose(out["score"][:, 0], out["score"][:, 1])


def test_object_tokens_are_single_tokens_owned_by_one_object(setup) -> None:
    _, processor, _, _, _ = setup
    tokens = object_token_ids(processor.tokenizer, {"dog": "dog", "puppy": "dog", "cat": "cat", "hot dog": "hot dog", "": ""})
    assert set(tokens) == {"dog", "cat"}
    decoded = {processor.tokenizer.decode([t]).strip().lower() for t in tokens["dog"]}
    assert {"dog", "dogs"} <= decoded
