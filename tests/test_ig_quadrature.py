"""IG must remain valid when the zero endpoint has an undefined derivative."""

from types import SimpleNamespace

import pytest
import torch

from vlm_grounding.baselines import _preference_attributions


def radial_adapter(*, nonfinite=False):
    layer = torch.nn.Identity()

    def forward(batch):
        hidden = layer(batch["hidden"])
        # sqrt(x*x+y*y) has a finite directional integral, but its automatic
        # derivative at exactly zero is NaN. Positive alpha has derivative
        # (x,y)/sqrt(x*x+y*y), so the preference objective is exactly 2*radius.
        radius = hidden.square().sum(-1).sqrt()
        if nonfinite:
            radius = radius * float("nan")
        return SimpleNamespace(logits=torch.stack((radius, -radius), dim=-1))

    batch = {"hidden": torch.tensor([[[3., 4.], [1., 1.]]])}
    positive = {**batch, "labels": torch.tensor([[-100, 0]])}
    negative = {**batch, "labels": torch.tensor([[-100, 1]])}
    return SimpleNamespace(layers=[layer], forward=forward), positive, negative


def test_ig_integrates_finite_path_with_singular_zero_endpoint():
    adapter, positive, negative = radial_adapter()
    scores = _preference_attributions(adapter, positive, negative, steps=16)
    torch.testing.assert_close(scores, torch.tensor([10.]), atol=1e-5, rtol=1e-5)
    reversed_scores = _preference_attributions(adapter, negative, positive, steps=16)
    torch.testing.assert_close(reversed_scores, -scores)
    assert not adapter.layers[0]._forward_pre_hooks


def test_ig_rejects_nonfinite_path_instead_of_creating_zero_masks():
    adapter, positive, negative = radial_adapter(nonfinite=True)
    with pytest.raises(ValueError, match="Non-finite GrAInS objective"):
        _preference_attributions(adapter, positive, negative, steps=2)
    assert not adapter.layers[0]._forward_pre_hooks
