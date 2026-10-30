"""Run small differential probes against clean, pinned upstream source.

This is an audit, not a reproduction runner or a benchmark. Selected upstream
AST definitions are executed unchanged; external model/data plumbing is replaced
with explicit tensor fixtures. No upstream implementation is copied into Git.
Install the project and scikit-learn, then fetch the four baseline repositories.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F

from vlm_grounding.baselines import (
    _masked_pixels, beta_gate, build_baseline, norm_preserving_add,
    orthogonalize, principal_direction, revis_update, sequence_log_probability,
)
from vlm_grounding.upstream import source_catalog


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def definitions(path, names, **extras):
    """Execute only named, unmodified definitions from the inspected source."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"Missing upstream definitions in {path}: {names}")
    namespace = {"torch": torch, "nn": nn, "F": F, "np": np, "PCA": PCA,
                 "Dict": Dict, "List": List, "Optional": Optional, "Image": Image,
                 "math": math, **extras}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def distance(left, right):
    return float((left.float() - right.float()).abs().max())


def directional_agreement(left, right):
    return float(F.cosine_similarity(left.flatten().float(), right.flatten().float(), dim=0))


def audit(root):
    report = {"scope": "synthetic differential probes; not pretrained-model parity",
              "python": platform.python_version(), "torch": torch.__version__,
              "sources": {}, "probes": {}}
    repository = Path(__file__).resolve().parents[1]
    report["local_commit"] = git(repository, "rev-parse", "HEAD")
    report["versions"] = {name: importlib.metadata.version(name)
                          for name in ("scikit-learn", "numpy", "transformers")}
    report["local_file_sha256"] = {
        name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
        for name in ("src/vlm_grounding/baselines.py", "scripts/audit_baseline_parity.py")
    }
    for name in ("revis", "rudder", "grains", "vti"):
        source = source_catalog()["sources"][name]
        checkout = root / name
        head = git(checkout, "rev-parse", "HEAD")
        if head != source["commit"] or git(checkout, "status", "--porcelain"):
            raise ValueError(f"{name}: expected a clean pinned checkout")
        report["sources"][name] = {"commit": head, "url": source["url"]}
    probes = report["probes"]

    # REVIS: original PCA helper on identical visual/prior samples.
    rv = definitions(root / "revis/steering_utils/vector_calculator.py",
                     ["get_pure_direction_with_pca"])
    pca = rv["get_pure_direction_with_pca"]
    visual = torch.tensor([[1., 1., 0.], [1., 3., 1.], [1., -1., 2.], [1., 2., -1.]])
    prior = torch.tensor([[2., 0., 0.], [2., 1., 0.], [2., -1., 0.], [2., 2., 0.]])
    prior_unit = F.normalize(pca(prior), dim=-1)
    author_vector = pca(visual - (visual @ prior_unit[:, None]) * prior_unit[None])
    local_vector = orthogonalize(visual.mean(0), prior.mean(0))
    probes["revis_vector"] = {
        "upstream": author_vector.tolist(), "local": local_vector.tolist(),
        "cosine": directional_agreement(author_vector, local_vector),
        "max_abs_difference": distance(author_vector, local_vector),
        "note": "Upstream PCA prior, samplewise purification, scaled PCA vs local means.",
    }
    rh = definitions(root / "revis/steering_utils/llm_layers.py",
                     ["MADRSteeringHook_visual_only"])
    hidden = torch.tensor([[[0., 1.], [0., 1.]]])
    vector = torch.tensor([1., 0.])
    author_hook = rh["MADRSteeringHook_visual_only"](
        {"visual": vector}, 0, 1., -2., 1.)
    upstream = author_hook.hook_fn(None, (), hidden.clone())
    block = nn.Identity()
    artifact = {"name": "revis", "layer": 0, "threshold": -2., "vectors": {0: vector}}
    with build_baseline("revis", SimpleNamespace(layers=[block]), artifact, {"alpha": 1.}):
        local = block(hidden.clone())
    torch.testing.assert_close(upstream[:, -1:], revis_update(hidden[:, -1:], vector, 1., -2.))
    probes["revis_prefill"] = {
        "same_last_token_formula": distance(upstream[:, -1:], local[:, -1:]) == 0,
        "upstream_changed_positions": int((upstream != hidden).any(-1).sum()),
        "local_changed_positions": int((local != hidden).any(-1).sum()),
    }

    # RUDDER: invoke actual capture/pooling function on a synthetic attention module.
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Identity()

        def forward(self, hidden_states):
            return (self.projection(hidden_states.flip(-1) * 0.25),)

    class ResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

        def forward(self, hidden_states):
            return hidden_states + self.self_attn(hidden_states)[0]

    class FixtureModel(nn.Module):
        device = torch.device("cpu")

        def __init__(self):
            super().__init__()
            self.block = ResidualBlock()

        def forward(self, hidden_states, **kwargs):
            return self.block(hidden_states)

    model = FixtureModel()
    hidden = torch.tensor([[[1., 2.], [3., 1.]]])
    carrier = SimpleNamespace(mask=None)
    rd = definitions(root / "rudder/Qwen/qwen_methods.py",
                     ["_CardCapture", "compute_card_vector_batch",
                      "BayesianGatingHookMaskedDynamic", "_align_mask_to_seq"],
                     mask_carrier=carrier, qwenvl_singleturn_prompt=lambda q: q,
                     get_qwenvl_self_attn=lambda model, layer: model.block.self_attn)
    def processor(**kwargs):
        return {"hidden_states": hidden.clone()}
    author_card = rd["compute_card_vector_batch"](model, processor, [None], ["fixture"], 0)
    adapter = SimpleNamespace(layers=[model.block],
                              attn_updates=[model.block.self_attn.projection])
    original = model(hidden)
    with build_baseline("rudder", adapter, config={
        "layer": 0, "alpha": 1., "gate_min": 1., "gate_max": 1.,
    }):
        changed = model(hidden)
    local_card = (changed - original)[:, -1]
    hook = rd["BayesianGatingHookMaskedDynamic"](clamp=(0.05, 1.))
    hook.set_vector(torch.tensor([[1., 0.]]))
    update = torch.tensor([[[0., 1.]]])
    hook(None, (), (update,))
    author_gate = hook.fetch_last_stats()["gate"]
    shared_similarity = F.cosine_similarity(update, torch.tensor([[[1., 0.]]]), dim=-1)
    torch.testing.assert_close(author_gate.squeeze(-1), beta_gate(shared_similarity))
    probes["rudder_card"] = {"upstream": author_card.tolist(), "local": local_card.tolist(),
                              "cosine": directional_agreement(author_card, local_card)}
    probes["rudder_gate"] = {
        "same_beta_formula_with_same_similarity": True,
        "upstream_gate_on_attention_output": float(author_gate.item()),
        "local_gate_on_block_input_for_example": float(beta_gate(torch.tensor(1.)).item()),
    }

    # VTI: execute original layer, original image mask, and original text PCA.
    vh = definitions(root / "vti/vti_utils/llm_layers.py", ["VTILayer"])
    hidden = torch.tensor([[[3., 4.], [4., 3.]]])
    vector = torch.tensor([0., 1.])
    upstream = vh["VTILayer"](vector[None], [0.5])(hidden.clone())
    block = nn.Identity()
    artifact = {"name": "vti", "vectors": {0: vector}, "vision_vectors": {}}
    with build_baseline("vti", SimpleNamespace(layers=[block]), artifact, {"alpha": 0.5}):
        local = block(hidden.clone())
    probes["vti_injection"] = {
        "original_norm": hidden.norm(dim=-1).tolist(),
        "upstream_norm": upstream.float().norm(dim=-1).tolist(),
        "local_norm": local.norm(dim=-1).tolist(),
        "max_abs_difference": distance(upstream, local),
        "note": "Same tensor input only; real hook boundaries also differ (MLP vs block).",
    }
    vp = definitions(root / "vti/vti_utils/pca.py", ["svd_flip", "PCA"])
    samples = torch.tensor([
        [[1., 2., 0.], [2., 0., 1.]], [[3., 1., 1.], [1., 2., 3.]],
        [[0., 4., 2.], [3., 3., 1.]], [[2., 0., 3.], [4., 1., 2.]],
    ])
    captured = [(torch.zeros_like(row), row) for row in samples]
    vi = definitions(root / "vti/vti_utils/icv_utils.py", ["obtain_textual_vti", "mask_patches"],
                     PCA=vp["PCA"], get_hiddenstates=lambda *args: captured)
    author_direction, _ = vi["obtain_textual_vti"](None, None, None)
    local_direction = torch.stack([principal_direction(samples[:, i]) for i in range(2)])
    probes["vti_text_pca"] = {
        "upstream": author_direction.tolist(), "local": local_direction.tolist(),
        "cosine": directional_agreement(author_direction, local_direction),
        "note": "Original flattened-layer PC1+mean vs independent unit PC1 for each layer.",
    }
    pixels = torch.arange(1., 49.).reshape(1, 3, 4, 4)
    chosen = torch.randperm(4, generator=torch.Generator().manual_seed(7))[:1]
    author_pixels = vi["mask_patches"](pixels[0], chosen, patch_size=2)[None]
    local_pixels = _masked_pixels({"pixel_values": pixels}, torch.Generator().manual_seed(7),
                                  0.25, 2)["pixel_values"]
    probes["vti_mask"] = {"max_abs_difference": distance(author_pixels, local_pixels),
                           "upstream_replacement": "per-channel image mean", "local_replacement": 0}

    # GrAInS: original public hook has a sequence-axis norm and a 10x multiplier.
    gs = definitions(root / "grains/src/utils/steering.py", ["add_steering_hook"])

    class TupleBlock(nn.Module):
        def forward(self, hidden_states):
            return (hidden_states.clone(),)

    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([TupleBlock()])
    hidden = torch.tensor([[[3., 4.]]])
    vector = torch.tensor([0., 1.])
    handles = gs["add_steering_hook"](model, 0, vector, 0.5)
    try:
        upstream = model.model.layers[0](hidden)[0]
    finally:
        for handle in handles:
            handle.remove()
    local = norm_preserving_add(hidden, vector, 0.5)
    probes["grains_single_token_hook"] = {
        "upstream": upstream.tolist(), "local": local.tolist(),
        "upstream_change": distance(upstream, hidden), "local_change": distance(local, hidden),
        "note": "Public dim=1 norm can cancel positive same-sign updates at decode length 1.",
    }
    logits = torch.tensor([[[0., 0., 0.], [0., 3., 0.], [0., 0., 3.], [3., 0., 0.]]])
    labels = torch.tensor([[-100, -100, 1, 2]])
    probes["grains_objective"] = {
        "upstream_last_input_token_at_final_logit": float(logits[0, -1].log_softmax(-1)[2]),
        "local_shifted_completion_log_probability": float(sequence_log_probability(logits, labels)),
        "note": "Numerical illustration of inspected objectives, not full upstream IG execution.",
    }
    steering_tree = ast.parse((root / "grains/src/utils/steering.py").read_text())
    defined = {node.name for node in steering_tree.body if isinstance(node, ast.FunctionDef)}
    missing = sorted({"generate_answer_dataset_vlm", "steer_and_generate_dataset_vlm"} - defined)
    probes["upstream_entrypoint_gaps"] = {
        "grains_generation_imports_missing_definitions": missing,
        "revis_missing_local_modules": [p for p in ("utils/text_halu.py", "utils/mmhal_bench.py")
                                         if not (root / "revis" / p).exists()],
        "rudder_chair_import_target_exists": bool(list((root / "rudder").rglob("methods_decodings_qwen.py"))),
    }
    report["source_file_sha256"] = {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
        if path.parts[len(root.parts)] in report["sources"]
    }
    mismatches = {
        "revis_vector": probes["revis_vector"]["max_abs_difference"] > 1e-5,
        "revis_prefill": probes["revis_prefill"]["upstream_changed_positions"]
                          != probes["revis_prefill"]["local_changed_positions"],
        "rudder_card": distance(author_card, local_card) > 1e-5,
        "vti_injection": probes["vti_injection"]["max_abs_difference"] > 1e-3,
        "vti_text_pca": distance(author_direction, local_direction) > 1e-5,
        "vti_mask": probes["vti_mask"]["max_abs_difference"] > 1e-5,
        "grains_single_token_hook": probes["grains_single_token_hook"]["upstream_change"]
                                    != probes["grains_single_token_hook"]["local_change"],
    }
    report["mismatched_probes"] = [name for name, differs in mismatches.items() if differs]
    report["conclusion"] = (
        "not_equivalent; successful audit execution is not a parity pass"
        if report["mismatched_probes"] else "no_difference_in_probes; full parity remains unverified"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.upstream_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"conclusion": report["conclusion"], "probes": report["probes"]}, indent=2))


if __name__ == "__main__":
    main()
