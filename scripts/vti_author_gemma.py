"""Gemma 4 I/O around directly imported, unmodified official VTI functions.

The upstream PCA, state aggregation, mean-patch masking and MLP intervention
are executed from the clean Git checkout. No local baseline implementation
is imported. This bridge supports Gemma 4 on one visible CUDA device.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

from vti_author import COMMIT, demos, sha, verify_source
from vlm_grounding.backend import load_adapter
from vlm_grounding.cli import (
    _config, _evaluation_protocol, _provenance, _reserve, _revision, _seed, _select,
    _synchronize, _validate_artifact, _versions, _write_json,
)
from vlm_grounding.data import read_manifest


def load_author(repo: Path) -> SimpleNamespace:
    """Import full author modules under a private package, without AST rewriting."""
    verify_source(repo)
    name = "_official_vti_" + sha(repo / "vti_utils/utils.py")[:12]
    package = ModuleType(name)
    package.__path__ = [str(repo / "vti_utils")]
    sys.modules[name] = package
    modules = {}
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        for stem in ("pca", "utils", "llm_layers"):
            key = f"{name}.{stem}"
            spec = importlib.util.spec_from_file_location(key, repo / "vti_utils" / f"{stem}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            spec.loader.exec_module(module)
            modules[stem] = module
    finally:
        sys.dont_write_bytecode = previous
    return SimpleNamespace(**modules)


def load_gemma(config: dict):
    model = dict(config["model"])
    if model.get("family") != "gemma4":
        raise ValueError("This architecture bridge supports Gemma 4 only")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES before extract/run")
    if str(model.get("device", "cuda")) not in ("cuda", "cuda:0"):
        raise ValueError("Use device=cuda or cuda:0 inside the single visible GPU")
    # Code preparation must never turn into an implicit checkpoint download.
    model["local_files_only"] = True
    adapter = load_adapter(model)
    torch.cuda.set_device(adapter.device)
    return adapter


class DtypeBridge(nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, value):
        # Upstream VTILayer returns FP16, including for BF16/FP32 model inputs.
        direction = self.operator.vti_direction
        if direction.ndim == 3 and value.shape[-2:] != direction.shape[-2:]:
            raise ValueError("Vision token grid differs from the VTI extraction grid")
        result = self.operator(value).to(dtype=value.dtype)
        if result.shape != value.shape:
            raise ValueError("VTI intervention changed the activation shape")
        return result


@contextmanager
def intervention(adapter, author, *, text=None, vision=None, alpha_text=0.9, alpha_image=0.9):
    """Call upstream add_vti_layers; restore original modules even on exceptions."""
    original = []
    try:
        for layers, directions, alpha in (
            (adapter.layers, text, alpha_text), (adapter.vision_layers, vision, alpha_image)
        ):
            if directions is None or alpha == 0:
                continue
            if directions.shape[0] != len(layers):
                raise ValueError("VTI direction count differs from the model's layer count")
            if not torch.isfinite(directions).all():
                raise ValueError("Non-finite VTI direction")
            for layer in layers:
                original.append((layer, layer.mlp))
            # Limit upstream layer discovery to this known text/vision stack.
            root = nn.Module()
            root.layers = nn.ModuleList(list(layers))
            if list(author.llm_layers.get_layers(root)) != list(layers):
                raise ValueError("Upstream layer discovery selected an unsupported nested stack")
            author.llm_layers.add_vti_layers(
                root, directions.unsqueeze(1).to(adapter.device), alpha=[alpha]
            )
            for layer in layers:
                layer.mlp[1] = DtypeBridge(layer.mlp[1])
        yield
    finally:
        for layer, mlp in reversed(original):
            layer.mlp = mlp


class VisionCall:
    def __init__(self, adapter):
        self.adapter = adapter
        self.positions = None

    def __call__(self, pixels, **kwargs):
        # Original state collection calls .half().cuda(); restore Gemma's dtype
        # and supply its native 2D patch positions without changing that source.
        # Gemma's generic hidden-state recorder includes the pooled, unbatched
        # final features. VTI requires the embedding and each encoder output on
        # the same unpooled patch grid, as provided by its original CLIP model.
        # Collect native tensors by hooks; no encoder output is changed.
        states = []
        layers = self.adapter.vision_layers
        handles = [layers[0].register_forward_pre_hook(
            lambda module, args: states.append(args[0].detach()) and None
        )]
        for layer in layers:
            def capture(module, args, output):
                states.append(output.detach())
            handles.append(layer.register_forward_hook(capture))
        try:
            self.adapter.model.model.vision_tower(
                pixel_values=pixels.to(device=self.adapter.device, dtype=self.adapter.dtype),
                pixel_position_ids=self.positions.to(self.adapter.device),
                **{**kwargs, 'output_hidden_states': False},
            )
        finally:
            for handle in handles:
                handle.remove()
        if len(states) != len(layers) + 1 or any(
                state.ndim != 3 or state.shape != states[0].shape for state in states):
            raise ValueError('Gemma vision encoder states must share one native patch grid')
        return SimpleNamespace(hidden_states=tuple(states))


class VisionInputs:
    """Lazily prepare one image's corruptions, preserving the original loop order."""
    def __init__(self, adapter, rows, author, call, trials, ratio):
        self.adapter, self.rows, self.author, self.call = adapter, rows, author, call
        self.trials, self.ratio = trials, ratio
        self.index, self.cached = None, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.index == index:
            return self.cached
        row = self.rows[index]
        batch = self.adapter.prepare(row["image_path"], row["question"])
        pixels = batch["pixel_values"].detach().cpu()
        positions = batch["image_position_ids"].detach().cpu()
        valid = (positions[0] >= 0).all(-1)
        pos, real = positions[0, valid], pixels[0, valid]
        gh, gw = (int(pos[:, axis].max()) + 1 for axis in (0, 1))
        patch = int(self.adapter.processor.image_processor.patch_size)
        if gh * gw != len(real) or real.shape[-1] != patch * patch * 3:
            raise ValueError("Unsupported Gemma image patch layout")
        order = (pos[:, 0] * gw + pos[:, 1]).argsort()
        grid = real[order].reshape(gh, gw, patch, patch, 3)
        chw = grid.permute(4, 0, 2, 1, 3).reshape(3, gh * patch, gw * patch)
        corruptions = []
        for _ in range(self.trials):
            ids = torch.randperm(gh * gw)[:int(self.ratio * gh * gw)]
            masked = self.author.utils.mask_patches(chw, ids, patch_size=patch)
            flat = masked.reshape(3, gh, patch, gw, patch).permute(1, 3, 2, 4, 0)
            changed = pixels.clone()
            changed[0, valid] = flat.reshape(gh * gw, -1)[order.argsort()]
            corruptions.append(changed[0])
        self.call.positions = positions
        self.index, self.cached = index, [corruptions, pixels[0]]
        return self.cached


class TextInputs:
    def __init__(self, adapter, rows):
        self.adapter, self.rows = adapter, rows
        self.index, self.cached = None, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.index == index:
            return self.cached
        adapter, row = self.adapter, self.rows[index]
        with Image.open(row["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if adapter.image_size is not None:
            size = int(adapter.image_size)
            image = image.resize((size, size), Image.Resampling.BICUBIC)
        # Upstream get_demos passes its final demo's question to every prompt.
        question = self.rows[-1]["question"]
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}
        ]}]
        prefix = adapter.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        pairs = []
        for caption in (row["h_value"], row["value"]):
            # Match upstream caption-state collection: no appended EOS.
            batch = adapter.processor(text=[prefix + caption], images=[image], return_tensors="pt")
            batch = {key: value.to(device=adapter.device,
                                  dtype=adapter.dtype if value.is_floating_point() else value.dtype)
                     for key, value in batch.items() if torch.is_tensor(value)}
            batch["use_cache"] = False
            pairs.append(batch)
        self.index, self.cached = index, pairs
        return pairs


def extract(args, source):
    _reserve([args.output], args.overwrite)
    if args.num_trials < 1 or not 0 < args.mask_ratio < 1:
        raise ValueError("Use num-trials >= 1 and 0 < mask-ratio < 1")
    if any(not math.isfinite(value) or value < 0 for value in (args.alpha_text, args.alpha_image)):
        raise ValueError("VTI strengths must be finite and nonnegative")
    rows = demos(args.repo, args.image_root, args.num_demos)
    config = _config(str(args.config))
    config.setdefault("generation", {})["seed"] = args.seed
    _seed(config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    author = load_author(args.repo)
    adapter = load_gemma(config)
    call = VisionCall(adapter)
    inputs = VisionInputs(adapter, rows, author, call, args.num_trials, args.mask_ratio)
    visual, _ = author.utils.obtain_visual_vti(SimpleNamespace(vision_model=call), inputs, rank=1)
    vision = visual[1:].contiguous()
    del inputs
    # Match the official runner: visual VTI is active during textual extraction.
    with intervention(adapter, author, vision=vision, alpha_image=args.alpha_image):
        text, _ = author.utils.obtain_textual_vti(
            adapter.model, TextInputs(adapter, rows), None, rank=1
        )
    text = text[1:].contiguous()
    for value, layers in ((text, adapter.layers), (vision, adapter.vision_layers)):
        if value.shape[0] != len(layers) or not torch.isfinite(value).all() or value.norm() == 0:
            raise ValueError("VTI produced invalid directions")
    if verify_source(args.repo) != source:
        raise ValueError("Author sources changed during extraction")
    training = [{"id": row["id"], "group_id": row["group_id"]} for row in rows]
    artifact = {
        "format_version": 1, "name": "vti_author", "author_source": source,
        "text_direction": text.cpu(), "vision_direction": vision.cpu(),
        "alpha_text": args.alpha_text, "alpha_image": args.alpha_image,
        "settings": {"num_demos": len(rows), "num_trials": args.num_trials,
                     "mask_ratio": args.mask_ratio, "rank": 1, "seed": args.seed},
        "training_images": [{key: row[key] for key in ("group_id", "image_sha256")} for row in rows],
        "architecture_bridge": ["Gemma processor and patch positions; padding preserved",
                                "Gemma MLP paths; upstream FP16 output cast to input dtype",
                                "Fixed processor token grid required at inference"],
        "text_extraction_vision_active": args.alpha_image != 0,
        "_provenance": _provenance(adapter, config, [str(args.repo / REQUIRED_DEMOS)], training),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    return {"status": "extracted", "artifact": str(args.output.resolve()),
            "settings": artifact["settings"], "author_commit": COMMIT}


REQUIRED_DEMOS = "experiments/data/hallucination_vti_demos.jsonl"


def run(args, source):
    sidecar = Path(str(args.output) + ".metadata.json")
    _reserve([args.output, sidecar], args.overwrite)
    config = _config(str(args.config))
    records = _select(read_manifest(args.manifest), args.split, args.limit, task=args.task)
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    if artifact.get("name") != "vti_author" or artifact.get("author_source") != source:
        raise ValueError("Artifact is not from these pinned official VTI sources")
    _validate_artifact(artifact, config["model"], records)
    training_groups = {row["group_id"] for row in artifact["training_images"]}
    training_hashes = {row["image_sha256"] for row in artifact["training_images"]}
    for record in records:
        if record["group_id"] in training_groups or sha(Path(record["image"])) in training_hashes:
            raise ValueError("Evaluation images overlap VTI direction-extraction images")
    _seed(config)
    author = load_author(args.repo)
    adapter = load_gemma(config)
    _validate_artifact(artifact, config["model"], records, adapter)
    protocol, fingerprint = _evaluation_protocol(adapter, config)
    cap = int(config.get("generation", {}).get("max_new_tokens", 128))
    metadata = {
        "method": "vti_author", "implementation": "official-author-code-gemma-bridge",
        "author_reproduction": False, "author_source": source, "config": config,
        "versions": _versions(), "evaluation_protocol": protocol,
        "checkpoint_revision": _revision(adapter),
        "protocol_fingerprint": fingerprint, "selected_split": args.split,
        "selected_task": args.task, "record_ids": [str(row["id"]) for row in records],
        "cohort_complete": args.limit is None, "status": "running", "completed_records": 0,
        "manifest_sha256": sha(args.manifest), "artifact_sha256": sha(args.artifact),
        "architecture_bridge": artifact["architecture_bridge"],
    }
    _write_json(sidecar, metadata)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as sink:
            for record in records:
                batch = adapter.prepare(record["image"], record["prompt"])
                _synchronize(adapter)
                started = time.perf_counter()
                with intervention(adapter, author, text=artifact["text_direction"],
                                  vision=artifact["vision_direction"],
                                  alpha_text=artifact["alpha_text"], alpha_image=artifact["alpha_image"]):
                    prediction = adapter.generate(batch, max_new_tokens=cap)
                _synchronize(adapter)
                row = {**record, "prediction": prediction, "method": "vti_author",
                       "generated_tokens": adapter.last_generated_tokens,
                       "elapsed_seconds": time.perf_counter() - started,
                       "protocol_fingerprint": fingerprint}
                sink.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                sink.flush()
                metadata["completed_records"] += 1
        if verify_source(args.repo) != source:
            raise ValueError("Author sources changed during generation")
        metadata["status"] = "complete"
    except BaseException:
        metadata["status"] = "failed"
        raise
    finally:
        _write_json(sidecar, metadata)
    return {"status": "complete", "output": str(args.output.resolve()), "records": len(records)}
