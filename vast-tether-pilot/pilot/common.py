"""Model, prompt, vocabulary and CHAIR helpers shared by the stages of the early-evidence pilot.

Layout (same locally and on the server): ROOT/pilot (this code), ROOT/repo (paper-tables-code-20261104),
ROOT/tools/chair (Maxlinn CHAIR port at 4087a26), ROOT/tools/vlm-spectra, ROOT/data/coco, ROOT/out.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str  # the paper's revisions
    model_class: str
    grid: int  # image tokens per side at 448x448
    depth_layers: tuple[int, ...]  # 0-based decoder layers at 10, 25, 50, 75 and 100% depth
    transformers_major: int = 4  # its environment: 4 is .venv (transformers 4.57.3, Spectra), 5 is .venv35 (5.22.2)
    uses_spectra: bool = True  # Spectra has adapters for Qwen2.5-VL and Qwen3-VL only
    chat_options: tuple[tuple[str, Any], ...] = ()  # extra arguments of the chat template
    stop_tokens: tuple[str, ...] = ()  # set as eos tokens when the checkpoint ships no generation config


MODELS = {
    "qwen3": ModelSpec("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                       "Qwen3VLForConditionalGeneration", 14, (3, 8, 17, 26, 35)),
    "qwen25": ModelSpec("Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5",
                        "Qwen2_5_VLForConditionalGeneration", 16, (2, 6, 13, 20, 27)),
    # Thinks by default; without thinking the prompt ends in an empty <think></think>. Its repository has no
    # generation_config.json, so generate would stop only at <|endoftext|>, not at the end of the answer.
    "qwen35": ModelSpec("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "Qwen3_5ForConditionalGeneration", 14,
                        (2, 7, 15, 23, 31), transformers_major=5, uses_spectra=False,
                        chat_options=(("enable_thinking", False),), stop_tokens=("<|im_end|>", "<|endoftext|>")),
}
PILOT_MODEL = os.environ.get("PILOT_MODEL", "qwen3")
SPEC = MODELS[PILOT_MODEL]
MODEL_ID, REVISION = SPEC.model_id, SPEC.revision

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / PILOT_MODEL
CHAIR_STATE = ROOT / "out" / "chair_state.pkl"  # the same for every model
COCO = ROOT / "data" / "coco"
CHAIR_PY = ROOT / "tools" / "chair" / "chair.py"
REPO_SCRIPTS = ROOT / "repo" / "scripts"

PROMPT = "Please describe this image in detail."
IMAGE_SIZE = 448
GRAY_RGB = (128, 128, 128)
MAX_NEW_TOKENS = 512
EARLY_STEPS = 16


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_image(path: str | Path) -> Image.Image:
    """The paper's preprocessing (backend.ModelAdapter.prepare): exif transpose, RGB, bicubic resize to a square."""
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)


def gray_image() -> Image.Image:
    return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), GRAY_RGB)


def fast_patch_embedding(model: Any) -> None:
    """Run the vision tower's Conv3d in float32 and round its output back to the model's dtype.

    In bf16 it takes about 8 s per image on Turing, 20 times the rest of the vision tower (tests/diagnose_gpu.py).
    """
    conv = model.model.visual.patch_embed.proj

    def forward(pixels: torch.Tensor) -> torch.Tensor:
        bias = None if conv.bias is None else conv.bias.float()
        output = torch.nn.functional.conv3d(pixels.float(), conv.weight.float(), bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return output.to(pixels.dtype)

    conv.forward = forward


def configure_generation(model: Any, processor: Any, spec: ModelSpec | None = None) -> None:
    """Stop at the end of the answer when the checkpoint ships no generation config, as Qwen3.5's does not."""
    spec = spec or SPEC
    if spec.stop_tokens:
        tokenizer = processor.tokenizer
        model.generation_config.eos_token_id = [tokenizer.convert_tokens_to_ids(token) for token in spec.stop_tokens]
        model.generation_config.pad_token_id = tokenizer.pad_token_id


def load_model_and_processor(attn: str = "sdpa") -> tuple[Any, Any]:
    import transformers

    major = int(transformers.__version__.split(".")[0])
    if major != SPEC.transformers_major:
        raise RuntimeError(f"{PILOT_MODEL} runs on transformers {SPEC.transformers_major}.x; this environment has {transformers.__version__}")
    precision = {"torch_dtype": torch.bfloat16} if major < 5 else {"dtype": torch.bfloat16}
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    model = getattr(transformers, SPEC.model_class).from_pretrained(
        MODEL_ID, revision=REVISION, attn_implementation=attn, **precision
    ).to("cuda")
    fast_patch_embedding(model)
    configure_generation(model, processor)
    return model.eval(), processor


def prepare_rows(processor: Any, images: Sequence[Image.Image], prefix: Sequence[int] = ()) -> tuple[Any, int]:
    """One row per image with the same prompt, followed by the same caption prefix. Returns (batch, prompt_length)."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **dict(SPEC.chat_options))
    batch = processor(text=[text] * len(images), images=list(images), return_tensors="pt")
    prompt_length = batch["input_ids"].shape[1]
    if not (batch["input_ids"] == batch["input_ids"][0]).all():
        raise ValueError("rows must share their prompt tokens: every image has to give the same number of image tokens")
    if len(prefix):
        add = torch.tensor([list(prefix)] * len(images), dtype=batch["input_ids"].dtype)
        for key, value in list(batch.items()):
            if key == "input_ids":
                batch[key] = torch.cat([value, add], 1)
            elif key == "attention_mask":
                batch[key] = torch.cat([value, torch.ones_like(add, dtype=value.dtype)], 1)
            elif torch.is_tensor(value) and value.dim() == 2 and value.shape == (len(images), prompt_length):
                batch[key] = torch.cat([value, torch.zeros_like(add, dtype=value.dtype)], 1)
    return batch, prompt_length


def object_vocabulary(tokenizer: Any, cache: Path) -> list[int]:
    """V_obj of the paper: rbd_common.build_vobj, unchanged, on this model's tokenizer."""
    if cache.exists():
        return json.loads(cache.read_text())
    sys.path.insert(0, str(REPO_SCRIPTS))
    from rbd_common import build_vobj

    ids = build_vobj(tokenizer)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(ids))
    return ids


def load_chair(annotations: Path, cache: Path) -> Any:
    """The unmodified CHAIR scorer. Its constructor reads every COCO annotation, so its state is cached once."""
    spec = importlib.util.spec_from_file_location("chair_scorer", CHAIR_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if cache.exists():
        scorer = module.CHAIR.__new__(module.CHAIR)
        scorer.__dict__.update(pickle.loads(cache.read_bytes()))
        return scorer
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
        scorer = module.CHAIR(str(annotations))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(scorer.__dict__))
    return scorer


def mention_token_positions(tokenizer: Any, ids: Sequence[int], scorer: Any) -> list[tuple[str, str, int]]:
    """(node, word, token index where the mention starts) for every mention. Copied from verified_decoding_v3.py."""
    import nltk

    text = tokenizer.decode(ids, skip_special_tokens=True)
    low = text.lower()
    words = nltk.word_tokenize(low)
    offsets, cursor = [], 0
    for word in words:
        found = low.find(word, cursor)
        if found < 0:
            found = cursor
        offsets.append(found)
        cursor = found + len(word)
    mention_words, nodes, indices, _ = scorer.caption_to_words(text)
    cumulative = [0]
    for i in range(1, len(ids) + 1):
        cumulative.append(len(tokenizer.decode(ids[:i], skip_special_tokens=True)))
    mentions = []
    for node, word, index in zip(nodes, mention_words, indices):
        char = offsets[index] if index < len(offsets) else len(text)
        t = next((i for i in range(len(ids)) if cumulative[i] <= char < cumulative[i + 1]), len(ids) - 1)
        mentions.append((node, word, t))
    return mentions


def label_caption(tokenizer: Any, ids: Sequence[int], scorer: Any, image_id: int) -> dict[str, Any]:
    """CHAIR's verdict on one caption, plus the first mention of each object with the token that starts it."""
    from pattern.en import singularize  # what CHAIR applies to words, so that ' people' matches the word 'person'

    ground_truth = set(scorer.imid_to_objects[image_id])
    mentions = mention_token_positions(tokenizer, ids, scorer)
    first, misaligned = [], 0
    seen: set[str] = set()
    for node, word, t in mentions:
        if node in seen:
            continue
        seen.add(node)
        token_text = singularize(tokenizer.decode([ids[t]]).strip().lower())
        if not token_text or not (word.startswith(token_text[:3]) or token_text.startswith(word[:3])):
            misaligned += 1
            continue
        first.append({"node": node, "word": word, "t": t, "token": int(ids[t]), "order": len(seen),
                      "hallucinated": node not in ground_truth})
    hallucinated = [node for node, _, _ in mentions if node not in ground_truth]
    return {"first_mentions": first, "misaligned": misaligned, "n_mentions": len(mentions),
            "n_hallucinated_mentions": len(hallucinated), "hallucinated_caption": bool(hallucinated),
            "ground_truth": sorted(ground_truth), "mentioned": sorted(seen),
            "n_correct_distinct": len(seen & ground_truth)}


def canonical_tokens(tokenizer: Any, scorer: Any, vocabulary: Iterable[int]) -> dict[str, int]:
    """One word-initial V_obj token per COCO object: its own name, the head noun of a two-word name, else a synonym.

    Objects with one-word names choose first, so 'hot dog' cannot take ' dog' from dog.
    """
    allowed = set(vocabulary)
    synonyms: dict[str, list[str]] = {}
    for word, node in scorer.inverse_synonym_dict.items():
        if node:  # blank lines of CHAIR's synonym list leave an empty entry
            synonyms.setdefault(node, []).append(word)
    assigned: dict[str, int] = {}
    taken: set[int] = set()
    for node in sorted(synonyms, key=lambda name: (" " in name, name)):
        candidates = [node.split()[-1]] + [w for w in synonyms[node] if " " not in w and w != node]
        for word in candidates:
            token = tokenizer.convert_tokens_to_ids("Ġ" + word)
            if token in allowed and token not in taken:
                assigned[node] = int(token)
                taken.add(int(token))
                break
    return assigned


def tracked_tokens(canonical: dict[str, int], first_mentions: Sequence[dict[str, Any]]) -> np.ndarray:
    """Tokens whose evidence is stored for one image: all canonical objects, then this caption's first mentions."""
    return np.array(sorted(set(canonical.values()) | {m["token"] for m in first_mentions}), dtype=np.int64)
