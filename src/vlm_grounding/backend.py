"""Explicit Hugging Face adapters; no downloads occur until ``load_adapter``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps


def _attribute(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _first_path(root: Any, paths: tuple[str, ...]) -> Any:
    for path in paths:
        try:
            return _attribute(root, path)
        except AttributeError:
            continue
    raise ValueError(f"Unsupported model layout: expected one of {paths}")


class ModelAdapter:
    """Batch-one image/text inference, differentiable margins and safe labels.

    Head interventions act on ``o_proj`` inputs: these contain query-head
    outputs even when the model uses grouped-query attention.
    """

    def __init__(self, model: Any, processor: Any, config: Mapping[str, Any]) -> None:
        self.model = model
        self.processor = processor
        self.config = dict(config)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.image_size = self.config.get("image_size")
        self.model_type = model.config.model_type
        self.layers = _first_path(
            model,
            (
                "model.language_model.layers",
                "language_model.model.layers",
                "model.layers",
            ),
        )
        if self.model_type == "qwen2_5_vl":
            self.vision_layers = _first_path(model, ("model.visual.blocks", "visual.blocks"))
        elif self.model_type == "llava":
            self.vision_layers = _first_path(
                model,
                (
                    "model.vision_tower.vision_model.encoder.layers",
                    "vision_tower.vision_model.encoder.layers",
                ),
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        text_config = getattr(model.config, "text_config", model.config)
        self.num_heads = int(text_config.num_attention_heads)
        self.hidden_size = int(text_config.hidden_size)
        self.attn_projections = [layer.self_attn.o_proj for layer in self.layers]
        for projection in self.attn_projections:
            if projection.in_features % self.num_heads:
                raise ValueError("Attention projection input is not divisible by query head count")
        self.yes_token_id = self._answer_token_id(str(self.config.get("yes_token", "Yes")))
        self.no_token_id = self._answer_token_id(str(self.config.get("no_token", "No")))
        if self.yes_token_id == self.no_token_id:
            raise ValueError("Yes and No must map to different tokens")
        self.model.eval()

    def _answer_token_id(self, text: str) -> int:
        ids = self.processor.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Answer token {text!r} must tokenize to exactly one token; got {ids}")
        return int(ids[0])

    def prepare(
        self, image_path: str | None, prompt: str, answer: str | None = None
    ) -> dict[str, torch.Tensor]:
        content: list[dict[str, str]] = []
        image = None
        if image_path is not None:
            with Image.open(Path(image_path)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                if self.image_size is not None:
                    size = int(self.image_size)
                    if size < 1:
                        raise ValueError("image_size must be positive")
                    image = image.resize((size, size), Image.Resampling.BICUBIC)
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        processor_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if image is not None:
            processor_kwargs["images"] = [image]
        prompt_batch = self.processor(text=[prompt_text], **processor_kwargs)
        if answer is None:
            batch = dict(prompt_batch)
        else:
            eos = self.processor.tokenizer.eos_token or ""
            batch = dict(self.processor(text=[prompt_text + answer + eos], **processor_kwargs))
            prompt_ids = prompt_batch["input_ids"]
            prefix = batch["input_ids"][:, : prompt_ids.shape[1]]
            if prefix.shape != prompt_ids.shape or not torch.equal(prefix, prompt_ids):
                raise ValueError("Answer tokenization changed the prompt prefix; labels are unsafe")
            labels = batch["input_ids"].clone()
            labels[:, : prompt_ids.shape[1]] = -100
            if not bool((labels != -100).any()):
                raise ValueError("Answer must contain at least one supervised token")
            batch["labels"] = labels
        return {
            key: value.to(
                device=self.device,
                dtype=self.dtype if value.is_floating_point() else value.dtype,
            )
            for key, value in batch.items()
            if torch.is_tensor(value)
        }

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Any:
        inputs = dict(batch)
        inputs["use_cache"] = False
        inputs["return_dict"] = True
        return self.model(**inputs)

    def answer_margin(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        logits = self.forward(batch).logits
        return logits[0, -1, self.yes_token_id] - logits[0, -1, self.no_token_id]

    @torch.no_grad()
    def generate(self, batch: Mapping[str, torch.Tensor], max_new_tokens: int) -> str:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        inputs = {key: value for key, value in batch.items() if key != "labels"}
        # HF generate does not inherit the use_cache=False of our forward wrapper.
        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        new_ids = outputs[:, inputs["input_ids"].shape[1] :]
        self.last_generated_tokens = int(new_ids.shape[1])
        return self.processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


def load_adapter(config: Mapping[str, Any]) -> ModelAdapter:
    """Load a user-selected checkpoint. Only this function can download weights."""
    from transformers import (
        AutoProcessor,
        LlavaForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
    )

    family = config.get("family", "qwen2_5_vl")
    classes = {
        "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
        "llava": LlavaForConditionalGeneration,
    }
    if family not in classes:
        raise ValueError(f"family must be one of {sorted(classes)}")
    checkpoint = str(config["checkpoint"])
    device = str(config.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run CPU tests now; use a GPU machine for the 7B experiment."
        )
    dtype_name = str(config.get("dtype", "bfloat16"))
    dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype_name not in dtypes:
        raise ValueError(f"dtype must be one of {sorted(dtypes)}")
    revision = str(config.get("revision", "main"))
    shared = {
        "revision": revision,
        "local_files_only": bool(config.get("local_files_only", False)),
        "trust_remote_code": False,
    }
    processor = AutoProcessor.from_pretrained(checkpoint, **shared)
    model = classes[family].from_pretrained(
        checkpoint,
        torch_dtype=dtypes[dtype_name],
        attn_implementation=str(config.get("attn_implementation", "eager")),
        **shared,
    ).to(device)
    return ModelAdapter(model, processor, config)
