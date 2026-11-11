"""Check the pinned Gemma cache without loading weights or contacting the Hub."""

import json
import os
from pathlib import Path
import shutil
import struct
import sys

from huggingface_hub import try_to_load_from_cache
from transformers import AutoProcessor


def check():
    repo = Path(__file__).resolve().parents[1]
    config = json.loads((repo / "configs/gemma4_e4b.local.json").read_text())
    model = config["model"]
    cache = Path(os.environ["HF_HUB_CACHE"])
    assert model["family"] == "gemma4" and model["local_files_only"] is True
    required = ["config.json", "generation_config.json", "processor_config.json",
                "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                "model.safetensors"]
    paths = {}
    for name in required:
        path = try_to_load_from_cache(
            model["checkpoint"], name, cache_dir=cache, revision=model["revision"],
        )
        if not isinstance(path, str) or not Path(path).is_file():
            raise RuntimeError(f"Pinned model file is missing: {name}")
        paths[name] = Path(path)
    with paths["model.safetensors"].open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < length < 100_000_000:
            raise ValueError("Invalid safetensors header length")
        header = json.loads(stream.read(length))
    end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
    expected = 8 + length + end
    if paths["model.safetensors"].stat().st_size != expected:
        raise ValueError("Model weights are incomplete")
    processor = AutoProcessor.from_pretrained(
        model["checkpoint"], revision=model["revision"], cache_dir=cache,
        local_files_only=True, trust_remote_code=False,
    )
    tokens = {word: processor.tokenizer.encode(word, add_special_tokens=False)
              for word in ("Yes", "No")}
    if any(len(ids) != 1 for ids in tokens.values()) or tokens["Yes"] == tokens["No"]:
        raise ValueError("Yes/No answer tokens do not match the adapter requirements")
    return {"status": "ready", "checkpoint": model["checkpoint"],
            "revision": model["revision"], "snapshot": str(paths["config.json"].parent),
            "model_bytes": expected, "tensor_count": len(header) - int("__metadata__" in header),
            "processor": type(processor).__name__, "answer_tokens": tokens,
            "free_disk_bytes": shutil.disk_usage(repo).free, "offline": True}


if __name__ == "__main__":
    try:
        print(json.dumps(check(), indent=2), flush=True)
    except Exception as error:
        print(f"NOT_READY: {error}", file=sys.stderr)
        raise SystemExit(1)
