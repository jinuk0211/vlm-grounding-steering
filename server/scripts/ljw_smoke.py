"""Real Gemma image inference and head-hook check on a synthetic fixture."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

from PIL import Image, ImageDraw
import torch

from ljw_check import check
from vlm_grounding.backend import load_adapter
from vlm_grounding.method import steer


def main():
    ready = check()
    repo = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = repo / "outputs" / f"gemma-smoke-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    image = Image.new("RGB", (448, 448), "white")
    ImageDraw.Draw(image).rectangle((100, 100, 348, 348), fill="red")
    image_path = output / "synthetic-red-square.png"
    image.save(image_path)
    config = json.loads((repo / "configs/gemma4_e4b.local.json").read_text())
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    started = time.monotonic()
    print("SMOKE_STAGE=loading", flush=True)
    adapter = load_adapter(config["model"])
    load_seconds = time.monotonic() - started
    prompt = "Is there a red square in this image? Answer yes or no."
    batch = adapter.prepare(str(image_path), prompt)
    layer = int(config["discovery"]["layers"][0])
    projection = adapter.attn_projections[layer]
    captured = {}

    def capture_input(name):
        def hook(_module, inputs):
            width = inputs[0].shape[-1] // adapter.num_heads
            captured[name] = inputs[0][0, -1, :width].detach().float().cpu().clone()
        return hook

    print("SMOKE_STAGE=image_forward", flush=True)
    with torch.no_grad():
        handle = projection.register_forward_pre_hook(capture_input("baseline"))
        try:
            baseline = float(adapter.answer_margin(batch))
        finally:
            handle.remove()
        artifact = {"stage": "fitted", "num_heads": adapter.num_heads,
                    "layer_count": len(adapter.layers), "max_gain": 0.25,
                    "positions": "last",
                    "selected_heads": [{"layer": layer, "head": 0, "delta": 0.1}]}
        hook_count = len(adapter.attn_projections[layer]._forward_pre_hooks)
        print("SMOKE_STAGE=head_intervention", flush=True)
        with steer(adapter, artifact):
            handle = projection.register_forward_pre_hook(capture_input("steered"))
            try:
                changed = float(adapter.answer_margin(batch))
            finally:
                handle.remove()
        restored = float(adapter.answer_margin(batch))
        assert len(adapter.attn_projections[layer]._forward_pre_hooks) == hook_count
        assert all(math.isfinite(x) for x in (baseline, changed, restored))
        assert math.isclose(baseline, restored, rel_tol=1e-3, abs_tol=1e-3)
        activation_difference = float((captured["steered"] - captured["baseline"]).norm())
        activation_ratio = float(captured["steered"].norm() / captured["baseline"].norm())
        assert activation_difference > 0 and math.isclose(activation_ratio, 1.1, abs_tol=0.02)
        print("SMOKE_STAGE=cached_generation", flush=True)
        response = adapter.generate(batch, max_new_tokens=12)
        assert 1 <= adapter.last_generated_tokens <= 12
    torch.cuda.synchronize()
    result = {"status": "passed", "synthetic_fixture": True,
              "research_metric": False, "checkpoint": ready["checkpoint"],
              "revision": ready["revision"], "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "cuda": torch.version.cuda, "load_seconds": load_seconds,
              "total_seconds": time.monotonic() - started,
              "input_tokens": batch["input_ids"].shape[1],
              "response": response, "generated_tokens": adapter.last_generated_tokens,
              "baseline_yes_no_margin": baseline, "steered_yes_no_margin": changed,
              "restored_yes_no_margin": restored,
              "head_activation_change_l2": activation_difference,
              "head_activation_norm_ratio": activation_ratio,
              "head_hook_restored": True,
              "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
