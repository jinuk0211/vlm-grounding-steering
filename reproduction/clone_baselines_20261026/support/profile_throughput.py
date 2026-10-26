"""Profile six Gemma methods on a fixed workload; never an efficacy evaluation."""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import torch

from vlm_grounding.backend import load_adapter
from vlm_grounding.baselines import build_baseline, extract_baseline
from vlm_grounding.method import _capture, _margin, _patch, fit, steer

METHODS = ("base", "ephs", "revis", "rudder", "grains", "vti")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class DeviceMemory:
    """Device-wide used-memory samples at 200 ms; allocator peaks are exact."""

    def __init__(self):
        self.lock = threading.Lock()
        self.window_start = 0.0
        self.maximum = None
        self.process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--loop-ms=200"],
            stdout=subprocess.PIPE, text=True,
        )
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        for line in self.process.stdout:
            observed = time.perf_counter()
            used = int(line.strip()) * 2**20
            with self.lock:
                if observed >= self.window_start:
                    self.maximum = used if self.maximum is None else max(self.maximum, used)

    def reset(self):
        with self.lock:
            self.window_start = time.perf_counter()
            self.maximum = None

    def peak(self):
        with self.lock:
            return self.maximum


def measure(operation, monitor):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    monitor.reset()
    start = time.perf_counter()
    try:
        result = operation()
        torch.cuda.synchronize()
        outcome = {"status": "passed", "result": result}
    except Exception as error:
        outcome = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    outcome.update(seconds=time.perf_counter() - start,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   device_peak_used_bytes_sampled=monitor.peak())
    return outcome


@contextmanager
def finite_gradients(diagnostics):
    original = torch.autograd.grad

    def checked(*args, **kwargs):
        values = original(*args, **kwargs)
        for value in values:
            if value is None or not bool(torch.isfinite(value).all()):
                raise ValueError("missing or nonfinite extraction gradient")
        diagnostics["finite_gradient_calls"] += 1
        return values

    torch.autograd.grad = checked
    try:
        yield
    finally:
        torch.autograd.grad = original


def summarize(rows):
    result = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        groups = {}
        for task in ("all", "pope", "caption"):
            successful = [row for row in selected if row["status"] == "passed"
                          and (task == "all" or row["task"] == task)]
            if not successful:
                continue
            times = np.array([row["seconds"] for row in successful])
            rng = np.random.default_rng(42)
            bootstrap = times[rng.integers(0, len(times), size=(2000, len(times)))].mean(1)
            device_peaks = [row["device_peak_used_bytes_sampled"] for row in successful
                            if row["device_peak_used_bytes_sampled"] is not None]
            groups[task] = {
                "count": len(successful), "mean_seconds": float(times.mean()),
                "mean_seconds_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
                "p50_seconds": float(np.median(times)), "p95_seconds": float(np.quantile(times, .95)),
                "max_seconds": float(times.max()), "total_seconds": float(times.sum()),
                "mean_generated_tokens": float(np.mean([row["result"]["generated_tokens"] for row in successful])),
                "cap_hit_count": sum(row["result"]["generated_tokens"] == 128 for row in successful),
                "peak_allocated_gib": max(row["peak_allocated_bytes"] for row in successful) / 2**30,
                "peak_reserved_gib": max(row["peak_reserved_bytes"] for row in successful) / 2**30,
                "device_peak_used_gib_sampled": max(device_peaks) / 2**30 if device_peaks else None,
                "device_peak_sampled_requests": len(device_peaks),
            }
        result[method] = {"attempted": len(selected), "failed": sum(row["status"] != "passed" for row in selected),
                          "groups": groups}
    return result


def main(args):
    config = json.loads(args.config.read_text())
    workload = json.loads(args.workload.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    signature = {"config_sha256": digest(args.config), "workload_sha256": digest(args.workload),
                 "source_sha256": {name: digest(Path(__import__("vlm_grounding").__file__).parent / name)
                                   for name in ("baselines.py", "backend.py", "method.py")},
                 "script_sha256": digest(__file__)}
    header_path = args.output / "header.json"
    if header_path.exists():
        if json.loads(header_path.read_text())["signature"] != signature:
            raise ValueError("Resume requires identical config, data, source and profiler")
    else:
        write_json(header_path, {"signature": signature, "config": config,
                                "purpose": "runtime-only; untuned pilot artifacts; no efficacy claims",
                                "methods": METHODS, "samples_per_method": len(workload["records"]),
                                "memory_sampling_ms": 200, "timing": "synchronized wall time including preprocessing and hooks",
                                "ephs_discovery": "fixed 8-head selection; causal selection not evaluated; compute-work proxy measured"})
    torch.manual_seed(42)
    monitor = DeviceMemory()
    setup_path = args.output / "setup.json"
    setup = json.loads(setup_path.read_text()) if setup_path.exists() else {}
    artifacts = {}
    try:
        load_start = time.perf_counter()
        adapter = load_adapter(config["model"])
        torch.cuda.synchronize()
        setup["model_load_seconds"] = time.perf_counter() - load_start
        setup["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}
        setup["hardware"] = {"gpu": torch.cuda.get_device_name(), "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
        if "discovery_head_work" not in setup:
            batch = adapter.prepare(workload["fit"][0]["image"], workload["fit"][0]["prompt"])
            layers = config["discovery"]["layers"]
            donors = {}

            def discovery_initial():
                with torch.no_grad(), _capture(adapter, layers) as captured:
                    _margin(adapter, batch)
                donors.update(captured)
                with torch.no_grad():
                    _margin(adapter, batch)
                    _margin(adapter, batch)
                return {"forwards": 3, "same_image_control_proxy": True}

            def discovery_head_work():
                with torch.no_grad():
                    for layer in layers:
                        for head in range(adapter.num_heads):
                            with _patch(adapter, layer, head, donors[layer][head]):
                                _margin(adapter, batch)
                                _margin(adapter, batch)
                return {"candidate_heads": len(layers) * adapter.num_heads,
                        "forwards": 2 * len(layers) * adapter.num_heads,
                        "same_image_control_proxy": True, "causal_selection_evaluated": False}

            setup["discovery_initial"] = measure(discovery_initial, monitor)
            setup["discovery_head_work"] = measure(discovery_head_work, monitor)
            write_json(setup_path, setup)
            donors.clear()
            print("DISCOVERY WORK", setup["discovery_head_work"], flush=True)
        preferences = [row for row in workload["preferences"] if row["role"] == "extraction"]
        artifact_root = args.output / "artifacts"
        artifact_root.mkdir(exist_ok=True)
        for method in ("revis", "grains", "vti", "ephs"):
            target = artifact_root / f"{method}.pt"
            if target.exists():
                artifacts[method] = torch.load(target, map_location="cpu", weights_only=True)
                continue
            gc.collect()
            torch.cuda.empty_cache()
            diagnostics = {"finite_gradient_calls": 0}

            def prepare_artifact():
                if method == "ephs":
                    layers = config["discovery"]["layers"]
                    selection = {"method": "EPHS", "stage": "discovery", "num_heads": adapter.num_heads,
                                 "layer_count": len(adapter.layers), "selection_mode": "synthetic-runtime-profile",
                                 "selected_heads": [{"layer": layers[i % len(layers)], "head": i // len(layers)}
                                                    for i in range(config["discovery"]["top_k"])], "discovery_ids": []}
                    artifact = fit(adapter, workload["fit"], selection, config["fit"])
                    details = {"fit_records": len(workload["fit"]), "fit_steps": artifact["fit_steps"],
                               "causal_discovery_profiled": False}
                else:
                    settings = dict(config["methods"][method])
                    records = preferences
                    if method == "revis":
                        settings["layer"] = len(adapter.layers) - 1
                        records = workload["preferences"]
                    with finite_gradients(diagnostics):
                        artifact = extract_baseline(method, adapter, records, settings)
                    vectors = [vector for key in ("vectors", "vision_vectors")
                               for vector in artifact.get(key, {}).values()]
                    if not all(bool(torch.isfinite(vector).all()) and vector.norm() > 0 for vector in vectors):
                        raise ValueError("nonfinite or zero direction")
                    details = {"extraction_pairs": len(preferences), "calibration_pairs": 4 if method == "revis" else 0,
                               "vector_count": len(vectors), **diagnostics}
                artifacts[method] = artifact
                torch.save(artifact, target)
                return details

            print("SETUP", method, flush=True)
            setup[method] = measure(prepare_artifact, monitor)
            write_json(setup_path, setup)
            print(method, setup[method], flush=True)
        missing = set(METHODS) - {"base", "rudder"} - artifacts.keys()
        if missing:
            raise RuntimeError(f"Cannot profile methods without successful preparation: {missing}")

        def generate(row, method):
            batch = adapter.prepare(row["image"], row["prompt"])
            context = (steer(adapter, artifacts[method]) if method == "ephs" else
                       build_baseline(method, adapter, artifacts.get(method), config["methods"].get(method)))
            with context:
                text = adapter.generate(batch, config["generation"]["max_new_tokens"])
            return {"text": text, "generated_tokens": adapter.last_generated_tokens,
                    "input_tokens": int(batch["input_ids"].shape[-1]),
                    "pixel_shape": list(batch["pixel_values"].shape)}

        for method in METHODS:
            for task in ("pope", "caption"):
                row = next(row for row in workload["records"] if row["task"] == task)
                generate(row, method)
        print("WARMUP COMPLETE", flush=True)
        journal = args.output / "samples.jsonl"
        rows = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        complete = {(row["method"], row["id"]) for row in rows}
        with journal.open("a", buffering=1) as stream:
            for index, row in enumerate(workload["records"]):
                order = METHODS[index % len(METHODS):] + METHODS[:index % len(METHODS)]
                for method in order:
                    if (method, row["id"]) in complete:
                        continue
                    entry = {"id": row["id"], "method": method, "task": row["task"],
                             "sample_index": index, **measure(lambda: generate(row, method), monitor)}
                    stream.write(json.dumps(entry, allow_nan=False) + "\n")
                    rows.append(entry)
                    if entry["status"] != "passed":
                        print("FAILED", method, row["id"], entry, flush=True)
                        gc.collect()
                        torch.cuda.empty_cache()
                if (index + 1) % 5 == 0:
                    write_json(args.output / "summary.json", summarize(rows))
                    print(f"PROGRESS {index + 1}/100 images, {len(rows)}/600 method-samples", flush=True)
        write_json(args.output / "summary.json", summarize(rows))
        print("COMPLETE", len(rows), flush=True)
        if any(row["status"] != "passed" for row in rows):
            raise RuntimeError("Profile completed with sample failures; inspect samples.jsonl")
    finally:
        monitor.process.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
