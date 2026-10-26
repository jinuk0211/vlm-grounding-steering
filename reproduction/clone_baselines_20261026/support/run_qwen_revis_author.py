"""Qwen REVIS reference check: author vectors/hooks, held-out calibration, paired pilot."""

import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_gemma_experiment import pilot_cohort
from vlm_grounding.backend import load_adapter
from vlm_grounding.data import read_manifest, validate_disjoint
from vlm_grounding.evaluation import evaluate_predictions, run_official_chair


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    previous = None
    for row in rows:
        payload = {k: v for k, v in row.items() if k != "row_sha256"}
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                           allow_nan=False).encode()).hexdigest()
        if row.get("previous_sha256") != previous or row.get("row_sha256") != actual:
            raise ValueError(f"Journal hash chain changed: {path}")
        previous = actual
    return rows


def append(path, row):
    # One writer. Chain each durable row so accidental edits/truncation within a row fail resume.
    previous = read_rows(path)
    row["previous_sha256"] = previous[-1]["row_sha256"] if previous else None
    row["row_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                                allow_nan=False).encode()).hexdigest()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def author_definitions(path, names, namespace=None):
    """Execute selected definitions from explicitly supplied, fingerprinted author sources."""
    scope = dict(torch=torch, nn=nn, F=F, np=np, Image=Image,
                 Dict=Dict, Optional=Optional, Tuple=Tuple, process_vision_info=None)
    scope.update(namespace or {})
    nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if set(names) != {node.name for node in nodes}:
        raise ValueError(f"Missing author definitions: {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def validate_row_inputs(rows, records):
    reference = {r["id"]: r for r in records}
    for row in rows:
        if row["id"] not in reference or any(row.get(k) != v for k, v in reference[row["id"]].items()):
            raise ValueError("Journal inputs differ from the sealed manifest")


def interleaved_pilot(records):
    """Keep the frozen 100-image cohort, order each ten as five POPE/five captions."""
    selected = pilot_cohort(records)
    pope = [r for r in selected if r["task"] == "pope"]
    captions = [r for r in selected if r["task"] == "caption"]
    return [row for i in range(50) for row in (pope[i], captions[i])]


def generate(adapter, batch, max_new_tokens):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        ids = adapter.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                     num_beams=1, use_cache=True)
    torch.cuda.synchronize()
    tail = ids[0, batch["input_ids"].shape[1]:]
    return dict(prediction=adapter.processor.decode(tail, skip_special_tokens=True).strip(),
                generated_tokens=tail.numel(), elapsed_seconds=time.perf_counter() - start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def output_checks(row, cap):
    text = row["prediction"]
    words = text.split()
    # Diagnostic flags, not a semantic correctness detector. Keep every output in scoring.
    return dict(empty=not text.strip(), fewer_than_four_words=len(words) < 4,
                repeated_word_fraction=(1 - len(set(words)) / len(words)) if words else None,
                token_cap_reached=row["generated_tokens"] == cap)


def score_progress(rows, args, count):
    report = dict(completed_pairs=count, preliminary=count < 100, conditions={})
    for condition in sorted({r["method"] for r in rows}):
        selected = [r for r in rows if r["method"] == condition]
        metrics = {}
        for task in ("pope", "caption"):
            cohort = [r for r in selected if r["task"] == task]
            if not cohort:
                continue
            if task == "caption":
                cohort = run_official_chair(cohort, args.chair, args.annotations,
                                           args.output / f"chair_{condition}_{count:03}.json")
            metrics[task] = evaluate_predictions(cohort)
        metrics["output_flags"] = {
            key: sum(r["output_checks"][key] for r in selected)
            for key in ("empty", "token_cap_reached")
        }
        report["conditions"][condition] = metrics
    write_json(args.output / f"progress_{count:03}.json", report)
    write_json(args.output / "latest_progress.json", report)
    print("SCORED_PROGRESS " + json.dumps(report, ensure_ascii=False), flush=True)
    return report


def calibrate(adapter, calibration, vectors, source, prepare, args, config):
    path = args.output / "calibration.jsonl"
    rows = read_rows(path)
    validate_row_inputs(rows, calibration)
    expected = [r["id"] for r in calibration]
    if [r["id"] for r in rows] != expected[:len(rows)]:
        raise ValueError("Calibration journal is not a prefix of the sealed cohort")
    observers = {i: source["DiagnosticsHook"](vectors[i + 1], i)
                 for i in range(10, len(adapter.layers))}
    try:
        for layer, observer in observers.items():
            observer.attach(adapter.layers[layer])
        for record in calibration[len(rows):]:
            for observer in observers.values():
                observer.clear()
            with Image.open(record["image"]) as image:
                batch, _ = prepare(adapter.model, adapter.processor, image.convert("RGB"),
                                   record["prompt"] + " Please answer yes or no.")
            measured = generate(adapter, batch, 5)
            text = measured["prediction"].lower()
            pred = "yes" if "yes" in text else ("no" if "no" in text else "unknown")
            category = ("TP" if record["label"] == pred == "yes" else
                        "FP" if record["label"] == "no" and pred == "yes" else
                        "TN" if record["label"] == pred == "no" else "Other")
            row = dict(**record, category=category, **measured,
                       risks={str(i): observer.captured_risk for i, observer in observers.items()})
            append(path, row)
            rows.append(row)
            if len(rows) % 10 == 0:
                print(f"CALIBRATION {len(rows)}/{len(calibration)} (not test scores)", flush=True)
    finally:
        for observer in observers.values():
            observer.detach()
    candidates = []
    for layer in range(len(adapter.layers) - 1, 9, -1):
        fact = [r["risks"][str(layer)] for r in rows if r["category"] in ("TP", "TN")]
        hall = [r["risks"][str(layer)] for r in rows if r["category"] == "FP"]
        if fact and hall:
            candidates.append(dict(layer=layer, separability=float(np.mean(hall) - np.mean(fact)),
                                   threshold_paper=float(np.percentile(fact, config["paper"]["risk_percentile"])),
                                   threshold_code=float(np.percentile(fact, config["released_code"]["risk_percentile"]))))
    chosen = next((r for r in candidates if r["separability"] > 0), None)
    if chosen is None:
        raise ValueError("Author calibration failed: no layer has positive separability")
    write_json(args.output / "calibrated.json", dict(selected=chosen, candidates=candidates,
               n=len(rows), observer_optimization="Independent read-only observers share the same unsteered pass"))
    print("CALIBRATED " + json.dumps(chosen), flush=True)
    return chosen


@contextmanager
def intervention(adapter, hook_class, vectors, calibration, condition, config):
    if condition == "base":
        yield
        return
    layer = calibration["layer"]
    alpha = config["paper" if condition == "revis_paper" else "released_code"]["alpha"]
    hook = hook_class(vectors[layer + 1], layer, alpha,
                      calibration["threshold_paper" if condition == "revis_paper" else "threshold_code"], 1.0)
    hook.attach(adapter.layers[layer])
    try:
        yield
    finally:
        hook.detach()


def main(args):
    from qwen_vl_utils import process_vision_info
    import qwen_vl_utils.vision_process as vision_process

    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    test = interleaved_pilot(read_manifest(args.data / "test.jsonl"))
    calibration_records = read_manifest(args.data / "revis_tune.jsonl")[:config["revis"]["calibration_records"]]
    validate_disjoint(test + calibration_records + read_manifest(args.data / "revis_fit.jsonl"))
    seal_path = args.data / "data_provenance.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    manifest_hashes = {name: digest(args.data / name) for name in
                       ("test.jsonl", "revis_tune.jsonl", "revis_fit.jsonl")}
    if any(seal["manifest_sha256"].get(name) != value for name, value in manifest_hashes.items()):
        raise ValueError("Manifest differs from the sealed source data")
    images = {Path(r["image"]) for r in test + calibration_records}
    image_hashes = {p.name: digest(p) for p in images}
    if any(seal["image_sha256"].get(name) != value for name, value in image_hashes.items()):
        raise ValueError("Pilot/calibration image differs from the sealed source data")
    sources = {
        "inputs": args.author / "main_all_visual_only.py",
        "observer": args.author / "layer_select.py",
        "hook": args.author / "steering_utils/llm_layers.py",
        "vectors": args.author / "vector/qwen2.5vl_none_image.pt",
    }
    runtime = Path(__import__("vlm_grounding").__file__).parent
    signature = dict(config=digest(args.config), runner=digest(__file__),
                     pilot_helper=digest(Path(__file__).with_name("run_gemma_experiment.py")),
                     sources={name: digest(path) for name, path in sources.items()},
                     runtime={name: digest(runtime / name) for name in ("backend.py", "evaluation.py")},
                     manifest=digest(args.data / "test.jsonl"), calibration=digest(args.data / "revis_tune.jsonl"),
                     data_seal=digest(seal_path), images=image_hashes,
                     sealed_manifests=manifest_hashes,
                     qwen_processor_source=digest(vision_process.__file__),
                     versions={name: importlib.metadata.version(name) for name in
                               ("torch", "transformers", "numpy", "nltk", "Pillow", "qwen-vl-utils")},
                     chair=digest(args.chair), annotations={p.name: digest(p) for p in args.annotations.glob("*.json")})
    header = dict(signature=signature, config=config, pilot_ids=[r["id"] for r in test],
                  calibration_ids=[r["id"] for r in calibration_records],
                  conditions=["base", "revis_paper", "revis_code"],
                  protocol="Native Qwen processor, original author vectors/hook. Paper alpha1.6/p80 and code alpha2.5/p85 kept separate. 50 captions/50 POPE; pilot, not paper-scale replication.")
    header_path = args.output / "header.json"
    if header_path.exists() and json.loads(header_path.read_text(encoding="utf-8")) != header:
        raise ValueError("Resume rejected: frozen sources/config/cohort/scorer changed")
    write_json(header_path, header)
    torch.manual_seed(42)
    adapter = load_adapter(config["model"])
    # Extraction/inference use no weight gradients. Model activations remain differentiable when needed elsewhere.
    adapter.model.requires_grad_(False)
    prepare = author_definitions(sources["inputs"], {"get_model_inputs"},
                                 {"process_vision_info": process_vision_info})["get_model_inputs"]
    observer = author_definitions(sources["observer"], {"DiagnosticsHook"})
    hook_class = author_definitions(sources["hook"], {"MADRSteeringHook_visual_only"})["MADRSteeringHook_visual_only"]
    vectors = torch.load(sources["vectors"], map_location="cpu", weights_only=True)
    calibrated = calibrate(adapter, calibration_records, vectors, observer, prepare, args, config["revis"])
    journal = args.output / "samples.jsonl"
    rows = read_rows(journal)
    validate_row_inputs(rows, test)
    conditions = header["conditions"]
    valid = {(c, r["id"]) for c in conditions for r in test}
    done = {(r["method"], r["id"]) for r in rows}
    if len(done) != len(rows) or not done <= valid:
        raise ValueError("Unexpected or duplicate prediction journal entries")
    for index, record in enumerate(test):
        prompt = "Describe this image in detail." if record["task"] == "caption" else record["prompt"] + " Please answer yes or no."
        cap = config["revis"]["caption_max_new_tokens" if record["task"] == "caption" else "pope_max_new_tokens"]
        with Image.open(record["image"]) as image:
            batch, _ = prepare(adapter.model, adapter.processor, image.convert("RGB"), prompt)
        order = conditions[index % 3:] + conditions[:index % 3]
        for condition in order:
            if (condition, record["id"]) in done:
                continue
            with intervention(adapter, hook_class, vectors, calibrated, condition, config["revis"]):
                row = {**record, "method": condition, "actual_prompt": prompt,
                       **generate(adapter, batch, cap)}
            row["output_checks"] = output_checks(row, cap)
            append(journal, row)
            rows.append(row)
            done.add((condition, record["id"]))
            if index < 2:
                print("FIRST_OUTPUT " + json.dumps(row, ensure_ascii=False), flush=True)
        if (index + 1) % 10 == 0 and not (args.output / f"progress_{index + 1:03}.json").exists():
            prefix = {r["id"] for r in test[:index + 1]}
            score_progress([r for r in rows if r["id"] in prefix], args, index + 1)
    write_json(args.output / "complete.json", dict(pairs=100, generations=len(rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "author", "chair", "annotations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
