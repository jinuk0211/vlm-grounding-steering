"""Resume a sealed six-condition Gemma experiment, measuring the pilot first."""

import argparse
import gc
import json
from pathlib import Path
import random
import time

import torch

from profile_throughput import DeviceMemory, METHODS, digest, measure, summarize, write_json
from report_profile import forecast
from vlm_grounding.backend import load_adapter
from vlm_grounding.baselines import build_baseline
from vlm_grounding.cli import _evaluation_protocol, _read_artifact, _validate_artifact, _versions
from vlm_grounding.data import read_manifest
from vlm_grounding.method import steer


def pilot_cohort(records: list[dict]) -> list[dict]:
    rng = random.Random(42)
    pool = [row for row in records if row["task"] == "pope"]
    rng.shuffle(pool)
    chosen, groups = [], set()
    for index in range(50):
        subset = ("random", "popular", "adversarial")[index % 3]
        label = "yes" if index % 2 == 0 else "no"
        row = next(row for row in pool if row["label"] == label
                   and row["metadata"]["subset"] == subset and row["group_id"] not in groups)
        chosen.append(row)
        groups.add(row["group_id"])
    captions = [row for row in records if row["task"] == "caption" and row["group_id"] not in groups]
    chosen.extend(rng.sample(captions, 50))
    rng.shuffle(chosen)
    if len({row["group_id"] for row in chosen}) != 100:
        raise ValueError("Pilot requires 100 distinct image groups")
    return chosen


def main(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text())
    records = read_manifest(args.manifest)
    counts = {task: sum(row["task"] == task for row in records) for task in ("caption", "pope", "mmhal")}
    subsets = {subset: sum(row["task"] == "pope" and row["metadata"].get("subset") == subset
                           for row in records) for subset in ("random", "popular", "adversarial")}
    if (counts != {"caption": 2000, "pope": 9000, "mmhal": 96}
            or len(records) != 11096 or any(row["split"] != "test" for row in records)
            or set(subsets.values()) != {3000}):
        raise ValueError(f"Incomplete requested cohort: {counts}")
    seal_path = args.manifest.parent / "data_provenance.json"
    seal = json.loads(seal_path.read_text())
    if seal["manifest_sha256"].get(args.manifest.name) != digest(args.manifest):
        raise ValueError("Evaluation manifest differs from its sealed source cohort")
    for name, expected in seal["image_sha256"].items():
        if digest(args.manifest.parent / "images" / name) != expected:
            raise ValueError(f"Sealed dataset image changed: {name}")
    artifacts, artifact_paths = {}, {}
    for method in ("ephs", "revis", "grains", "vti"):
        path = args.artifacts / (method + (".json" if method == "ephs" else ".pt"))
        artifact_paths[method] = path
        artifacts[method] = _read_artifact(str(path), method)
        _validate_artifact(artifacts[method], config["model"], records)
    sources = Path(__import__("vlm_grounding").__file__).parent
    signature = {"config": digest(args.config), "manifest": digest(args.manifest),
                 "data_seal": digest(seal_path),
                 "artifacts": {method: digest(path) for method, path in artifact_paths.items()},
                 "sources": {name: digest(sources / name) for name in ("backend.py", "baselines.py", "method.py", "cli.py")},
                 "runner": digest(__file__), "versions": _versions(),
                 "helpers": {name: digest(Path(__file__).with_name(name))
                             for name in ("profile_throughput.py", "report_profile.py")}}
    args.output.mkdir(parents=True, exist_ok=True)
    header_path = args.output / "header.json"
    if header_path.exists():
        if json.loads(header_path.read_text())["signature"] != signature:
            raise ValueError("Resume rejected: source, config, manifest or artifact changed")
    else:
        write_json(header_path, {"signature": signature, "config": config,
                                 "counts_per_method": counts, "methods": METHODS, "versions": _versions()})
    journal = args.output / "samples.jsonl"
    rows = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    done = {(row["method"], row["id"]) for row in rows}
    valid_ids = {row["id"] for row in records}
    if len(done) != len(rows) or any(row["method"] not in METHODS or row["id"] not in valid_ids
                                     or row["status"] != "passed" for row in rows):
        raise ValueError("Invalid or duplicate resume journal")
    by_id = {row["id"]: row for row in records}
    for row in rows:
        if any(row.get(key) != value for key, value in by_id[row["id"]].items()):
            raise ValueError("Resume journal inputs differ from the sealed manifest")
        if row.get("prediction") != row.get("result", {}).get("text"):
            raise ValueError("Resume journal prediction differs from its measured result")
    torch.manual_seed(int(config["generation"]["seed"]))
    adapter = load_adapter(config["model"])
    for artifact in artifacts.values():
        _validate_artifact(artifact, config["model"], records, adapter)
    protocol, fingerprint = _evaluation_protocol(adapter, config)
    if any(row.get("protocol_fingerprint") != fingerprint for row in rows):
        raise ValueError("Resume journal generation protocol differs")
    monitor = DeviceMemory()
    pilot = pilot_cohort(records)
    pilot_ids = {row["id"] for row in pilot}
    ordered = pilot + [row for row in records if row["id"] not in pilot_ids]
    try:
        # Every fresh model load, including resume, receives excluded warmups.
        for method in METHODS:
            for task in ("pope", "caption"):
                record = next(row for row in pilot if row["task"] == task)
                batch = adapter.prepare(record["image"], record["prompt"])
                context = (steer(adapter, artifacts[method], config["methods"].get(method)) if method == "ephs" else
                           build_baseline(method, adapter, artifacts.get(method),
                                          config["methods"].get(method), batch=batch, task=task))
                with context:
                    adapter.generate(batch, config["generation"]["max_new_tokens"])
        with journal.open("a", encoding="utf-8") as stream:
            for index, record in enumerate(ordered):
                rotated = METHODS[index % len(METHODS):] + METHODS[:index % len(METHODS)]
                for method in rotated:
                    if (method, record["id"]) in done:
                        continue
                    cleanup_start = time.perf_counter()
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    cleanup_seconds = time.perf_counter() - cleanup_start

                    def generate():
                        batch = adapter.prepare(record["image"], record["prompt"])
                        context = (steer(adapter, artifacts[method], config["methods"].get(method)) if method == "ephs" else
                                   build_baseline(method, adapter, artifacts.get(method),
                                                  config["methods"].get(method), batch=batch, task=record["task"]))
                        with context:
                            text = adapter.generate(batch, config["generation"]["max_new_tokens"])
                        return {"text": text, "generated_tokens": adapter.last_generated_tokens}

                    measured = measure(generate, monitor)
                    measured["generation_and_preprocessing_seconds"] = measured["seconds"]
                    measured["cleanup_seconds"] = cleanup_seconds
                    measured["seconds"] += cleanup_seconds
                    row = {**record, "method": method, "protocol_fingerprint": fingerprint, **measured}
                    if row["status"] != "passed":
                        write_json(args.output / "last_failure.json", row)
                        raise RuntimeError(row["error"])
                    row["prediction"] = row["result"]["text"]
                    row["elapsed_seconds"] = row["seconds"]
                    row["generated_tokens"] = row["result"]["generated_tokens"]
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    rows.append(row)
                    done.add((method, record["id"]))
                if index == 99:
                    pilot_rows = [row for row in rows if row["id"] in pilot_ids]
                    write_json(args.output / "pilot_summary.json", summarize(pilot_rows))
                    estimate = forecast(pilot_rows)
                    estimate["limitations"] = ["MMHal uses caption runtime proxy", "judge API and additional tuning excluded"]
                    write_json(args.output / "forecast.json", estimate)
                    print("PILOT_COMPLETE", json.dumps(estimate), flush=True)
                if (index + 1) % 25 == 0:
                    print(json.dumps({"completed_images_or_questions": index + 1,
                                      "completed_generations": len(done), "total_generations": len(records) * 6}), flush=True)
        for method in METHODS:
            selected = [row for row in rows if row["method"] == method]
            path = args.output / f"{method}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
            write_json(Path(str(path) + ".metadata.json"), {
                "status": "complete", "method": method, "record_ids": [row["id"] for row in selected],
                "config": config, "checkpoint_revision": protocol["checkpoint_revision"],
                "protocol_fingerprint": fingerprint, "evaluation_protocol": protocol,
                "signature": signature, "cohort_complete": True})
        write_json(args.output / "complete.json", {"generations": len(done), "methods": METHODS, "counts_per_method": counts})
    finally:
        monitor.process.terminate()
        monitor.process.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
