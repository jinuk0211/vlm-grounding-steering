"""Reproducible local-data commands; only model commands load checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any


METHODS = ("base", "ephs", "revis", "rudder", "grains", "vti")
SPLITS = ("discovery", "fit", "tune", "test")


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object.")
    return value


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8")


def _reserve(paths: list[str | Path], overwrite: bool) -> None:
    for path in paths:
        if Path(path).exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite explicitly.")


def _versions() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "torchvision", "transformers", "numpy", "Pillow",
                 "vlm-grounding-steering"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _load_adapter(model_config: dict):
    from .backend import load_adapter
    return load_adapter(model_config)


def _config(path: str) -> dict:
    value = _json(path)
    if not isinstance(value.get("model"), dict) or not value["model"].get("checkpoint"):
        raise ValueError("Config must contain model.checkpoint.")
    return value


def _seed(config: dict) -> None:
    import torch
    seed = int(config.get("generation", {}).get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _synchronize(adapter: Any) -> None:
    device = getattr(adapter, "device", None)
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def _model_signature(config: dict) -> dict:
    return {key: value for key, value in config.items()
            if key not in ("device", "local_files_only")}


def _revision(adapter: Any) -> str | None:
    return getattr(adapter.model.config, "_commit_hash", None)


def _evaluation_protocol(adapter: Any, config: dict) -> tuple[dict, str]:
    protocol = {
        "model": _model_signature(config["model"]),
        "checkpoint_revision": _revision(adapter),
        "generation": {**config.get("generation", {}), "do_sample": False, "num_beams": 1,
                       "max_new_tokens": int(config.get("generation", {}).get("max_new_tokens", 128)),
                       "use_cache": True},
        "batch_size": 1,
    }
    fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return protocol, fingerprint


def _provenance(adapter: Any, config: dict, manifests: list[str],
                records: list[dict], parent: dict | None = None) -> dict:
    groups = {str(record["group_id"]) for record in records}
    sources = [{"path": str(Path(path).resolve()), "sha256": _sha(path)}
               for path in manifests]
    if parent:
        groups.update(parent.get("training_group_ids", []))
        sources = list(parent.get("manifests", [])) + sources
    return {
        "model_config": config["model"],
        "checkpoint_revision": _revision(adapter),
        "manifests": sources,
        "record_ids": [str(record["id"]) for record in records],
        "training_group_ids": sorted(groups),
        "versions": _versions(),
        "full_config": config,
    }


def _validate_artifact(artifact: dict, model_config: dict,
                       records: list[dict] | None = None, adapter: Any = None) -> None:
    provenance = artifact.get("_provenance")
    if not isinstance(provenance, dict) or "model_config" not in provenance:
        raise ValueError("Artifact has no model/data provenance; recreate it with this CLI.")
    if _model_signature(provenance["model_config"]) != _model_signature(model_config):
        raise ValueError("Artifact model config differs from the requested checkpoint/preprocessing.")
    if records:
        test_groups = {str(row["group_id"]) for row in records if row["split"] == "test"}
        overlap = test_groups.intersection(provenance.get("training_group_ids", []))
        if overlap:
            raise ValueError(f"Test image groups overlap artifact training/calibration: {sorted(overlap)[:5]}")
    if adapter is not None:
        expected = provenance.get("checkpoint_revision")
        if expected and _revision(adapter) != expected:
            raise ValueError("Resolved checkpoint revision differs from the artifact.")


def _select(records: list[dict], split: str, limit: int | None,
            *, task: str | None = None, preferences: bool = False) -> list[dict]:
    selected = [record for record in records if record["split"] == split
                and (task is None or record["task"] == task)
                and (not preferences or ("chosen" in record and "rejected" in record))]
    if not selected:
        raise ValueError(f"No matching records for split={split}, task={task}, preferences={preferences}.")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive.")
        if split == "test":
            raise ValueError("--limit is for development smoke tests only; test cohorts must be complete.")
        selected = selected[:limit]
    return selected


def _read_artifact(path: str, method: str) -> dict:
    if method == "ephs":
        return _json(path)
    import torch
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("Baseline artifact must be a dictionary.")
    return value


def _save_artifact(path: str, artifact: dict, *, tensor: bool = False) -> None:
    if tensor:
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, path)
    else:
        _write_json(path, artifact)


def _doctor(_args) -> dict:
    result = _versions()
    try:
        import torch
        result.update({"cuda_available": torch.cuda.is_available(),
                       "cuda_device_count": torch.cuda.device_count(),
                       "model_downloaded": False})
    except ImportError:
        result.update({"cuda_available": False, "model_downloaded": False})
    result["status"] = "CPU checks available; pretrained model experiments require configured compute."
    print(json.dumps(result, indent=2))
    return result


def _prepare(args) -> dict:
    from . import data
    _reserve([args.output], args.overwrite)
    if args.command == "prepare-pope":
        records = data.prepare_pope(args.input, args.image_root, args.split, subset=args.subset)
    elif args.command == "prepare-causal":
        records = data.prepare_causal(args.input, args.image_root, args.split)
    elif args.command == "prepare-mme":
        records = data.prepare_mme(args.input, args.split)
    elif args.command == "prepare-mmhal":
        records = data.prepare_mmhal(args.input, args.image_root, args.split)
    else:
        excluded = set()
        for path in args.exclude_manifests:
            excluded.update(str(row["group_id"]) for row in data.read_manifest(path))
        records = data.prepare_coco(
            args.annotations, args.image_root, args.work_dir, seed=args.seed,
            discovery_count=args.discovery_count, fit_count=args.fit_count,
            tune_count=args.tune_count, test_count=args.test_count,
            exclude_group_ids=excluded,
        )
    data.write_manifest(records, args.output)
    result = {"output": str(Path(args.output).resolve()), "records": len(records),
              "splits": data.validate_disjoint(records)}
    print(json.dumps(result, indent=2))
    return result


def _verify(args) -> dict:
    from .data import validate_disjoint
    result = validate_disjoint(args.manifests)
    print(json.dumps({"split_counts": result, "group_isolation": "passed"}, indent=2))
    return result


def _fetch(args) -> dict:
    from .upstream import fetch_sources
    result = fetch_sources(args.names, args.output_dir)
    print(json.dumps(result, indent=2))
    return result


def _extract(args) -> dict:
    from .baselines import extract_baseline
    from .data import read_manifest, validate_disjoint
    _reserve([args.output], args.overwrite)
    config = _config(args.config)
    records, manifests = [], []
    if args.manifest:
        records = _select(read_manifest(args.manifest), "fit", args.limit, preferences=True)
        manifests.append(args.manifest)
    elif args.method != "rudder":
        raise ValueError("--manifest is required for offline baseline extraction.")
    if args.calibration:
        if args.method != "revis":
            raise ValueError("--calibration is only used by REVIS.")
        calibration = _select(read_manifest(args.calibration), "tune", args.limit, preferences=True)
        records += [{**record, "role": "calibration"} for record in calibration]
        manifests.append(args.calibration)
    validate_disjoint(records)
    _seed(config)
    adapter = _load_adapter(config["model"])
    artifact = extract_baseline(args.method, adapter, records,
                                config.get("methods", {}).get(args.method, {}))
    artifact["_provenance"] = _provenance(adapter, config, manifests, records)
    _save_artifact(args.output, artifact, tensor=True)
    result = {"artifact": str(Path(args.output).resolve()), "method": args.method,
              "records": len(records), "author_reproduction": False}
    print(json.dumps(result, indent=2))
    return result


def _discover(args) -> dict:
    from .data import read_manifest
    from .method import discover
    _reserve([args.output], args.overwrite)
    config = _config(args.config)
    records = _select(read_manifest(args.manifest), "discovery", args.limit, task="pope")
    _seed(config)
    adapter = _load_adapter(config["model"])
    artifact = discover(adapter, records, config.get("discovery", {}))
    artifact["_provenance"] = _provenance(adapter, config, [args.manifest], records)
    _save_artifact(args.output, artifact)
    result = {"artifact": str(Path(args.output).resolve()), "selected_heads": artifact["selected_heads"]}
    print(json.dumps(result, indent=2))
    return result


def _fit(args) -> dict:
    from .data import read_manifest
    from .method import fit
    _reserve([args.output], args.overwrite)
    config, selection = _config(args.config), _json(args.selection)
    records = _select(read_manifest(args.manifest), "fit", args.limit, task="pope")
    _validate_artifact(selection, config["model"])
    previous_groups = set(selection["_provenance"].get("training_group_ids", []))
    if previous_groups.intersection(str(row["group_id"]) for row in records):
        raise ValueError("Fit image groups overlap discovery images.")
    _seed(config)
    adapter = _load_adapter(config["model"])
    _validate_artifact(selection, config["model"], adapter=adapter)
    artifact = fit(adapter, records, selection, config.get("fit", {}))
    artifact["_provenance"] = _provenance(adapter, config, [args.manifest], records,
                                         selection["_provenance"])
    artifact["_provenance"]["selection_sha256"] = _sha(args.selection)
    _save_artifact(args.output, artifact)
    result = {"artifact": str(Path(args.output).resolve()), "fit_steps": artifact["fit_steps"]}
    print(json.dumps(result, indent=2))
    return result


def _run(args) -> dict:
    from .baselines import build_baseline
    from .data import read_manifest
    from .method import steer
    output = Path(args.output)
    sidecar = Path(str(output) + ".metadata.json")
    _reserve([output, sidecar], args.overwrite)
    config = _config(args.config)
    records = _select(read_manifest(args.manifest), args.split, args.limit, task=args.task)
    artifact = _read_artifact(args.artifact, args.method) if args.artifact else None
    if args.method not in ("base", "rudder") and artifact is None:
        raise ValueError(f"--artifact is required for {args.method}.")
    if args.method == "base" and artifact is not None:
        raise ValueError("Base inference must not receive a steering artifact.")
    if artifact is not None:
        _validate_artifact(artifact, config["model"], records)
    _seed(config)
    adapter = _load_adapter(config["model"])
    if artifact is not None:
        _validate_artifact(artifact, config["model"], records, adapter)
    method_config = config.get("methods", {}).get(args.method, {})
    max_tokens = int(config.get("generation", {}).get("max_new_tokens", 128))
    protocol, fingerprint = _evaluation_protocol(adapter, config)
    metadata = {
        "method": args.method, "config": config, "versions": _versions(),
        "checkpoint_revision": _revision(adapter),
        "manifest": str(Path(args.manifest).resolve()), "manifest_sha256": _sha(args.manifest),
        "artifact": str(Path(args.artifact).resolve()) if args.artifact else None,
        "artifact_sha256": _sha(args.artifact) if args.artifact else None,
        "record_ids": [str(row["id"]) for row in records], "selected_split": args.split,
        "selected_task": args.task, "requested_limit": args.limit,
        "cohort_complete": args.limit is None, "status": "running", "completed_records": 0,
        "implementation": "local-adaptation" if args.method in ("revis", "rudder", "grains", "vti")
                          else "base" if args.method == "base" else "proposed-ephs",
        "author_reproduction": False,
        "evaluation_protocol": protocol, "protocol_fingerprint": fingerprint,
    }
    _write_json(sidecar, metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            for index, record in enumerate(records):
                batch = adapter.prepare(record["image"], record["prompt"])
                context = (steer(adapter, artifact, method_config) if args.method == "ephs"
                           else build_baseline(args.method, adapter, artifact, method_config))
                _synchronize(adapter)
                started = time.perf_counter()
                with context:
                    prediction = adapter.generate(batch, max_new_tokens=max_tokens)
                _synchronize(adapter)
                row = {**record, "prediction": prediction, "method": args.method,
                       "elapsed_seconds": time.perf_counter() - started,
                       "generated_tokens": getattr(adapter, "last_generated_tokens", None),
                       "protocol_fingerprint": fingerprint}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                metadata["completed_records"] = index + 1
        metadata["status"] = "complete"
    except BaseException:
        metadata["status"] = "failed"
        _write_json(sidecar, metadata)
        raise
    _write_json(sidecar, metadata)
    result = {"output": str(output.resolve()), "records": len(records), "status": "complete"}
    print(json.dumps(result, indent=2))
    return result


def _prediction_rows(path: str) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Predictions must contain JSON objects.")
                rows.append(value)
    sidecar = Path(str(path) + ".metadata.json")
    if sidecar.is_file():
        metadata = _json(sidecar)
        if metadata.get("status") != "complete":
            raise ValueError(f"{path}: run did not complete; partial outputs cannot be compared.")
        if sorted(str(row["id"]) for row in rows) != sorted(metadata["record_ids"]):
            raise ValueError(f"{path}: predictions do not match the declared cohort.")
        if metadata.get("protocol_fingerprint") and any(
                row.get("protocol_fingerprint") != metadata["protocol_fingerprint"] for row in rows):
            raise ValueError(f"{path}: prediction protocol fingerprints differ from run metadata.")
    return rows


def _export_mmhal(args) -> dict:
    from .evaluation import mmhal_responses

    _reserve([args.output], args.overwrite)
    rows = mmhal_responses(_prediction_rows(args.predictions), args.template)
    _write_json(args.output, rows)
    result = {"output": str(Path(args.output).resolve()), "responses": len(rows)}
    print(json.dumps(result, indent=2))
    return result


def _compare(args) -> dict:
    from .evaluation import (
        attach_mmhal_scores, compare_methods, run_official_causal, run_official_chair,
        write_comparison_csv,
    )
    output_dir = Path(args.output_dir)
    report_path, csv_path = output_dir / "comparison.json", output_dir / "metrics.csv"
    _reserve([report_path, csv_path], args.overwrite)
    if bool(args.chair_module) != bool(args.annotations):
        raise ValueError("--chair-module and --annotations must be supplied together.")
    records = [row for path in args.predictions for row in _prediction_rows(path)]
    if not records:
        raise ValueError("No predictions were found.")
    fingerprints = {row["protocol_fingerprint"] for row in records if row.get("protocol_fingerprint")}
    if len(fingerprints) > 1:
        raise ValueError("Prediction evaluation protocol fingerprints differ.")
    sidecars = []
    for path in args.predictions:
        sidecar = Path(str(path) + ".metadata.json")
        if sidecar.is_file():
            metadata = _json(sidecar)
            sidecars.append(metadata)
    if sidecars:
        reference = sidecars[0]
        for metadata in sidecars[1:]:
            if (_model_signature(metadata["config"]["model"])
                    != _model_signature(reference["config"]["model"])
                    or metadata["config"].get("generation", {}) != reference["config"].get("generation", {})
                    or metadata.get("checkpoint_revision") != reference.get("checkpoint_revision")):
                raise ValueError("Prediction runs differ in model, resolved revision, or generation settings.")
    by_method = defaultdict(list)
    for record in records:
        by_method[record.get("method", "base")].append(record)
    mmhal_evaluations = {}
    for method, response, judgement in args.mmhal_evaluation:
        if method in mmhal_evaluations:
            raise ValueError(f"Duplicate MMHal evaluation for {method}")
        mmhal_evaluations[method] = (response, judgement)
    mmhal_methods = {method for method, rows in by_method.items()
                     if any(row["task"] == "mmhal" for row in rows)}
    if mmhal_evaluations and set(mmhal_evaluations) != mmhal_methods:
        raise ValueError("MMHal evaluations must cover every method with MMHal predictions")
    judge_models = {}
    for method, (_, judgement) in mmhal_evaluations.items():
        output = json.loads(Path(judgement).read_text(encoding="utf-8-sig"))
        models = {entry.get("model") for entry in output if isinstance(entry, dict)}
        if len(models) != 1:
            raise ValueError(f"{method}: MMHal judge responses must declare one model")
        judge_models[method] = models.pop()
    if len({model for model in judge_models.values() if model}) > 1:
        raise ValueError("MMHal judge model differs across methods")
    enriched, official = [], {}
    for method, rows in by_method.items():
        captions = [row for row in rows if row["task"] == "caption"]
        if captions and args.chair_module:
            scored = run_official_chair(captions, args.chair_module, args.annotations)
            rows = [row for row in rows if row["task"] != "caption"] + scored
        causal = [row for row in rows if row["task"] == "causal"]
        if causal and args.causal_module:
            official[method] = run_official_causal(causal, args.causal_module)
        if method in mmhal_evaluations:
            response, judgement = mmhal_evaluations[method]
            scored = attach_mmhal_scores(
                [row for row in rows if row["task"] == "mmhal"], response, judgement
            )
            rows = [row for row in rows if row["task"] != "mmhal"] + scored
        enriched.extend(rows)
    report = compare_methods(enriched, baseline=args.baseline,
                             bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    report["prediction_files"] = [{"path": str(Path(path).resolve()), "sha256": _sha(path)}
                                  for path in args.predictions]
    report["official_causal"] = official
    report["mmhal_evaluations"] = {
        method: {"responses": {"path": str(Path(response).resolve()), "sha256": _sha(response)},
                 "judge_output": {"path": str(Path(judgement).resolve()),
                                  "sha256": _sha(judgement)},
                 "judge_model": judge_models[method],
                 "protocol": "MMHal-Bench official eval_gpt4.py raw response format; rating 0-6"}
        for method, (response, judgement) in mmhal_evaluations.items()
    }
    report["mmhal_judge_model_verified"] = bool(judge_models) and all(judge_models.values())
    report["versions"] = _versions()
    report["run_protocol_metadata"] = sidecars
    report["all_protocols_verified"] = len(sidecars) == len(args.predictions)
    _write_json(report_path, report)
    write_comparison_csv(report, csv_path)
    print(json.dumps({"report": str(report_path.resolve()), "csv": str(csv_path.resolve())}, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlm-grounding", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Inspect dependencies and CUDA without loading a model.").set_defaults(func=_doctor)
    for name in ("pope", "causal", "mme", "mmhal"):
        sub = subparsers.add_parser(f"prepare-{name}", help=f"Convert locally downloaded {name} data.")
        sub.add_argument("--input", required=True)
        if name != "mme":
            sub.add_argument("--image-root", required=True)
        if name == "pope":
            sub.add_argument("--subset")
        sub.add_argument("--split", choices=SPLITS, default="test")
        sub.add_argument("--output", required=True)
        sub.add_argument("--overwrite", action="store_true")
        sub.set_defaults(func=_prepare)
    sub = subparsers.add_parser("prepare-coco", help="Build disjoint calibration and diagnostic COCO records.")
    sub.add_argument("--annotations", required=True)
    sub.add_argument("--image-root", required=True)
    sub.add_argument("--work-dir", required=True)
    sub.add_argument("--output", required=True)
    for split, default in (("discovery", 200), ("fit", 200), ("tune", 200), ("test", 0)):
        sub.add_argument(f"--{split}-count", type=int, default=default)
    sub.add_argument("--exclude-manifests", nargs="*", default=[])
    sub.add_argument("--seed", type=int, default=42)
    sub.add_argument("--overwrite", action="store_true")
    sub.set_defaults(func=_prepare)
    sub = subparsers.add_parser("verify-splits", help="Reject image/group leakage and missing files.")
    sub.add_argument("--manifests", nargs="+", required=True)
    sub.set_defaults(func=_verify)
    sub = subparsers.add_parser("fetch-upstream", help="Opt in to fetching pinned author/scorer source.")
    sub.add_argument("--names", nargs="+", required=True,
                     choices=("revis", "rudder", "grains", "vti", "chair", "causal"))
    sub.add_argument("--output-dir", default="upstream")
    sub.set_defaults(func=_fetch)
    sub = subparsers.add_parser("export-mmhal", help="Make the official MMHal response JSON.")
    sub.add_argument("--predictions", required=True)
    sub.add_argument("--template", required=True)
    sub.add_argument("--output", required=True)
    sub.add_argument("--overwrite", action="store_true")
    sub.set_defaults(func=_export_mmhal)
    for command, function in (("extract", _extract), ("discover", _discover),
                              ("fit", _fit), ("run", _run)):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", required=True)
        sub.add_argument("--manifest", required=command != "extract")
        sub.add_argument("--output", required=True)
        sub.add_argument("--limit", type=int, help="Development-only smoke subset; forbidden for test.")
        sub.add_argument("--overwrite", action="store_true")
        if command == "extract":
            sub.add_argument("--method", choices=METHODS[2:], required=True)
            sub.add_argument("--calibration", help="REVIS tune-split calibration manifest.")
        if command == "fit":
            sub.add_argument("--selection", required=True)
        if command == "run":
            sub.add_argument("--method", choices=METHODS, required=True)
            sub.add_argument("--artifact")
            sub.add_argument("--split", choices=SPLITS, default="test")
            sub.add_argument("--task", choices=("pope", "caption", "causal", "mme", "mmhal"))
        sub.set_defaults(func=function)
    sub = subparsers.add_parser("compare", help="Paired image-group bootstrap and optional upstream scorers.")
    sub.add_argument("--predictions", nargs="+", required=True)
    sub.add_argument("--output-dir", required=True)
    sub.add_argument("--baseline", default="base")
    sub.add_argument("--bootstrap-samples", type=int, default=1000)
    sub.add_argument("--seed", type=int, default=42)
    sub.add_argument("--chair-module")
    sub.add_argument("--annotations")
    sub.add_argument("--causal-module")
    sub.add_argument("--mmhal-evaluation", nargs=3, action="append", default=[],
                     metavar=("METHOD", "RESPONSES", "JUDGE_OUTPUT"),
                     help="Repeat per method; consumes official eval_gpt4.py raw JSON outputs.")
    sub.add_argument("--overwrite", action="store_true")
    sub.set_defaults(func=_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
