"""Calculate explicit experiment-budget scenarios from completed GPU profiles."""

import argparse
import json
from pathlib import Path

import numpy as np


def forecast(rows, caption_count=2096, binary_count=9000):
    tasks = {}
    for task in ("pope", "caption"):
        by_id = {}
        for row in rows:
            if row["task"] == task and row["status"] == "passed":
                values = by_id.setdefault(row["id"], {})
                if row["method"] in values:
                    raise ValueError("Duplicate sample/method result")
                values[row["method"]] = row["seconds"]
        expected = {"base", "ephs", "revis", "rudder", "grains", "vti"}
        if len(by_id) != 50 or any(set(values) != expected for values in by_id.values()):
            raise ValueError("Forecast requires 50 complete six-method samples per task")
        tasks[task] = np.array([sum(values.values()) for values in by_id.values()])
    rng = np.random.default_rng(42)
    draws = {}
    for task, values in tasks.items():
        draws[task] = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(1)
    total = binary_count * draws["pope"] + caption_count * draws["caption"]
    return {"evaluation_seconds": float(binary_count * tasks["pope"].mean()
                                        + caption_count * tasks["caption"].mean()),
            "evaluation_seconds_ci95": np.quantile(total, [.025, .975]).tolist()}


def main(args):
    folder = args.input
    summary = json.loads((folder / "summary.json").read_text())
    setup = json.loads((folder / "setup.json").read_text())
    rows = [json.loads(line) for line in (folder / "samples.jsonl").read_text().splitlines()]
    if any(value["attempted"] != 100 or value["failed"] for value in summary.values()):
        raise ValueError("Incomplete or failed profile; refusing a complete-experiment forecast")
    result = forecast(rows, args.captions + args.mmhal, args.binary)
    offline = sum(setup[method]["seconds"] * (args.preference_pairs / 4)
                  for method in ("revis", "grains", "vti"))
    offline += setup["ephs"]["seconds"] * args.fit_questions / 8
    initial = setup["discovery_initial"]["seconds"] * args.discovery_examples
    heads = setup["discovery_head_work"]["seconds"] * args.discovery_examples
    total = result["evaluation_seconds"] + offline + setup["model_load_seconds"] + initial
    result.update(
        workload={"caption": args.captions, "pope": args.binary, "mmhal_caption_time_proxy": args.mmhal,
                  "methods": 6, "seeds": 1, "preference_pairs": args.preference_pairs,
                  "calibration_pairs": args.preference_pairs, "fit_questions": args.fit_questions,
                  "fit_epochs": 2, "discovery_candidates": args.discovery_examples},
        offline_preparation_seconds_estimated=offline,
        discovery_seconds_compute_proxy_range=[initial, initial + heads],
        planned_seconds_range_before_allowance=[total, total + heads],
        planned_seconds_range_with_20_percent_allowance=[total * 1.2, (total + heads) * 1.2],
        excludes=["data acquisition", "MMHal external judge API", "hyperparameter search", "extra seeds and ablations"],
        limitations=["Gemma adaptations, not author reproductions", "MMHal runtime uses COCO caption proxy",
                     "offline times extrapolated from four long caption pairs", "discovery uses same-image controls for compute only",
                     "95% interval covers sampled inference means only; not domain shift or setup extrapolation"],
    )
    (folder / "forecast.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Gemma 4 E4B throughput pilot", "", "100 images × 6 execution conditions = 600 measured generations.",
             "BF16, batch 1, 448 px, greedy decoding, 128-token cap. Warmups excluded.",
             "Four baselines: REVIS, RUDDER, GrAInS, VTI. Comparators: Base and EPHS.",
             "", "| Method | Short mean s | Caption mean s | Overall p95 s | Allocated max GiB | Reserved max GiB | Device max GiB (sampled) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, entry in summary.items():
        groups = entry["groups"]
        all_rows = groups["all"]
        lines.append(f"| {method} | {groups['pope']['mean_seconds']:.3f} | {groups['caption']['mean_seconds']:.3f} | "
                     f"{all_rows['p95_seconds']:.3f} | {all_rows['peak_allocated_gib']:.2f} | "
                     f"{all_rows['peak_reserved_gib']:.2f} | {all_rows['device_peak_used_gib_sampled']:.2f} |")
    lines += ["", "## One-time preparation measurements", "",
              "| Stage | Workload | Seconds | Allocated max GiB |", "|---|---|---:|---:|"]
    for method in ("revis", "grains", "vti", "ephs"):
        entry = setup[method]
        workload = "8 questions, 2 epochs, fixed 8 heads" if method == "ephs" else "4 caption pairs" + (" + 4 calibration pairs" if method == "revis" else "")
        lines.append(f"| {method} | {workload} | {entry['seconds']:.2f} | {entry['peak_allocated_bytes'] / 2**30:.2f} |")
    interval = result["evaluation_seconds_ci95"]
    planned = result["planned_seconds_range_with_20_percent_allowance"]
    lines += ["", "## Declared full-run scenario", "",
              f"{args.captions:,} COCO captions + {args.binary:,} POPE questions + {args.mmhal} MMHal questions; six conditions, one seed.",
              f"Inference estimate: **{result['evaluation_seconds'] / 3600:.2f} GPU hours**, empirical mean interval "
              f"{interval[0] / 3600:.2f}–{interval[1] / 3600:.2f} hours.",
              f"Preparation scenario: {args.preference_pairs} caption preference pairs and the same number of REVIS calibration pairs; "
              f"{args.fit_questions} EPHS questions for two epochs; {args.discovery_examples} discovery candidates.",
              f"Estimated preparation excluding discovery: {offline / 3600:.2f} hours. Discovery compute proxy: "
              f"{initial / 3600:.2f}–{(initial + heads) / 3600:.2f} hours depending on the fraction passing the mask-effect gate.",
              f"Planning range with an explicit 20% operational allowance: **{planned[0] / 3600:.2f}–{planned[1] / 3600:.2f} hours**.",
              "", "These are forecasts for the current Gemma ports. MMHal uses caption throughput as a proxy. "
              "Offline extrapolation from four pairs is provisional; the causal selection itself was not evaluated. "
              "Dataset downloads, external judge calls, tuning, additional seeds and ablations are excluded.", ""]
    (folder / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--captions", type=int, default=2000)
    parser.add_argument("--binary", type=int, default=9000)
    parser.add_argument("--mmhal", type=int, default=96)
    parser.add_argument("--preference-pairs", type=int, default=200)
    parser.add_argument("--fit-questions", type=int, default=400)
    parser.add_argument("--discovery-examples", type=int, default=100)
    main(parser.parse_args())
