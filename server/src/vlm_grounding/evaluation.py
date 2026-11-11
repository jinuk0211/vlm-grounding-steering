"""Shared evaluation with strict output coverage and image-cluster uncertainty."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import random
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

Record = dict[str, Any]


def parse_yes_no(text: str) -> str | None:
    """Accept a leading standalone Yes/No, optionally after 'Answer:'; else invalid."""
    match = re.match(
        r"^\s*(?:answer\s*:\s*)?(yes|no)(?=\s|[.!?,;:]|$)", text, flags=re.IGNORECASE
    )
    return match.group(1).lower() if match else None


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _validate_predictions(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise ValueError("No predictions provided")
    keys = set()
    for record in records:
        missing = {"id", "group_id", "task", "prediction"} - record.keys()
        if missing:
            raise ValueError(f"Prediction is missing fields: {sorted(missing)}")
        key = (str(record.get("method", "base")), str(record["id"]))
        if key in keys:
            raise ValueError(f"Duplicate prediction: {key}")
        keys.add(key)
        if not isinstance(record["prediction"], str):
            raise ValueError(f"{record['id']}: prediction must be a string")


def match_predictions(
    manifest: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> list[Record]:
    """Join ID-only external outputs to trusted manifest data, failing on missing/extra IDs."""
    expected = {str(row["id"]): row for row in manifest}
    if len(expected) != len(manifest):
        raise ValueError("Manifest has duplicate IDs")
    by_method: defaultdict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in predictions:
        method, identifier = str(row.get("method", "base")), str(row["id"])
        if identifier in by_method[method]:
            raise ValueError(f"Duplicate prediction: {method}/{identifier}")
        by_method[method][identifier] = row
    if not by_method:
        raise ValueError("No predictions provided")
    joined = []
    protected = ("group_id", "task", "split", "image", "prompt", "label")
    for method, output in by_method.items():
        if output.keys() != expected.keys():
            raise ValueError(
                f"{method}: prediction coverage mismatch; "
                f"missing={len(expected.keys() - output.keys())}, "
                f"extra={len(output.keys() - expected.keys())}"
            )
        for identifier, reference in expected.items():
            prediction = output[identifier]
            for field in protected:
                if field in prediction and prediction[field] != reference.get(field):
                    raise ValueError(f"{method}/{identifier}: changed manifest {field}")
            joined.append({
                **dict(reference),
                **{key: value for key, value in prediction.items() if key not in protected},
                "metadata": dict(reference.get("metadata", {})),
                "method": method,
            })
    _validate_predictions(joined)
    return joined


def evaluate_predictions(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate one homogeneous task/subset. Metric fractions are in [0, 1]."""
    _validate_predictions(records)
    tasks = {row["task"] for row in records}
    methods = {row.get("method", "base") for row in records}
    if len(tasks) != 1 or len(methods) != 1:
        raise ValueError("evaluate_predictions requires one task and method")
    task = next(iter(tasks))
    metrics: dict[str, Any] = {
        "n": len(records),
        "n_image_groups": len({row["group_id"] for row in records}),
        "mean_words": sum(len(row["prediction"].split()) for row in records) / len(records),
    }
    durations = [row.get("elapsed_seconds") for row in records]
    if all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0 for t in durations):
        metrics["mean_seconds"] = sum(durations) / len(durations)
        tokens = [row.get("generated_tokens") for row in records]
        if all(isinstance(n, int) and n >= 0 for n in tokens):
            metrics["tokens_per_second"] = _ratio(sum(tokens), sum(durations))
            metrics["mean_generated_tokens"] = sum(tokens) / len(tokens)
    if task == "caption":
        has_chair = ["chair" in row for row in records]
        if any(has_chair) and not all(has_chair):
            raise ValueError("Partial CHAIR coverage: every caption needs scorer output")
        if all(has_chair):
            chairs = [row["chair"] for row in records]
            metrics.update({
                "CHAIR_s": sum(row["hallucinated_mentions"] > 0 for row in chairs) / len(chairs),
                "CHAIR_i": _ratio(
                    sum(row["hallucinated_mentions"] for row in chairs),
                    sum(row["generated_mentions"] for row in chairs),
                ),
                "object_recall": _ratio(
                    sum(row["correct_distinct_objects"] for row in chairs),
                    sum(row["ground_truth_objects"] for row in chairs),
                ),
                "mean_correct_distinct_objects": sum(
                    row["correct_distinct_objects"] for row in chairs
                ) / len(chairs),
            })
        else:
            metrics["chair_status"] = "not_scored"
        return metrics
    if task == "mmhal":
        has_score = ["mmhal_score" in row for row in records]
        if any(has_score) and not all(has_score):
            raise ValueError("Partial MMHal coverage: every response needs a judge rating")
        if not all(has_score):
            metrics["mmhal_status"] = "not_scored"
            return metrics
        scores = [row["mmhal_score"] for row in records]
        if any(type(score) is not int or score not in range(7) for score in scores):
            raise ValueError("MMHal ratings must be integers from 0 through 6")
        by_type: defaultdict[str, list[int]] = defaultdict(list)
        for row, score in zip(records, scores):
            category = row.get("metadata", {}).get("question_type")
            if not isinstance(category, str) or not category:
                raise ValueError("MMHal records require metadata.question_type")
            by_type[category].append(score)
        metrics.update({
            "mmhal_status": "scored",
            "mmhal_average_score": sum(scores) / len(scores),
            "mmhal_hallucination_rate": sum(score < 3 for score in scores) / len(scores),
            "mmhal_type_scores": {
                category: sum(values) / len(values) for category, values in sorted(by_type.items())
            },
        })
        return metrics
    if any(row.get("label") not in {"yes", "no"} for row in records):
        raise ValueError("Binary-task evaluation requires yes/no ground truth")
    parsed = [parse_yes_no(row["prediction"]) for row in records]
    tp = sum(p == "yes" and r["label"] == "yes" for p, r in zip(parsed, records))
    fp = sum(p == "yes" and r["label"] == "no" for p, r in zip(parsed, records))
    positives = sum(r["label"] == "yes" for r in records)
    correct = [p == r["label"] for p, r in zip(parsed, records)]
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, positives)
    metrics.update({
        "accuracy": sum(correct) / len(records),
        "precision": precision,
        "recall": recall,
        # Invalid positive answers are misses; an all-invalid system cannot gain recall.
        "F1": _ratio(2 * tp, 2 * tp + fp + positives - tp),
        "yes_ratio": parsed.count("yes") / len(records),
        "invalid_count": parsed.count(None),
        "invalid_ratio": parsed.count(None) / len(records),
        "positive_accuracy": recall,
        "negative_accuracy": _ratio(
            sum(p == "no" and r["label"] == "no" for p, r in zip(parsed, records)),
            len(records) - positives,
        ),
    })
    if task == "mme":
        grouped: defaultdict[tuple[str, str], list[bool]] = defaultdict(list)
        for row, is_correct in zip(records, correct):
            category = row.get("metadata", {}).get("category")
            if not category:
                raise ValueError("MME records require metadata.category")
            grouped[(category, row["group_id"])].append(is_correct)
        if any(len(values) != 2 for values in grouped.values()):
            raise ValueError("MME acc+ requires exactly two questions per category/image")
        category_rows: defaultdict[str, list[bool]] = defaultdict(list)
        category_pairs: defaultdict[str, list[bool]] = defaultdict(list)
        for (category, _), values in grouped.items():
            category_rows[category].extend(values)
            category_pairs[category].append(all(values))
        scores = {
            category: 100 * (
                sum(values) / len(values)
                + sum(category_pairs[category]) / len(category_pairs[category])
            )
            for category, values in category_rows.items()
        }
        metrics["mme_category_scores"] = scores
        metrics["mme_score_sum"] = sum(scores.values())
        metrics["mme_protocol"] = "accuracy + paired accuracy, strict leading yes/no parser"
    return metrics


def _load_external_module(path: str | Path, name: str):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _caption_image_id(record: Mapping[str, Any]) -> int:
    value = record.get("metadata", {}).get("coco_image_id")
    if value is None and str(record["group_id"]).startswith("coco:"):
        value = str(record["group_id"]).split(":", 1)[1]
    if value is None:
        raise ValueError(f"{record['id']}: missing metadata.coco_image_id")
    return int(value)


def attach_chair_scores(
    records: Sequence[Mapping[str, Any]], output: Mapping[str, Any] | str | Path
) -> list[Record]:
    """Consume CHAIR's object normalization/output, never guess synonyms locally."""
    _validate_predictions(records)
    if isinstance(output, (str, Path)):
        output = json.loads(Path(output).read_text(encoding="utf-8"))
    sentences = output["sentences"]
    scored = {int(row["image_id"]): row for row in sentences}
    ids = [_caption_image_id(row) for row in records]
    if len(set(ids)) != len(ids) or len(scored) != len(sentences):
        raise ValueError("CHAIR input/output must contain one caption per image")
    if set(ids) != set(scored):
        raise ValueError("CHAIR image coverage does not match predictions")
    enriched = []
    for record in records:
        row = scored[_caption_image_id(record)]
        if record["prediction"] != row["caption"]:
            raise ValueError(f"{record['id']}: CHAIR caption differs from actual prediction")
        generated = row["mscoco_generated_words"]
        ground_truth = set(row["mscoco_gt_words"])
        hallucinated = row["mscoco_hallucinated_words"]
        enriched.append({
            **dict(record),
            "chair": {
                "hallucinated_mentions": len(hallucinated),
                "generated_mentions": len(generated),
                "correct_distinct_objects": len(set(generated) & ground_truth),
                "ground_truth_objects": len(ground_truth),
                "source": output.get("scorer_provenance", "external CHAIR per-sentence output"),
            },
        })
    return enriched


def run_official_chair(
    records: Sequence[Mapping[str, Any]],
    chair_module_path: str | Path,
    annotation_dir: str | Path,
    output_path: str | Path | None = None,
) -> list[Record]:
    """Run an explicitly supplied Maxlinn Python3 port of original CHAIR.

    Not a new local approximation. The upstream port and its dependencies must be
    installed separately. Build annotations from JSON; never load upstream pickle.
    """
    _validate_predictions(records)
    if {row["task"] for row in records} != {"caption"}:
        raise ValueError("CHAIR accepts caption records only")
    if len({row.get("method", "base") for row in records}) != 1:
        raise ValueError("Score each method separately with CHAIR")
    annotation_dir = Path(annotation_dir).resolve()
    for kind in ("captions", "instances"):
        for split in ("train", "val"):
            required = annotation_dir / f"{kind}_{split}2014.json"
            if not required.is_file():
                raise FileNotFoundError(required)
    module = _load_external_module(chair_module_path, "_vlm_grounding_chair")
    evaluator = module.CHAIR(str(annotation_dir))
    payload = [
        {"image_id": _caption_image_id(row), "caption": row["prediction"]}
        for row in records
    ]
    if len({r["image_id"] for r in payload}) != len(payload):
        raise ValueError("CHAIR requires one caption per image")
    if any(row["image_id"] not in evaluator.imid_to_objects for row in payload):
        raise ValueError("CHAIR annotations do not contain every evaluated image")
    with tempfile.TemporaryDirectory(prefix="vlm-chair-") as temp_dir:
        captions_path = Path(temp_dir) / "captions.json"
        captions_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            output = evaluator.compute_chair(str(captions_path), "image_id", "caption")
        except ZeroDivisionError as exc:
            raise ValueError(
                "External CHAIR scorer divided by zero (possibly no recognized object mentions). "
                "Report CHAIR_i as undefined; do not substitute a perfect score."
            ) from exc
    output["scorer_provenance"] = {
        "implementation": "Maxlinn Python3 port of LisaAnne/Hallucination CHAIR",
        "path": str(Path(chair_module_path).resolve()),
        "sha256": hashlib.sha256(Path(chair_module_path).read_bytes()).hexdigest(),
        "annotation_dir": str(annotation_dir),
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    return attach_chair_scores(records, output)


def run_official_causal(
    records: Sequence[Mapping[str, Any]], metric_module_path: str | Path
) -> dict[str, Any]:
    """Run Causal-HalBench metric.py on complete manifest-aligned predictions."""
    _validate_predictions(records)
    if {row["task"] for row in records} != {"causal"}:
        raise ValueError("Causal-HalBench scorer accepts causal records only")
    if len({row.get("method", "base") for row in records}) != 1:
        raise ValueError("Score each method separately")
    qa, responses = [], []
    for row in records:
        source = dict(row.get("metadata", {}).get("causal_qa", {}))
        if not {"image_name", "type", "tag", "id", "answer"} <= source.keys():
            raise ValueError(f"{row['id']}: missing original Causal-HalBench QA metadata")
        if source["answer"].lower() != row["label"]:
            raise ValueError("Causal-HalBench metadata label mismatch")
        qa.append(source)
        responses.append({
            **source,
            "answer": parse_yes_no(row["prediction"]) or "__invalid__",
        })
    module = _load_external_module(metric_module_path, "_vlm_grounding_causal")
    with tempfile.TemporaryDirectory(prefix="vlm-causal-") as temp_dir:
        qa_path, response_path = Path(temp_dir) / "qa.json", Path(temp_dir) / "responses.json"
        qa_path.write_text(json.dumps(qa), encoding="utf-8")
        response_path.write_text(json.dumps(responses), encoding="utf-8")
        output = module.evaluate(str(qa_path), str(response_path))
    if not output:
        raise ValueError("Official Causal-HalBench scorer returned no metrics")
    return {
        **output,
        "scorer_sha256": hashlib.sha256(Path(metric_module_path).read_bytes()).hexdigest(),
        "coverage": len(records),
        "protocol": "official metric.py; strict leading yes/no; invalid answers retained",
    }


def mmhal_responses(
    records: Sequence[Mapping[str, Any]], template: str | Path
) -> list[Record]:
    """Restore the official 96-row response schema in its required original order."""
    _validate_predictions(records)
    rows = json.loads(Path(template).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or len(rows) != 96 or any(not isinstance(r, dict) for r in rows):
        raise ValueError("Official MMHal response_template.json must contain 96 objects")
    if len(records) != 96 or any(row["task"] != "mmhal" for row in records):
        raise ValueError("MMHal export requires all 96 benchmark predictions")
    methods = {row.get("method", "base") for row in records}
    if len(methods) != 1:
        raise ValueError("MMHal export requires predictions from one method")
    indexed = {row.get("metadata", {}).get("mmhal_index"): row for row in records}
    if len(indexed) != 96 or set(indexed) != set(range(96)):
        raise ValueError("MMHal predictions must cover every original template index")
    output = []
    for index, source in enumerate(rows):
        prediction = indexed[index]
        metadata = prediction.get("metadata", {})
        fields = {
            "image_id": metadata.get("image_id"),
            "image_src": metadata.get("image_src"),
            "image_content": metadata.get("image_content"),
            "gt_answer": metadata.get("gt_answer"),
            "question_type": metadata.get("question_type"),
            "question_topic": metadata.get("question_topic"),
            "question": prediction["prompt"],
        }
        if any(source.get(field) != value for field, value in fields.items()):
            raise ValueError(f"MMHal template differs from prediction at index {index}")
        output.append({**source, "model_answer": prediction["prediction"]})
    return output


def attach_mmhal_scores(
    records: Sequence[Mapping[str, Any]],
    response_path: str | Path,
    evaluation_path: str | Path,
) -> list[Record]:
    """Join ordered official GPT-4 raw responses only after verifying exact model answers."""
    _validate_predictions(records)
    responses = json.loads(Path(response_path).read_text(encoding="utf-8-sig"))
    evaluations = json.loads(Path(evaluation_path).read_text(encoding="utf-8-sig"))
    if not isinstance(responses, list) or len(responses) != 96:
        raise ValueError("MMHal response file must contain 96 ordered entries")
    if not isinstance(evaluations, list) or len(evaluations) != 96:
        raise ValueError("MMHal judge output must contain 96 ordered entries")
    if len(records) != 96 or any(row["task"] != "mmhal" for row in records):
        raise ValueError("MMHal scoring requires all 96 benchmark predictions")
    indexed = {row.get("metadata", {}).get("mmhal_index"): row for row in records}
    if len(indexed) != 96 or set(indexed) != set(range(96)):
        raise ValueError("MMHal scoring requires all original template indices")
    scored = []
    for index, (response, evaluation) in enumerate(zip(responses, evaluations)):
        prediction = indexed[index]
        metadata = prediction.get("metadata", {})
        if not isinstance(response, dict) or any((
            response.get("model_answer") != prediction["prediction"],
            response.get("question") != prediction["prompt"],
            response.get("image_id") != metadata.get("image_id"),
            response.get("image_src") != metadata.get("image_src"),
            response.get("image_content") != metadata.get("image_content"),
            response.get("gt_answer") != metadata.get("gt_answer"),
            response.get("question_type") != metadata.get("question_type"),
            response.get("question_topic") != metadata.get("question_topic"),
        )):
            raise ValueError(f"MMHal judged response differs from prediction at index {index}")
        try:
            judge_text = evaluation["choices"][0]["message"]["content"]
        except (TypeError, KeyError, IndexError) as exc:
            raise ValueError(f"MMHal judge entry {index} has no response text") from exc
        if not isinstance(judge_text, str):
            raise ValueError(f"MMHal judge entry {index} must be text")
        ratings = re.findall(r"\brating\s*:\s*([0-6])\b", judge_text, flags=re.IGNORECASE)
        if len(ratings) != 1:
            raise ValueError(f"MMHal judge entry {index} needs exactly one rating 0-6")
        scored.append({**dict(prediction), "mmhal_score": int(ratings[0])})
    return scored


def _cohort(record: Mapping[str, Any]) -> str:
    subset = record.get("metadata", {}).get("subset")
    return f"{record['task']}:{subset}" if subset else str(record["task"])


def _quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compare_methods(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline: str = "base",
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired comparisons require identical IDs/inputs; resample original image groups."""
    _validate_predictions(records)
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    fingerprints = {row["protocol_fingerprint"] for row in records if row.get("protocol_fingerprint")}
    if len(fingerprints) > 1:
        raise ValueError("Mismatched evaluation protocol fingerprints across predictions")
    protocol_verified = bool(fingerprints) and all(
        row.get("protocol_fingerprint") for row in records
    )
    methods: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        methods[str(row.get("method", "base"))].append(row)
    if baseline not in methods:
        raise ValueError(f"Missing baseline: {baseline}")
    reference = {str(row["id"]): row for row in methods[baseline]}
    all_scores: dict[str, dict[str, Any]] = {}
    cohorts: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for method, rows in methods.items():
        current = {str(row["id"]): row for row in rows}
        if current.keys() != reference.keys():
            raise ValueError(f"{method}: IDs differ from {baseline}; missing outputs cannot be dropped")
        for identifier, row in current.items():
            for field in ("group_id", "task", "split", "image", "prompt", "label"):
                if row.get(field) != reference[identifier].get(field):
                    raise ValueError(f"{method}/{identifier}: unmatched {field}")
            if _cohort(row) != _cohort(reference[identifier]):
                raise ValueError(f"{method}/{identifier}: unmatched subset")
            cohorts[(method, _cohort(row))].append(row)
        all_scores[method] = {
            cohort: evaluate_predictions(values)
            for (name, cohort), values in cohorts.items() if name == method
        }
    deltas = []
    rng = random.Random(seed)
    for method in methods:
        if method == baseline:
            continue
        for cohort in all_scores[baseline]:
            base_rows = {str(r["id"]): r for r in cohorts[(baseline, cohort)]}
            test_rows = {str(r["id"]): r for r in cohorts[(method, cohort)]}
            groups: defaultdict[str, list[str]] = defaultdict(list)
            for identifier, row in base_rows.items():
                groups[str(row["group_id"])].append(identifier)
            group_ids = sorted(groups)
            task = next(iter(base_rows.values()))["task"]
            metric_names = (
                ("CHAIR_s", "CHAIR_i", "object_recall", "mean_words") if task == "caption"
                else ("mmhal_average_score", "mmhal_hallucination_rate", "mean_words")
                if task == "mmhal" else ("accuracy", "recall", "F1")
            )
            distributions: defaultdict[str, list[float]] = defaultdict(list)
            # Duplicated sampled groups receive distinct IDs for validation.
            for _ in range(bootstrap_samples):
                sampled_base, sampled_method = [], []
                for draw_index in range(len(group_ids)):
                    group_id = rng.choice(group_ids)
                    for identifier in groups[group_id]:
                        for target, source in ((sampled_base, base_rows), (sampled_method, test_rows)):
                            target.append({
                                **source[identifier],
                                "id": f"{draw_index}:{identifier}",
                                "group_id": f"{draw_index}:{group_id}",
                            })
                base_metrics = evaluate_predictions(sampled_base)
                method_metrics = evaluate_predictions(sampled_method)
                for metric in metric_names:
                    a, b = base_metrics.get(metric), method_metrics.get(metric)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        distributions[metric].append(b - a)
            for metric in metric_names:
                a, b = all_scores[baseline][cohort].get(metric), all_scores[method][cohort].get(metric)
                if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                    continue
                values = distributions[metric]
                deltas.append({
                    "method": method, "baseline": baseline, "cohort": cohort,
                    "metric": metric, "delta": b - a,
                    "ci_low": _quantile(values, 0.025) if values else None,
                    "ci_high": _quantile(values, 0.975) if values else None,
                    "valid_bootstrap_samples": len(values),
                    "image_groups": len(group_ids),
                })
    return {
        "methods": all_scores, "paired_deltas": deltas,
        "bootstrap": {"unit": "original image group", "samples": bootstrap_samples, "seed": seed},
        "protocol_verified": protocol_verified,
        "protocol_fingerprint": next(iter(fingerprints)) if protocol_verified else None,
    }


def write_comparison_csv(report: Mapping[str, Any], path: str | Path) -> None:
    """Write comparable scalar metrics in long format; nested MME scores are separate rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "cohort", "metric", "value"])
        writer.writeheader()
        for method, cohorts in report["methods"].items():
            for cohort, metrics in cohorts.items():
                for metric, value in metrics.items():
                    if isinstance(value, dict):
                        for key, subvalue in value.items():
                            writer.writerow({
                                "method": method, "cohort": cohort,
                                "metric": f"{metric}.{key}", "value": subvalue,
                            })
                    else:
                        writer.writerow({
                            "method": method, "cohort": cohort, "metric": metric, "value": value
                        })
