"""Validate VTI's paper strength grid and visual-conditioned extraction coverage."""

import math
import re


def strength_grid() -> list[dict[str, float]]:
    """Paper section 5.1: each strength is 0.1, 0.2, ..., 1.0."""
    return [{"vision_alpha": vision / 10, "text_alpha": text / 10}
            for vision in range(1, 11) for text in range(1, 11)]


def validate_complete_grid(profiles: list[dict], expected_ids: list[str]) -> dict:
    """Check full coverage; native extraction, scoring and winner selection are separate."""
    if (not expected_ids or any(not isinstance(rid, str) or not rid for rid in expected_ids)
            or len(expected_ids) != len(set(expected_ids))):
        raise ValueError("Expected a nonempty unique development cohort")
    grid = strength_grid()
    expected = {(p["vision_alpha"], p["text_alpha"]) for p in grid}
    if len(profiles) != len(expected):
        raise ValueError("Expected all 100 strength combinations")
    required = {"split", "vision_alpha", "text_alpha", "coverage_ids", "vision_vectors_sha256",
                "text_vectors_sha256", "text_fit_vision_alpha"}
    seen, text_by_vision = set(), {}
    visual_sha = None
    for profile in profiles:
        if not required <= profile.keys():
            raise ValueError("Missing candidate coverage or extraction provenance")
        pair = (profile["vision_alpha"], profile["text_alpha"])
        if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in pair) or pair not in expected or pair in seen):
            raise ValueError("Duplicate or non-paper strength combination")
        if profile["split"] != "tune":
            raise ValueError("Strength selection requires development profiles")
        if not isinstance(profile["coverage_ids"], list) or profile["coverage_ids"] != expected_ids:
            raise ValueError("Every candidate must finish the same ordered development cohort")
        for name in ("vision_vectors_sha256", "text_vectors_sha256"):
            if not isinstance(profile[name], str) or not re.fullmatch(r"[0-9a-f]{64}", profile[name]):
                raise ValueError("Missing extraction tensor SHA256")
        fit_alpha = profile["text_fit_vision_alpha"]
        if (isinstance(fit_alpha, bool) or not isinstance(fit_alpha, (int, float))
                or not math.isfinite(fit_alpha) or fit_alpha != pair[0]):
            raise ValueError("Text direction was not fitted at the candidate visual strength")
        if visual_sha is not None and visual_sha != profile["vision_vectors_sha256"]:
            raise ValueError("Visual directions changed between strength candidates")
        visual_sha = profile["vision_vectors_sha256"]
        previous = text_by_vision.setdefault(pair[0], profile["text_vectors_sha256"])
        if previous != profile["text_vectors_sha256"]:
            raise ValueError("Candidates at one visual strength must share its fitted text direction")
        seen.add(pair)
    if seen != expected:
        raise ValueError("Incomplete strength grid")
    return {"candidate_count": len(seen), "development_count": len(expected_ids),
            "coverage_ids": list(expected_ids), "vision_vectors_sha256": visual_sha,
            "text_vectors_by_vision_alpha": dict(sorted(text_by_vision.items())),
            "scope": "Structural coverage and provenance consistency; not native extraction or scores"}


def select_strengths(profiles: list[dict], base: dict, expected_ids: list[str]) -> dict:
    """Predeclared reconstruction: healthy outputs, caption recall floor, task-specific metrics."""
    coverage = validate_complete_grid(profiles, expected_ids)
    if base["coverage_ids"] != expected_ids or base["health_failures"]:
        raise ValueError("Development Base is incomplete or unhealthy")
    for profile in [base, *profiles]:
        for task, keys in (("caption", ("CHAIR_s", "CHAIR_i", "object_recall")),
                           ("pope", ("official_pope_F1", "official_pope_accuracy"))):
            metrics = profile["metrics"][task]
            count = metrics["n"]
            if (isinstance(count, bool) or not isinstance(count, int) or count <= 0
                    or count != base["metrics"][task]["n"]):
                raise ValueError("Incomplete task metric coverage")
            for key in keys:
                value = metrics[key]
                if value is None and key in ("CHAIR_i", "official_pope_F1"):
                    continue
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError("Invalid development metric")
        if sum(profile["metrics"][task]["n"] for task in ("caption", "pope")) != len(expected_ids):
            raise ValueError("Task counts differ from development coverage")
    selected = {}
    for task in ("caption", "pope"):
        eligible = [p for p in profiles if p["health_failures"][task] == 0]
        if task == "caption":
            eligible = [p for p in eligible if p["metrics"][task]["CHAIR_i"] is not None
                        and p["metrics"][task]["object_recall"] >=
                        .95 * base["metrics"][task]["object_recall"]]
            def key(p):
                return (p["metrics"][task]["CHAIR_s"], p["metrics"][task]["CHAIR_i"],
                        round(p["vision_alpha"] * 10) + round(p["text_alpha"] * 10),
                        p["vision_alpha"], p["text_alpha"])
        else:
            eligible = [p for p in eligible if p["metrics"][task]["official_pope_F1"] is not None]
            def key(p):
                return (-p["metrics"][task]["official_pope_F1"],
                        -p["metrics"][task]["official_pope_accuracy"],
                        round(p["vision_alpha"] * 10) + round(p["text_alpha"] * 10),
                        p["vision_alpha"], p["text_alpha"])
        if not eligible:
            raise ValueError(f"No healthy eligible {task} candidate; no winner")
        best = min(eligible, key=key)
        selected[task] = {name: best[name] for name in ("vision_alpha", "text_alpha",
                         "vision_vectors_sha256", "text_vectors_sha256", "text_fit_vision_alpha")}
        selected[task]["metrics"] = best["metrics"][task]
    return {"coverage": coverage, "selected": selected,
            "selection_rule": "Explicit reconstruction; no test-score selection or improvement guarantee"}
