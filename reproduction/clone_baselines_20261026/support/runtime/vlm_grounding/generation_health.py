"""Generation validity gates; these diagnostics never replace benchmark scoring."""

import re


def failure_reasons(row: dict, cap: int) -> list[str]:
    """Inspect raw whitespace, repeated symbols and unfinished binary answers."""
    raw = row.get("raw_decoded", row["prediction"])
    reasons = []
    if not raw.strip():
        reasons.append("empty")
    trailing = raw[len(raw.rstrip()):]
    if len(trailing) > 100 and trailing.count("\n") > 20:
        reasons.append("whitespace_repetition")
    words = re.findall(r"\w+|[^\w\s]", raw.lower())
    if len(words) >= 32 and 1 - len(set(words)) / len(words) > .85:
        reasons.append("word_or_symbol_repetition")
    if (row["task"] == "pope" and row["generated_tokens"] >= cap
            and not row.get("ended_with_eos", False)):
        reasons.append("answer_length_cap")
    return reasons
