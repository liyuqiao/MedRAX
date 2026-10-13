from __future__ import annotations

from enum import Enum
from typing import Any, Iterable


class RetrievalMode(str, Enum):
    REPORT = "report"
    COMBINED_PHRASES = "combined_phrases"
    INDEPENDENT_PHRASES = "independent_phrases"
    COMPARE = "compare"


def deduplicate_phrase_results(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated phrase text while retaining its first ranking and provenance."""
    unique: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        phrase = str(row["phrase"]).strip()
        key = phrase.casefold()
        study_id = row.get("study_id")
        if key not in unique:
            item = {
                "rank": len(order) + 1,
                "phrase": phrase,
                "score": row.get("score", row.get("similarity")),
                "study_id": study_id,
                "source_study_ids": [],
                "duplicate_count": 0,
            }
            unique[key] = item
            order.append(key)
        item = unique[key]
        if study_id not in item["source_study_ids"]:
            item["source_study_ids"].append(study_id)
        item["duplicate_count"] += 1
    return [unique[key] for key in order]
