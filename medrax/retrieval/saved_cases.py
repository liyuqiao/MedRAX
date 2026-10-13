from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import RetrievalMode, deduplicate_phrase_results


class SavedCaseRetriever:
    """Replay experiments 1, 2, and 4 from their per-image JSONL artifacts."""

    def __init__(self, report_path: str | Path, combined_path: str | Path,
                 independent_path: str | Path):
        self.paths = {
            RetrievalMode.REPORT: Path(report_path),
            RetrievalMode.COMBINED_PHRASES: Path(combined_path),
            RetrievalMode.INDEPENDENT_PHRASES: Path(independent_path),
        }
        self._indexes: dict[RetrievalMode, dict[str, dict[str, Any]]] = {}

    @staticmethod
    def _sample_from_image_path(image_path: str) -> str:
        sample_id = Path(image_path).stem
        if not sample_id:
            raise ValueError("image_path does not contain a sample identifier")
        return sample_id

    def _index(self, mode: RetrievalMode) -> dict[str, dict[str, Any]]:
        if mode in self._indexes:
            return self._indexes[mode]
        path = self.paths[mode]
        if not path.is_file():
            raise FileNotFoundError(f"{mode.value} result file not found: {path}")
        records: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path} at line {line_number}") from exc
                sample_id = str(record.get("sample_id", ""))
                if not sample_id:
                    raise ValueError(f"missing sample_id in {path} at line {line_number}")
                if sample_id in records:
                    raise ValueError(f"duplicate sample_id {sample_id!r} in {path}")
                records[sample_id] = record
        self._indexes[mode] = records
        return records

    def _get(self, mode: RetrievalMode, sample_id: str) -> dict[str, Any]:
        try:
            return self._index(mode)[sample_id]
        except KeyError as exc:
            raise KeyError(f"sample_id {sample_id!r} not found in {mode.value} results") from exc

    @staticmethod
    def _text_result(record: dict[str, Any], top_k: int, combined: bool,
                     include_reference: bool = False) -> dict[str, Any]:
        rows = []
        for position, source in enumerate(record.get("top_k", [])[:top_k], 1):
            text = source.get("text", "")
            row = {
                "rank": source.get("rank", position),
                "text": text,
                "score": source.get("similarity", source.get("score")),
                "study_ids": source.get("study_ids", []),
                "is_correct": source.get("is_correct"),
            }
            if combined:
                row["phrases"] = [part.strip().rstrip(".") for part in text.split(",")
                                  if part.strip()]
            rows.append(row)
        result = {
            "correct_rank": record.get("correct_rank"),
            "top_k": rows,
        }
        if include_reference:
            result["reference_text"] = record.get("correct_text")
        return result

    @staticmethod
    def _phrase_result(record: dict[str, Any], top_k: int,
                       include_reference: bool = False) -> dict[str, Any]:
        raw = list(record.get("top_k", [])[:top_k])
        result = {
            "first_correct_rank": record.get("first_correct_rank"),
            "raw_top_k": raw,
            "semantic_top_k": deduplicate_phrase_results(raw),
        }
        if include_reference:
            result["reference_phrases"] = record.get("correct_phrases", [])
        return result

    def retrieve(self, mode: RetrievalMode | str, *, sample_id: str | None = None,
                 image_path: str | None = None, top_k: int = 10,
                 include_reference: bool = False) -> dict[str, Any]:
        mode = RetrievalMode(mode)
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        resolved_id = sample_id or (self._sample_from_image_path(image_path) if image_path else None)
        if not resolved_id:
            raise ValueError("sample_id or image_path is required")
        if mode is RetrievalMode.COMPARE:
            return self.compare(sample_id=resolved_id, top_k=top_k,
                                include_reference=include_reference)
        record = self._get(mode, resolved_id)
        if mode is RetrievalMode.REPORT:
            return self._text_result(record, top_k, combined=False,
                                     include_reference=include_reference)
        if mode is RetrievalMode.COMBINED_PHRASES:
            return self._text_result(record, top_k, combined=True,
                                     include_reference=include_reference)
        return self._phrase_result(record, top_k, include_reference=include_reference)

    def compare(self, *, sample_id: str | None = None, image_path: str | None = None,
                top_k: int = 10, include_reference: bool = False) -> dict[str, Any]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        resolved_id = sample_id or (self._sample_from_image_path(image_path) if image_path else None)
        if not resolved_id:
            raise ValueError("sample_id or image_path is required")
        report = self._get(RetrievalMode.REPORT, resolved_id)
        combined = self._get(RetrievalMode.COMBINED_PHRASES, resolved_id)
        independent = self._get(RetrievalMode.INDEPENDENT_PHRASES, resolved_id)
        study_ids = {report.get("study_id"), combined.get("study_id"), independent.get("study_id")}
        if len(study_ids) != 1:
            raise ValueError(f"study mismatch for sample {resolved_id!r}: {sorted(study_ids)}")
        return {
            "sample_id": resolved_id,
            "study_id": report.get("study_id"),
            "image_path": report.get("image_path") or independent.get("image_path"),
            "report_retrieval": self._text_result(
                report, top_k, combined=False, include_reference=include_reference),
            "combined_phrase_retrieval": self._text_result(
                combined, top_k, combined=True, include_reference=include_reference),
            "independent_phrase_retrieval": self._phrase_result(
                independent, top_k, include_reference=include_reference),
        }
