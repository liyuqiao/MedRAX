import json

import pytest

from medrax.retrieval.saved_cases import SavedCaseRetriever


def _write(path, record):
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def _paths(tmp_path, study_id=7):
    report = tmp_path / "report.jsonl"
    combined = tmp_path / "combined.jsonl"
    phrases = tmp_path / "phrases.jsonl"
    common = {"sample_id": "image-a", "study_id": study_id, "image_path": "/x/image-a.jpg"}
    _write(report, {**common, "correct_rank": 12, "correct_text": "reference report", "top_k": [
        {"rank": 1, "text": "retrieved report", "similarity": .8, "study_ids": [8]}]})
    _write(combined, {**common, "correct_rank": 20, "correct_text": "reference phrases", "top_k": [
        {"rank": 1, "text": "edema located_at pulmonary, tip located_at esophagus",
         "similarity": .7, "study_ids": [9]}]})
    _write(phrases, {**common, "first_correct_rank": 3, "correct_phrases": [], "top_k": [
        {"rank": 1, "phrase": "tip located_at esophagus", "score": .9, "study_id": 8},
        {"rank": 2, "phrase": "tip located_at esophagus", "score": .8, "study_id": 9}]})
    return report, combined, phrases


def test_compare_returns_same_image_results_and_semantic_phrases(tmp_path):
    retriever = SavedCaseRetriever(*_paths(tmp_path))

    result = retriever.compare(sample_id="image-a", top_k=10)

    assert result["sample_id"] == "image-a"
    assert result["study_id"] == 7
    assert result["report_retrieval"]["top_k"][0]["text"] == "retrieved report"
    assert result["combined_phrase_retrieval"]["top_k"][0]["phrases"] == [
        "edema located_at pulmonary", "tip located_at esophagus"]
    assert result["independent_phrase_retrieval"]["semantic_top_k"][0]["duplicate_count"] == 2
    assert "reference_text" not in result["report_retrieval"]
    assert "reference_phrases" not in result["independent_phrase_retrieval"]


def test_compare_includes_reference_only_when_requested(tmp_path):
    retriever = SavedCaseRetriever(*_paths(tmp_path))
    result = retriever.compare(sample_id="image-a", include_reference=True)
    assert result["report_retrieval"]["reference_text"] == "reference report"
    assert result["independent_phrase_retrieval"]["reference_phrases"] == []


def test_compare_accepts_image_path_basename(tmp_path):
    retriever = SavedCaseRetriever(*_paths(tmp_path))
    assert retriever.compare(image_path="/uploads/image-a.jpg")["sample_id"] == "image-a"


def test_compare_rejects_cross_experiment_study_mismatch(tmp_path):
    report, combined, phrases = _paths(tmp_path)
    phrase_record = json.loads(phrases.read_text())
    phrase_record["study_id"] = 99
    _write(phrases, phrase_record)

    with pytest.raises(ValueError, match="study mismatch"):
        SavedCaseRetriever(report, combined, phrases).compare(sample_id="image-a")


def test_compare_reports_missing_sample(tmp_path):
    retriever = SavedCaseRetriever(*_paths(tmp_path))
    with pytest.raises(KeyError, match="not found"):
        retriever.compare(sample_id="missing")
