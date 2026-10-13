from medrax.retrieval.schemas import deduplicate_phrase_results


def test_deduplicate_phrase_results_preserves_first_rank_and_sources():
    rows = [
        {"rank": 1, "phrase": "tip located_at esophagus", "study_id": 10, "score": 0.9},
        {"rank": 2, "phrase": "tip located_at esophagus", "study_id": 20, "score": 0.8},
        {"rank": 3, "phrase": "tube located_at esophagus", "study_id": 30, "score": 0.7},
    ]

    assert deduplicate_phrase_results(rows) == [
        {
            "rank": 1,
            "phrase": "tip located_at esophagus",
            "score": 0.9,
            "study_id": 10,
            "source_study_ids": [10, 20],
            "duplicate_count": 2,
        },
        {
            "rank": 2,
            "phrase": "tube located_at esophagus",
            "score": 0.7,
            "study_id": 30,
            "source_study_ids": [30],
            "duplicate_count": 1,
        },
    ]
