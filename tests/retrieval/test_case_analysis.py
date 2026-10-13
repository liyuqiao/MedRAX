from medrax.retrieval.case_analysis import render_case_markdown


def test_render_case_markdown_contains_three_experiment_sections():
    comparison = {
        "sample_id": "case-a", "study_id": 7,
        "report_retrieval": {"top_k": [{"rank": 1, "text": "pulmonary edema"}]},
        "combined_phrase_retrieval": {"top_k": [{"rank": 1, "text": "edema located_at pulmonary"}]},
        "independent_phrase_retrieval": {"semantic_top_k": [
            {"rank": 1, "phrase": "tip located_at esophagus", "score": .8}]},
    }
    rendered = render_case_markdown(comparison)
    assert "Experiment 1" in rendered
    assert "Experiment 2" in rendered
    assert "Experiment 4" in rendered
    assert "tip located_at esophagus" in rendered
