from run_cxr_retrieval_reports import build_comparison_prompt, build_report_prompt


def test_report_prompt_forces_one_mode_and_no_reference_use():
    prompt = build_report_prompt("sample-a", "independent_phrases")
    assert "sample-a" in prompt
    assert "independent_phrases" in prompt
    assert "include_reference: false" in prompt
    assert "不要讨论其他实验模式" in prompt


def test_comparison_prompt_contains_all_reports_and_reference():
    prompt = build_comparison_prompt(
        "sample-a", "reference", {
            "experiment_1": "report one",
            "experiment_2": "report two",
            "experiment_4": "report four",
        })
    assert "reference" in prompt
    assert "report one" in prompt
    assert "report two" in prompt
    assert "report four" in prompt
