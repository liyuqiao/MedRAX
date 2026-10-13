from medrax.tools.cxr_retrieval import CXRClinicalRetrievalTool


class FakeRetriever:
    def compare(self, **kwargs):
        return {
            "sample_id": kwargs.get("sample_id") or "from-image",
            "study_id": 7,
            "include_reference": kwargs["include_reference"],
            "modes": [1, 2, 4],
        }


def test_tool_replays_saved_case():
    tool = CXRClinicalRetrievalTool(retriever=FakeRetriever())
    result, metadata = tool._run(sample_id="case-1", top_k=5)
    assert result["sample_id"] == "case-1"
    assert metadata["analysis_status"] == "completed"
    assert metadata["experiments"] == [1, 2, 4]
    assert result["include_reference"] is False


def test_tool_returns_contextual_failure():
    class BrokenRetriever:
        def compare(self, **kwargs):
            raise KeyError("missing")

    result, metadata = CXRClinicalRetrievalTool(retriever=BrokenRetriever())._run(sample_id="x")
    assert "error" in result
    assert metadata["analysis_status"] == "failed"
