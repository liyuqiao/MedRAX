from __future__ import annotations

from typing import Any, Optional, Type

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from medrax.retrieval.saved_cases import SavedCaseRetriever
from medrax.retrieval.schemas import RetrievalMode
from medrax.tracing import trace_event


class CXRClinicalRetrievalInput(BaseModel):
    image_path: Optional[str] = Field(
        None, description="Path to a known evaluation chest X-ray; its filename is used as sample_id."
    )
    sample_id: Optional[str] = Field(
        None, description="MIMIC-CXR sample_id for replaying the same saved case across experiments."
    )
    top_k: int = Field(10, ge=1, le=100, description="Number of raw results per experiment.")
    mode: RetrievalMode = Field(
        RetrievalMode.COMPARE,
        description="One retrieval representation or compare all experiments 1, 2, and 4.",
    )


class CXRClinicalRetrievalTool(BaseTool):
    """Compare full-report, combined-phrase, and independent-phrase CXR retrieval."""

    name: str = "cxr_clinical_retrieval"
    description: str = (
        "Compare CXR-CLIP experiment 1 (complete reports), experiment 2 (combined relation "
        "phrases), and experiment 4 (independent phrases) for the same chest X-ray. Use this "
        "to obtain retrieval evidence and analyze whether global report matching omits a focal "
        "finding. Provide either sample_id or an evaluation image path whose filename is the "
        "sample_id. Independent phrase output includes raw and semantic-deduplicated rankings."
    )
    args_schema: Type[BaseModel] = CXRClinicalRetrievalInput
    retriever: Any = None

    def __init__(self, retriever: Any = None, *, report_path: str | None = None,
                 combined_path: str | None = None, independent_path: str | None = None):
        if retriever is None:
            missing = [name for name, value in (("report_path", report_path),
                       ("combined_path", combined_path), ("independent_path", independent_path))
                       if not value]
            if missing:
                raise ValueError("missing saved retrieval paths: " + ", ".join(missing))
            retriever = SavedCaseRetriever(report_path, combined_path, independent_path)
        super().__init__(retriever=retriever)

    def _run(self, image_path: str | None = None, sample_id: str | None = None, top_k: int = 10,
             mode: RetrievalMode | str = RetrievalMode.COMPARE,
             run_manager: Optional[CallbackManagerForToolRun] = None):
        try:
            mode = RetrievalMode(mode)
            if mode is RetrievalMode.COMPARE:
                result = self.retriever.compare(
                    image_path=image_path, sample_id=sample_id, top_k=top_k)
                experiments = [1, 2, 4]
            else:
                result = self.retriever.retrieve(
                    mode, image_path=image_path, sample_id=sample_id, top_k=top_k)
                experiments = [{
                    RetrievalMode.REPORT: 1,
                    RetrievalMode.COMBINED_PHRASES: 2,
                    RetrievalMode.INDEPENDENT_PHRASES: 4,
                }[mode]]
            trace_event("cxr_retrieval_result", "Compared CXR retrieval experiments 1, 2, and 4",
                        {"sample_id": sample_id, "mode": mode.value})
            return result, {
                "analysis_status": "completed", "experiments": experiments,
                "sample_id": result.get("sample_id", sample_id),
                "study_id": result.get("study_id"), "mode": mode.value,
            }
        except Exception as exc:
            trace_event("cxr_retrieval_error", "CXR clinical retrieval failed",
                        {"error": str(exc)}, level="ERROR")
            return {"error": str(exc)}, {
                "analysis_status": "failed", "experiments": [1, 2, 4],
                "sample_id": sample_id, "image_path": image_path,
            }

    async def _arun(self, image_path: str | None = None, sample_id: str | None = None,
                    top_k: int = 10, mode: RetrievalMode | str = RetrievalMode.COMPARE,
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None):
        return self._run(image_path=image_path, sample_id=sample_id, top_k=top_k, mode=mode)
