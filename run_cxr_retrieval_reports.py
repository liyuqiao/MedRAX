"""Generate experiment 1/2/4 MedRAX reports and a final comparison on AutoDL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from medrax.retrieval.saved_cases import SavedCaseRetriever


DEFAULT_CASE_IDS = (
    "9bff71ab-e64fa16c-fb22884b-26584905-9f6b4cb7",
    "070b58a0-da9b8080-6eeeaf5a-46226e7b-2f9453fa",
    "60781ae0-7016f7ed-54a825ab-7509c1b0-9b9b2725",
    "f05b9731-d6bf3b29-6197f242-4cc974a3-fe0f5b56",
)

EXPERIMENT_MODES = {
    "experiment_1": "report",
    "experiment_2": "combined_phrases",
    "experiment_4": "independent_phrases",
}


def build_report_prompt(sample_id: str, mode: str, top_k: int = 10) -> str:
    return f"""
分析MIMIC-CXR测试病例 {sample_id}。

必须调用 cxr_clinical_retrieval，且严格使用以下参数：
- sample_id: {sample_id}
- mode: {mode}
- top_k: {top_k}
- include_reference: false

只使用本次工具返回的检索证据生成报告。相似病例中的内容不是当前图像的既定事实。
汇总多条结果共同支持的内容，检查病变类型、解剖位置、左右侧、严重程度和设备位置。
若结果互相冲突或证据不足，必须明确说明，不要补充工具结果未支持的内容。
不要讨论其他实验模式，也不要猜测参考报告。

严格按以下结构输出：

## Evidence Summary
列出主要检索证据及一致性。

## Findings
根据证据生成结构化胸片所见。

## Impression
给出最重要的1至3项结论。

## Uncertainty
列出证据冲突、缺失和不能确定的内容。
""".strip()


def build_comparison_prompt(sample_id: str, reference_report: str,
                            reports: dict[str, str]) -> str:
    return f"""
对同一胸片病例的三份MedRAX报告进行回顾性比较。此阶段不要调用任何工具。

病例：{sample_id}

## Reference Report
{reference_report}

## Experiment 1 Final Report
{reports['experiment_1']}

## Experiment 2 Final Report
{reports['experiment_2']}

## Experiment 4 Final Report
{reports['experiment_4']}

逐项评价：
1. 关键异常是否出现；
2. 病变类型是否正确；
3. 解剖位置和左右侧是否正确；
4. 严重程度是否正确；
5. 医疗设备位置是否正确；
6. 是否引入参考报告中不存在的内容；
7. 哪些结论得到检索证据直接支持。

严格按以下结构输出：

# Case Comparison

## Experiment 1
说明正确内容、遗漏内容和错误内容，并引用具体文本。

## Experiment 2
说明相对实验1的改善与退化，并引用具体文本。

## Experiment 4
说明相对实验2的改善与退化，并引用具体文本。

## Structured Evaluation
使用Markdown表格比较关键征象、位置、严重程度、设备位置和错误引入。

## Conclusion
判断从实验1到实验2再到实验4是否改善了MedRAX的证据获取能力。不要因报告更长而判定更好。
""".strip()


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def invoke_agent(agent: Any, prompt: str, thread_id: str) -> str:
    from langchain_core.messages import HumanMessage

    state = agent.workflow.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        {"configurable": {"thread_id": thread_id}},
    )
    return _message_text(state["messages"][-1].content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MedRAX reports for CXR retrieval experiments 1, 2, and 4")
    parser.add_argument("--report-results", required=True)
    parser.add_argument("--combined-results", required=True)
    parser.add_argument("--independent-results", required=True)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--output-dir", default="outputs/agent-retrieval-reports")
    parser.add_argument("--prompt-file", default="medrax/docs/system_prompts.txt")
    parser.add_argument("--model", default="qwen3-vl-flash")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from main import initialize_agent

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    openai_kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    agent, _ = initialize_agent(
        args.prompt_file,
        tools_to_use=["CXRClinicalRetrievalTool"],
        retrieval_report_path=args.report_results,
        retrieval_combined_path=args.combined_results,
        retrieval_independent_path=args.independent_results,
        model=args.model,
        temperature=0.0,
        top_p=1.0,
        openai_kwargs=openai_kwargs,
    )
    evaluator = SavedCaseRetriever(
        args.report_results, args.combined_results, args.independent_results)
    sample_ids = args.sample_ids or list(DEFAULT_CASE_IDS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        reports: dict[str, str] = {}
        for experiment, mode in EXPERIMENT_MODES.items():
            print(f"Generating {sample_id} / {experiment}", flush=True)
            report = invoke_agent(
                agent,
                build_report_prompt(sample_id, mode, args.top_k),
                thread_id=f"{sample_id}-{experiment}",
            )
            reports[experiment] = report
            (output_dir / f"{sample_id}-{experiment}.md").write_text(
                report, encoding="utf-8")

        saved = evaluator.compare(
            sample_id=sample_id, top_k=args.top_k, include_reference=True)
        reference_report = saved["report_retrieval"].get("reference_text") or "Unavailable"
        comparison = invoke_agent(
            agent,
            build_comparison_prompt(sample_id, reference_report, reports),
            thread_id=f"{sample_id}-comparison",
        )
        (output_dir / f"{sample_id}-comparison.md").write_text(
            comparison, encoding="utf-8")
        all_results.append({
            "sample_id": sample_id,
            "study_id": saved["study_id"],
            "reference_report": reference_report,
            "reports": reports,
            "comparison": comparison,
        })

    (output_dir / "all-results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(sample_ids)} case analyses to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
