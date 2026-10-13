from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .saved_cases import SavedCaseRetriever


def render_case_markdown(comparison: dict[str, Any]) -> str:
    def text_rows(rows):
        return "\n".join(f"{row['rank']}. {row['text']}" for row in rows) or "No results"

    def phrase_rows(rows):
        return "\n".join(f"{row['rank']}. {row['phrase']} (score={row.get('score')})"
                         for row in rows) or "No results"

    return (
        f"# CXR Retrieval Case: {comparison['sample_id']}\n\n"
        f"Study: {comparison['study_id']}\n\n"
        "## Reference report\n\n"
        f"{comparison['report_retrieval'].get('reference_text') or 'Unavailable'}\n\n"
        "## Experiment 1 — Complete reports\n\n"
        f"{text_rows(comparison['report_retrieval']['top_k'])}\n\n"
        "## Experiment 2 — Combined phrases\n\n"
        f"{text_rows(comparison['combined_phrase_retrieval']['top_k'])}\n\n"
        "## Experiment 4 — Independent phrases (semantic deduplication)\n\n"
        f"{phrase_rows(comparison['independent_phrase_retrieval']['semantic_top_k'])}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render saved CXR retrieval comparisons")
    parser.add_argument("--report-results", required=True)
    parser.add_argument("--combined-results", required=True)
    parser.add_argument("--independent-results", required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    retriever = SavedCaseRetriever(
        args.report_results, args.combined_results, args.independent_results)
    sections = [render_case_markdown(retriever.compare(sample_id=sample_id, top_k=args.top_k))
                for sample_id in args.sample_id]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n---\n\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
