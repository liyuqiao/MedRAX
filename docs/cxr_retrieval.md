# CXR-CLIP retrieval in MedRAX

`CXRClinicalRetrievalTool` lets the MedRAX agent compare three saved CXR-CLIP retrieval
experiments for exactly the same evaluation image:

- experiment 1: original CXR-CLIP with complete reports;
- experiment 2: original CXR-CLIP with concatenated relation phrases;
- experiment 4: original CXR-CLIP with independently ranked phrases.

The current integration deliberately replays the per-image JSONL artifacts. It is intended for
controlled agent case analysis on AutoDL and does not claim to retrieve arbitrary images that are
absent from those files. An evaluation image path works when its filename stem is the MIMIC-CXR
`sample_id`; callers may also pass `sample_id` directly.

## Agent initialization

```python
agent, tools = initialize_agent(
    "medrax/docs/system_prompts.txt",
    tools_to_use=["CXRClinicalRetrievalTool"],
    retrieval_report_path="/root/autodl-tmp/cxr-clip/outputs/badcase/base-report.jsonl",
    retrieval_combined_path="/root/autodl-tmp/cxr-clip/outputs/badcase/base-triple.jsonl",
    retrieval_independent_path=(
        "/root/autodl-tmp/cxr-clip/outputs/phrase-retrieval-badcase/"
        "baseline/retrieval-details.jsonl"
    ),
)
```

Ask the agent to analyze an uploaded known test image, or invoke the tool with:

```json
{"sample_id": "9bff71ab-e64fa16c-fb22884b-26584905-9f6b4cb7", "top_k": 10}
```

The response includes the raw study-aware phrase ranking and `semantic_top_k`, which collapses
identical phrase text from different studies while retaining source study IDs.

Ground-truth text is hidden by default. Keep `include_reference=false` while the agent generates
reports. It may be enabled only during retrospective evaluation after all experiment-specific
reports have been generated.

## Render existing cases

```bash
python -m medrax.retrieval.case_analysis \
  --report-results /root/autodl-tmp/cxr-clip/outputs/badcase/base-report.jsonl \
  --combined-results /root/autodl-tmp/cxr-clip/outputs/badcase/base-triple.jsonl \
  --independent-results /root/autodl-tmp/cxr-clip/outputs/phrase-retrieval-badcase/baseline/retrieval-details.jsonl \
  --sample-id 9bff71ab-e64fa16c-fb22884b-26584905-9f6b4cb7 \
  --sample-id 070b58a0-da9b8080-6eeeaf5a-46226e7b-2f9453fa \
  --sample-id 60781ae0-7016f7ed-54a825ab-7509c1b0-9b9b2725 \
  --sample-id f05b9731-d6bf3b29-6197f242-4cc974a3-fe0f5b56 \
  --output outputs/cxr-retrieval-cases.md
```

## AutoDL verification

Run from the MedRAX repository root after installing its environment:

```bash
pytest tests/retrieval tests/tools/test_cxr_retrieval.py -q
python -m medrax.retrieval.case_analysis --help
```

## Generate final MedRAX reports

The batch runner invokes the agent independently for experiments 1, 2, and 4, then supplies the
three completed reports and the reference report to a separate retrospective comparison turn:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="..."  # omit for the default OpenAI endpoint

python run_cxr_retrieval_reports.py \
  --report-results /root/autodl-tmp/cxr-clip/outputs/badcase/base-report.jsonl \
  --combined-results /root/autodl-tmp/cxr-clip/outputs/badcase/base-triple.jsonl \
  --independent-results /root/autodl-tmp/cxr-clip/outputs/phrase-retrieval-badcase/baseline/retrieval-details.jsonl \
  --model qwen3-vl-flash \
  --output-dir outputs/agent-retrieval-reports
```

With the four default case IDs this creates 12 independent experiment reports, four retrospective
comparison reports, and `all-results.json`. Pass one or more `--sample-id` arguments to select a
different subset.
