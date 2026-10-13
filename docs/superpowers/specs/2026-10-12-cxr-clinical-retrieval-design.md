# CXR Clinical Retrieval Tool Design

## Goal

Add a self-contained MedRAX tool that exposes the three established CXR-CLIP retrieval variants to the existing agent and can replay saved, same-image results for case analysis.

## Scope

The tool supports `report`, `combined_phrases`, `independent_phrases`, and `compare` modes. Live retrieval is delegated to an injected backend so model and index loading remain independent from LangChain. Saved-result replay reads the existing JSONL artifacts for experiments 1, 2, and 4 by `sample_id` and requires all returned records to describe the same image.

Fine-tuned experiments 3 and 5 are outside this first integration. The design leaves the backend interface open to adding them later without changing the agent-facing schema.

## Components

- `medrax/retrieval/schemas.py`: mode enum and normalized result shapes.
- `medrax/retrieval/backend.py`: retrieval backend protocol plus an in-memory backend used in tests and lightweight deployments.
- `medrax/retrieval/saved_cases.py`: indexed JSONL replay, consistency validation, and semantic phrase deduplication.
- `medrax/tools/cxr_retrieval.py`: MedRAX/LangChain tool with one stable input schema.
- `medrax/tools/__init__.py` and `main.py`: export and optional agent registration.
- `medrax/docs/system_prompts.txt`: instruct the agent to treat retrieved text as evidence, compare modes on the same image, and never copy unsupported details from a similar report.

## Data flow

For live use, the tool sends the image path, requested mode, and Top-K to a backend. `compare` runs all three modes and returns a single normalized dictionary. Independent-phrase output contains both raw study-aware ranking and a phrase-text-deduplicated `semantic_top_k` intended for the agent.

For saved cases, the tool locates one record in each configured JSONL file. It validates matching `sample_id` and `study_id`, then returns normalized experiment 1/2/4 results. This makes existing evidence reproducible without loading a GPU model.

## Errors

Invalid modes and non-positive Top-K values are rejected by Pydantic. Missing files, missing sample IDs, duplicate sample IDs, malformed JSON, or cross-experiment study mismatches produce contextual errors. The tool catches operational errors and returns `analysis_status: failed` metadata, matching existing MedRAX tool behavior.

## Testing

Unit tests use temporary JSONL files and an in-memory backend. They cover mode dispatch, compare output, phrase deduplication, same-image validation, missing samples, tool output, and registration in `initialize_agent` without loading large models.

## Success criteria

- The existing agent can initialize `CXRClinicalRetrievalTool` selectively.
- One call can compare experiments 1, 2, and 4 for the same image.
- Existing saved cases can be replayed by `sample_id`.
- Raw rankings remain available while semantically duplicate phrases are collapsed for agent consumption.
- Tests run without a GPU, checkpoints, MIMIC-CXR images, or network access.
