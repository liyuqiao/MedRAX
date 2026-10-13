# CXR Clinical Retrieval Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a MedRAX agent tool that compares CXR-CLIP experiments 1, 2, and 4 and replays existing same-image cases.

**Architecture:** A small retrieval package normalizes mode-specific results and isolates saved JSONL access from the LangChain tool. The tool accepts an injected backend, while `main.py` constructs a saved-case backend from configured result paths so tests never load GPU models.

**Tech Stack:** Python, Pydantic, LangChain `BaseTool`, pytest, JSONL.

---

### Task 1: Retrieval schemas and semantic phrase deduplication

**Files:**
- Create: `medrax/retrieval/__init__.py`
- Create: `medrax/retrieval/schemas.py`
- Test: `tests/retrieval/test_schemas.py`

- [ ] Write tests showing valid mode parsing and stable phrase-text deduplication that preserves first rank and records duplicate study IDs.
- [ ] Run `pytest tests/retrieval/test_schemas.py -q` and confirm failure because the package is missing.
- [ ] Implement `RetrievalMode` and `deduplicate_phrase_results` with no model dependencies.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Saved experiment replay

**Files:**
- Create: `medrax/retrieval/saved_cases.py`
- Test: `tests/retrieval/test_saved_cases.py`

- [ ] Write temporary-JSONL tests for a successful same-sample comparison, missing samples, and mismatched study IDs.
- [ ] Run the focused tests and confirm the expected import/behavior failures.
- [ ] Implement lazy JSONL indexing and normalized experiment 1/2/4 output with raw and semantic phrase rankings.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Agent-facing tool

**Files:**
- Create: `medrax/tools/cxr_retrieval.py`
- Modify: `medrax/tools/__init__.py`
- Test: `tests/tools/test_cxr_retrieval.py`

- [ ] Write tests for `saved_case` and live mode dispatch, successful metadata, and contextual failure output.
- [ ] Run the focused tests and confirm failure because the tool is absent.
- [ ] Implement the Pydantic input schema and `CXRClinicalRetrievalTool` using dependency injection.
- [ ] Export the tool and run the focused tests.

### Task 4: Agent registration and prompting

**Files:**
- Modify: `main.py`
- Modify: `medrax/docs/system_prompts.txt`
- Test: `tests/test_retrieval_registration.py`

- [ ] Write a registration test that replaces the tool constructor and language model, then verifies selective initialization.
- [ ] Run it and confirm failure because the tool is not registered.
- [ ] Add optional result-path arguments, register `CXRClinicalRetrievalTool`, and add concise evidence-handling instructions to the medical prompt.
- [ ] Run the registration and focused retrieval tests.

### Task 5: Case-analysis command and documentation

**Files:**
- Create: `medrax/retrieval/case_analysis.py`
- Create: `docs/cxr_retrieval.md`
- Test: `tests/retrieval/test_case_analysis.py`

- [ ] Write a test that renders a deterministic Markdown comparison for a replayed case.
- [ ] Run it and confirm failure because the renderer is missing.
- [ ] Implement the renderer and a CLI accepting three JSONL paths, sample IDs, Top-K, and output path.
- [ ] Document agent initialization and the four established case IDs.
- [ ] Run all retrieval tests and the CLI help command.

### Task 6: Verification

**Files:**
- Verify all files above.

- [ ] Run `pytest tests/retrieval tests/tools/test_cxr_retrieval.py tests/test_retrieval_registration.py -q`.
- [ ] Run the existing test suite with `pytest -q`.
- [ ] Run `git diff --check` and review `git status --short`.
