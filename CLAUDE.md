# CLAUDE.md

Keep your replies extremely concise and focus on conveying the key information. No unnecessary fluff, no long code snippets.

Whenever working with any third-party library or something similar, you MUST look up the official documentation to ensure that you're working with up-to-date information.
Use the DocsExplorer subagent for efficient documentation lookup.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Uses [uv](https://docs.astral.sh/uv/), Python 3.11+. Needs `OPENAI_API_KEY` in `.env` (copy `.env.example`).

```bash
uv sync               # install dependencies
uv run ingest         # build vector_db/ from knowledge-base/ (run first, and after any chunking/embedding change)
uv run app            # Gradio chat UI
uv run eval <index>   # evaluate one test case (retrieval + LLM-judged answer), e.g. `uv run eval 0`
uv run evaluator      # Gradio dashboard that evaluates all 150 test cases
```

The entry points are defined in `[project.scripts]` in `pyproject.toml`. There is no unit test suite, linter, or formatter configured. "Tests" here means the RAG evaluation in `evaluation/`, and every run calls the OpenAI API.

## Architecture

Top-level modules are imported flat (`from config import ...`, `from answer import ...`). Hatch packages the repo root as `.`, so run everything through `uv run` from the repo root.

- **`config.py`**: settings that ingestion and retrieval must agree on: `DB_NAME` (`vector_db/`), `EMBEDDING_MODEL`, and `RETRIEVAL_K`. If you change `EMBEDDING_MODEL`, re-ingest.
- **`ingest.py`**: loads `knowledge-base/<folder>/**/*.md` and tags each document with `metadata["doc_type"] = <folder name>`. It splits documents with `RecursiveCharacterTextSplitter` (`CHUNK_SIZE`/`CHUNK_OVERLAP` are defined here) and deletes and rebuilds the whole Chroma collection on every run.
- **`answer.py`**: the RAG pipeline. It opens Chroma, builds the retriever and LLM, and raises `RuntimeError` if the store is empty, all **at import time**. So importing `answer` (and therefore `app`, `evaluator`, `evaluation.eval`) requires a populated `vector_db/`. Retrieval runs on `combined_question()`, which joins all prior *user* messages with the current question. The retrieved chunks are formatted into `SYSTEM_PROMPT`, and the full history goes to the chat model. `answer_question()` returns `(answer, docs)`, and both the UI and the evaluation depend on that shape.
- **`evaluation/eval.py`**: retrieval metrics are keyword-based. MRR, nDCG, and coverage check whether each `TestCase.keywords` entry appears as a case-insensitive substring of the retrieved chunks. Retrieval eval calls `fetch_context(question)` directly, with no history. Answer eval runs `answer_question()`, then an LLM judge via **litellm** (`openai/gpt-4.1-nano`, structured output into the `AnswerEval` pydantic model). The app itself uses LangChain `ChatOpenAI`. `evaluate_*_all()` are generators yielding `(test_case, result, progress_fraction)`, which the dashboard consumes.
- **`evaluator.py`**: Gradio dashboard. It averages metrics, breaks them down by `test_case.category`, and color-codes them using `THRESHOLDS` keyed by the `MetricType` enum.
- **`evaluation/test_cases.jsonl`**: one JSON object per line with `question`, `keywords`, `reference_answer`, and `category` (e.g. `direct_fact`, `temporal`, `relationship`, `spanning`, `holistic`).

## Notes

- The knowledge base is a fictional insurance company, Insurellm. The folder names under `knowledge-base/` become `doc_type` values.
- When pipeline changes affect quality, re-run the evaluation and update the Results table in `README.md`. The baseline on `main` is MRR 0.79, nDCG 0.79, keyword coverage 93%, and accuracy 4.21/5. `spanning` and `holistic` are the weakest categories.
- `.claude/skills/` mirrors `.agents/skills/` (tracked in `skills-lock.json`). Use the `caveman-commit` skill style for commit messages (Conventional Commits).
