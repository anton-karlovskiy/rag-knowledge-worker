# CLAUDE.md

Keep your replies extremely concise and focus on conveying the key information. No unnecessary fluff, no long code snippets.

Whenever working with any third-party library or something similar, you MUST look up the official documentation to ensure that you're working with up-to-date information.
Use the DocsExplorer subagent for efficient documentation lookup.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Uses [uv](https://docs.astral.sh/uv/), Python 3.11+. Needs `OPENAI_API_KEY` in `.env` (copy `.env.example`). You also need another provider's key (e.g. `GROQ_API_KEY`) if a `MODEL` is switched to that provider.

```bash
uv sync               # install dependencies
uv run ingest         # build preprocessed_db/ from knowledge-base/ (one LLM call per document, takes minutes)
uv run app            # Gradio chat UI
uv run eval <index>   # evaluate one test case (retrieval + LLM-judged answer), e.g. `uv run eval 0`
uv run evaluator      # Gradio dashboard that evaluates all 150 test cases
```

The entry points are defined in `[project.scripts]` in `pyproject.toml`. There is no unit test suite, linter, or formatter configured. "Tests" here means the RAG evaluation in `evaluation/`, and every run makes several LLM calls per question.

## Architecture

Top-level modules are imported flat (`from config import ...`, `from answer import ...`). Hatch packages the repo root as `.`, so run everything through `uv run` from the repo root. There is no LangChain: storage uses `chromadb.PersistentClient` directly, embeddings use the `openai` SDK, and all chat/structured-output calls go through **litellm** `completion()` (`MODEL` values are LiteLLM names such as `openai/gpt-4.1-nano`).

- **`config.py`**: settings that ingestion and retrieval must agree on: `DB_NAME` (`preprocessed_db/`), `COLLECTION_NAME`, `EMBEDDING_MODEL`, `RETRIEVAL_K` (candidates per query, 20) and `FINAL_K` (chunks kept after rerank, 10). If you change `EMBEDDING_MODEL` or the ingest `MODEL`, re-ingest.
- **`ingest.py`**: LLM chunking. For each `knowledge-base/<folder>/**/*.md` file, the LLM returns `Chunks` (structured output). Each chunk has a `headline`, `summary` and `original_text`, and all three are concatenated into the `page_content` that gets embedded. It asks for roughly `len(text) // AVERAGE_CHUNK_SIZE + 1` chunks. Documents are processed in parallel (`WORKERS`, set to 1 on rate limits), with `tenacity` retries. Metadata is `{source, type}`, where `type` is the folder name. Each run deletes and rebuilds the collection.
- **`answer.py`**: the RAG pipeline. It opens Chroma and raises `RuntimeError` if the collection is empty, **at import time**. So importing `answer` (and therefore `app`, `evaluator`, `evaluation.eval`) requires a populated `preprocessed_db/`. `fetch_context(question, history)` does the following:
  1. `rewrite_query` uses the LLM and the history to produce a short query.
  2. It retrieves `RETRIEVAL_K` chunks for both the original question and the rewritten query.
  3. `merge_chunks` merges them and removes duplicates by `page_content`.
  4. `rerank` has the LLM return a `RankOrder` of 1-based chunk ids.
  5. It keeps the top `FINAL_K`.

  `answer_question()` returns `(answer, chunks)`, where the chunks are pydantic `Result` objects with `page_content` and `metadata`. The UI and the evaluation depend on that shape. `llm_retry` retries only transient API errors (rate limit, connection, 5xx, max 5 attempts), so other exceptions surface immediately.
- **`evaluation/eval.py`**: retrieval metrics are keyword-based. MRR, nDCG@`FINAL_K` and coverage check whether each `TestCase.keywords` entry appears as a case-insensitive substring of the retrieved chunks. Retrieval eval calls `fetch_context(question)` with no history, but that still runs the rewrite and rerank LLM calls. Answer eval runs `answer_question()`, then an LLM judge via litellm (structured output into `AnswerEval`). `evaluate_*_all()` are generators yielding `(test_case, result, progress_fraction)`, which the dashboard consumes.
- **`evaluator.py`**: Gradio dashboard. It averages metrics, breaks them down by `test_case.category`, and color-codes them using `THRESHOLDS` keyed by the `MetricType` enum.
- **`evaluation/test_cases.jsonl`**: one JSON object per line with `question`, `keywords`, `reference_answer`, and `category` (e.g. `direct_fact`, `temporal`, `relationship`, `comparative`, `spanning`, `holistic`).

## Notes

- The knowledge base is a fictional insurance company, Insurellm. The folder names under `knowledge-base/` become the chunk `type` metadata.
- When pipeline changes affect quality, re-run the evaluation and update the Results table in `README.md`. That table compares the current pipeline with the basic one. The current scores are MRR 0.90, nDCG 0.87, keyword coverage 95.8%, and accuracy 4.67/5. `holistic` is the weakest category (~3.4/5 accuracy).
- `.claude/skills/` mirrors `.agents/skills/` (tracked in `skills-lock.json`). Use the `caveman-commit` skill style for commit messages (Conventional Commits).
