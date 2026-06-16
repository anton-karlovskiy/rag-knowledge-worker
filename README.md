# RAG Knowledge Worker

A RAG-based AI assistant that answers questions about a company using its internal documents. Built on top of ChromaDB, OpenAI embeddings, and a Gradio chat interface.

The pipeline uses LLM-generated chunking (with headline, summary, and original text per chunk), dual-query retrieval, and LLM reranking to get higher-quality answers than basic similarity search alone.

The included knowledge base covers a fictional insurance company called Insurellm, with documents across four categories: company info, products, employees, and contracts. You can swap in your own markdown documents to adapt this to any company.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/your-username/rag-knowledge-worker
cd rag-knowledge-worker
uv sync
cp .env.example .env
# add your OPENAI_API_KEY to .env
```

## Usage

**Step 1: Ingest documents**

This reads all markdown files from `knowledge-base/`, chunks them with an LLM, embeds the chunks, and stores everything in a local ChromaDB database.

```bash
uv run ingest
```

Ingestion uses 3 parallel workers by default. If you hit rate limits, set `WORKERS = 1` in `ingest.py`.

**Step 2: Run the app**

```bash
uv run app
```

Opens a Gradio chat interface in your browser. Ask anything about the company. The right panel shows the retrieved context chunks that informed each answer.

## Evaluation

The `evaluation/` folder contains a set of test questions with reference answers and expected keywords.

To evaluate a single test case by index:

```bash
uv run eval 0
```

This runs both retrieval evaluation (MRR, nDCG, keyword coverage) and answer quality evaluation (accuracy, completeness, relevance scored by an LLM judge).

## Project structure

```
├── app.py              # Gradio chat UI
├── answer.py           # RAG pipeline: query rewriting, retrieval, reranking, generation
├── ingest.py           # Document loading, LLM chunking, embedding, ChromaDB storage
├── models.py           # Shared data models
├── knowledge-base/
│   ├── company/        # General company documents
│   ├── contracts/      # Customer contracts
│   ├── employees/      # Employee profiles
│   └── products/       # Product descriptions
└── evaluation/
    ├── eval.py         # Retrieval and answer quality evaluation
    ├── test.py         # Test data loader
    └── tests.jsonl     # Test questions with reference answers
```

## How the RAG pipeline works

1. The user's question is rewritten into a focused knowledge base query.
2. Both the original and rewritten questions are used to retrieve chunks independently (dual-query retrieval).
3. The two result sets are merged and deduplicated.
4. An LLM reranks the merged chunks by relevance to the original question.
5. The top 10 chunks are injected into the system prompt, and the model generates an answer.

## Configuration

Key settings are at the top of each file:

| Setting | File | Default |
|---|---|---|
| `MODEL` | `answer.py`, `ingest.py` | `openai/gpt-4.1-nano` |
| `EMBEDDING_MODEL` | `answer.py`, `ingest.py` | `text-embedding-3-large` |
| `RETRIEVAL_K` | `answer.py` | 20 (per query) |
| `CONTEXT_K` | `answer.py` | 10 (after reranking) |
| `WORKERS` | `ingest.py` | 3 |

The `MODEL` value is passed to [LiteLLM](https://docs.litellm.ai/), so you can point it at any supported provider by changing the prefix (e.g. `groq/...`, `anthropic/...`).
