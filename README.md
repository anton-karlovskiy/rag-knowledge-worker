# RAG Knowledge Worker

A RAG-based AI assistant that answers questions about a company using its internal documents. Built with ChromaDB, OpenAI embeddings, LiteLLM, LangChain, and a Gradio chat interface.

It ships two interchangeable RAG pipelines, selectable in the UI:

- **LLM**: an LLM splits each document into overlapping chunks, each with a headline and summary. For each question, the pipeline retrieves candidates for both the original and an LLM-rewritten query, has an LLM rerank them, and passes the best chunks to the model as context.
- **LangChain**: the baseline. Fixed-size 500-character chunks and plain similarity search, no query rewriting or reranking.

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

This reads all markdown files from `knowledge-base/`, has an LLM split each one into chunks, embeds the chunks, and stores everything in a local ChromaDB database. It makes one LLM call per document, so it takes a few minutes. If you hit rate limits, set `WORKERS` in `ingest_llm.py` to 1.

```bash
uv run ingest-llm         # LLM pipeline -> llm_db/
uv run ingest-langchain   # LangChain pipeline -> langchain_db/ (fast, no LLM calls)
```

Ingest the store for each pipeline you want to use. A pipeline is loaded only when first selected, so a missing store only breaks that pipeline.

**Step 2: Run the app**

```bash
uv run app
```

Opens a Gradio chat interface in your browser. Ask anything about the company. The right panel shows the retrieved context chunks that informed each answer. The **RAG Pipeline** switch at the top picks the LangChain or LLM (chunking + rerank) pipeline per question.

**Optional: Visualize the vector store**

```bash
uv run visualize                        # 2D t-SNE of the LLM store
uv run visualize --dimensions 3         # 3D t-SNE
uv run visualize --pipeline langchain   # LangChain store
```

Projects every chunk embedding with t-SNE and opens an interactive Plotly scatter in your browser, colored by document type. Hover a point to see its text. No LLM calls.

## Evaluation

The `evaluation/` folder contains a set of test cases, each with a question, reference answer, and expected keywords.

Evaluation needs the vector store from Step 1 for the selected pipeline (`llm_db/` or `langchain_db/`, not tracked in git). If it is missing or empty, the run fails with an error telling you which ingest command to run.

To evaluate a single test case by index:

```bash
uv run eval 0                         # LLM pipeline
uv run eval 0 --pipeline langchain    # LangChain pipeline
```

This runs both retrieval evaluation (MRR, nDCG, keyword coverage) and answer quality evaluation (accuracy, completeness, relevance scored by an LLM judge).

To evaluate all test cases at once, launch the evaluation dashboard:

```bash
uv run evaluator
```

Opens a Gradio dashboard with a pipeline switch and two sections, Retrieval and Answer, each with its own "Run Evaluation" button. Each section shows the averaged metrics across all test cases, color-coded green/amber/red against the thresholds in `evaluator.py`, plus a bar chart breaking down MRR (retrieval) or accuracy (answer) by question category. Answer evaluation runs the full RAG pipeline plus an LLM judge for every test case, so it takes noticeably longer than retrieval evaluation.

### Results

Full runs over all 150 test cases with the default configuration.

**LLM pipeline:**

![RAG evaluation dashboard, LLM pipeline](docs/evaluator-dashboard-llm.png)

**LangChain pipeline:**

![RAG evaluation dashboard, LangChain pipeline](docs/evaluator-dashboard-langchain.png)

The LLM pipeline compared with the LangChain pipeline (500-character chunks, no query rewriting or reranking):

| Metric | LangChain | LLM |
|---|---|---|
| MRR | 0.7919 | 0.8977 |
| nDCG | 0.7949 | 0.8696 |
| Keyword coverage | 93.0% | 95.8% |
| Accuracy | 4.21 / 5 | 4.67 / 5 |
| Completeness | 3.92 / 5 | 4.25 / 5 |
| Relevance | 4.63 / 5 | 4.89 / 5 |

Every metric improved. The biggest gains are on the categories that were weakest before: `spanning` MRR rose from ~0.47 to ~0.69 and `holistic` from ~0.57 to ~0.68. `temporal` and `comparative` questions now score 5 / 5 accuracy on average.

`holistic` is still the weakest category (~3.4 / 5 accuracy). These questions need information pulled from many documents at once, and ten chunks still cover only part of it.

## Project structure

```
├── app.py              # Gradio chat UI
├── evaluator.py        # Gradio evaluation dashboard over all test cases
├── pipelines.py        # Pipeline registry: lazily loads the module selected in the UI
├── answer_llm.py       # LLM RAG pipeline: query rewriting, retrieval, reranking, generation
├── ingest_llm.py       # LLM ingestion: LLM chunking, embedding, ChromaDB storage
├── answer_langchain.py # LangChain RAG pipeline
├── ingest_langchain.py # LangChain ingestion (500-character chunks)
├── config.py           # Settings shared by ingestion and retrieval
├── docs/               # README images
├── knowledge-base/
│   ├── company/        # General company documents
│   ├── contracts/      # Customer contracts
│   ├── employees/      # Employee profiles
│   └── products/       # Product descriptions
└── evaluation/
    ├── eval.py         # Retrieval and answer quality evaluation
    ├── test.py         # Test case model and loader
    └── test_cases.jsonl # Test cases: questions, reference answers, and keywords
```

## How the RAG pipeline works

Both pipelines expose the same interface, `answer_question(question, history) -> (answer, chunks)` and `fetch_context(question, history) -> chunks`, so the app and evaluation switch between them through `pipelines.get_pipeline(name)`. The LangChain pipeline splits documents into 500-character chunks and retrieves the top 10 for the question combined with prior user messages. The LLM pipeline works as follows:


1. At ingestion, an LLM splits each document into overlapping chunks. Each chunk gets a headline, a summary, and the original text, and all three are embedded together.
2. An LLM rewrites the user's question into a short, specific query, using the conversation history to resolve references like "they" or "it".
3. The 20 chunks most similar to the original question and the 20 most similar to the rewritten query are retrieved from ChromaDB and merged, with duplicates removed.
4. An LLM reranks the merged chunks by relevance to the original question, and the top 10 are kept.
5. The chunks are injected into the system prompt, and the model generates an answer.

## Configuration

Key settings are at the top of each file:

| Setting | File | Default |
|---|---|---|
| `MODEL` | `answer_llm.py` | `openai/gpt-4.1-nano` |
| `MODEL` | `ingest_llm.py` | `openai/gpt-4.1-nano` |
| `MODEL` | `answer_langchain.py` | `gpt-4.1-nano` |
| `EMBEDDING_MODEL` | `config.py` | `text-embedding-3-large` |
| `LLM_RETRIEVAL_K` | `config.py` | 20 (candidates per query) |
| `FINAL_K` | `config.py` | 10 (chunks kept after reranking, or retrieved by the LangChain pipeline) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `ingest_langchain.py` | 500 / 200 characters |
| `AVERAGE_CHUNK_SIZE` | `ingest_llm.py` | 100 (sets the chunk count the LLM is asked for) |
| `WORKERS` | `ingest_llm.py` | 3 |

In the LLM pipeline, `MODEL` values are [LiteLLM model names](https://docs.litellm.ai/docs/providers), so you can switch providers (e.g. `groq/openai/gpt-oss-120b`) by changing the name and adding that provider's API key to `.env`.

If you change `EMBEDDING_MODEL`, re-run both ingest commands. If you change the `ingest_llm.py` `MODEL`, re-run `uv run ingest-llm`.
