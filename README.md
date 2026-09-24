# RAG Knowledge Worker

A RAG-based AI assistant that answers questions about a company using its internal documents. Built with LangChain, ChromaDB, OpenAI embeddings, and a Gradio chat interface.

The pipeline splits documents into fixed-size overlapping chunks, retrieves the most similar chunks for each question, and passes them to the model as context.

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

This reads all markdown files from `knowledge-base/`, splits them into chunks, embeds the chunks, and stores everything in a local ChromaDB database.

```bash
uv run ingest
```

**Step 2: Run the app**

```bash
uv run app
```

Opens a Gradio chat interface in your browser. Ask anything about the company. The right panel shows the retrieved context chunks that informed each answer.

## Evaluation

The `evaluation/` folder contains a set of test cases, each with a question, reference answer, and expected keywords.

Evaluation needs the vector store from Step 1 (`vector_db/`, not tracked in git). If it is missing or empty, `app`, `eval`, and `evaluator` exit at startup with an error telling you to run `uv run ingest`.

To evaluate a single test case by index:

```bash
uv run eval 0
```

This runs both retrieval evaluation (MRR, nDCG, keyword coverage) and answer quality evaluation (accuracy, completeness, relevance scored by an LLM judge).

To evaluate all test cases at once, launch the evaluation dashboard:

```bash
uv run evaluator
```

Opens a Gradio dashboard with two sections, Retrieval and Answer, each with its own "Run Evaluation" button. Each section shows the averaged metrics across all test cases, color-coded green/amber/red against the thresholds in `evaluator.py`, plus a bar chart breaking down MRR (retrieval) or accuracy (answer) by question category. Answer evaluation runs the full RAG pipeline plus an LLM judge for every test case, so it takes noticeably longer than retrieval evaluation.

### Results

A full run over all 150 test cases with the default configuration:

![RAG evaluation dashboard](docs/evaluator-dashboard.png)

| Retrieval | Score | Answer | Score |
|---|---|---|---|
| MRR | 0.7919 | Accuracy | 4.21 / 5 |
| nDCG | 0.7949 | Completeness | 3.92 / 5 |
| Keyword coverage | 93.0% | Relevance | 4.63 / 5 |

Performance is strongest on `direct_fact`, `temporal`, and `relationship` questions. The weakest categories are `spanning` (lowest MRR, ~0.47) and `holistic` (lowest MRR at ~0.57 and lowest accuracy at ~3.0 / 5). Both need information pulled from many documents at once. Ten 500-character chunks only cover a small part of that.

## Project structure

```
├── app.py              # Gradio chat UI
├── evaluator.py        # Gradio evaluation dashboard over all test cases
├── answer.py           # RAG pipeline: retrieval and generation
├── ingest.py           # Document loading, chunking, embedding, ChromaDB storage
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

1. At ingestion, each document is split into 500-character chunks with 200 characters of overlap.
2. The user's current question is combined with their earlier questions in the conversation.
3. The 10 chunks most similar to the combined question are retrieved from ChromaDB.
4. The chunks are injected into the system prompt, and the model generates an answer.

## Configuration

Key settings are at the top of each file:

| Setting | File | Default |
|---|---|---|
| `MODEL` | `answer.py` | `gpt-4.1-nano` |
| `EMBEDDING_MODEL` | `config.py` | `text-embedding-3-large` |
| `RETRIEVAL_K` | `answer.py` | 10 |
| `CHUNK_SIZE` | `ingest.py` | 500 |
| `CHUNK_OVERLAP` | `ingest.py` | 200 |

If you change `EMBEDDING_MODEL`, re-run `uv run ingest`.
