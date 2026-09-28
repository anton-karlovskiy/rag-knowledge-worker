from importlib import import_module
from types import ModuleType


# Pipeline name -> module exposing answer_question(question, history) and fetch_context(question, history)
PIPELINES = {
    "langchain": "answer_langchain",
    "llm": "answer_llm",
}
LABELS = {
    "langchain": "LangChain",
    "llm": "LLM (chunking + rerank)",
}
DEFAULT_PIPELINE = "llm"


def get_pipeline(name: str = DEFAULT_PIPELINE) -> ModuleType:
    """
    Import the pipeline module on first use, so only the selected pipeline needs a populated vector store.
    """
    if name not in PIPELINES:
        raise ValueError(f"Unknown pipeline {name!r}, expected one of {list(PIPELINES)}")
    return import_module(PIPELINES[name])
