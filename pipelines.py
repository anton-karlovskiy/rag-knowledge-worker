from importlib import import_module
from types import ModuleType


# Pipeline name -> module exposing answer_question(question, history) and fetch_context(question, history)
PIPELINE_MODULES = {
    "langchain": "answer_langchain",
    "llm": "answer_llm",
}
PIPELINE_LABELS = {
    "langchain": "LangChain",
    "llm": "LLM (chunking + rerank)",
}
DEFAULT_PIPELINE = "llm"
# (label, value) pairs for the UI pipeline selector
PIPELINE_CHOICES = [(label, name) for name, label in PIPELINE_LABELS.items()]


def get_pipeline(pipeline_name: str = DEFAULT_PIPELINE) -> ModuleType:
    """
    Import the pipeline module on first use, so only the selected pipeline needs a populated vector store.
    """
    if pipeline_name not in PIPELINE_MODULES:
        raise ValueError(f"Unknown pipeline {pipeline_name!r}, expected one of {list(PIPELINE_MODULES)}")
    return import_module(PIPELINE_MODULES[pipeline_name])
