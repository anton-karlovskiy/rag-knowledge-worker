from pathlib import Path


# Shared by both pipelines and the evaluation
EMBEDDING_MODEL = "text-embedding-3-large"
FINAL_K = 10

# LLM pipeline
LLM_DB_PATH = str(Path(__file__).parent / "llm_db")
LLM_COLLECTION_NAME = "llm"
LLM_RETRIEVAL_K = 20

# LangChain pipeline keeps its own store so both pipelines can be served side by side
LANGCHAIN_DB_PATH = str(Path(__file__).parent / "langchain_db")
LANGCHAIN_COLLECTION_NAME = "langchain"
