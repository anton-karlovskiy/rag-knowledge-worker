from pathlib import Path


ROOT_DIR = Path(__file__).parent

# Source documents: each subfolder name becomes the chunk `type` metadata
KNOWLEDGE_BASE_DIR_NAME = "knowledge-base"
KNOWLEDGE_BASE_PATH = ROOT_DIR / KNOWLEDGE_BASE_DIR_NAME
DOCUMENT_GLOB = "**/*.md"

# Shared by both pipelines and the evaluation
EMBEDDING_MODEL = "text-embedding-3-large"
FINAL_K = 10

# LLM pipeline
LLM_DB_PATH = str(ROOT_DIR / "llm_db")
LLM_COLLECTION_NAME = "llm"
LLM_RETRIEVAL_K = 20

# LangChain pipeline keeps its own store so both pipelines can be served side by side
LANGCHAIN_DB_PATH = str(ROOT_DIR / "langchain_db")
LANGCHAIN_COLLECTION_NAME = "langchain"
