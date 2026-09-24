from pathlib import Path


DB_NAME = str(Path(__file__).parent / "preprocessed_db")
COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "text-embedding-3-large"
RETRIEVAL_K = 20
FINAL_K = 10
