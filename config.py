from pathlib import Path


DB_NAME = str(Path(__file__).parent / "vector_db")
EMBEDDING_MODEL = "text-embedding-3-large"
RETRIEVAL_K = 10
