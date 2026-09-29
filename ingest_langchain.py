from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import LANGCHAIN_COLLECTION_NAME, LANGCHAIN_DB_PATH, EMBEDDING_MODEL


load_dotenv(override=True)

KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge-base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


def load_documents():
    documents = []
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        if not folder.is_dir():
            continue
        loader = DirectoryLoader(
            str(folder), glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        for doc in loader.load():
            doc.metadata["type"] = folder.name
            documents.append(doc)
    print(f"Loaded {len(documents)} documents")
    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_documents(documents)


def build_vector_store(chunks):
    if Path(LANGCHAIN_DB_PATH).exists():
        Chroma(
            collection_name=LANGCHAIN_COLLECTION_NAME, persist_directory=LANGCHAIN_DB_PATH, embedding_function=embeddings
        ).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name=LANGCHAIN_COLLECTION_NAME, persist_directory=LANGCHAIN_DB_PATH
    )

    collection = vectorstore._collection
    count = collection.count()
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


def main():
    documents = load_documents()
    chunks = create_chunks(documents)
    build_vector_store(chunks)
    print("Ingestion complete")


if __name__ == "__main__":
    main()
