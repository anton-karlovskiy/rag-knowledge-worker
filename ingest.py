from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential

from models import Result


load_dotenv(override=True)

MODEL = "openai/gpt-4.1-nano"

DB_NAME = str(Path(__file__).parent / "preprocessed_db")
COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "text-embedding-3-large"
KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 100
EMBEDDING_BATCH_SIZE = 512

RETRY_WAIT = wait_exponential(multiplier=1, min=10, max=240)

WORKERS = 3

client = OpenAI()


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


def fetch_documents():
    documents = []
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        if not folder.is_dir():
            continue
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append({"type": doc_type, "source": file.as_posix(), "text": f.read()})
    print(f"Loaded {len(documents)} documents")
    return documents


def make_prompt(document):
    estimated_chunks = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {estimated_chunks} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


def make_messages(document):
    return [{"role": "user", "content": make_prompt(document)}]


@retry(wait=RETRY_WAIT)
def process_document(document):
    messages = make_messages(document)
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    doc_as_chunks = Chunks.model_validate_json(response.choices[0].message.content).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]


def create_chunks(documents):
    """Use parallel workers to chunk documents. Set WORKERS=1 if you hit rate limits."""
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks


def create_embeddings(chunks):
    chroma_client = PersistentClient(path=DB_NAME)
    if COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(COLLECTION_NAME)

    texts = [chunk.page_content for chunk in chunks]

    embeddings = []
    for i in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="Embedding batches"):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = client.embeddings.create(model=EMBEDDING_MODEL, input=batch).data
        embeddings.extend(e.embedding for e in batch_embeddings)

    docs_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    ids = [str(i) for i in range(len(chunks))]
    metadatas = [chunk.metadata for chunk in chunks]

    docs_collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Vectorstore created with {docs_collection.count()} documents")


def main():
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")


if __name__ == "__main__":
    main()
