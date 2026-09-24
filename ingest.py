from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from chromadb import PersistentClient
from dotenv import load_dotenv
from litellm import completion
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from tqdm import tqdm

from config import COLLECTION_NAME, DB_NAME, EMBEDDING_MODEL


load_dotenv(override=True)

MODEL = "openai/gpt-4.1-nano"
KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 100
EMBEDDING_BATCH_SIZE = 500
# If you hit rate limits, set WORKERS to 1
WORKERS = 3

wait = wait_exponential(multiplier=1, min=10, max=240)
openai = OpenAI()


class Result(BaseModel):
    page_content: str
    metadata: dict


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

    def as_result(self, document: dict) -> Result:
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


def fetch_documents() -> list[dict]:
    documents = []
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        if not folder.is_dir():
            continue
        for file in folder.rglob("*.md"):
            documents.append(
                {"type": folder.name, "source": file.as_posix(), "text": file.read_text(encoding="utf-8")}
            )
    print(f"Loaded {len(documents)} documents")
    return documents


def make_prompt(document: dict) -> str:
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


@retry(wait=wait)
def process_document(document: dict) -> list[Result]:
    messages = [{"role": "user", "content": make_prompt(document)}]
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]


def create_chunks(documents: list[dict]) -> list[Result]:
    """
    Have the LLM split each document into chunks, running several documents in parallel.
    """
    chunks = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for result in tqdm(executor.map(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = openai.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def create_embeddings(chunks: list[Result]) -> None:
    chroma = PersistentClient(path=DB_NAME)
    if COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(COLLECTION_NAME)

    texts = [chunk.page_content for chunk in chunks]
    vectors = embed(texts)
    ids = [str(i) for i in range(len(chunks))]
    metadatas = [chunk.metadata for chunk in chunks]

    collection = chroma.get_or_create_collection(COLLECTION_NAME)
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    print(f"There are {collection.count():,} vectors with {len(vectors[0]):,} dimensions in the vector store")


def main():
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")


if __name__ == "__main__":
    main()
