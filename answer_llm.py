from chromadb import PersistentClient
from dotenv import load_dotenv
from litellm import ServiceUnavailableError, completion
from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import LLM_COLLECTION_NAME, LLM_DB_PATH, EMBEDDING_MODEL, FINAL_K, LLM_RETRIEVAL_K


load_dotenv(override=True)

MODEL = "openai/gpt-4.1-nano"

ANSWER_SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""

RERANK_SYSTEM_PROMPT = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""

REWRITE_QUERY_PROMPT = """
You are in a conversation with a user.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Since the conversation is contextual, understand the meaning of the user question and add details based on the history.
Condense everything in a single contextually-rich VERY short and specific question, most likely to surface content.

EXAMPLE:
user: Who is the founder? -> Query: who is the founder?
assistant: The founder is FooBar
user: What role covers? -> Query: What role FooBar covers?
...

IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""

# Retry only transient API failures (rate limits, network errors, 5xx); let bugs surface immediately.
# litellm's exceptions subclass openai's, so these also cover completion() calls.
llm_retry = retry(
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, InternalServerError, ServiceUnavailableError)
    ),
    wait=wait_exponential(multiplier=1, min=10, max=240),
    stop=stop_after_attempt(5),
    reraise=True,
)
openai_client = OpenAI()

chroma = PersistentClient(path=LLM_DB_PATH)
collection = chroma.get_or_create_collection(LLM_COLLECTION_NAME)
if collection.count() == 0:
    raise RuntimeError(f"Vector store at {LLM_DB_PATH} is empty. Run `uv run ingest-llm` first.")


class RetrievedChunk(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


@llm_retry
def rerank(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Have the LLM reorder the chunks by relevance to the question.
    """
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": RERANK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    order = RankOrder.model_validate_json(response.choices[0].message.content).order
    return [chunks[i - 1] for i in order]


@llm_retry
def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    """
    Rewrite the question into a short, specific query more likely to surface relevant chunks.
    """
    message = REWRITE_QUERY_PROMPT.format(history=history or [], question=question)
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content


def merge_chunks(chunks: list[RetrievedChunk], extra: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Append the chunks from extra that are not already in chunks.
    """
    merged = chunks[:]
    existing = {chunk.page_content for chunk in chunks}
    for chunk in extra:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def retrieve_candidates(question: str) -> list[RetrievedChunk]:
    query_embedding = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=LLM_RETRIEVAL_K)
    return [
        RetrievedChunk(page_content=document, metadata=metadata)
        for document, metadata in zip(results["documents"][0], results["metadatas"][0])
    ]


def fetch_context(question: str, history: list[dict] | None = None) -> list[RetrievedChunk]:
    """
    Retrieve chunks for the original and rewritten question, rerank them, and keep the top FINAL_K.
    """
    rewritten_query = rewrite_query(question, history)
    chunks = merge_chunks(retrieve_candidates(question), retrieve_candidates(rewritten_query))
    return rerank(question, chunks)[:FINAL_K]


def make_rag_messages(question: str, history: list[dict], chunks: list[RetrievedChunk]) -> list[dict]:
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = ANSWER_SYSTEM_PROMPT.format(context=context)
    return [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]


@llm_retry
def answer_question(question: str, history: list[dict] | None = None) -> tuple[str, list[RetrievedChunk]]:
    """
    Answer the given question with RAG; return the answer and the context chunks.
    """
    history = [{"role": message["role"], "content": message["content"]} for message in history or []]
    chunks = fetch_context(question, history)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks
