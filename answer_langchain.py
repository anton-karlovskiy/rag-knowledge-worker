from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from config import LANGCHAIN_COLLECTION_NAME, LANGCHAIN_DB_PATH, EMBEDDING_MODEL, FINAL_K


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"

ANSWER_SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    collection_name=LANGCHAIN_COLLECTION_NAME, persist_directory=LANGCHAIN_DB_PATH, embedding_function=embeddings
)
if vectorstore._collection.count() == 0:
    raise RuntimeError(f"Vector store at {LANGCHAIN_DB_PATH} is empty. Run `uv run ingest-langchain` first.")
retriever = vectorstore.as_retriever(search_kwargs={"k": FINAL_K})
llm = ChatOpenAI(temperature=0, model=MODEL)


def fetch_context(question: str, history: list[dict] | None = None) -> list[Document]:
    """
    Retrieve relevant context chunks for a question.
    history is accepted only to match the LLM pipeline's interface; answer_question folds it into the question.
    """
    return retriever.invoke(question)


def combine_user_messages(question: str, history: list[dict] | None = None) -> str:
    """
    Combine all the user's messages into a single string.
    """
    history = history or []
    prior_user_messages = "\n".join(message["content"] for message in history if message["role"] == "user")
    return prior_user_messages + "\n" + question


def answer_question(question: str, history: list[dict] | None = None) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context chunks.
    """
    history = [{"role": message["role"], "content": message["content"]} for message in history or []]
    combined_question = combine_user_messages(question, history)
    chunks = fetch_context(combined_question)
    context = "\n\n".join(chunk.page_content for chunk in chunks)
    system_prompt = ANSWER_SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, chunks
