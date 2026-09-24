from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from config import DB_NAME, EMBEDDING_MODEL, RETRIEVAL_K


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
if vectorstore._collection.count() == 0:
    raise RuntimeError(f"Vector store at {DB_NAME} is empty. Run `uv run ingest` first.")
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
llm = ChatOpenAI(temperature=0, model=MODEL)


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question)


def combined_question(question: str, history: list[dict] | None = None) -> str:
    """
    Combine all the user's messages into a single string.
    """
    history = history or []
    prior = "\n".join(message["content"] for message in history if message["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] | None = None) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    history = history or []
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
