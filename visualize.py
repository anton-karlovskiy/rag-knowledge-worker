import argparse
import numpy as np
import plotly.graph_objects as go
from chromadb import PersistentClient
from sklearn.manifold import TSNE

from config import LANGCHAIN_COLLECTION_NAME, LANGCHAIN_DB_NAME, LLM_COLLECTION_NAME, LLM_DB_NAME
from pipelines import DEFAULT_PIPELINE, PIPELINES


COLORS = {"products": "blue", "employees": "green", "contracts": "red", "company": "orange"}
STORES = {
    "llm": (LLM_DB_NAME, LLM_COLLECTION_NAME),
    "langchain": (LANGCHAIN_DB_NAME, LANGCHAIN_COLLECTION_NAME),
}


def load_store(pipeline: str):
    db_name, collection_name = STORES[pipeline]
    collection = PersistentClient(path=db_name).get_or_create_collection(collection_name)
    result = collection.get(include=["embeddings", "documents", "metadatas"])
    if not result["ids"]:
        raise RuntimeError(f"Collection '{collection_name}' is empty. Run `uv run ingest-{pipeline}` first.")
    types = [metadata["type"] for metadata in result["metadatas"]]
    return np.array(result["embeddings"]), result["documents"], types


def plot(vectors, documents, types, dimensions):
    reduced_vectors = TSNE(n_components=dimensions, random_state=42).fit_transform(vectors)
    fig = go.Figure()
    for doc_type in sorted(set(types)):
        idx = [i for i, t in enumerate(types) if t == doc_type]
        points = reduced_vectors[idx]
        coords = dict(x=points[:, 0], y=points[:, 1])
        if dimensions == 3:
            coords["z"] = points[:, 2]
        trace = go.Scatter3d if dimensions == 3 else go.Scatter
        fig.add_trace(trace(
            **coords,
            mode="markers",
            name=doc_type,
            marker=dict(size=5, color=COLORS.get(doc_type, "gray"), opacity=0.8),
            text=[f"Type: {doc_type}<br>Text: {documents[i][:100]}..." for i in idx],
            hoverinfo="text",
        ))
    fig.update_layout(
        title=f"{dimensions}D Chroma Vector Store Visualization",
        width=900,
        height=700,
        margin=dict(r=10, b=10, l=10, t=40),
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize the vector store with t-SNE")
    parser.add_argument("--dimensions", type=int, choices=[2, 3], default=2)
    parser.add_argument("--pipeline", choices=list(PIPELINES), default=DEFAULT_PIPELINE)
    args = parser.parse_args()
    vectors, documents, types = load_store(args.pipeline)
    print(f"Running t-SNE on {len(vectors)} chunks...")
    plot(vectors, documents, types, args.dimensions).show()


if __name__ == "__main__":
    main()
