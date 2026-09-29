import argparse
import numpy as np
import plotly.graph_objects as go
from chromadb import PersistentClient
from sklearn.manifold import TSNE

from config import LANGCHAIN_COLLECTION_NAME, LANGCHAIN_DB_PATH, LLM_COLLECTION_NAME, LLM_DB_PATH
from pipelines import DEFAULT_PIPELINE, PIPELINE_MODULES


TYPE_COLORS = {"products": "blue", "employees": "green", "contracts": "red", "company": "orange"}
STORES = {
    "llm": (LLM_DB_PATH, LLM_COLLECTION_NAME),
    "langchain": (LANGCHAIN_DB_PATH, LANGCHAIN_COLLECTION_NAME),
}


def load_embeddings(pipeline_name: str):
    db_path, collection_name = STORES[pipeline_name]
    collection = PersistentClient(path=db_path).get_or_create_collection(collection_name)
    result = collection.get(include=["embeddings", "documents", "metadatas"])
    if not result["ids"]:
        raise RuntimeError(f"Collection '{collection_name}' is empty. Run `uv run ingest-{pipeline_name}` first.")
    chunk_types = [metadata["type"] for metadata in result["metadatas"]]
    return np.array(result["embeddings"]), result["documents"], chunk_types


def plot_tsne(vectors, chunk_texts, chunk_types, dimensions):
    reduced_vectors = TSNE(n_components=dimensions, random_state=42).fit_transform(vectors)
    fig = go.Figure()
    for type_name in sorted(set(chunk_types)):
        indices = [i for i, chunk_type in enumerate(chunk_types) if chunk_type == type_name]
        points = reduced_vectors[indices]
        coords = dict(x=points[:, 0], y=points[:, 1])
        if dimensions == 3:
            coords["z"] = points[:, 2]
        scatter_class = go.Scatter3d if dimensions == 3 else go.Scatter
        fig.add_trace(scatter_class(
            **coords,
            mode="markers",
            name=type_name,
            marker=dict(size=5, color=TYPE_COLORS.get(type_name, "gray"), opacity=0.8),
            text=[f"Type: {type_name}<br>Text: {chunk_texts[i][:100]}..." for i in indices],
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
    parser.add_argument("--pipeline", choices=list(PIPELINE_MODULES), default=DEFAULT_PIPELINE)
    args = parser.parse_args()
    vectors, chunk_texts, chunk_types = load_embeddings(args.pipeline)
    print(f"Running t-SNE on {len(vectors)} chunks...")
    plot_tsne(vectors, chunk_texts, chunk_types, args.dimensions).show()


if __name__ == "__main__":
    main()
