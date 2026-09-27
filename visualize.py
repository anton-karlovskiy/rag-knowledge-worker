import argparse
import numpy as np
import plotly.graph_objects as go
from chromadb import PersistentClient
from sklearn.manifold import TSNE

from config import COLLECTION_NAME, DB_NAME


COLORS = {"products": "blue", "employees": "green", "contracts": "red", "company": "orange"}


def load_store():
    collection = PersistentClient(path=DB_NAME).get_or_create_collection(COLLECTION_NAME)
    result = collection.get(include=["embeddings", "documents", "metadatas"])
    if not result["ids"]:
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' is empty. Run `uv run ingest` first.")
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
    parser.add_argument("--dims", type=int, choices=[2, 3], default=2)
    args = parser.parse_args()
    vectors, documents, types = load_store()
    print(f"Running t-SNE on {len(vectors)} chunks...")
    plot(vectors, documents, types, args.dims).show()


if __name__ == "__main__":
    main()
