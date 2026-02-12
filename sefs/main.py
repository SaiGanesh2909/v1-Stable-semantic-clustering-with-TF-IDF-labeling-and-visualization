import sys
from embedder import Embedder
from vector_store import VectorStore
from cluster_engine import ClusterEngine
from watcher import start_watching


if __name__ == "__main__":

    embedder = Embedder()
    sample_embedding = embedder.get_embedding("dimension check")
    embedding_dim = len(sample_embedding)

    # 🔁 REINDEX MODE
    if "--reindex" in sys.argv:
        print("🔄 Reindex mode activated...")
        store = VectorStore(embedding_dim)
        store.reset()
        print("✅ Index cleared successfully.")
        exit()

    # 🔁 RECALCULATE ALL MODE
    if "--recalculate" in sys.argv:
        print("🔄 Recalculating all clusters...")
        store = VectorStore(embedding_dim)
        embeddings = store.get_all_embeddings()

        if embeddings is None:
            print("No embeddings found.")
            exit()

        labels = ClusterEngine().cluster(embeddings)
        print("✅ Recalculation complete.")
        exit()

    # 🚀 NORMAL MODE
    start_watching()
