from pyvis.network import Network
from vector_store import VectorStore
from embedder import Embedder
from cluster_engine import ClusterEngine
from label_generator import LabelGenerator
from extractor import extract_text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from collections import defaultdict
import os
import json


def visualize():

    embedder = Embedder()
    sample_embedding = embedder.get_embedding("dimension check")
    embedding_dim = len(sample_embedding)

    store = VectorStore(embedding_dim)
    embeddings = store.get_all_embeddings()
    file_map = store.get_file_map()

    if embeddings is None:
        print("No data to visualize.")
        return

    embeddings = normalize(embeddings)

    cluster_engine = ClusterEngine()
    labels = cluster_engine.cluster(embeddings)

    similarity_matrix = cosine_similarity(embeddings)

    label_generator = LabelGenerator()

    cluster_text_map = defaultdict(list)
    cluster_file_map = defaultdict(list)

    for file, cluster_id in zip(file_map, labels):
        cluster_text_map[cluster_id].append(extract_text(file))
        cluster_file_map[cluster_id].append(file)

    net = Network(
        height="850px",
        width="100%",
        bgcolor="#121212",
        font_color="white"
    )

    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=200,
        spring_strength=0.08,
        damping=0.4
    )

    colors = [
        "#ff4d4d",
        "#4da6ff",
        "#66ff66",
        "#ffcc00",
        "#cc66ff",
        "#00ffff",
        "#ff66b2"
    ]

    for idx, cluster_id in enumerate(cluster_file_map):

        label = label_generator.generate_label(
            cluster_text_map[cluster_id]
        )

        cluster_node_id = f"cluster_{cluster_id}"

        # Cluster node
        net.add_node(
            cluster_node_id,
            label=label,
            color=colors[idx % len(colors)],
            size=45,
            shape="dot",
            title=f"<b>Cluster:</b> {label}"
        )

        for file in cluster_file_map[cluster_id]:

            filename = os.path.basename(file)
            preview = extract_text(file)[:300].replace("\n", " ")

            net.add_node(
                filename,
                label=filename,
                color=colors[idx % len(colors)],
                size=20,
                title=f"<b>File:</b> {filename}<br><br><b>Preview:</b><br>{preview}"
            )

            net.add_edge(
                cluster_node_id,
                filename,
                value=2
            )

    # Add similarity edges (light grey lines)
    threshold = 0.80

    for i in range(len(file_map)):
        for j in range(i + 1, len(file_map)):
            if similarity_matrix[i][j] > threshold:
                net.add_edge(
                    os.path.basename(file_map[i]),
                    os.path.basename(file_map[j]),
                    color="#888888",
                    width=1,
                    title=f"Similarity: {similarity_matrix[i][j]:.2f}"
                )

    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "stabilization": false
        },
        "interaction": {
            "hover": true
        },
        "nodes": {
            "borderWidth": 1,
            "borderWidthSelected": 2
        }
    }
    """)


    net.write_html("semantic_graph.html")

    print("✅ Enhanced interactive graph generated: semantic_graph.html")


if __name__ == "__main__":
    visualize()
