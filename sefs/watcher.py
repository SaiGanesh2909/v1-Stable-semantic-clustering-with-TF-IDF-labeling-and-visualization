from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os

from extractor import extract_text
from embedder import Embedder
from vector_store import VectorStore
from cluster_engine import ClusterEngine
from folder_manager import move_file_to_cluster
from config import ROOT_DIRECTORY, SUPPORTED_EXTENSIONS
from file_registry import FileRegistry
from label_generator import LabelGenerator


class SEFSHandler(FileSystemEventHandler):

    def __init__(self):
        self.embedder = Embedder()
        sample_embedding = self.embedder.get_embedding("dimension check")
        embedding_dim = len(sample_embedding)

        self.vector_store = VectorStore(embedding_dim)
        self.cluster_engine = ClusterEngine()
        self.registry = FileRegistry()
        self.label_generator = LabelGenerator()


    def process_file(self, file_path):

        if not os.path.exists(file_path):
            return

        if os.path.dirname(file_path) != ROOT_DIRECTORY:
            return

        if not any(file_path.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            return

        if not self.registry.needs_update(file_path):
            return

        print(f"\n📂 Processing: {os.path.basename(file_path)}")

        text = extract_text(file_path)

        embedding = self.embedder.get_embedding(text)
        if embedding is None:
            return

        self.vector_store.add(embedding, file_path)

        embeddings = self.vector_store.get_all_embeddings()
        labels = self.cluster_engine.cluster(embeddings)

        if labels is None:
            return

        file_map = self.vector_store.get_file_map()

        cluster_text_map = {}
        cluster_file_map = {}

        # Group files by cluster
        for file, cluster_id in zip(file_map, labels):

            if cluster_id not in cluster_text_map:
                cluster_text_map[cluster_id] = []
                cluster_file_map[cluster_id] = []

            cluster_text_map[cluster_id].append(extract_text(file))
            cluster_file_map[cluster_id].append(file)

        # Generate semantic folder names + move files
        for cluster_id in cluster_file_map:

            label = self.label_generator.generate_label(
                cluster_text_map[cluster_id]
            )

            print(f"🧠 Cluster {cluster_id} labeled as: {label}")

            for file in cluster_file_map[cluster_id]:
                print(
                    f"   ➜ {os.path.basename(file)} → {label}"
                )
                move_file_to_cluster(ROOT_DIRECTORY, file, label)


    def on_created(self, event):
        if not event.is_directory:
            self.process_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self.process_file(event.src_path)


def start_watching():
    event_handler = SEFSHandler()
    observer = Observer()
    observer.schedule(event_handler, ROOT_DIRECTORY, recursive=True)

    print("Monitoring directory:", ROOT_DIRECTORY)
    observer.start()
    print("🚀 SEFS Monitoring Started...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
