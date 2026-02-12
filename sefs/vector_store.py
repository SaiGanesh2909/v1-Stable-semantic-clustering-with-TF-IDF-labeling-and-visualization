import faiss
import numpy as np
import os
import json


class VectorStore:
    def __init__(self, embedding_dim):
        self.dim = embedding_dim
        self.index_file = "faiss_index.bin"
        self.map_file = "file_map.json"

        # Always recreate index if dimension mismatch
        if os.path.exists(self.index_file):
            index = faiss.read_index(self.index_file)

            if index.d != self.dim:
                print("Embedding dimension changed. Rebuilding index.")
                os.remove(self.index_file)
                if os.path.exists(self.map_file):
                    os.remove(self.map_file)
                self.index = faiss.IndexFlatL2(self.dim)
                self.file_map = []
            else:
                self.index = index
                if os.path.exists(self.map_file):
                    with open(self.map_file, "r") as f:
                        self.file_map = json.load(f)
                else:
                    self.file_map = []
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.file_map = []

    # -------------------------
    def add(self, embedding, file_path):
        if file_path in self.file_map:
            return

        embedding = np.array([embedding]).astype("float32")
        self.index.add(embedding)
        self.file_map.append(file_path)
        self.save()

    # -------------------------
    def get_all_embeddings(self):
        if self.index.ntotal == 0:
            return None
        return self.index.reconstruct_n(0, self.index.ntotal)

    # -------------------------
    def get_file_map(self):
        return self.file_map

    # -------------------------
    def save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.map_file, "w") as f:
            json.dump(self.file_map, f)

    # -------------------------
    def reset(self):
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.map_file):
            os.remove(self.map_file)

        self.index = faiss.IndexFlatL2(self.dim)
        self.file_map = []
