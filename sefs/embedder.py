from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def get_embedding(self, text):
        if not text.strip():
            return None
        return self.model.encode(text)
