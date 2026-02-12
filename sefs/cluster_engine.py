from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np


class ClusterEngine:
    def cluster(self, embeddings):

        if embeddings is None or len(embeddings) < 5:
            return None

        embeddings = normalize(embeddings)

        k = 5  # demo clarity mode

        clustering = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = clustering.fit_predict(embeddings)

        return labels
