from sklearn.feature_extraction.text import TfidfVectorizer
import re

class LabelGenerator:
    def __init__(self):
        self.generated_labels = set()

    def generate_label(self, texts):

        if not texts:
            return self._unique_label("Misc")

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=15,
            ngram_range=(1, 2),   # Allow meaningful phrases
            min_df=1
        )

        try:
            X = vectorizer.fit_transform(texts)
        except:
            return self._unique_label("Cluster")

        feature_array = vectorizer.get_feature_names_out()
        tfidf_scores = X.sum(axis=0).A1
        sorted_indices = tfidf_scores.argsort()[::-1]

        generic_words = {"document", "documents", "file", "data", "information"}

        top_keywords = []
        for idx in sorted_indices:
            word = feature_array[idx]

            if word.lower() not in generic_words:
                cleaned = re.sub(r"[^A-Za-z0-9_]", "", word)
                if cleaned:
                    top_keywords.append(cleaned)

            if len(top_keywords) == 3:
                break

        if not top_keywords:
            return self._unique_label("Cluster")

        label = "_".join(top_keywords)

        return self._unique_label(label)

    def _unique_label(self, base_label):
        base_label = base_label.strip("_")

        if base_label not in self.generated_labels:
            self.generated_labels.add(base_label)
            return base_label

        counter = 1
        while f"{base_label}_{counter}" in self.generated_labels:
            counter += 1

        unique_label = f"{base_label}_{counter}"
        self.generated_labels.add(unique_label)
        return unique_label
