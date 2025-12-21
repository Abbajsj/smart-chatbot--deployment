from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class TextSearchEngine:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )
        self.doc_vectors = self.vectorizer.fit_transform(documents)

    def search(self, query, min_score=0.15):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]

        best_index = similarities.argmax()
        best_score = similarities[best_index]

        if best_score < min_score:
            return None, 0.0

        return self.documents[best_index], float(best_score)

    def search_multiple(self, query, min_score=0.15):
        parts = re.split(r"\band\b|\?|\.|,", query.lower())
        parts = [p.strip() for p in parts if len(p.strip()) > 3]

        matched_questions = []

        for part in parts:
            q, score = self.search(part, min_score)
            if q and q not in matched_questions:
                matched_questions.append(q)

        if not matched_questions:
            return None, 0.0

        return matched_questions, 1.0