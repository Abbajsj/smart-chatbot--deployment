from sentence_transformers import SentenceTransformer
import faiss


class SemanticSearchEngine:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def _split_query(self, query):
        query = query.lower()
        for sep in [" and ", "&", ",", " also ", " plus "]:
            query = query.replace(sep, "|")
        return [q.strip() for q in query.split("|") if len(q.strip()) > 3]

    def search(self, query, top_k=2, min_score=0.45):
        """
        Returns only HIGH confidence matches
        """
        sub_queries = self._split_query(query)
        results = []

        for sq in sub_queries:
            q_emb = self.model.encode(
                [sq],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            scores, idxs = self.index.search(q_emb, top_k)

            for score, idx in zip(scores[0], idxs[0]):
                if score >= min_score:
                    results.append(self.documents[idx])

        return list(dict.fromkeys(results))