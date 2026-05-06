import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class HybridRetriever:
    def __init__(self, documents):
        self.docs = documents
        self.texts = [d.text for d in documents]

        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        embeddings = self.model.encode(self.texts)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

        tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, k=5):
        q_emb = self.model.encode([query])
        _, dense_idx = self.index.search(np.array(q_emb), k)

        bm25_scores = self.bm25.get_scores(query.split())
        bm25_idx = np.argsort(bm25_scores)[-k:]

        # Filter out -1 indices FAISS returns when fewer docs than k
        valid_dense = [i for i in dense_idx[0] if i >= 0]
        indices = list(set(valid_dense) | set(int(i) for i in bm25_idx))

        return [self.docs[i] for i in indices if 0 <= i < len(self.docs)]
