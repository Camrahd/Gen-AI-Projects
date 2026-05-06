from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SemanticMemory:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.texts = []
        self.index = None

    def add(self, text):
        emb = self.model.encode([text])

        if self.index is None:
            self.index = faiss.IndexFlatL2(len(emb[0]))

        self.index.add(np.array(emb))
        self.texts.append(text)

    def search(self, query, k=2):
        if self.index is None:
            return []

        q = self.model.encode([query])
        _, idx = self.index.search(np.array(q), k)

        return [self.texts[i] for i in idx[0]]