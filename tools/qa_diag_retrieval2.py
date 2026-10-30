"""Почему поиск не находит нужную статью, хотя векторы совместимы.

Сравниваем три пути: Chroma similarity_search, MMR-ретривер (как в коде),
BM25, и честный перебор косинусов в numpy по тем же самым векторам.
"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for line in open(os.path.expanduser("~/.agent-secrets/keys.env"), encoding="utf-8", errors="ignore"):
    if line.startswith("PROXYAPI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
os.environ["OPENAI_BASE_URL"] = "https://api.proxyapi.ru/openai/v1"
os.environ["CHROMA_PERSIST_DIR"] = os.path.join(ROOT, "chroma_legal_db")
os.environ["PRODUCTION"] = "true"

import tiktoken
class _S:
    def _t(self, t): return list(range(max(1, len(t)//4)))
    def encode(self, t, **k): return self._t(t)
    def encode_ordinary(self, t, **k): return self._t(t)
    def encode_batch(self, ts, **k): return [self._t(t) for t in ts]
    def decode(self, t, **k): return ""
tiktoken.get_encoding = lambda n=None: _S()
tiktoken.encoding_for_model = lambda n=None: _S()

import numpy as np
from rag_engine import get_rag_engine

rag = get_rag_engine()
db = rag.db
coll = db.vector_db._collection
print("документов:", coll.count(), "| метаданные:", db.vector_db._collection.metadata)

# все векторы и метаданные
data = coll.get(include=["embeddings", "metadatas"])
emb = np.array(data["embeddings"], dtype=float)
arts = [str(m.get("article_num")) for m in data["metadatas"]]
print("матрица:", emb.shape)

Q = "какой срок исковой давности"
print(f"\nвопрос: «{Q}»")

# 1) честный перебор косинусов
qv = np.array(db.embeddings.embed_query(Q), dtype=float)
qv /= np.linalg.norm(qv)
sims = emb @ qv / np.linalg.norm(emb, axis=1)
top = np.argsort(-sims)[:5]
print("\n[1] numpy-перебор (эталон):")
for i in top:
    print(f"    {sims[i]:.4f}  ст. {arts[i]}")

# 2) Chroma similarity_search
print("\n[2] Chroma similarity_search_with_score:")
for doc, score in db.vector_db.similarity_search_with_score(Q, k=5):
    print(f"    dist={score:.4f}  ст. {doc.metadata.get('article_num')}")

# 3) MMR-ретривер, как в коде
print("\n[3] MMR (semantic_retriever):")
print("   ", [d.metadata.get("article_num") for d in rag.semantic_retriever.invoke(Q)])

# 4) BM25
print("\n[4] BM25:")
print("   ", [d.metadata.get("article_num") for d in rag.bm25_retriever.invoke(Q)])

print("\nнормы векторов: min=%.4f max=%.4f" % (np.linalg.norm(emb, axis=1).min(), np.linalg.norm(emb, axis=1).max()))
