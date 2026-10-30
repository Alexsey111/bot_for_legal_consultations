"""Диагностика поиска: почему на разные вопросы приходят одни и те же статьи."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for line in open(os.path.expanduser("~/.agent-secrets/keys.env"), encoding="utf-8", errors="ignore"):
    if line.startswith("PROXYAPI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
os.environ.setdefault("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
os.environ["CHROMA_PERSIST_DIR"] = os.path.join(ROOT, "chroma_legal_db")
os.environ.setdefault("PRODUCTION", "true")

import tiktoken
class _S:
    def _t(self, t): return list(range(max(1, len(t)//4)))
    def encode(self, t, **k): return self._t(t)
    def encode_ordinary(self, t, **k): return self._t(t)
    def encode_batch(self, ts, **k): return [self._t(t) for t in ts]
    def decode(self, t, **k): return ""
tiktoken.get_encoding = lambda name=None: _S()
tiktoken.encoding_for_model = lambda name=None: _S()

import numpy as np
from rag_engine import get_rag_engine

rag = get_rag_engine()
db = rag.db
print("документов в коллекции:", db.vector_db._collection.count())

qs = ["дарение что это", "какой срок исковой давности"]
vecs = []
for q in qs:
    v = db.embeddings.embed_query(q)
    v = np.array(v, dtype=float)
    vecs.append(v)
    print(f"«{q[:30]}»: размерность={v.shape[0]}, норма={np.linalg.norm(v):.4f}, первые 3={np.round(v[:3],4)}")

cos = float(np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])))
print(f"косинус между векторами двух разных вопросов: {cos:.4f}")

print("\n--- поиск по сходству (с score) ---")
for q in qs:
    res = db.vector_db.similarity_search_with_score(q, k=5)
    print(f"\n«{q[:34]}»")
    for doc, score in res:
        print(f"   {score:8.4f}  ст. {doc.metadata.get('article_num')} | {str(doc.metadata.get('reference'))[:40]}")

print("\n--- MMR-ретривер (как в коде) ---")
for q in qs:
    docs = rag.semantic_retriever.invoke(q)
    print(f"«{q[:34]}» -> {[d.metadata.get('article_num') for d in docs]}")
