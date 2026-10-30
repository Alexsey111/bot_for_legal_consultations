"""Качество чанков и слияния: где теряется правильная статья."""
import os, sys
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
data = rag.db.vector_db.get(include=["embeddings", "documents", "metadatas"])
docs, metas, emb = data["documents"], data["metadatas"], np.array(data["embeddings"], float)
lens = np.array([len(d) for d in docs])
print("чанков:", len(docs))
for lim in (50, 100, 200, 400):
    print(f"  короче {lim:4} символов: {int((lens < lim).sum()):5}  ({(lens < lim).mean()*100:.0f}%)")

print("\nпримеры коротких чанков:")
short = [i for i in range(len(docs)) if lens[i] < 80][:5]
for i in short:
    print(f"   ст.{metas[i].get('article_num')}: {docs[i]!r}")

Q = "какой срок исковой давности"
qv = np.array(rag.db.embeddings.embed_query(Q), float)
qv /= np.linalg.norm(qv)
sims = emb @ qv / np.linalg.norm(emb, axis=1)

print(f"\nзапрос: «{Q}»")
print("[A] top-5 по всем чанкам:")
for i in np.argsort(-sims)[:5]:
    print(f"    {sims[i]:.3f} ст.{metas[i].get('article_num'):6} len={lens[i]:5} {docs[i][:60]!r}")

mask = lens > 400
print("[B] top-5 только по чанкам длиннее 400 символов:")
for i in np.argsort(-sims)[:len(sims)][:5]:
    pass
idx = np.where(mask)[0]
for i in idx[np.argsort(-sims[idx])[:5]]:
    print(f"    {sims[i]:.3f} ст.{metas[i].get('article_num'):6} len={lens[i]:5} {docs[i][:60]!r}")

print("[C] BM25 top-5:", [d.metadata.get("article_num") for d in rag.bm25_retriever.invoke(Q)][:5])
print("[D] первое вхождение ст.196 в семантическом списке:",
      next((rank for rank, i in enumerate(np.argsort(-sims)) if str(metas[i].get('article_num')) == '196'), None))
