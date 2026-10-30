"""Разложение по ветвям: кто именно находит статью — BM25 или семантика."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_boot  # noqa: F401  — ключ, Chroma, эмбеддинги без токенизации

import numpy as np
from rag_engine import get_rag_engine
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

CASES = [
    ("что такое договор дарения", {"572"}),
    ("какой срок исковой давности", {"196"}),
    ("можно ли расторгнуть договор аренды досрочно", {"619", "620"}),
    ("что такое договор займа", {"807"}),
    ("что такое неустойка", {"330"}),
    ("что такое договор купли-продажи", {"454"}),
    ("как оформить доверенность", {"185", "185.1"}),
    ("что такое сервитут", {"274"}),
    ("ответственность за вред от источника повышенной опасности", {"1079"}),
    ("что такое публичный договор", {"426"}),
]

rag = get_rag_engine()
data = rag.db.vector_db.get(include=["embeddings", "documents", "metadatas"])
docs, metas = data["documents"], data["metadatas"]
emb = np.array(data["embeddings"], float)
norms = np.linalg.norm(emb, axis=1)
arts = [str(m.get("article_num")) for m in metas]
lens = np.array([len(d) for d in docs])
key2idx = {}
for i, d in enumerate(docs):
    key2idx.setdefault((d, arts[i]), i)

bm25 = BM25Retriever.from_documents([Document(page_content=d, metadata=m) for d, m in zip(docs, metas)])
bm25.k = 5


def top_k(scores, keep=None, k=5):
    out = []
    for i in np.argsort(-scores):
        if keep is not None and not keep[i]:
            continue
        out.append(int(i))
        if len(out) >= k:
            break
    return out


def hit(ranked, expect):
    return any(arts[i] in expect for i in ranked[:5])


print(f"{'вопрос':48} {'BM25':>6} {'сем. всё':>9} {'сем.>300':>9}")
c = {"bm": 0, "sem": 0, "semf": 0}
for q, expect in CASES:
    qv = np.array(rag.db.embeddings.embed_query(q), float)
    sims = emb @ qv / (norms * np.linalg.norm(qv))
    bm = [key2idx[(d.page_content, str(d.metadata.get("article_num")))]
          for d in bm25.invoke(q)
          if (d.page_content, str(d.metadata.get("article_num"))) in key2idx]
    h_bm = hit(bm, expect)
    h_sem = hit(top_k(sims), expect)
    h_semf = hit(top_k(sims, keep=lens >= 300), expect)
    c["bm"] += h_bm
    c["sem"] += h_sem
    c["semf"] += h_semf
    print(f"{q[:48]:48} {'OK' if h_bm else '—':>6} {'OK' if h_sem else '—':>9} {'OK' if h_semf else '—':>9}")

print(f"\ntop-5: BM25 {c['bm']}/10 · семантика по всему индексу {c['sem']}/10 · "
      f"семантика без обрубков {c['semf']}/10")
