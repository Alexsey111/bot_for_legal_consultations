"""Эксперимент: даёт ли фильтр мусорных чанков + слияние по рангам (RRF) рабочий поиск.

Работаем поверх уже собранного индекса, проект не меняем.
Стоимость: 10 запросов эмбеддингов, копейки.
"""
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

corpus = [Document(page_content=d, metadata=m) for d, m in zip(docs, metas)]
bm25 = BM25Retriever.from_documents(corpus)
bm25.k = 30


def top_k(scores, keep=None, k=30):
    order = np.argsort(-scores)
    out = []
    for i in order:
        if keep is not None and not keep[i]:
            continue
        out.append(int(i))
        if len(out) >= k:
            break
    return out


def hit(ranked, expect):
    return any(arts[i] in expect for i in ranked[:5])


print(f"{'вопрос':52} {'как в боте':>10} {'фильтр+RRF':>10}")
base_ok = fix_ok = 0
for q, expect in CASES:
    qv = np.array(rag.db.embeddings.embed_query(q), float)
    sims = emb @ qv / (norms * np.linalg.norm(qv))

    bm_idx = [key2idx[(d.page_content, str(d.metadata.get("article_num")))]
              for d in bm25.invoke(q)
              if (d.page_content, str(d.metadata.get("article_num"))) in key2idx]

    base = top_k(sims, k=15) + bm_idx          # как в боте: сначала семантика, потом BM25

    keep = lens >= 300                        # выкидываем заголовки-обрубки
    sem_ok = top_k(sims, keep=keep, k=30)
    bm_ok = [i for i in bm_idx if keep[i]]
    rrf = {}
    for lst in (sem_ok, bm_ok):
        for r, i in enumerate(lst):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + r)
    fix = [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])]

    hb, hf = hit(base, expect), hit(fix, expect)
    base_ok += hb
    fix_ok += hf
    print(f"{q[:52]:52} {'OK' if hb else '—':>10} {'OK' if hf else '—':>10}")

print(f"\nпопаданий в top-5: как в боте {base_ok}/{len(CASES)}, фильтр+RRF {fix_ok}/{len(CASES)}")
