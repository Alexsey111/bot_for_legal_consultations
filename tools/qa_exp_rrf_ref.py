"""Проверка: даёт ли слияние по рангам (RRF) + склейка повторов 6/6 на эталонах."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import qa_boot  # noqa: F401
import numpy as np
from rag_engine import get_rag_engine
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

CASES = [
    ("дарение что это", {"572"}),
    ("какой срок исковой давности", {"196"}),
    ("можно ли расторгнуть договор аренды досрочно", {"619", "620"}),
    ("что такое договор займа", {"807"}),
    ("наём жилого помещения права нанимателя", {"671"}),
    ("возмещение вреда имуществу", {"1064"}),
]

rag = get_rag_engine()
res = rag.db.vector_db.get(include=["embeddings", "documents", "metadatas"])
docs, metas = res["documents"], res["metadatas"]
emb = np.array(res["embeddings"], float)
norms = np.linalg.norm(emb, axis=1)
arts = [str(m.get("article_num")) for m in metas]
key2idx = {}
for i, d in enumerate(docs):
    key2idx.setdefault((d, arts[i]), i)

bm25 = BM25Retriever.from_documents([Document(page_content=d, metadata=m) for d, m in zip(docs, metas)])
bm25.k = 15


def dedupe(order, k=5):
    seen, out = set(), []
    for i in order:
        if arts[i] in seen:
            continue
        seen.add(arts[i])
        out.append(i)
        if len(out) >= k:
            break
    return out


print(f"{'вопрос':46} {'сем':>5} {'RRF':>5} {'RRF+склейка':>12}")
c_sem = c_rrf = c_both = 0
for q, expect in CASES:
    qv = np.array(rag.db.embeddings.embed_query(q), float)
    sims = emb @ qv / (norms * np.linalg.norm(qv))
    sem = [int(i) for i in np.argsort(-sims)[:15]]
    bm = [key2idx[(d.page_content, str(d.metadata.get("article_num")))]
          for d in bm25.invoke(q)
          if (d.page_content, str(d.metadata.get("article_num"))) in key2idx]

    rrf = {}
    for lst, w in ((sem, 1.0), (bm, 1.0)):
        for r, i in enumerate(lst):
            rrf[i] = rrf.get(i, 0.0) + w / (60 + r)
    fused = [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])]

    h_sem = any(arts[i] in expect for i in sem[:5])
    h_rrf = any(arts[i] in expect for i in fused[:5])
    clean = dedupe(fused)
    h_both = any(arts[i] in expect for i in clean)
    c_sem += h_sem
    c_rrf += h_rrf
    c_both += h_both
    print(f"{q[:46]:46} {'OK' if h_sem else '—':>5} {'OK' if h_rrf else '—':>5} "
          f"{'OK' if h_both else '—':>12}   {[arts[i] for i in clean]}")

print(f"\nhit@5: семантика {c_sem}/6 · RRF {c_rrf}/6 · RRF+склейка {c_both}/6")
