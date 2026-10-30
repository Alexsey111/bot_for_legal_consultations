"""Подбор веса BM25 в слиянии: чтобы и 1064 находилась, и 671 не терялась."""
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
    ("что такое неустойка", {"330"}),
    ("что такое сервитут", {"274"}),
    ("как делится наследство по закону", {"1111", "1142"}),
    ("как продать квартиру", {"549", "550"}),
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

WEIGHTS = [1.0, 0.8, 0.6, 0.5, 0.3]
hits = {w: 0 for w in WEIGHTS}
print(f"{'вопрос':44} " + " ".join(f"{w:>5}" for w in WEIGHTS))
for q, expect in CASES:
    qv = np.array(rag.db.embeddings.embed_query(q), float)
    sims = emb @ qv / (norms * np.linalg.norm(qv))
    sem = [int(i) for i in np.argsort(-sims)[:15]]
    bm = [key2idx[(d.page_content, str(d.metadata.get("article_num")))]
          for d in bm25.invoke(q)
          if (d.page_content, str(d.metadata.get("article_num"))) in key2idx]

    row = []
    for w in WEIGHTS:
        rrf = {}
        for lst, weight in ((sem, 1.0), (bm, w)):
            for r, i in enumerate(lst):
                rrf[i] = rrf.get(i, 0.0) + weight / (60 + r)
        seen, clean = set(), []
        for i, _ in sorted(rrf.items(), key=lambda x: -x[1]):
            if arts[i] in seen:
                continue
            seen.add(arts[i])
            clean.append(i)
            if len(clean) == 5:
                break
        hit = any(arts[i] in expect for i in clean)
        hits[w] += hit
        row.append("OK" if hit else "—")
    print(f"{q[:44]:44} " + " ".join(f"{c:>5}" for c in row))

print("\nhit@5: " + " · ".join(f"bm25×{w} = {hits[w]}/{len(CASES)}" for w in WEIGHTS))
