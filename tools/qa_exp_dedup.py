"""Проверка: помогает ли склейка повторов одной статьи в top-5 попаданию (hit@5).

Сейчас в выдаче дубли: «дарение» → 572, 572, 573, 574, 572 — три слота из пяти
занимает одна и та же статья.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import qa_boot  # noqa: F401
from rag_engine import get_rag_engine

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
import numpy as np
emb = np.array(res["embeddings"], float)
norms = np.linalg.norm(emb, axis=1)
arts = [str(m.get("article_num")) for m in res["metadatas"]]

print(f"{'вопрос':46} {'как есть':>10} {'без повторов':>13}")
plain = dedup = 0
for q, expect in CASES:
    qv = np.array(rag.db.embeddings.embed_query(q), float)
    sims = emb @ qv / (norms * np.linalg.norm(qv))
    order = np.argsort(-sims)
    top = [int(i) for i in order[:15]]
    seen, clean = set(), []
    for i in top:
        if arts[i] in seen:
            continue
        seen.add(arts[i])
        clean.append(i)
    hit_plain = any(arts[i] in expect for i in top[:5])
    hit_clean = any(arts[i] in expect for i in clean[:5])
    plain += hit_plain
    dedup += hit_clean
    print(f"{q[:46]:46} {'OK' if hit_plain else '—':>10} {'OK' if hit_clean else '—':>13}"
          f"   {[arts[i] for i in clean[:5]]}")

print(f"\nhit@5: как есть {plain}/6 · без повторов {dedup}/6")
