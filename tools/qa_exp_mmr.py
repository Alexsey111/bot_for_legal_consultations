"""Сравнение семантической ветки: MMR (как сейчас) против обычного топ-15 по близости."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import qa_boot  # noqa: F401
from rag_engine import get_rag_engine, SEMANTIC_TOP_K

CASES = [
    ("дарение что это", {"572"}),
    ("какой срок исковой давности", {"196"}),
    ("можно ли расторгнуть договор аренды досрочно", {"619", "620"}),
    ("что такое договор займа", {"807"}),
    ("наём жилого помещения права нанимателя", {"671"}),
    ("возмещение вреда имуществу", {"1064"}),
]

rag = get_rag_engine()
mmr = rag.semantic_retriever
plain = rag.db.vector_db.as_retriever(
    search_type="similarity", search_kwargs={"k": SEMANTIC_TOP_K}
)

print(f"{'вопрос':44} {'MMR':>6} {'близость':>9}")
h_mmr = h_plain = 0
for q, expect in CASES:
    out = {}
    for name, retr in (("mmr", mmr), ("plain", plain)):
        rag.semantic_retriever = retr
        docs = rag._hybrid_retrieve(q)
        arts = [str(d.metadata.get("article_num")) for d in docs[:5]]
        out[name] = (any(a in expect for a in arts), arts)
    h_mmr += out["mmr"][0]
    h_plain += out["plain"][0]
    print(f"{q[:44]:44} {'OK' if out['mmr'][0] else '—':>6} {'OK' if out['plain'][0] else '—':>9}"
          f"   MMR {out['mmr'][1]} | близость {out['plain'][1]}")

print(f"\nhit@5: MMR {h_mmr}/6 · обычная близость {h_plain}/6")
