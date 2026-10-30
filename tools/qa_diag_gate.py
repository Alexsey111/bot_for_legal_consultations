"""Почему «кража» всё ещё уточняет: печатаем маршрут, признаки темы и число документов."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import qa_boot  # noqa: F401
from rag_engine import get_rag_engine, QueryClarifier

rag = get_rag_engine()
for q in ["кража", "как приготовить борщ", "дело Иванова", "что такое сервитут"]:
    qt = rag.enhanced_analyzer.analyze(q)
    route, direct = rag.router.route(q, qt)
    rel = rag.check_topic_relevance(q)
    docs, ctx = rag.retrieve_documents(q, qt)
    clar = QueryClarifier.needs_clarification(q, docs) if docs else None
    print(f"\n«{q}»")
    print(f"   маршрут ......... {route} | тип: {qt['type']}")
    print(f"   тема ............ relevant={rel['relevant']} best_cos={rel['best_cosine']:.3f} cov={rel['coverage']:.2f}")
    print(f"   документов ...... {len(docs)}")
    print(f"   уточнение ....... {'ДА' if clar else 'нет'}")
    if docs:
        print(f"   части ГК в top-5: {sorted({d.metadata.get('part') for d in docs[:5]})}")
        print(f"   статьи .......... {[d.metadata.get('article_num') for d in docs]}")
