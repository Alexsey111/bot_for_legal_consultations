"""Точечное воспроизведение DEF-001 без LLM, без сети и без базы.

Смысл: проверить именно контракт функции retrieve_documents — возвращает ли она
кортеж (docs, context) в ветке общего вопроса. Окружение подменяется заглушками,
поэтому тест детерминированный и бесплатный.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_legal_db")

from rag_engine import LegalRAG


class FakeDoc:
    metadata = {"article_num": "454", "part": "2"}
    page_content = "Текст статьи для проверки контракта."


class FakeDB:
    def get_article_by_number(self, num):
        return [FakeDoc()]


def make_engine():
    rag = LegalRAG.__new__(LegalRAG)      # без __init__: без сети, без Chroma
    rag.db = FakeDB()
    rag._hybrid_retrieve = lambda q: [FakeDoc(), FakeDoc()]
    rag._get_reranker = lambda: None       # production-режим
    rag._build_context = lambda docs, qt=None, show_full=False: "контекст"
    return rag


rag = make_engine()

for qtype in ({"type": "general"}, {"type": "specific_article", "article_num": "454"}):
    result = rag.retrieve_documents("вопрос", qtype)
    kind = type(result).__name__
    ok = isinstance(result, tuple) and len(result) == 2
    print(f"{qtype['type']:18} -> {kind:8} контракт (docs, context): {'ДА' if ok else 'НЕТ'}")

print()
print("Вывод:", "DEF-001 подтверждён: ветка general возвращает None" if
      not isinstance(rag.retrieve_documents("в", {"type": "general"}), tuple) else
      "DEF-001 не подтвердился")
