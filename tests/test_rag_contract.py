"""Регрессионный тест на контракт поиска (дефект DEF-001).

Проверяет ровно то, чего не хватало: в ветке общего вопроса
LegalRAG.retrieve_documents обязан вернуть кортеж (docs, context),
а не None. Тест не ходит в сеть и не требует ChromaDB.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_legal_db")
os.environ.setdefault("PRODUCTION", "true")


class FakeDoc:
    metadata = {"article_num": "454", "part": "2"}
    page_content = "Текст статьи для проверки контракта."


class FakeDB:
    def get_article_by_number(self, num):
        return [FakeDoc()]


@pytest.fixture()
def engine():
    from rag_engine import LegalRAG

    rag = LegalRAG.__new__(LegalRAG)          # без __init__: без сети и без Chroma
    rag.db = FakeDB()
    rag._hybrid_retrieve = lambda question: [FakeDoc(), FakeDoc()]
    rag._get_reranker = lambda: None           # production-режим: без реранкера
    rag._build_context = lambda docs, query_type=None, show_full=False: "контекст"
    return rag


def test_general_returns_tuple(engine):
    result = engine.retrieve_documents("что будет, если покупатель не оплатил товар?",
                                       {"type": "general"})
    assert isinstance(result, tuple), f"ожидался кортеж, получено {type(result).__name__}"
    assert len(result) == 2
    docs, context = result
    assert docs and isinstance(context, str)


def test_specific_article_returns_tuple(engine):
    result = engine.retrieve_documents("статья 454", {"type": "specific_article",
                                                      "article_num": "454"})
    assert isinstance(result, tuple) and len(result) == 2


def test_general_without_docs_returns_empty(engine):
    engine._hybrid_retrieve = lambda question: []
    docs, context = engine.retrieve_documents("вопрос", {"type": "general"})
    assert docs == [] and context == ""
