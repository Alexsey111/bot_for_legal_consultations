"""Тесты порога темы (DEF-002, DEF-017): «это вообще про ГК?».

Без сети и без Chroma: подменяем векторный поиск.
Chroma лежит в L2-пространстве при единичных векторах, поэтому cos = 1 - dist/2.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_legal_db")
os.environ.setdefault("PRODUCTION", "true")


class FakeDoc:
    def __init__(self, text):
        self.page_content = text


class FakeVectorDB:
    def __init__(self, hits):
        self._hits = hits          # [(текст, расстояние Chroma), ...]

    def similarity_search_with_score(self, question, k=None):
        return [(FakeDoc(t), d) for t, d in self._hits]


class FakeDB:
    def __init__(self, hits):
        self.vector_db = FakeVectorDB(hits)


def make_engine(hits):
    from rag_engine import LegalRAG

    rag = LegalRAG.__new__(LegalRAG)
    rag.db = FakeDB(hits)
    return rag


def test_far_documents_are_refused():
    # cos = 1 - 1.30/2 = 0.35 < 0.45 и ни одного слова из вопроса в тексте
    engine = make_engine([("Статья 333. Уменьшение неустойки", 1.30)] * 3)
    res = engine.check_topic_relevance("как приготовить борщ")
    assert res["relevant"] is False
    assert res["coverage"] == 0.0


def test_very_far_documents_are_refused_even_with_word_overlap():
    # cos = 1 - 1.60/2 = 0.20 < 0.33 — жёсткий отказ независимо от слов
    engine = make_engine([("Статья 210. Кража в тексте кодекса", 1.60)] * 3)
    res = engine.check_topic_relevance("кража")
    assert res["best_cosine"] == pytest.approx(0.20, abs=1e-6)
    assert res["relevant"] is False


def test_near_documents_pass():
    # cos = 1 - 0.60/2 = 0.70 — уверенно своя тема
    engine = make_engine([("Статья 572. Договор дарения", 0.60)] * 3)
    res = engine.check_topic_relevance("что такое договор дарения")
    assert res["relevant"] is True


def test_middle_distance_saved_by_word_overlap():
    # cos = 0.40: ниже мягкого порога, но слово вопроса есть в тексте
    engine = make_engine([("Статья 274. Содержание сервитута", 1.20)] * 3)
    res = engine.check_topic_relevance("что такое сервитут")
    assert res["coverage"] == 1.0
    assert res["relevant"] is True


def test_middle_distance_without_word_overlap_is_refused():
    engine = make_engine([("Статья 243. Конфискация", 1.20)] * 3)
    res = engine.check_topic_relevance("дело Иванова")
    assert res["coverage"] == 0.0
    assert res["relevant"] is False


def test_probe_failure_fails_open():
    class BrokenDB:
        class vector_db:
            @staticmethod
            def similarity_search_with_score(question, k=None):
                raise RuntimeError("Chroma недоступна")

    from rag_engine import LegalRAG

    rag = LegalRAG.__new__(LegalRAG)
    rag.db = BrokenDB()
    res = rag.check_topic_relevance("что такое неустойка")
    assert res["relevant"] is True, "проверка темы не должна ронять ответ"


def test_only_stopwords_delegated_to_clarifier():
    engine = make_engine([("Статья 1. Основные начала", 1.30)] * 3)
    res = engine.check_topic_relevance("можно ли")
    assert res["relevant"] is True


def test_other_branch_refused_even_if_close():
    # «какое наказание за кражу»: по косинусу похоже на свои вопросы (0.40),
    # но это уголовное право, а не ГК
    engine = make_engine([("Статья 1064. Общие основания ответственности", 1.20)] * 3)
    res = engine.check_topic_relevance("какое наказание за кражу")
    assert res["relevant"] is False
    assert res.get("reason") == "other_branch"


def test_civil_aspect_with_crime_word_passes():
    # гражданские последствия того же события — это ГК, отказывать нельзя
    engine = make_engine([("Статья 1064. Возмещение вреда", 0.70)] * 3)
    res = engine.check_topic_relevance("как взыскать ущерб от кражи")
    assert res["relevant"] is True
