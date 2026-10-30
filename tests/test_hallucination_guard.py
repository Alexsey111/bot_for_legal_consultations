"""Тесты ссылок вне контекста: выдуманная статья не должна проходить.

Раньше правило было «до двух выдуманных статей можно, если рядом есть верная»
— то есть ответ с придуманной нормой уходил пользователю.
Разделяем: статьи нет в кодексе — отказ; статья есть, но не в выборке — пометка.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_legal_db")
os.environ.setdefault("PRODUCTION", "true")

from rag_engine import AnswerValidator  # noqa: E402

LONG = "Общий срок исковой давности составляет три года и исчисляется со дня, " \
       "когда лицо узнало или должно было узнать о нарушении права. "


class FakeDoc:
    def __init__(self, article_num):
        self.metadata = {"article_num": article_num, "reference": f"ст. {article_num} ГК РФ"}
        self.page_content = "текст"


class FakeCode:
    """Кодекс, в котором есть только перечисленные статьи."""
    def __init__(self, numbers):
        self.numbers = set(numbers)

    def get_article_by_number(self, num):
        return [FakeDoc(num)] if str(num) in self.numbers else []


def test_invented_article_rejected_even_next_to_real_one():
    v = AnswerValidator(FakeCode({"454"}))
    ok, err = v.validate("что такое купля-продажа",
                         LONG + "Смотрите статью 454 и статью 999.", [FakeDoc("454")])
    assert ok is False
    assert v.reason_code == "hallucination"
    assert "999" in err


def test_real_article_outside_context_is_allowed_with_warning():
    v = AnswerValidator(FakeCode({"454", "15"}))
    ok, err = v.validate("что такое купля-продажа",
                         LONG + "Смотрите статью 454 и статью 15.", [FakeDoc("454")])
    assert ok is True, "реальная статья кодекса — не выдумка"
    assert "cited_outside_context" in v.warnings


def test_check_failure_does_not_blame_the_model():
    class BrokenCode:
        def get_article_by_number(self, num):
            raise RuntimeError("Chroma недоступна")

    v = AnswerValidator(BrokenCode())
    ok, _ = v.validate("что такое купля-продажа",
                       LONG + "Смотрите статью 454.", [FakeDoc("454")])
    assert ok is True
