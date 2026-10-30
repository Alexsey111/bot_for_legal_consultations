"""Тесты уточняющего вопроса: он не должен мешать нормальному вопросу.

Раньше правило «документы из 3+ частей ГК → уточнить» срабатывало и на
«можно ли расторгнуть договор аренды досрочно»: поиск честно приносит статьи
из разных частей, а человек вместо ответа получал встречный вопрос.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_legal_db")
os.environ.setdefault("PRODUCTION", "true")

from rag_engine import QueryClarifier  # noqa: E402


class Doc:
    def __init__(self, part):
        self.metadata = {"part": part}


def test_long_question_with_diverse_parts_is_answered():
    docs = [Doc(1), Doc(2), Doc(3), Doc(2), Doc(1)]
    q = "можно ли расторгнуть договор аренды досрочно"
    assert QueryClarifier.needs_clarification(q, docs) is None


def test_short_generic_question_asks_for_details():
    docs = [Doc(1), Doc(2), Doc(3)]
    q = "как составить договор"
    assert QueryClarifier.needs_clarification(q, docs) is not None


def test_short_question_with_diverse_parts_asks_for_details():
    docs = [Doc(1), Doc(2), Doc(3)]
    assert QueryClarifier.needs_clarification("аренда и залог", docs) is not None


def test_no_documents_means_no_clarification():
    assert QueryClarifier.needs_clarification("как приготовить борщ", []) is None
