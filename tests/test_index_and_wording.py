"""Тесты на проверку индекса после сборки (DEF-018) и оговорок (DEF-016).

DEF-018: сборка индекса обязана падать, если векторы не соответствуют текстам.
Именно этот случай произошёл в приёмке: «✅ загрузка завершена успешно»,
а в базе мусор.
DEF-016: список запрещённых фраз больше не молчит — он помечает ответ,
а строгий режим (блокировка) включается флагом.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_legal_db")
os.environ.setdefault("PRODUCTION", "true")

import rag_engine  # noqa: E402
from database import LegalVectorDB  # noqa: E402
from rag_engine import AnswerValidator  # noqa: E402


class FakeDoc:
    def __init__(self, article_num):
        self.metadata = {"article_num": article_num, "reference": f"ст. {article_num}"}
        self.page_content = "текст"


LONG = "Общий срок исковой давности составляет три года и исчисляется со дня, " \
       "когда лицо узнало или должно было узнать о нарушении права. "


# ---------- DEF-018 ----------

class FakeVectorDB:
    def __init__(self, docs, embeddings):
        self._docs, self._emb = docs, embeddings

    def get(self, include=None):
        return {"documents": self._docs, "embeddings": self._emb}


class FakeEmbeddings:
    def __init__(self, output):
        self._out = output

    def embed_documents(self, texts):
        return [self._out for _ in texts]


def make_db(docs, stored, fresh):
    db = LegalVectorDB.__new__(LegalVectorDB)      # без __init__: без Chroma
    db.vector_db = FakeVectorDB(docs, stored)
    db.embeddings = FakeEmbeddings(fresh)
    return db


def test_index_check_passes_when_vectors_match():
    vec = [1.0, 0.0, 0.0]
    db = make_db(["текст"] * 3, [vec] * 3, vec)
    db._verify_embeddings_alignment(sample_size=2)     # не должно бросить


def test_index_check_fails_when_vectors_do_not_match():
    stored = [[1.0, 0.0, 0.0]] * 3
    fresh = [0.0, 1.0, 0.0]                            # косинус 0 — мусор
    db = make_db(["a", "b", "c"], stored, fresh)
    with pytest.raises(RuntimeError, match="вектор не соответствует тексту"):
        db._verify_embeddings_alignment(sample_size=2)


# ---------- DEF-016 ----------

def test_uncertain_wording_is_flagged():
    v = AnswerValidator(None)
    ok, _ = v.validate("вопрос", LONG + "Возможно, статья 454 применяется. См. ст. 454.",
                       [FakeDoc("454")])
    assert ok is True
    assert "uncertain_wording" in v.warnings


def test_uncertain_wording_blocked_in_strict_mode(monkeypatch):
    monkeypatch.setattr(rag_engine, "STRICT_UNCERTAIN_WORDING", True)
    v = AnswerValidator(None)
    ok, err = v.validate("вопрос", LONG + "Возможно, статья 454 применяется. См. ст. 454.",
                         [FakeDoc("454")])
    assert ok is False
    assert v.reason_code == "uncertain_wording"
