"""Тесты валидатора: ссылки в любом падеже и отказ без внутренней терминологии.

DEF-005 — регулярка не видела косвенных падежей («статьёй 196», «статьям 454»).
DEF-013 — ответ без ссылки выбрасывался вместо того, чтобы быть отданным.
DEF-014 — пользователю показывалась внутренняя причина отказа.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_legal_db")
os.environ.setdefault("PRODUCTION", "true")

from rag_engine import AnswerValidator, LegalRAG  # noqa: E402


class FakeDoc:
    def __init__(self, article_num):
        self.metadata = {"article_num": article_num, "reference": f"ст. {article_num} ГК РФ"}
        self.page_content = "текст"


LONG = "Общий срок исковой давности составляет три года и исчисляется со дня, " \
       "когда лицо узнало или должно было узнать о нарушении права. "


@pytest.mark.parametrize("text, expected", [
    ("статья 454", {"454"}),
    ("статьи 454", {"454"}),
    ("статьёй 196", {"196"}),
    ("статьям 454 и 455", {"454", "455"}),
    ("статей 196, 200", {"196", "200"}),
    ("ст. 454", {"454"}),
    ("ст 454", {"454"}),
    ("статья № 454", {"454"}),
    ("статье 123.16", {"123.16"}),
])
def test_article_forms_recognised(text, expected):
    assert AnswerValidator.articles_in(text) == expected


def test_answer_with_indirect_case_passes():
    v = AnswerValidator(None)
    answer = LONG + "Это следует из статьёй 196 ГК РФ."
    ok, err = v.validate("какой срок исковой давности", answer, [FakeDoc("196")])
    assert ok is True and err is None
    assert "no_citation" not in v.warnings


def test_answer_without_citation_is_returned_with_warning():
    v = AnswerValidator(None)
    ok, err = v.validate("что такое дарение", LONG + "Даритель передаёт вещь безвозмездно.",
                         [FakeDoc("572")])
    assert ok is True, "ответ без ссылки нельзя выбрасывать (DEF-013)"
    assert v.reason_code == "no_citation"
    assert "no_citation" in v.warnings


def test_short_answer_still_rejected():
    v = AnswerValidator(None)
    ok, err = v.validate("что такое дарение", "Дарение — договор.", [FakeDoc("572")])
    assert ok is False and v.reason_code == "too_short"


def test_hallucinated_article_still_rejected():
    v = AnswerValidator(None)
    answer = LONG + "Смотрите статьи 1001, 1002, 1003 и 572 ГК РФ."
    ok, err = v.validate("что такое дарение", answer, [FakeDoc("572")])
    assert ok is False
    assert v.reason_code == "hallucination"


def test_refusal_message_has_no_internal_wording():
    for code in ("too_short", "too_many_ellipsis", "hallucination", "no_citation", None):
        msg = LegalRAG._refusal_message(code).lower()
        assert "причина" not in msg, f"внутренняя причина видна пользователю: {code}"
        assert "не содержит ссылок" not in msg
        assert "что можно попробовать" in msg
