"""Общая затравка для прогонов: ключ proxyapi, локальный Chroma, эмбеддинги без tiktoken.

Импортировать первым делом:
    import qa_boot  # noqa: F401
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_key = None
with open(os.path.expanduser("~/.agent-secrets/keys.env"), encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.strip().startswith("PROXYAPI_API_KEY="):
            _key = line.strip().split("=", 1)[1].strip()

os.environ["OPENAI_API_KEY"] = _key or ""
os.environ["OPENAI_BASE_URL"] = "https://api.proxyapi.ru/openai/v1"
os.environ.setdefault("CHROMA_PERSIST_DIR", os.path.join(ROOT, "chroma_legal_db"))
os.environ.setdefault("PRODUCTION", "true")

# ВАЖНО: OpenAIEmbeddings по умолчанию токенизирует вход через tiktoken.
# Без словаря BPE (офлайн) любой суррогат вместо него делает все тексты
# одинаковой длины одинаковым входом для модели → мусорные векторы.
# Поэтому везде работаем с отправкой исходного текста.
import tiktoken


class _StubEncoding:
    """«Токены» как длина/4 — только для оценки размера контекста в rag_engine."""
    def _t(self, text):
        return list(range(max(1, len(text) // 4)))
    def encode(self, text, **kw):
        return self._t(text)
    def encode_ordinary(self, text, **kw):
        return self._t(text)
    def encode_batch(self, texts, **kw):
        return [self._t(t) for t in texts]
    def decode(self, tokens, **kw):
        return ""


tiktoken.get_encoding = lambda name=None: _StubEncoding()
tiktoken.encoding_for_model = lambda name=None: _StubEncoding()

import database as _dbmod
from langchain_openai import OpenAIEmbeddings


class _RawText(OpenAIEmbeddings):
    check_embedding_ctx_length: bool = False


_dbmod.OpenAIEmbeddings = _RawText
