"""Переиндексация без tiktoken: текст уходит в API как есть.

OpenAIEmbeddings по умолчанию (check_embedding_ctx_length=True) перед запросом
переводит текст в токены через tiktoken. В офлайне словарь BPE не скачивается,
и любая заглушка вместо него превращает все тексты одинаковой длины
в одинаковый вход для модели — векторы получаются мусорными, без единой ошибки.

Здесь поле выключено: в API уходит исходный текст.
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

key = None
with open(os.path.expanduser("~/.agent-secrets/keys.env"), encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.strip().startswith("PROXYAPI_API_KEY="):
            key = line.strip().split("=", 1)[1].strip()
if not key:
    sys.exit("не нашёл PROXYAPI_API_KEY")

os.environ["OPENAI_API_KEY"] = key
os.environ["OPENAI_BASE_URL"] = "https://api.proxyapi.ru/openai/v1"
os.environ["CHROMA_PERSIST_DIR"] = os.path.join(ROOT, "chroma_legal_db")
os.environ["PRODUCTION"] = "true"

# подменяем класс, которым пользуется database.py, на версию без токенизации
import database as dbmod
from langchain_openai import OpenAIEmbeddings

class _RawText(OpenAIEmbeddings):
    check_embedding_ctx_length: bool = False

dbmod.OpenAIEmbeddings = _RawText

print(f"[индексация] endpoint={os.environ['OPENAI_BASE_URL']}")
print("[индексация] check_embedding_ctx_length=False (текст без токенизации)")

import ingest_data
ingest_data.main()
