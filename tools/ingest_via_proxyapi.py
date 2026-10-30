"""Переиндексация ГК РФ с эмбеддингами через proxyapi.

Зачем отдельный скрипт: в ingest_data.py модель берётся из окружения, а .env
проекта указывает на счёт OpenAI, который пуст. Здесь подставляется ключ
proxyapi и локальный путь к Chroma.

Запуск:
    .venv/Scripts/python.exe tools/ingest_via_proxyapi.py
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# ключ читаем точечно: файл секретов содержит строки без '=', парсить их нельзя
key = None
with open(os.path.expanduser("~/.agent-secrets/keys.env"), encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if line.startswith("PROXYAPI_API_KEY="):
            key = line.split("=", 1)[1].strip()

if not key:
    sys.exit("не нашёл PROXYAPI_API_KEY в ~/.agent-secrets/keys.env")

os.environ["OPENAI_API_KEY"] = key
os.environ["OPENAI_BASE_URL"] = "https://api.proxyapi.ru/openai/v1"
os.environ["CHROMA_PERSIST_DIR"] = os.path.join(ROOT, "chroma_legal_db")
os.environ["PRODUCTION"] = "true"

print(f"[индексация] endpoint={os.environ['OPENAI_BASE_URL']}")
print(f"[индексация] каталог={os.environ['CHROMA_PERSIST_DIR']}")

# OpenAIEmbeddings при подсчёте длины входа использует tiktoken, а он тянет
# BPE-словарь из сети (openaipublic.blob.core.windows.net) — в офлайне таймаут.
# Заглушка влияет только на нарезку слишком длинных входов, не на сами векторы.
import tiktoken

class _Stub:
    def _t(self, text): return list(range(max(1, len(text) // 4)))
    def encode(self, text, **kw): return self._t(text)
    def encode_ordinary(self, text, **kw): return self._t(text)
    def encode_batch(self, texts, **kw): return [self._t(t) for t in texts]
    def decode(self, tokens, **kw): return ""

tiktoken.get_encoding = lambda name=None: _Stub()
tiktoken.encoding_for_model = lambda name=None: _Stub()

import ingest_data

ingest_data.main()
