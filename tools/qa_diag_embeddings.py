"""Проверка: совпадает ли модель эмбеддингов в коллекции с моделью на запросе."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for line in open(os.path.expanduser("~/.agent-secrets/keys.env"), encoding="utf-8", errors="ignore"):
    if line.startswith("PROXYAPI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
os.environ.setdefault("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
os.environ["CHROMA_PERSIST_DIR"] = os.path.join(ROOT, "chroma_legal_db")
os.environ.setdefault("PRODUCTION", "true")

import tiktoken
class _S:
    def _t(self, t): return list(range(max(1, len(t)//4)))
    def encode(self, t, **k): return self._t(t)
    def encode_ordinary(self, t, **k): return self._t(t)
    def encode_batch(self, ts, **k): return [self._t(t) for t in ts]
    def decode(self, t, **k): return ""
tiktoken.get_encoding = lambda name=None: _S()
tiktoken.encoding_for_model = lambda name=None: _S()

import numpy as np
from rag_engine import get_rag_engine

rag = get_rag_engine()
db = rag.db
coll = db.vector_db._collection
print("метаданные коллекции:", db.vector_db._collection.metadata)
print("документов:", coll.count())

# берём документ статьи 572 (дарение), если есть
res = coll.get(where={"article_num": "572"}, limit=1, include=["documents", "embeddings", "metadatas"])
if not res["ids"]:
    res = coll.get(limit=1, include=["documents", "embeddings", "metadatas"])
text = res["documents"][0]
stored = np.array(res["embeddings"][0], dtype=float)
print("\nдокумент:", str(res["metadatas"][0])[:120])
print("текст:", text[:120].replace("\n", " "))

fresh = np.array(db.embeddings.embed_documents([text])[0], dtype=float)
cos = float(np.dot(stored, fresh) / (np.linalg.norm(stored) * np.linalg.norm(fresh)))
print(f"\nразмерность: в хранилище {stored.shape[0]}, у новой модели {fresh.shape[0]}")
print(f"норма сохранённого вектора: {np.linalg.norm(stored):.4f}")
print(f"косинус(сохранённый вектор, пересчитанный тем же текстом): {cos:.4f}")
print("ВЫВОД:", "модель эмбеддингов та же" if cos > 0.95 else "ВЕКТОРЫ НЕСОВМЕСТИМЫ — коллекция собрана другой моделью")

# насколько близок запрос к «своей» статье
q = np.array(db.embeddings.embed_query("дарение что это"), dtype=float)
print(f"\nкосинус(запрос «дарение», сохранённый вектор ст.572): "
      f"{float(np.dot(q, stored)/(np.linalg.norm(q)*np.linalg.norm(stored))):.4f}")
