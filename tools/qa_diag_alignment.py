"""Совпадают ли сохранённые векторы со своими текстами.

1) сколько в индексе уникальных векторов (дубли считаются без обращений к API);
2) у случайной выборки документов сверяем сохранённый вектор с пересчитанным.
"""
import os, sys, random, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for line in open(os.path.expanduser("~/.agent-secrets/keys.env"), encoding="utf-8", errors="ignore"):
    if line.startswith("PROXYAPI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
os.environ["OPENAI_BASE_URL"] = "https://api.proxyapi.ru/openai/v1"
os.environ["CHROMA_PERSIST_DIR"] = os.path.join(ROOT, "chroma_legal_db")
os.environ["PRODUCTION"] = "true"
import tiktoken
class _S:
    def _t(self, t): return list(range(max(1, len(t)//4)))
    def encode(self, t, **k): return self._t(t)
    def encode_ordinary(self, t, **k): return self._t(t)
    def encode_batch(self, ts, **k): return [self._t(t) for t in ts]
    def decode(self, t, **k): return ""
tiktoken.get_encoding = lambda n=None: _S()
tiktoken.encoding_for_model = lambda n=None: _S()

import numpy as np
from rag_engine import get_rag_engine

rag = get_rag_engine()
data = rag.db.vector_db.get(include=["embeddings", "documents", "metadatas"])
docs, emb = data["documents"], np.array(data["embeddings"], float)
print("документов:", len(docs))

hashes = {hashlib.md5(np.round(v, 6).tobytes()).hexdigest() for v in emb}
print(f"уникальных векторов: {len(hashes)} из {len(docs)}  → дублей {len(docs)-len(hashes)}")

# дубликаты текстов
tdup = len(docs) - len({d for d in docs})
print(f"уникальных текстов: {len({d for d in docs})}  → повторов текста {tdup}")

random.seed(7)
idx = random.sample(range(len(docs)), 40)
fresh = rag.db.embeddings.embed_documents([docs[i] for i in idx])
bad = []
for i, fv in zip(idx, fresh):
    a, b = emb[i], np.array(fv, float)
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    if c < 0.99:
        bad.append((round(c, 4), len(docs[i]), docs[i][:60].replace("\n", " ")))

print(f"\nвыборка 40 документов: несовпадение (cos < 0.99) у {len(bad)}")
for c, ln, t in bad[:8]:
    print(f"    cos={c:.4f} len={ln:5} {t!r}")
