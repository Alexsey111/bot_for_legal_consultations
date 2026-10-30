"""Проверка гипотезы: сам endpoint возвращает одинаковые векторы на разные тексты."""
import os, sys, json, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
key = None
for line in open(os.path.expanduser("~/.agent-secrets/keys.env"), encoding="utf-8", errors="ignore"):
    if line.startswith("PROXYAPI_API_KEY="):
        key = line.split("=", 1)[1].strip()
os.environ["OPENAI_API_KEY"] = key
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
data = rag.db.vector_db.get(include=["documents", "metadatas"])
docs, metas = data["documents"], data["metadatas"]

# берём 5 текстов одинаковой длины из одной группы индекса
pick = [i for i, d in enumerate(docs) if len(d) == 297][:5]
texts = [docs[i] for i in pick]
print("выбраны статьи:", [metas[i].get("article_num") for i in pick])
for t in texts:
    print(f"   len={len(t)} {t[:60]!r}...")


def embed_batch(ts):
    req = urllib.request.Request(
        "https://api.proxyapi.ru/openai/v1/embeddings",
        data=json.dumps({"model": "text-embedding-3-small", "input": ts}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    return [np.array(x["embedding"], float) for x in d["data"]]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\n--- один пакет, 5 разных текстов ---")
vs = embed_batch(texts)
for i in range(len(vs)):
    for j in range(i + 1, len(vs)):
        c = cos(vs[i], vs[j])
        if c > 0.9:
            print(f"    cos({i},{j}) = {c:.4f}  <-- одинаковые")

print("\n--- те же тексты по одному ---")
solo = [embed_batch([t])[0] for t in texts]
print(f"    cos(пакет[0], одиночный[0]) = {cos(vs[0], solo[0]):.4f}")
print(f"    cos(одиночный[0], одиночный[1]) = {cos(solo[0], solo[1]):.4f}")

print("\n--- через langchain-клиент проекта (как на индексации) ---")
lc = rag.db.embeddings.embed_documents(texts)
print(f"    cos(lc[0], lc[1]) = {cos(np.array(lc[0], float), np.array(lc[1], float)):.4f}")
print(f"    cos(lc[0], пакет[0]) = {cos(np.array(lc[0], float), vs[0]):.4f}")
