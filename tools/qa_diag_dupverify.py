"""Точная проверка: одинаковые векторы у разных текстов — реальны или артефакт хеша."""
import os, sys, json, urllib.request, hashlib
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
key = None
for line in open(os.path.expanduser("~/.agent-secrets/keys.env"), encoding="utf-8", errors="ignore"):
    if line.startswith("PROXYAPI_API_KEY="):
        key = line.split("=", 1)[1].strip()
os.environ["OPENAI_API_KEY"] = key or ""
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
docs, metas = data["documents"], data["metadatas"]
emb = np.array(data["embeddings"], float)

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def embed(texts):
    req = urllib.request.Request(
        "https://api.proxyapi.ru/openai/v1/embeddings",
        data=json.dumps({"model": "text-embedding-3-small", "input": texts}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return [np.array(x["embedding"], float) for x in json.load(r)["data"]]

groups = defaultdict(list)
for i, v in enumerate(emb):
    groups[hashlib.md5(np.round(v, 6).tobytes()).hexdigest()].append(i)
big = max(groups.values(), key=len)[:4]

print("члены группы:", [metas[i].get("article_num") for i in big])
print("\nкосинусы между сохранёнными векторами членов группы:")
for a in range(len(big)):
    for b in range(a + 1, len(big)):
        print(f"   cos(ст.{metas[big[a]].get('article_num')}, ст.{metas[big[b]].get('article_num')}) = "
              f"{cos(emb[big[a]], emb[big[b]]):.6f}")

fresh = embed([docs[i] for i in big])
print("\nсохранённый против пересчитанного по тому же тексту:")
for i, fv in zip(big, fresh):
    print(f"   ст.{metas[i].get('article_num'):8} cos = {cos(emb[i], fv):.4f}")

print("\nпересчитанные между собой:")
for a in range(len(fresh)):
    for b in range(a + 1, len(fresh)):
        print(f"   cos(fresh[{a}], fresh[{b}]) = {cos(fresh[a], fresh[b]):.4f}")

print("\nполные тексты двух членов группы:")
print(repr(docs[big[0]]))
print(repr(docs[big[1]]))
