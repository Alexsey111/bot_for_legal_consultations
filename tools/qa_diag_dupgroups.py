"""Кто делит один вектор: смотрим самую большую группу."""
import os, sys, hashlib
from collections import defaultdict
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
docs, metas = data["documents"], data["metadatas"]
emb = np.array(data["embeddings"], float)
print("shape:", emb.shape)

groups = defaultdict(list)
for i, v in enumerate(emb):
    groups[hashlib.md5(np.round(v, 6).tobytes()).hexdigest()].append(i)

sizes = sorted((len(g) for g in groups.values()), reverse=True)
print("размеры групп (топ-10):", sizes[:10])
big = max(groups.values(), key=len)
print(f"\nсамая большая группа: {len(big)} документов")
for i in big[:5]:
    print(f"  ст.{metas[i].get('article_num'):8} len={len(docs[i]):5} {docs[i][:70]!r}")
