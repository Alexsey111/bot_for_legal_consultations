"""Что за тексты приходят по запросу «какой срок исковой давности»."""
import os, sys
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
res = rag.db.vector_db.similarity_search_with_score("какой срок исковой давности", k=3)
for doc, score in res:
    print("=" * 70)
    print(f"dist={score:.4f} ст.{doc.metadata.get('article_num')} ref={doc.metadata.get('reference')}")
    print("текст:", doc.page_content[:300].replace("\n", " "))

print("\n=== сколько документов реально про исковую давность ===")
allm = rag.db.vector_db.get(include=["metadatas", "documents"])
cnt = sum(1 for d in allm["documents"] if "исковой давност" in d.lower())
print("документов со словами «исковой давност»:", cnt, "из", len(allm["documents"]))
print("статьи:", sorted({m.get("article_num") for m, d in zip(allm["metadatas"], allm["documents"]) if "исковой давност" in d.lower()})[:12])

# прямая проверка: косинус запроса и текста статьи 196 из хранилища
import numpy as np
qv = np.array(rag.db.embeddings.embed_query("какой срок исковой давности"), dtype=float)
for art in ("196", "333"):
    idx = [i for i, m in enumerate(allm["metadatas"]) if str(m.get("article_num")) == art][:1]
    if not idx: 
        print(f"ст.{art}: нет в хранилище"); continue
    i = idx[0]
    dv = np.array(rag.db.embeddings.embed_documents([allm["documents"][i]])[0], dtype=float)
    c = float(qv @ dv / (np.linalg.norm(qv) * np.linalg.norm(dv)))
    print(f"cos(запрос, текст ст.{art}) = {c:.4f} | начало текста: {allm['documents'][i][:80]!r}")
