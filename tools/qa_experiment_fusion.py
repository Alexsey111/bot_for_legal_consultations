"""Эксперимент: даёт ли фильтр мусорных чанков + слияние по рангам (RRF) рабочий поиск.

Ничего не меняем в проекте: работаем поверх уже собранного индекса,
деньги на эмбеддинги не тратятся (запросы по 10 вопросов).
"""
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

# вопрос -> какие статьи считаются правильным ответом
CASES = [
    ("что такое договор дарения", {"572"}),
    ("какой срок исковой давности", {"196"}),
    ("можно ли расторгнуть договор аренды досрочно", {"619", "620"}),
    ("что такое договор займа", {"807"}),
    ("что такое неустойка", {"330"}),
    ("что такое договор купли-продажи", {"454"}),
    ("как оформить доверенность", {"185", "185.1"}),
    ("что такое сервитут", {"274"}),
    ("ответственность за вред от источника повышенной опасности", {"1079"}),
    ("что такое публичный договор", {"426"}),
]

rag = get_rag_engine()
data = rag.db.vector_db.get(include=["embeddings", "documents", "metadatas"])
docs, metas, emb = data["documents"], data["metadatas"], np.array(data["embeddings"], float)
norms = np.linalg.norm(emb, axis=1)
arts = [str(m.get("article_num")) for m in metas]
lens = np.array([len(d) for d in docs])

# индекс BM25 строим один раз по тем же документам (langchain BM25Retriever)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
corpus = [Document(page_content=d, metadata=m) for d, m in zip(docs, metas)]
bm25 = BM25Retriever.from_documents(corpus)
bm25.k = 30

def ranks_from_scores(scores, keep=None, k=30):
    order = np.argsort(-scores)
    out = []
    for i in order:
        if keep is not None and not keep[i]:
            continue
        out.append(i)
        if len(out) >= k:
            break
    return out

def hit(ranked_docs, expect):
    return any(arts[i] in expect for i in ranked_docs[:5])

print(f"{'вопрос':52} {'как в боте':>11} {'фильтр+RRF':>11}")
wins = {"base": 0, "fix": 0}
for q, expect in CASES:
    qv = np.array(rag.db.embeddings.embed_query(q), float)
    sims = emb @ qv / (norms * np.linalg.norm(qv))

    # как сейчас: 15 семантических (MMR ~ по score) + BM25, top-5 из начала
    sem_all = ranks_from_scores(sims, k=15)
    bm = [i for i, d in enumerate(docs)
          if d.page_content if False else False]  # placeholder
    bm_idx = []
    for d in bm25.invoke(q):
        # найти индекс по тексту+статье
        for i, (dd, mm) in enumerate(zip(docs, metas)):
            if dd is d.page_content:
                pass
        bm_idx.append(None)
    # BM25 отдаёт объекты Document — сопоставим по (текст, статья)
    key2idx = {}
    for i, (d, m) in enumerate(zip(docs, metas)):
        key2idx.setdefault((d, str(m.get("article_num"))), i)
    bm_idx = [key2idx.get((d.page_content, str(d.metadata.get("article_num")))) for d in bm25.invoke(q)]
    bm_idx = [i for i in bm_idx if i is not None]
    base = sem_all + bm_idx

    # исправление: выкинуть короткие чанки и слить по рангам
    keep = lens >= 300
    sem_ok = ranks_from_scores(sims, keep=keep, k=30)
    bm_ok = [i for i in bm_idx if keep[i]]
    rrf = {}
    for lst in (sem_ok, bm_ok):
        for r, i in enumerate(lst):
            rrf[i] = rrf.get(i, 0) + 1.0 / (60 + r)
    fix = [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])]

    hb, hf = hit(base, expect), hit(fix, expect)
    wins["base"] += hb
    wins["fix"] += hf
    print(f"{q[:52]:52} {'OK' if hb else '—':>11} {'OK' if hf else '—':>11}")

print("\nпопаданий в top-5: как в боте %d/%d, после фильтра+RRF %d/%d"
      % (wins["base"], len(CASES), wins["fix"], len(CASES)))
