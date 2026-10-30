"""Проверка эмбеддингов proxyapi: детерминированы ли они и различают ли смысл."""
import os, sys, json, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
key = None
for line in open(os.path.expanduser("~/.agent-secrets/keys.env"), encoding="utf-8", errors="ignore"):
    if line.startswith("PROXYAPI_API_KEY="):
        key = line.split("=", 1)[1].strip()

import numpy as np

def embed(texts, model="text-embedding-3-small"):
    req = urllib.request.Request(
        "https://api.proxyapi.ru/openai/v1/embeddings",
        data=json.dumps({"model": model, "input": texts}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return [np.array(x["embedding"], dtype=float) for x in d["data"]]

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

same = "какой срок исковой давности"
print("--- один и тот же текст, три отдельных запроса ---")
v1 = embed([same])[0]
v2 = embed([same])[0]
v3 = embed([same])[0]
print(f"cos(1,2) = {cos(v1,v2):.4f}")
print(f"cos(1,3) = {cos(v1,v3):.4f}")
print(f"норма = {np.linalg.norm(v1):.4f}")

print("\n--- один запрос, текст дважды в одном пакете ---")
b = embed([same, same])
print(f"cos(batch[0], batch[1]) = {cos(b[0], b[1]):.4f}")

print("\n--- разные тексты ---")
a, c, d = embed(["какой срок исковой давности", "как приготовить борщ", "договор дарения"])
print(f"cos(срок давности, борщ) = {cos(a,c):.4f}")
print(f"cos(срок давности, дарение) = {cos(a,d):.4f}")
print(f"cos(борщ, дарение) = {cos(c,d):.4f}")
