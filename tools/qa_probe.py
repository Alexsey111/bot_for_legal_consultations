"""Прогон двух вопросов через rag_engine напрямую, без Telegram.

Цель: подтвердить или опровергнуть DEF-001 (общий вопрос падает) и заодно
проверить контрольный случай — прямой запрос статьи.
"""
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from rag_engine import get_rag_engine

QUESTIONS = [
    ("direct", "что говорит статья 454?"),
    ("general", "что будет, если покупатель не оплатил товар?"),
    ("general", "как приготовить борщ"),
]

rag = get_rag_engine()

for kind, q in QUESTIONS:
    print("=" * 70)
    print(f"[{kind}] {q}")
    try:
        ans = rag.generate_answer(q)
        print("OK, длина:", len(ans))
        print(ans[:400].replace("\n", " "))
    except Exception as e:
        print("ИСКЛЮЧЕНИЕ:", type(e).__name__, e)
        traceback.print_exc(limit=3)
