"""Сравнение правил отсечения «не моя тема» на размеченном наборе.

Метрики на один вопрос (k=20):
  c1   — косинус лучшего чанка
  cm   — средний косинус top-20 (фон)
  gap  — c1 - cm (насколько пик выше фона)
  cov  — доля значимых слов вопроса, встретившихся в top-5 текстах
"""
import os, sys, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import qa_boot  # noqa: F401
from rag_engine import get_rag_engine

ON = ["что такое договор дарения", "какой срок исковой давности",
      "можно ли расторгнуть договор аренды досрочно", "что такое договор займа",
      "что такое неустойка", "что такое договор купли-продажи",
      "как оформить доверенность", "что такое сервитут",
      "ответственность за вред от источника повышенной опасности",
      "что такое публичный договор", "наём жилого помещения права нанимателя",
      "возмещение вреда имуществу", "как делится наследство по закону",
      "что такое залог и ипотека", "какие бывают юридические лица",
      "что такое оферта и акцепт", "как взыскать долг по расписке",
      "что такое исковая давность", "как продать квартиру", "что такое неустойка по договору"]

OFF = ["как приготовить борщ", "какая погода завтра в Барнауле",
       "кто выиграл чемпионат мира по футболу", "кража", "дело Иванова",
       "напиши код на питоне для сортировки списка", "как настроить роутер",
       "что такое любовь", "сколько стоит билет в Москву", "посоветуй фильм на вечер",
       "как похудеть за месяц", "что делать если болит голова"]

STOP = set("что такое как какие какой какая кто где когда можно ли это для или по на в не"
           "мне мой моя мои его её их мы вы они ты я".split())

rag = get_rag_engine()


def metrics(q):
    res = rag.db.vector_db.similarity_search_with_score(q, k=20)
    cs = [1.0 - sc / 2.0 for _, sc in res]
    c1, cm = cs[0], sum(cs) / len(cs)
    words = [w for w in re.findall(r"[а-яёa-z]{4,}", q.lower()) if w not in STOP]
    joined = " ".join(d.page_content.lower() for d, _ in res[:5])
    cov = sum(1 for w in words if w[:5] in joined) / len(words) if words else 0.0
    return c1, cm, c1 - cm, cov


RULES = {
    "A: c1 < 0.45": lambda c1, cm, gap, cov: c1 < 0.45,
    "B: gap < 0.06": lambda c1, cm, gap, cov: gap < 0.06,
    "C: cov < 0.34": lambda c1, cm, gap, cov: cov < 0.34,
    "D: gap<0.06 и cov<0.6": lambda c1, cm, gap, cov: gap < 0.06 and cov < 0.6,
    "E: c1<0.33 или gap<0.05": lambda c1, cm, gap, cov: c1 < 0.33 or gap < 0.05,
    "F: c1<0.33 или (c1<0.50 и gap<0.08)": lambda c1, cm, gap, cov: c1 < 0.33 or (c1 < 0.50 and gap < 0.08),
}

print(f"{'вопрос':50} {'c1':>5} {'cm':>5} {'gap':>6} {'cov':>5}")
data = {}
for label, qs in (("ON", ON), ("OFF", OFF)):
    for q in qs:
        m = metrics(q)
        data[q] = (label, m)
        print(f"{q[:50]:50} {m[0]:5.3f} {m[1]:5.3f} {m[2]:6.3f} {m[3]:5.2f}")

print()
for name, rule in RULES.items():
    fp = [q for q, (lab, m) in data.items() if lab == "ON" and rule(*m)]
    tn = [q for q, (lab, m) in data.items() if lab == "OFF" and rule(*m)]
    print(f"{name:38} откажет своим: {len(fp):2}/{len(ON)}  поймает чужих: {len(tn):2}/{len(OFF)}")
    if fp:
        print(f"{'':40}ошибочно откажет: {', '.join(fp[:4])}")
    if len(tn) < len(OFF):
        print(f"{'':40}пропустит: {', '.join(q for q in OFF if q not in tn)}")
