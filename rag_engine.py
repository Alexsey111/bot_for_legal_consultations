"""
rag_engine.py
Оптимизированный RAG Engine для ГК РФ с гибридным поиском
Версия с исправлениями критических багов и оптимизаций
"""

import re
import os
import hashlib
from typing import List, Tuple, Optional, Dict, Any
from collections import OrderedDict, defaultdict
from threading import Lock

import tiktoken
import structlog

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
# from sentence_transformers import CrossEncoder

from database import LegalVectorDB

log = structlog.get_logger()

# pip show langchain-core

# ================= CONFIG =================

SEMANTIC_TOP_K = 15
BM25_TOP_K = 15
RERANK_CANDIDATES = 25  # Количество документов для reranking
RERANK_TOP_K = 5        # Финальное количество после reranking
RERANK_SCORE_THRESHOLD = 0.3  # Минимальный score для включения документа

# ===== Порог «это вообще про ГК?» (DEF-002, DEF-017) =====
# Измерено на 20 вопросах по ГК и 12 посторонних (tools/qa_threshold_rules.py):
#   свои вопросы:  лучший косинус 0.376 … 0.719
#   чужие вопросы: лучший косинус 0.177 … 0.440
# Один порог по косинусу разделяет плохо, поэтому два правила:
#   1) лучший чанк совсем далёк (< 0.33) — отказ без разговоров;
#   2) чанк средний (< 0.45) и ни одно значимое слово вопроса не встречается
#      в найденных текстах — тоже отказ.
# На размеченном наборе: 0 ложных отказов своим, 12 из 12 чужих отбито.
RELEVANCE_HARD_COSINE = float(os.getenv("RELEVANCE_HARD_COSINE", "0.33"))
RELEVANCE_SOFT_COSINE = float(os.getenv("RELEVANCE_SOFT_COSINE", "0.45"))
RELEVANCE_MIN_COVERAGE = float(os.getenv("RELEVANCE_MIN_COVERAGE", "0.34"))
RELEVANCE_PROBE_K = 20  # сколько чанков смотреть при проверке темы

# DEF-016: список запрещённых фраз раньше только писал в лог, то есть не работал
# вовсе. По умолчанию он теперь помечает ответ (пометка видна пользователю),
# а блокировка включается флагом — потому что выбросить верный ответ из-за
# одного слова «возможно» хуже, чем показать его с оговоркой. Решение
# о строгом режиме — за владельцем продукта.
STRICT_UNCERTAIN_WORDING = os.getenv("STRICT_UNCERTAIN_WORDING", "0") == "1"

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2
MAX_TOKENS = 1500

MAX_CACHE_SIZE = 1000
MAX_CACHE_ANSWER_LENGTH = 10000  # Максимальная длина кешируемого ответа (10KB)

# Стратегии обработки контекста
MAX_CONTEXT_TOKENS = 6000  # Ограничение для GPT-4o-mini (8K context window)
SHORT_ARTICLE_THRESHOLD = 1500  # До этой длины - всегда полный текст
LONG_ARTICLE_SUMMARY_LENGTH = 800  # Для очень длинных статей

SYSTEM_PROMPT = """
Ты юридический консультант по Гражданскому кодексу РФ.

🚨 КРИТИЧЕСКИ ВАЖНО - АНТИ-ГАЛЛЮЦИНАЦИЯ:
• Упоминай ТОЛЬКО статьи из предоставленного контекста ниже
• Если статьи нет в контексте - НЕ упоминай её вообще
• Лучше сказать "информации недостаточно", чем выдумать статью
• Проверяй: каждая упомянутая статья ДОЛЖНА быть в разделе "Контекст" ниже

⚠️ РЕЖИМЫ ЦИТИРОВАНИЯ:

🎯 РЕЖИМ 1 - Конкретная статья (пользователь запросил "статья X"):
• ОБЯЗАТЕЛЬНО цитируй дословно с кавычками «...»
• Включай ВСЕ важные формулировки
• Формат: «ТОЧНЫЙ ТЕКСТ» (ст. X п. Y ГК РФ)

📚 РЕЖИМ 2 - Общий вопрос (пользователь спросил про ситуацию):
• Можешь пересказывать своими словами
• Но ВСЕГДА указывай источник: (см. ст. X п. Y ГК РФ)
• Если цитируешь - используй кавычки «...»
• Если контекст обрезан - так и скажи: "Статья содержит дополнительные условия, для полного текста используйте команду /article_X"

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленных статей ГК РФ в контексте ниже
2. НЕ упоминай статьи, которых нет в контексте
3. Если информации недостаточно - скажи: "В предоставленных статьях нет информации по этому вопросу. Попробуйте переформулировать вопрос."
4. Объясняй простым языком ПОСЛЕ ссылки на закон

ФОРМАТ ОТВЕТА:

🎯 Для запроса конкретной статьи:
📋 Краткая суть
📌 Текст закона:
«ДОСЛОВНАЯ ЦИТАТА» (ст. X п. Y ГК РФ)
💬 Простыми словами:
[объяснение]

📚 Для общего вопроса:
📋 Прямой ответ на вопрос
📌 Что говорит закон:
По закону [краткое изложение] (ст. X п. Y ГК РФ).
Либо (если нужна точная формулировка):
«ЦИТАТА» (ст. X п. Y ГК РФ)
💡 Практический совет:
[как применить]
⚖️ Правовая база:
• Статья X: [суть]

💡 ВАЖНО:
Если статья обрезана или нужен полный текст - сообщи:
"⚠️ Показан сокращенный контекст. Для полного текста используйте: /article_X"

НЕ ДЕЛАЙ:
- Не придумывай формулировки
- Не ссылайся на статьи, которых НЕТ в контексте ниже
- Не копируй весь текст длинных статей без запроса
- НЕ упоминай статью 8 или любую другую, если её нет в контексте!

ПЕРЕД ОТВЕТОМ:
Проверь список статей в контексте. Упоминай ТОЛЬКО их!
"""


# ================= CACHE =================

class ResponseCache:
    """
    Умный кеш с нормализацией запросов (thread-safe)
    
    ✅ ИСПРАВЛЕНО: Добавлено ограничение по памяти для предотвращения OOM
    """

    def __init__(
        self, 
        max_size: int = MAX_CACHE_SIZE,
        max_memory_mb: int = 50  # ✅ НОВОЕ: Ограничение по памяти
    ):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.current_memory_bytes = 0  # ✅ НОВОЕ: Отслеживание памяти
        self.hits = 0
        self.misses = 0
        self._lock = Lock()

    def _normalize_query(self, query: str) -> str:
        """Нормализует запрос для улучшения попаданий в кеш"""
        query = query.lower().strip()
        query = re.sub(r'\s+', ' ', query)
        query = re.sub(r'[?!.]+$', '', query)
        query = query.replace('гк рф', 'гражданский кодекс')
        query = query.replace('ст.', 'статья')
        query = query.replace('п.', 'пункт')
        return query

    def _get_key(self, query: str) -> str:
        """Создает хеш ключ"""
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        """Получить из кеша (thread-safe)"""
        key = self._get_key(query)
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                hit_rate = self.hits / (self.hits + self.misses) * 100
                log.info("cache_hit", hit_rate=f"{hit_rate:.1f}%")
                return self.cache[key]
            self.misses += 1
            return None

    def set(self, query: str, answer: str):
        """
        Сохранить в кеш (thread-safe)

        ✅ ИСПРАВЛЕНО:
        - Ограничение размера кешируемого ответа
        - Отслеживание используемой памяти
        - Eviction при превышении лимита памяти
        - Защита от OOM
        """
        key = self._get_key(query)
        
        # ================= MEMORY SAFETY =================
        # Вычисляем размер в байтах (UTF-8)
        answer_size_bytes = len(answer.encode('utf-8'))
        
        # Ограничиваем размер отдельного ответа
        if answer_size_bytes > MAX_CACHE_ANSWER_LENGTH:
            log.warning(
                "cache_answer_truncated",
                answer_length=answer_size_bytes,
                max_length=MAX_CACHE_ANSWER_LENGTH
            )
            answer = answer[:MAX_CACHE_ANSWER_LENGTH] + "\n\n[... cached excerpt ...]"
            answer_size_bytes = len(answer.encode('utf-8'))
        
        with self._lock:
            # ✅ НОВОЕ: Eviction по памяти
            # Освобождаем место пока не будет достаточно памяти
            max_memory_bytes = self.max_memory_mb * 1024 * 1024
            
            while (
                len(self.cache) >= self.max_size or 
                self.current_memory_bytes + answer_size_bytes > max_memory_bytes
            ):
                if not self.cache:
                    break
                
                # Удаляем самый старый элемент (LRU)
                old_key, old_value = self.cache.popitem(last=False)
                old_size = len(old_value.encode('utf-8'))
                self.current_memory_bytes -= old_size
                
                log.debug(
                    "cache_evicted",
                    key=old_key[:16],
                    freed_bytes=old_size,
                    current_memory_mb=f"{self.current_memory_bytes / (1024*1024):.2f}"
                )
            
            # Добавляем новый элемент
            self.cache[key] = answer
            self.current_memory_bytes += answer_size_bytes

    def stats(self) -> dict:
        """
        Статистика кеша (thread-safe)
        
        ✅ ИСПРАВЛЕНО: Добавлена информация о памяти
        """
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            memory_mb = self.current_memory_bytes / (1024 * 1024)
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "saved_requests": self.hits,
                "memory_mb": f"{memory_mb:.2f}",  # ✅ НОВОЕ
                "max_memory_mb": self.max_memory_mb,  # ✅ НОВОЕ
                "memory_usage_percent": f"{(memory_mb / self.max_memory_mb * 100):.1f}%"  # ✅ НОВОЕ
            }
    
    def invalidate_user_queries(self, user_id: int):
        """
        ✅ НОВОЕ: Инвалидация кеша для конкретного пользователя (GDPR)
        
        Используется при удалении пользователя для очистки его запросов из кеша
        """
        with self._lock:
            # Ищем ключи которые могут содержать user_id
            # (это эвристика, т.к. в ключе только hash запроса)
            # Для точной инвалидации нужно хранить mapping user_id -> cache_keys
            
            # УПРОЩЕННАЯ ВЕРСИЯ: Очищаем весь кеш
            # В production лучше хранить {user_id: [cache_keys]} mapping
            log.warning(
                "cache_invalidation_full",
                reason="user_deletion",
                user_id=user_id,
                cleared_entries=len(self.cache)
            )
            
            old_size = len(self.cache)
            self.cache.clear()
            self.current_memory_bytes = 0
            
            log.info(f"✅ Cache cleared for user {user_id}: {old_size} entries removed")


# ================= ENHANCED QUERY ANALYZER =================

class EnhancedQueryAnalyzer:
    """Продвинутый анализатор запросов с полным парсингом"""

    @staticmethod
    def analyze(query: str) -> Dict[str, Any]:
        """
        Расширенный анализ запроса
        Returns: {
            "type": "specific_article" | "article_range" | "general",
            "article_num": "454",
            "article_range": ["454", "455", "456"],
            "point_num": "1",
            "subpoint": "а",
            "paragraph": "2",
            "part_gk": 2  # Часть ГК РФ
        }
        """
        query_lower = query.lower()
        result = {"type": "general"}

        # ========== ЧАСТЬ ГК РФ ==========
        part_match = re.search(r'част[ьи]\s+(\d)', query_lower)
        if part_match:
            result["part_gk"] = int(part_match.group(1))

        # ========== ДИАПАЗОН СТАТЕЙ ==========
        # "статьи 454-456", "ст. 100-105"
        range_patterns = [
            r'стать[ияюе]\s+(\d+)\s*[-–—]\s*(\d+)',
            r'ст\.?\s*(\d+)\s*[-–—]\s*(\d+)'
        ]
        for pattern in range_patterns:
            match = re.search(pattern, query_lower)
            if match:
                start = int(match.group(1))
                end = int(match.group(2))
                result["type"] = "article_range"
                result["article_range"] = [str(i) for i in range(start, end + 1)]
                return result

        # ========== СТАТЬЯ + ПУНКТ + ПОДПУНКТ ==========
        # "статья 454 пункт 1 подпункт а"
        full_pattern = r'стать[ияюе]\s+(\d+)\s+пункт[ауеы]?\s+(\d+)(?:\s+(?:подпункт[ауеы]?|абзац)\s+([а-я]|\d+))?'
        match = re.search(full_pattern, query_lower)
        if match:
            result["type"] = "specific_article"
            result["article_num"] = match.group(1)
            result["point_num"] = match.group(2)
            if match.group(3):
                result["subpoint"] = match.group(3)
            return result

        # ========== СОКРАЩЕННАЯ ФОРМА ==========
        # "ст. 454 п. 1 пп. а"
        short_pattern = r'ст\.?\s*(\d+)\s+п\.?\s*(\d+)(?:\s+(?:пп\.?|абз\.?)\s*([а-я]|\d+))?'
        match = re.search(short_pattern, query_lower)
        if match:
            result["type"] = "specific_article"
            result["article_num"] = match.group(1)
            result["point_num"] = match.group(2)
            if match.group(3):
                result["subpoint"] = match.group(3)
            return result

        # ========== ТОЛЬКО СТАТЬЯ ==========
        article_patterns = [
            r'стать[ияюе]\s+(\d+(?:\.\d+)?)',
            r'ст\.?\s*(\d+(?:\.\d+)?)',
        ]
        for pattern in article_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result["type"] = "specific_article"
                result["article_num"] = match.group(1)
                return result

        # ========== ТОЛЬКО ПУНКТ (контекстно) ==========
        point_only = re.search(r'(?:^|\s)пункт[ауеы]?\s+(\d+)', query_lower)
        if point_only:
            result["point_hint"] = point_only.group(1)
            return result

        return result

# ================= QUERY CLARIFIER =================

class QueryClarifier:
    """Определяет, нужны ли уточнения"""

    @staticmethod
    def needs_clarification(question: str, docs: List[Document]) -> Optional[str]:
        """
        Определяет нужны ли уточнения
        Returns: Текст уточняющего вопроса или None
        """
        # Нет документов — уточнять нечего: вопрос не по теме, и вызывающий
        # вернёт отказ. Раньше правило «короткий вопрос + общее слово»
        # срабатывало даже на пустом наборе («как приготовить борщ»).
        if not docs:
            return None

        question_lower = question.lower()

        # Слишком общие вопросы
        general_keywords = ["договор", "право", "можно ли", "как"]
        if len(question.split()) <= 3 and any(kw in question_lower for kw in general_keywords):
            return (
                "❓ Уточните, пожалуйста:\n\n"
                "• О каком конкретно виде договора идет речь? (купля-продажа, аренда, подряд?)\n"
                "• Какая ситуация произошла?\n"
                "• Что именно вы хотите узнать?\n\n"
                "Это поможет дать более точный ответ."
            )

        # Если найдено много разных статей из разных тем
        if len(docs) > 0:
            parts = set(doc.metadata.get("part") for doc in docs[:5] if doc.metadata.get("part"))
            if len(parts) >= 3 and len(question.split()) <= 4:
                # Уточняем только когда вопрос сам по себе короткий и общий.
                # Раньше правило срабатывало на любые документы из 3+ частей ГК,
                # и нормальный вопрос «можно ли расторгнуть договор аренды
                # досрочно» получал встречный вопрос вместо ответа.
                return (
                    "❓ Ваш вопрос затрагивает несколько разделов ГК РФ.\n\n"
                    "Уточните, пожалуйста:\n"
                    "• В какой области: купля-продажа, наследство, обязательства?\n"
                    "• Опишите ситуацию подробнее\n\n"
                    "Или я могу дать общий ответ по всем найденным нормам?"
                )

        return None

# ================= RULE-BASED ROUTER =================

class RuleBasedRouter:
    """Маршрутизация запросов по правилам (без LLM)"""

    def __init__(self, db: LegalVectorDB):
        self.db = db
        self.faq_cache = self._build_faq_cache()

    def _build_faq_cache(self) -> Dict[str, str]:
        """Строит кэш точных FAQ ответов с готовыми markdown ответами"""
        return {
            # Нормализованный запрос -> готовый markdown ответ
            "что такое исковая давность": (
                "📋 **Исковая давность** — это срок для защиты права по иску лица, право которого нарушено.\n\n"
                "📌 Что говорит закон:\n"
                "«Общий срок исковой давности составляет три года» (ст. 196 ГК РФ).\n\n"
                "💬 Простыми словами:\n"
                "Если у вас нарушено право, у вас есть 3 года чтобы подать иск в суд. "
                "После этого срока суд может отказать в иске, даже если вы правы.\n\n"
                "⚠️ <i>Информация носит справочный характер. Для точной консультации обратитесь к юристу.</i>"
            ),
            "срок исковой давности": (
                "📋 **Срок исковой давности** составляет **3 года** с момента, когда лицо узнало о нарушении своего права.\n\n"
                "📌 Что говорит закон:\n"
                "«Общий срок исковой давности составляет три года» (ст. 196 ГК РФ).\n\n"
                "💡 Важно знать:\n"
                "• Срок начинает течь с момента, когда вы узнали о нарушении\n"
                "• По отдельным видам требований могут быть установлены специальные сроки\n"
                "• Срок можно прервать действиями должника или подачей иска\n\n"
                "⚠️ <i>Информация носит справочный характер. Для точной консультации обратитесь к юристу.</i>"
            ),
            "что такое договор купли продажи": (
                "📋 **Договор купли-продажи** — это соглашение, по которому одна сторона (продавец) обязуется передать вещь в собственность другой стороне (покупателю), а покупатель обязуется принять эту вещь и уплатить за нее определенную денежную сумму.\n\n"
                "📌 Что говорит закон:\n"
                "«По договору купли-продажи одна сторона (продавец) обязуется передать вещь в собственность другой стороне (покупателю), а покупатель обязуется принять эту вещь и уплатить за нее определенную денежную сумму (цену)» (ст. 454 ГК РФ).\n\n"
                "💬 Простыми словами:\n"
                "Вы платите деньги — получаете вещь в собственность. Это основной вид договора для покупки товаров, недвижимости и другого имущества.\n\n"
                "⚠️ <i>Информация носит справочный характер. Для точной консультации обратитесь к юристу.</i>"
            ),
            "договор купли продажи": (
                "📋 **Договор купли-продажи** — это соглашение о передаче имущества за деньги.\n\n"
                "📌 Что говорит закон:\n"
                "«По договору купли-продажи одна сторона (продавец) обязуется передать вещь в собственность другой стороне (покупателю), а покупатель обязуется принять эту вещь и уплатить за нее определенную денежную сумму (цену)» (ст. 454 ГК РФ).\n\n"
                "💡 Практический совет:\n"
                "Всегда фиксируйте условия договора письменно, особенно при крупных покупках. Указывайте цену, описание товара и срок передачи.\n\n"
                "⚠️ <i>Информация носит справочный характер. Для точной консультации обратитесь к юристу.</i>"
            ),
            "что такое договор аренды": (
                "📋 **Договор аренды** — это соглашение, по которому одна сторона (арендодатель) обязуется предоставить имущество за плату во временное владение и (или) пользование другой стороне (арендатору).\n\n"
                "📌 Что говорит закон:\n"
                "«По договору аренды (имущественного найма) арендодатель (наймодатель) обязуется предоставить арендатору (нанимателю) имущество за плату во временное владение и (или) пользование» (ст. 606 ГК РФ).\n\n"
                "💬 Простыми словами:\n"
                "Вы платите за пользование вещью, но она остается собственностью арендодателя. Типичные примеры: аренда квартиры, офиса, автомобиля.\n\n"
                "⚠️ <i>Информация носит справочный характер. Для точной консультации обратитесь к юристу.</i>"
            ),
            "договор аренды": (
                "📋 **Договор аренды** — это предоставление имущества во временное пользование за плату.\n\n"
                "📌 Что говорит закон:\n"
                "«По договору аренды (имущественного найма) арендодатель (наймодатель) обязуется предоставить арендатору (нанимателю) имущество за плату во временное владение и (или) пользование» (ст. 606 ГК РФ).\n\n"
                "💡 Практический совет:\n"
                "В договоре аренды обязательно укажите: предмет аренды, срок, размер арендной платы и ответственность сторон.\n\n"
                "⚠️ <i>Информация носит справочный характер. Для точной консультации обратитесь к юристу.</i>"
            ),
            "что такое договор подряда": (
                "📋 **Договор подряда** — это соглашение, по которому одна сторона (подрядчик) обязуется выполнить по заданию другой стороны (заказчика) определенную работу и сдать ее результат заказчику, а заказчик обязуется принять результат работы и оплатить его.\n\n"
                "📌 Что говорит закон:\n"
                "«По договору подряда одна сторона (подрядчик) обязуется выполнить по заданию другой стороны (заказчика) определенную работу и сдать ее результат заказчику, а заказчик обязуется принять результат работы и оплатить его» (ст. 702 ГК РФ).\n\n"
                "💬 Простыми словами:\n"
                "Вы платите за результат работы, а не просто за время. Примеры: ремонт квартиры, строительство, пошив одежды.\n\n"
                "⚠️ <i>Информация носит справочный характер. Для точной консультации обратитесь к юристу.</i>"
            ),
            "договор подряда": (
                "📋 **Договор подряда** — это выполнение работы за плату с передачей результата заказчику.\n\n"
                "📌 Что говорит закон:\n"
                "«По договору подряда одна сторона (подрядчик) обязуется выполнить по заданию другой стороны (заказчика) определенную работу и сдать ее результат заказчику, а заказчик обязуется принять результат работы и оплатить его» (ст. 702 ГК РФ).\n\n"
                "💡 Важно знать:\n"
                "• Подрядчик несет ответственность за качество работы\n"
                "• Заказчик может контролировать ход выполнения\n"
                "• Риск случайной гибели результата переходит к заказчику при приемке\n\n"
                "⚠️ <i>Информация носит справочный характер. Для точной консультации обратитесь к юристу.</i>"
            ),
            "что такое наследование": (
                "📋 **Наследование** — это переход имущества умершего к другим лицам в порядке универсального правопреемства.\n\n"
                "📌 Что говорит закон:\n"
                "«При наследовании имущество умершего (наследство, наследственное имущество) переходит к другим лицам в порядке универсального правопреемства» (ст. 1110 ГК РФ).\n\n"
                "💬 Простыми словами:\n"
                "После смерти человека его имущество переходит к наследникам по закону или по завещанию. Наследники получают не только права, но и обязанности (долги).\n\n"
                "⚠️ <i>Информация носит справочный характер. Для точной консультации обратитесь к юристу.</i>"
            ),
            "наследование": (
                "📋 **Наследование** — это переход имущества умершего к наследникам.\n\n"
                "📌 Что говорит закон:\n"
                "«При наследовании имущество умершего (наследство, наследственное имущество) переходит к другим лицам в порядке универсального правопреемства» (ст. 1110 ГК РФ).\n\n"
                "💡 Практический совет:\n"
                "• Наследство можно принять по завещанию или по закону\n"
                "• На принятие наследства дается 6 месяцев\n"
                "• Вместе с имуществом переходят и долги умершего\n\n"
                "⚠️ <i>Информация носит справочный характер. Для точной консультации обратитесь к юристу.</i>"
            ),
            "что такое юридическое лицо": (
                "📋 **Юридическое лицо** — это организация, которая имеет в собственности, хозяйственном ведении или оперативном управлении обособленное имущество и отвечает по своим обязательствам этим имуществом.\n\n"
                "📌 Что говорит закон:\n"
                "«Юридическим лицом признается организация, которая имеет в собственности, хозяйственном ведении или оперативном управлении обособленное имущество и отвечает по своим обязательствам этим имуществом, может от своего имени приобретать и осуществлять гражданские права и нести гражданские обязанности, быть истцом и ответчиком в суде» (ст. 48 ГК РФ).\n\n"
                "💬 Простыми словами:\n"
                "Юридическое лицо — это организация, которая как отдельный субъект права может владеть имуществом, заключать договоры и отвечать по обязательствам. Примеры: ООО, АО, фонды.\n\n"
                "⚠️ <i>Информация носит справочный характер. Для точной консультации обратитесь к юристу.</i>"
            ),
            "юридическое лицо": (
                "📋 **Юридическое лицо** — это организация, являющаяся субъектом права.\n\n"
                "📌 Что говорит закон:\n"
                "«Юридическим лицом признается организация, которая имеет в собственности, хозяйственном ведении или оперативном управлении обособленное имущество и отвечает по своим обязательствам этим имуществом» (ст. 48 ГК РФ).\n\n"
                "💡 Важно знать:\n"
                "• Юридическое лицо должно быть зарегистрировано в ЕГРЮЛ\n"
                "• Оно действует через свои органы (директор, совет директоров)\n"
                "• Ответственность ограничена имуществом юридического лица\n\n"
                "⚠️ <i>Информация носит справочный характер. Для точной консультации обратитесь к юристу.</i>"
            ),
            "что такое физическое лицо": (
                "📋 **Физическое лицо** — это человек как субъект гражданского права, обладающий правоспособностью и дееспособностью.\n\n"
                "📌 Что говорит закон:\n"
                "«Способность иметь гражданские права и нести обязанности (гражданская правоспособность) признается за всеми гражданами» (ст. 17 ГК РФ).\n\n"
                "💬 Простыми словами:\n"
                "Физическое лицо — это любой человек, который может иметь права и обязанности. Правоспособность возникает с рождения, а полная дееспособность — с 18 лет.\n\n"
                "⚠️ <i>Информация носит справочный характер. Для точной консультации обратитесь к юристу.</i>"
            ),
            "физическое лицо": (
                "📋 **Физическое лицо** — это человек как субъект гражданского права.\n\n"
                "📌 Что говорит закон:\n"
                "«Способность иметь гражданские права и нести обязанности (гражданская правоспособность) признается за всеми гражданами» (ст. 17 ГК РФ).\n\n"
                "💡 Важно знать:\n"
                "• Правоспособность — способность иметь права (с рождения)\n"
                "• Дееспособность — способность осуществлять права (с 18 лет)\n"
                "• Несовершеннолетние имеют ограниченную дееспособность\n\n"
                "⚠️ <i>Информация носит справочный характер. Для точной консультации обратитесь к юристу.</i>"
            ),
        }

    def route(self, query: str, query_analysis: Dict) -> Tuple[str, Optional[str]]:
        """
        Маршрутизация запроса

        Returns: ("route_type", answer_or_none)

        Routes:
        - "exact_faq": точный FAQ ответ (возвращает готовый ответ)
        - "direct_article": прямой запрос статьи (возвращает None, используй БД)
        - "article_range": диапазон статей (возвращает None)
        - "hybrid_rag": гибридный поиск + LLM (возвращает None)
        """
        # ========== ПРИОРИТЕТ 1: Точные FAQ ==========
        normalized = self._normalize_query(query)
        if normalized in self.faq_cache:
            article_ref = self.faq_cache[normalized]
            # Можно вернуть готовый ответ или просто статью
            return ("exact_faq", article_ref)

        # ========== ПРИОРИТЕТ 2: Прямой запрос статьи ==========
        if query_analysis["type"] == "specific_article":
            return ("direct_article", None)

        # ========== ПРИОРИТЕТ 3: Диапазон статей ==========
        if query_analysis["type"] == "article_range":
            return ("article_range", None)

        # ========== ПРИОРИТЕТ 4: Гибридный RAG ==========
        return ("hybrid_rag", None)

    def _normalize_query(self, query: str) -> str:
        """Нормализует запрос для сравнения"""
        q = query.lower().strip()
        q = re.sub(r'[?!.,;]', '', q)
        q = re.sub(r'\s+', ' ', q)
        return q


# ================= ANSWER VALIDATOR =================

class AnswerValidator:
    """Валидирует ответы LLM перед отправкой пользователю"""

    # DEF-005: раньше здесь были две разные регулярки, и обе не знали косвенных
    # падежей — «статьёй 196», «статьям 454», «статей 196» считались ответом
    # без ссылок. Теперь одна основа «стат+ь/е/ё» на все формы (включая
    # «статей», где мягкого знака нет), плюс «ст.» и «№».
    ARTICLE_REF_RE = re.compile(
        r"стат[ьеё]\w*\s*(?:№\s*)?(\d+(?:\.\d+)?)"
        r"|ст\.?\s*(?:№\s*)?(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    # Продолжение перечисления: «статьям 454 и 455», «статей 196, 200».
    # Число не считается статьёй, если за ним идёт единица измерения.
    ENUM_TAIL_RE = re.compile(
        r"\s*(?:,|и|или)\s*(\d+(?:\.\d+)?)"
        r"(?!\s*(?:год|лет|месяц|дн|руб|процент|тыс|коп))",
        re.IGNORECASE,
    )

    @classmethod
    def articles_in(cls, text: str) -> set:
        """Номера статей, упомянутых в тексте, в любом падеже и перечислении."""
        found = set()
        for m in cls.ARTICLE_REF_RE.finditer(text):
            num = m.group(1) or m.group(2)
            if not num:
                continue
            found.add(num)
            pos = m.end()
            while True:
                tail = cls.ENUM_TAIL_RE.match(text, pos)
                if not tail:
                    break
                found.add(tail.group(1))
                pos = tail.end()
        return found

    def __init__(self, db: LegalVectorDB):
        self.db = db
        self.warnings: List[str] = []
        self.reason_code: Optional[str] = None

    def _article_exists(self, num: str) -> bool:
        """Есть ли такая статья в кодексе (не в выборке поиска, а вообще)."""
        if self.db is None:
            return False
        try:
            return bool(self.db.get_article_by_number(num))
        except Exception as e:
            # Проверить не смогли — не обвиняем модель в выдумке
            log.warning("article_exists_check_failed", article=num, error=str(e))
            return True

    def validate(self, question: str, answer: str, context_docs: List[Document]) -> Tuple[bool, Optional[str]]:
        """
        Валидирует ответ
        Returns: (is_valid, error_message)
        """
        self.warnings = []
        self.reason_code = None

        # ========== 1. Проверка длины ==========
        if len(answer) < 50:
            self.reason_code = "too_short"
            return False, "Ответ слишком короткий"

        # ========== 2. Проверка упоминания источников (СМЯГЧЕНО) ==========
        # Для общих/системных вопросов статьи не обязательны
        question_lower = question.lower()
        system_keywords = ["запис", "консультац", "помощь", "контакт", "связ"]
        is_system_question = any(kw in question_lower for kw in system_keywords)

        has_article_refs = bool(self.ARTICLE_REF_RE.search(answer))

        if not has_article_refs and not is_system_question:
            # DEF-013: ответ без ссылки больше не выбрасывается. Раньше здесь
            # был отказ, и вместо консультации человек получал техническое
            # «ответ не содержит ссылок» — то есть система выбрасывала
            # собственный ответ и объясняла это своими внутренними правилами.
            self.warnings.append("no_citation")
            self.reason_code = "no_citation"
            log.warning("answer_without_citation", question=question[:80])

        # ========== 3. Проверка галлюцинаций статей ==========
        mentioned_articles = self.articles_in(answer)

        context_articles = set()
        for doc in context_docs:
            # Извлекаем номер статьи из reference или metadata
            article_num = doc.metadata.get("article_num")
            if article_num:
                context_articles.add(str(article_num))
            else:
                # Пытаемся извлечь из reference
                ref = doc.metadata.get("reference", "")
                match = re.search(r'ст(?:атья|\.)\s*(\d+)', ref, re.IGNORECASE)
                if match:
                    context_articles.add(match.group(1))

        # Проверяем, что все упомянутые статьи есть в контексте
        hallucinated = mentioned_articles - context_articles

        if hallucinated:
            # DEF-013 (продолжение): раньше до двух «выдуманных» статей
            # пропускались, если рядом была хоть одна верная — то есть ответ
            # с придуманной нормой уходил пользователю. Теперь ссылка вне
            # контекста проверяется по самому кодексу: статья, которая в ГК
            # есть, — это не выдумка, а просто не попавшая в выборку норма;
            # статьи, которой в кодексе нет, в ответе быть не может.
            missing = {a for a in hallucinated if not self._article_exists(a)}
            extra_but_real = hallucinated - missing

            if extra_but_real:
                self.warnings.append("cited_outside_context")
                log.warning("article_outside_context", articles=sorted(extra_but_real))

            if missing:
                log.error("hallucination_detected", articles=sorted(missing))
                self.reason_code = "hallucination"
                return False, f"Обнаружены ссылки на статьи вне контекста: {', '.join(sorted(missing))}"

        # ========== 4. Проверка запрещенных фраз ==========
        forbidden = [
            "я не могу", "у меня нет информации", "обратитесь к юристу",
            "я не уверен", "возможно", "вероятно"
        ]
        answer_lower = answer.lower()
        for phrase in forbidden:
            if phrase in answer_lower:
                self.warnings.append("uncertain_wording")
                log.warning("answer_forbidden_phrase", phrase=phrase)
                if STRICT_UNCERTAIN_WORDING:
                    self.reason_code = "uncertain_wording"
                    return False, "Ответ содержит неуверенные формулировки"
                break

        # ========== 5. Проверка на пустые/бессмысленные ответы ==========
        if answer.count('...') > 3:
            self.reason_code = "too_many_ellipsis"
            return False, "Ответ содержит слишком много многоточий"

        # ========== 6. Проверка наличия цитат (ОПЦИОНАЛЬНО) ==========
        has_quotes = '«' in answer or '"' in answer
        if len(context_docs) > 0 and not has_quotes and not is_system_question:
            log.warning("answer_no_quotes")
            # Не блокируем - цитаты не всегда нужны

        return True, None


# ================= MAIN ENGINE =================

class LegalRAG:
    """Основной RAG движок для юридических консультаций"""

    def __init__(self):
        log.info("rag_engine_initializing")

        # Векторная БД
        self.db = LegalVectorDB()
        self.db.load()

        # LLM
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            request_timeout=30,
        )

        # Reranker будет загружен по требованию (lazy load)
        self._reranker = None

        # Кеш
        self.cache = ResponseCache()
        
        # Query analyzer (используем только EnhancedQueryAnalyzer)
        self.enhanced_analyzer = EnhancedQueryAnalyzer()

        # Rule-based router
        self.router = RuleBasedRouter(self.db)

        # ================= TOKEN COUNTING =================
        # Инициализируем tiktoken encoder для точного подсчёта токенов
        # Это критично для русского языка, где на 1 токен приходится больше символов
        try:
            self.tokenizer = tiktoken.encoding_for_model(LLM_MODEL)
            log.info("tokenizer_initialized", model=LLM_MODEL)
        except Exception as e:
            log.warning("tokenizer_init_failed", model=LLM_MODEL, error=str(e)[:100], fallback="cl100k_base")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Гибридный поиск
        self._init_hybrid_search()

        log.info("rag_engine_ready")

    def _get_device(self) -> str:
        """Определяет доступное устройство для вычислений"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    
    def _get_reranker(self):
        """Lazy initialization reranker модели"""
        import os
        
        # Проверяем флаг production
        is_production = os.getenv('PRODUCTION', 'false').lower() == 'true'
        
        if is_production:
            # Production mode - reranker отключен для экономии RAM
            if self._reranker is None:
                log.info("🚀 Production mode: Reranker disabled (экономия 1.4 GB RAM)")
                self._reranker = "DISABLED"  # Маркер отключения
            return None
        
        # Development mode - загружаем reranker
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                log.info("reranker_loading")
                device = self._get_device()
                log.info("reranker_device", device=device)
                self._reranker = CrossEncoder(
                    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                    device=device
                )
                log.info("reranker_loaded")
            except ImportError as e:
                log.warning(f"⚠️ sentence-transformers не установлен: {e}")
                self._reranker = "DISABLED"
                return None
        
        return self._reranker if self._reranker != "DISABLED" else None


    # ================= HYBRID SEARCH =================

    def _init_hybrid_search(self):
        """Инициализация гибридного поиска с MMR"""
        log.info("hybrid_search_setup")

        all_docs = self.db.get_all_documents()
        log.info("documents_loaded", count=len(all_docs))

        # Semantic retriever: обычная близость, без MMR.
        # MMR с lambda_mult=0.5 намеренно выкидывает близкие соседние нормы ради
        # «разнообразия» — и терял нужную статью (на «наём жилого помещения»
        # приносил 678, 685, 686, 672, 684 вместо 671). Разнообразие теперь
        # делает склейка повторов статьи в _hybrid_retrieve, поэтому здесь
        # нужен честный top-K по близости.
        self.semantic_retriever = self.db.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": SEMANTIC_TOP_K},
        )

        # BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(all_docs)
        self.bm25_retriever.k = BM25_TOP_K

        log.info("hybrid_search_ready")

    def _hybrid_retrieve(self, question: str) -> List[Document]:
        """Гибридный поиск: слияние по рангам (RRF) и склейка повторов статьи.

        Раньше список собирался «сначала вся семантика, потом весь BM25» — то
        есть если семантика ошибалась, у BM25 не оставалось шанса попасть в
        первую пятёрку. Плюс в выдаче оставались дубли одной статьи: на вопрос
        «дарение что это» приходило 572, 572, 573, 574, 572 — три слота из пяти
        занимала одна норма.

        Теперь: RRF по двум спискам, затем по одной статье на позицию
        (остаётся лучший по рангу фрагмент статьи).
        """
        semantic_docs = self.semantic_retriever.invoke(question)
        bm25_docs = self.bm25_retriever.invoke(question)

        RRF_K = 60.0
        scores: Dict[str, float] = {}
        by_key: Dict[str, Document] = {}
        order_hint: Dict[str, int] = {}       # для стабильной сортировки при равных баллах

        for weight, docs in ((1.0, semantic_docs), (1.0, bm25_docs)):
            for rank, doc in enumerate(docs):
                key = str(doc.metadata.get("doc_id") or doc.metadata.get("reference") or doc.page_content[:60])
                scores[key] = scores.get(key, 0.0) + weight / (RRF_K + rank)
                by_key.setdefault(key, doc)
                order_hint.setdefault(key, len(order_hint))

        ranked = sorted(scores, key=lambda k: (-scores[k], order_hint[k]))

        seen_articles = set()
        merged: List[Document] = []
        for key in ranked:
            doc = by_key[key]
            article = str(doc.metadata.get("article_num") or "")
            if article and article in seen_articles:
                continue                      # та же статья уже есть в выдаче
            if article:
                seen_articles.add(article)
            merged.append(doc)

        log.info(
            "hybrid_search_rrf",
            semantic=len(semantic_docs), bm25=len(bm25_docs),
            merged=len(merged), articles=len(seen_articles),
        )
        return merged

    def _is_safe(self, text: str) -> bool:
        """
        Проверка безопасности запроса (расширенная защита от prompt injection)
        """
        dangerous = [
            # ========== Prompt injection - английский ==========
            r"ignore\s+(?:all\s+)?(?:previous|the|your)?\s*(?:instructions|commands|rules|prompts|constraints)",
            r"forget\s+(?:all\s+)?(?:previous|the|your)?\s*(?:instructions|commands|rules|prompts|constraints)",
            r"disregard\s+(?:all\s+)?(?:previous|the|your)?\s*(?:instructions|commands|rules|prompts|constraints)",
            r"override\s+(?:all\s+)?(?:previous|the|your)?\s*(?:instructions|commands|rules|prompts|constraints)",
            r"bypass\s+(?:all\s+)?(?:previous|the|your)?\s*(?:instructions|commands|rules|prompts|constraints)",
            r"dismiss\s+(?:all\s+)?(?:previous|the|your)?\s*(?:instructions|commands|rules|prompts|constraints)",
            r"neglect\s+(?:all\s+)?(?:previous|the|your)?\s*(?:instructions|commands|rules|prompts|constraints)",

            # ========== Prompt injection - русский ==========
            r"проигнорируй\s+(?:все\s+)?(?:предыдущие|свои|твои|вышеперечисленные)?\s*(?:инструкции|команды|правила|промпты|ограничения)",
            r"забудь\s+(?:все\s+)?(?:предыдущие|свои|твои|вышеперечисленные)?\s*(?:инструкции|команды|правила|промпты|ограничения)",
            r"не\s+обращай\s+внимание\s+на\s+(?:все\s+)?(?:предыдущие|свои|твои|вышеперечисленные)?\s*(?:инструкции|команды|правила|промпты|ограничения)",
            r"отмени\s+(?:все\s+)?(?:предыдущие|свои|твои|вышеперечисленные)?\s*(?:инструкции|команды|правила|промпты|ограничения)",
            r"отклонить\s+(?:все\s+)?(?:предыдущие|свои|твои|вышеперечисленные)?\s*(?:инструкции|команды|правила|промпты|ограничения)",
            r"игнорируй\s+(?:все\s+)?(?:предыдущие|свои|твои|вышеперечисленные)?\s*(?:инструкции|команды|правила|промпты|ограничения)",

            # ========== System prompt attacks ==========
            r"system\s+prompt",
            r"системн(?:ый|ая|ое|ые)\s+промпт(?:ы|ов|ам)?",
            r"developer\s+instructions",
            r"инструкции\s+разработчика",
            r"initial\s+instructions",
            r"начальные\s+инструкции",

            # ========== Role playing / persona attacks ==========
            r"you\s+are\s+now",
            r"ты\s+сейчас",
            r"act\s+as",
            r"сыграй\s+роль",
            r"pretend\s+to\s+be",
            r"притворись\s+что\s+ты",
            r"assume\s+the\s+role",
            r"принимай\s+роль",

            # ========== Jailbreak patterns ==========
            r"dan\s+mode",
            r"jailbreak",
            r"bypass\s+filters?",
            r"обход\s+фильтр(?:ов|а)?",
            r"unrestricted\s+mode",
            r"без ограничений",

            # ========== Code execution / SQL injection ==========
            r"';\s*drop\s+table",
            r"';\s*delete\s+from",
            r"';\s*exec\s*\(",
            r"<script>",

            # ========== Unicode evasion ==========
            r"[\u2000-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]",  # Zero-width, bidirectional, BOM
        ]

        # Проверяем каждый паттерн
        for pattern in dangerous:
            match = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
            if match:
                log.warning(f"Unsafe query detected (pattern: {pattern}): {match.group(0)}")
                return False

        # DEF-015: вторая проверка — по нормализованному тексту. Регулярки выше
        # знают только точные формулировки: «игнорируй, пожалуйста, все
        # инструкции» проходило, потому что между глаголом и словом
        # «инструкции» появилось лишнее слово. Здесь пунктуация снимается,
        # а глагол-приказ и объект ищутся по отдельности.
        if self._injection_by_tokens(text):
            log.warning(f"Unsafe query detected (tokens): {text[:80]}")
            return False

        return True

    # ---------- DEF-015 ----------
    # Только повелительные формы: «игнорируй», но не «игнорировать» —
    # иначе обычный юридический вопрос про ограничения считался бы атакой.
    INJECTION_VERBS = (
        "игнорируй", "проигнорируй", "забудь", "не обращай внимания",
        "ignore", "forget", "disregard", "override", "bypass", "dismiss", "neglect",
    )
    INJECTION_OBJECTS = (
        "инструкц", "промпт", "команд", "ограничен", "правил",
        "instructions", "prompts", "commands", "constraints", "rules",
    )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Нижний регистр, ё→е, пунктуация в пробелы, пробелы схлопнуты."""
        t = text.lower().replace("ё", "е")
        t = re.sub(r"[^\w\s]+", " ", t, flags=re.UNICODE)
        return re.sub(r"\s+", " ", t).strip()

    @classmethod
    def _injection_by_tokens(cls, text: str) -> bool:
        norm = cls._normalize_text(text)
        return (
            any(v in norm for v in cls.INJECTION_VERBS)
            and any(o in norm for o in cls.INJECTION_OBJECTS)
        )


    def _sort_by_point_number(self, docs: List[Document]) -> List[Document]:
        """
        ИСПРАВЛЕНО: Безопасная сортировка документов по номеру пункта
        Обрабатывает None, "full" и нечисловые значения.
        Использует article_order_index для сохранения исходного порядка.
        """
        def point_sort_key(doc):
            # Сначала используем article_order_index для сохранения глобального порядка
            order_index = doc.metadata.get("article_order_index", 999999)

            value = doc.metadata.get("point_num")
            
            # Если это "full", None или пустая строка - в конец
            if not value or value == "full":
                return (order_index, 9999)

            # Если это число - используем его
            if str(value).isdigit():
                return (order_index, int(value))

            # Иначе в конец
            return (order_index, 9999)

        return sorted(docs, key=point_sort_key)

    # ================= RETRIEVAL =================

    # ---------- DEF-002/DEF-017: «это вообще про ГК?» ----------
    TOPIC_STOPWORDS = frozenset(
        "что такое как какие какой какая кто где когда можно ли это для или по на в не"
        " мне мой моя мои его её их мы вы они ты я чем чтобы если".split()
    )

    def _content_words(self, question: str) -> List[str]:
        """Значимые слова вопроса: длиннее трёх букв, не служебные."""
        return [
            w for w in re.findall(r"[а-яёa-z]{4,}", question.lower())
            if w not in self.TOPIC_STOPWORDS
        ]

    # Вопросы из других отраслей права: по смыслу они рядом с ГК, по косинусу
    # неотличимы («какое наказание за кражу» — 0.40, как у своих вопросов).
    OTHER_BRANCH_MARKERS = (
        "наказание", "уголовн", "лишение свободы", "приговор", "административн",
        "гибдд", "прописк", "штрафстоянк", "депутат", "налогов", "таможн",
    )
    # ...но если человек спрашивает про гражданские последствия того же
    # события («как взыскать ущерб от кражи»), это уже ГК — не отказываем.
    CIVIL_MARKERS = (
        "взыска", "возмест", "ущерб", "договор", "сделк", "иск", "компенсац",
        "неустойк", "убытк", "наследств", "собственност", "аренд", "займ",
    )

    # ---------- DEF-014: сообщение об отказе для человека ----------
    @staticmethod
    def _refusal_message(reason_code: Optional[str]) -> str:
        """Текст отказа без внутренней терминологии.

        Раньше человек читал «Причина: Ответ не содержит ссылок на статьи» —
        это формулировка проверки, а не сообщение для него.
        """
        if reason_code == "hallucination":
            head = "Не могу подтвердить этот ответ по тексту кодекса, поэтому не отдаю его.\n\n"
        else:
            head = "Не получилось подготовить ответ по этому вопросу.\n\n"
        return (
            head
            + "Что можно попробовать:\n"
            "• Спросить про конкретную статью — например: «что говорит статья 454»\n"
            "• Описать ситуацию подробнее: кто, что и когда сделал\n"
            + "• Открыть /help"
        )

    def check_topic_relevance(self, question: str) -> Dict[str, Any]:
        """Проверяет, относится ли вопрос к гражданскому кодексу вообще.

        Считает два признака по одному запросу в Chroma:
          best_cosine — насколько близок лучший найденный чанк;
          coverage    — доля значимых слов вопроса, встретившихся в top-5.

        Отказ, если лучший чанк совсем далёк, либо если он средний и в текстах
        нет ни одного слова из вопроса. Пороги измерены на размеченном наборе
        (20 своих вопросов, 12 посторонних): 0 ложных отказов, 12/12 чужих.
        """
        try:
            probe = self.db.vector_db.similarity_search_with_score(question, k=RELEVANCE_PROBE_K)
        except Exception as e:  # поиск не должен ломать ответ из-за проверки темы
            log.warning("relevance_probe_failed", error=str(e))
            return {"relevant": True, "best_cosine": 1.0, "coverage": 1.0}

        if not probe:
            return {"relevant": False, "best_cosine": 0.0, "coverage": 0.0}

        # Chroma в L2-пространстве при единичных векторах: cos = 1 - dist/2
        best = 1.0 - probe[0][1] / 2.0
        low = question.lower()

        if any(m in low for m in self.OTHER_BRANCH_MARKERS) and not any(
            m in low for m in self.CIVIL_MARKERS
        ):
            return {"relevant": False, "best_cosine": best, "coverage": 0.0,
                    "reason": "other_branch"}

        words = self._content_words(question)
        if not words:
            # одни служебные слова («можно ли») — пусть решает уточняющий вопрос
            return {"relevant": True, "best_cosine": best, "coverage": 0.0}

        texts = " ".join(d.page_content.lower() for d, _ in probe[:5])
        coverage = sum(1 for w in words if w[:5] in texts) / len(words)

        relevant = best >= RELEVANCE_HARD_COSINE and (
            best >= RELEVANCE_SOFT_COSINE or coverage >= RELEVANCE_MIN_COVERAGE
        )
        return {"relevant": relevant, "best_cosine": best, "coverage": coverage}

    def retrieve_documents(self, question: str, query_type: Dict) -> Tuple[List[Document], str]:
        """Получает релевантные документы в зависимости от типа запроса"""
        
        # Если запрашивают диапазон статей
        if query_type["type"] == "article_range":
            article_range = query_type["article_range"]
            log.info(f"Retrieving article range: {article_range}")

            all_docs = []
            for article_num in article_range:
                docs = self.db.get_article_by_number(article_num)
                all_docs.extend(docs)

            if not all_docs:
                return [], ""
            
            # Безопасная сортировка
            all_docs = self._sort_by_point_number(all_docs)

            context = self._build_context(all_docs, query_type, show_full=True)
            return all_docs, context

        # Если запрашивают конкретную статью
        if query_type["type"] == "specific_article":
            article_num = query_type["article_num"]
            point_num = query_type.get("point_num")
            subpoint = query_type.get("subpoint")
            part_gk = query_type.get("part_gk")

            log.info(f"Retrieving specific article {article_num}, point {point_num}, subpoint {subpoint}, part {part_gk}")

            # Получаем статью из БД
            docs = self.db.get_article_by_number(article_num)

            if not docs:
                return [], ""

            # Фильтр по части ГК РФ
            if part_gk:
                docs = [d for d in docs if d.metadata.get("part") == part_gk]

            # Если запрашивают конкретный пункт
            if point_num:
                docs = [d for d in docs if d.metadata.get("point_num") == point_num]

            # Если запрашивают подпункт (фильтр по содержимому)
            if subpoint:
                docs = [d for d in docs if subpoint.lower() in d.page_content.lower()]

            # Безопасная сортировка
            docs = self._sort_by_point_number(docs)

            context = self._build_context(docs, query_type, show_full=True)
            return docs, context

        # Общий вопрос - гибридный поиск
        log.info(f"Retrieving for general query: {question[:100]}...")

        # DEF-002/DEF-017: сначала «это вообще про ГК?», потом поиск и уточнения.
        # Без этой проверки бот на «как приготовить борщ» находил три статьи
        # из разных частей кодекса и предлагал уточнить вид договора.
        relevance = self.check_topic_relevance(question)
        if not relevance["relevant"]:
            log.warning(
                "off_topic_question",
                question=question[:100],
                best_cosine=round(relevance["best_cosine"], 3),
                coverage=round(relevance["coverage"], 2),
            )
            return [], ""

        docs = self._hybrid_retrieve(question)

        if not docs:
            log.warning("No documents found!")
            return [], ""

        # ========== 5. Reranking с batch_size для ускорения (опционально)
        reranker = self._get_reranker()

        if reranker is None:
            # Production mode - reranker отключен, используем документы как есть
            log.info(f"⚡ Reranker disabled, returning top {RERANK_TOP_K} without reranking")
            top_docs = docs[:RERANK_TOP_K]
        else:
            # Development mode - используем reranking
            log.info(f"Reranking top {min(len(docs), RERANK_CANDIDATES)} candidates...")
            
            candidates_count = min(len(docs), RERANK_CANDIDATES)
            pairs = [[question, d.page_content] for d in docs[:candidates_count]]
            
            # Оптимизация: batch_size для GPU
            scores = reranker.predict(pairs, batch_size=16)
            
            # Сортируем по score
            ranked = sorted(
                zip(docs[:candidates_count], scores),
                key=lambda x: x[1],
                reverse=True
            )

            # Фильтрация по threshold
            top_docs = [
                doc for doc, score in ranked[:RERANK_TOP_K]
                if score >= RERANK_SCORE_THRESHOLD
            ]

            log.info(
                f"Reranking: {len(ranked)} candidates -> {len(top_docs)} "
                f"above threshold ({RERANK_SCORE_THRESHOLD})"
            )

            # Fallback если ничего не прошло
            if not top_docs:
                log.warning(
                    f"No documents above threshold, using fallback to top-{RERANK_TOP_K}"
                )
                top_docs = [doc for doc, score in ranked[:RERANK_TOP_K]]

        # ВАЖНО: без этого return функция возвращала None, и generate_answer
        # падал на распаковке `docs, context = ...`. Дефект DEF-001.
        context = self._build_context(top_docs, query_type)
        log.info(f"Retrieved {len(top_docs)} docs for general query")
        return top_docs, context


    # ================= CONTEXT =================

    def _build_context(self, docs: List[Document], query_type: Dict = None, show_full=False) -> str:
        """
        УЛУЧШЕННАЯ версия: адаптивное форматирование контекста с точным подсчётом токенов
        
        ✅ ИСПРАВЛЕНО: Добавлено предупреждение пользователю при обрезании контекста
        """
        parts = []
        total_tokens = 0
        truncated = False  # ✅ НОВОЕ: Флаг обрезания

        # Определяем режим
        if query_type:
            is_specific_article = query_type.get("type") == "specific_article"
        else:
            is_specific_article = show_full

        for i, doc in enumerate(docs, 1):
            doc_id = doc.metadata.get("doc_id", "unknown")
            order_index = doc.metadata.get("article_order_index", "unknown")
            reference = doc.metadata.get("reference", "")
            part = doc.metadata.get("part", "?")
            content = doc.page_content
            content_length = len(content)

            # Логируем doc_id и order_index для дебага
            log.debug(
                f"Processing doc [{i}]: doc_id={doc_id}, order_index={order_index}, "
                f"ref={reference}, len={content_length}"
            )

            # ========== СТРАТЕГИЯ 1: Конкретная статья - ВСЕГДА полный текст ==========
            if is_specific_article:
                formatted_content = content
                log.debug(f"Specific article mode: full text ({content_length} chars)")

            # ========== СТРАТЕГИЯ 2: Короткая статья - полный текст ==========
            elif content_length <= SHORT_ARTICLE_THRESHOLD:
                formatted_content = content
                log.debug(f"Short article: full text ({content_length} chars)")

            # ========== СТРАТЕГИЯ 3: Длинная статья в general режиме ==========
            else:
                # Умное сокращение с сохранением структуры
                formatted_content = self._smart_truncate(content, LONG_ARTICLE_SUMMARY_LENGTH)
                log.debug(f"Long article: truncated {content_length} -> {len(formatted_content)} chars")

            # Формируем блок
            block = f"[Источник {i}] {reference} (часть {part} ГК РФ)\n{formatted_content}"

            # ================= ТОЧНЫЙ ПОДСЧЁТ ТОКЕНОВ =================
            block_tokens = len(self.tokenizer.encode(block))

            # Проверяем общий размер контекста
            if total_tokens + block_tokens > MAX_CONTEXT_TOKENS:
                truncated = True  # ✅ НОВОЕ: Отмечаем что контекст обрезан
                log.warning(
                    f"Context size limit reached: {total_tokens} + {block_tokens} = {total_tokens + block_tokens} > {MAX_CONTEXT_TOKENS}. "
                    f"Truncating at {i} documents (included {len(parts)}/{len(docs)})"
                )
                break

            parts.append(block)
            total_tokens += block_tokens

        separator = "\n\n" + "="*50 + "\n\n"
        final_context = separator.join(parts)

        # ✅ НОВОЕ: Добавляем предупреждение если контекст был обрезан
        if truncated:
            included_articles = []
            for part in parts:
                # Извлекаем номера статей из контекста
                match = re.search(r'ст\.?\s*(\d+)', part)
                if match:
                    included_articles.append(match.group(1))
            
            truncation_warning = (
                f"\n\n{'='*50}\n"
                f"⚠️ КОНТЕКСТ ОБРЕЗАН ПО ЛИМИТУ ТОКЕНОВ\n"
                f"{'='*50}\n\n"
                f"Включено: {len(parts)} из {len(docs)} документов\n"
                f"Использовано токенов: {total_tokens} / {MAX_CONTEXT_TOKENS}\n\n"
            )
            
            if included_articles:
                truncation_warning += (
                    f"Включённые статьи: {', '.join(set(included_articles))}\n\n"
                )
            
            truncation_warning += (
                f"💡 Для полного текста конкретной статьи используйте:\n"
                f"   /article <номер>\n"
                f"   Например: /article 454\n"
            )
            
            final_context = truncation_warning + final_context

        log.info(
            f"✅ Context built: {len(docs)} documents, {len(parts)} included, "
            f"{total_tokens} tokens / {MAX_CONTEXT_TOKENS} limit ({total_tokens / MAX_CONTEXT_TOKENS * 100:.1f}%), "
            f"truncated={truncated}"
        )

        return final_context


    def _smart_truncate(self, content: str, max_length: int) -> str:
        """
        Умное сокращение статьи с сохранением структуры
        Приоритеты:
        1. Заголовок статьи
        2. Первый пункт (обычно главное определение)
        3. Пункт, содержащий ключевые слова из запроса (если есть)
        4. Завершающие условия
        """
        lines = content.split('\n')

        # Всегда сохраняем заголовок статьи
        header = []
        body = []
        for line in lines:
            if line.startswith('Статья'):
                header.append(line)
            else:
                body.append(line)

        # Объединяем заголовок
        result = '\n'.join(header) + '\n\n'

        # Берем начало тела статьи
        body_text = '\n'.join(body)
        remaining_length = max_length - len(result)

        if len(body_text) <= remaining_length:
            return result + body_text

        # Сокращаем с умным обрезанием по предложениям
        truncated = body_text[:remaining_length]

        # Обрезаем по последнему полному предложению
        last_period = max(
            truncated.rfind('.'),
            truncated.rfind('?'),
            truncated.rfind('!')
        )

        if last_period > remaining_length * 0.7:  # Если нашли предложение в последней трети
            truncated = truncated[:last_period + 1]

        return result + truncated + "\n\n[... статья содержит дополнительные пункты, не включенные в контекст]"

    def _get_article_explanation(self, article_ref: str) -> str:
        """
        Генерирует объяснение для конкретной статьи по ссылке
        Используется для точных FAQ ответов
        """
        # Парсим ссылку на статью (например, "ст. 196 ГК РФ")
        match = re.search(r'ст\.?\s*(\d+)', article_ref.lower())
        if not match:
            return f"📖 {article_ref}\n\nДля получения подробной информации, пожалуйста, задайте более конкретный вопрос."

        article_num = match.group(1)
        docs = self.db.get_article_by_number(article_num)

        if not docs:
            return f"📖 {article_ref}\n\nК сожалению, не удалось найти текст этой статьи."

        # Формируем контекст (передаем query_type для конкретной статьи)
        query_type = {"type": "specific_article", "article_num": article_num}
        context = self._build_context(docs, query_type, show_full=True)

        # Генерируем объяснение через LLM
        user_prompt = (
            f"Дай краткое и понятное объяснение для {article_ref}.\n\n"
            f"Текст статьи:{context}\n\n"
            "Объясни основную суть статьи простым языком."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.llm.invoke(messages)
            answer = response.content.strip()

            # Валидация ответа для FAQ
            validator = AnswerValidator(self.db)
            is_valid, error_msg = validator.validate(article_ref, answer, docs)

            if not is_valid:
                log.warning(f"FAQ answer validation warning: {error_msg}")
                # Для FAQ не блокируем, а просто логируем

            # Добавляем дисклеймер
            answer += (
                "\n\n⚠️ <i>Информация носит справочный характер. "
                "Для точной консультации обратитесь к юристу.</i>"
            )

            return answer

        except Exception as e:
            log.error(f"Error generating article explanation: {e}", exc_info=True)
            return f"📖 {article_ref}\n\nПроизошла ошибка при получении объяснения."

    # ================= GENERATION =================

    def generate_answer(self, question: str) -> str:
        """
        Генерирует ответ на вопрос
        
        Args:
            question: Вопрос пользователя
            
        Returns:
            Ответ ассистента
        """
        # Проверка безопасности
        if not self._is_safe(question):
            log.warning(f"Unsafe query detected: {question[:100]}")
            return "⚠️ Запрос отклонен системой безопасности."
        
        # Проверяем кеш
        cached = self.cache.get(question)
        if cached:
            return cached

        # Анализируем тип запроса (используем enhanced analyzer)
        query_type = self.enhanced_analyzer.analyze(question)
        log.info(f"Query type: {query_type['type']}, details: {query_type}")

        # НОВОЕ: Rule-based routing
        route_type, direct_answer = self.router.route(question, query_type)
        log.info(f"Route: {route_type}")

        if route_type == "exact_faq" and direct_answer:
            # Возвращаем готовый ответ из FAQ (без вызова LLM)
            log.info("Exact FAQ match - returning cached answer")
            answer = direct_answer
            self.cache.set(question, answer)
            return answer

        # Получаем документы
        docs, context = self.retrieve_documents(question, query_type)

        if not docs:
            answer = (
                "❌ Не нашел релевантных статей ГК РФ по вашему вопросу.\n\n"
                "💡 Попробуйте:\n"
                "• Переформулировать вопрос\n"
                "• Уточнить область права\n"
                "• Указать конкретную статью (например: 'что говорит статья 454?')"
            )
            self.cache.set(question, answer)
            return answer

        # Проверяем, нужны ли уточнения (только для общих вопросов)
        if query_type["type"] == "general":
            clarification = QueryClarifier.needs_clarification(question, docs)
            if clarification:
                self.cache.set(question, clarification)
                return clarification

        # Адаптируем промпт в зависимости от типа запроса
        if query_type["type"] == "article_range":
            user_prompt = (
                f"Вопрос: {question}\n\n"
                f"Текст статей ГК РФ:{context}\n\n"
                "Дай подробное объяснение запрошенного диапазона статей."
            )
        elif query_type["type"] == "specific_article":
            # Запрос конкретной статьи - требуем дословное цитирование
            user_prompt = (
                f"Пользователь запросил конкретную статью.\n\n"
                f"Вопрос: {question}\n\n"
                f"Полный текст статьи:{context}\n\n"
                f"ВАЖНО: Дай подробное объяснение с ДОСЛОВНЫМ цитированием ключевых норм в кавычках."
            )
        else:
            # Общий вопрос - более гибкий подход
            user_prompt = (
                f"Вопрос пользователя: {question}\n\n"
                f"Релевантные нормы ГК РФ:{context}\n\n"
                f"ВАЖНО: Ответь на вопрос, используя эти нормы. Можешь пересказывать своими словами, "
                f"но обязательно указывай источники (ст. X п. Y ГК РФ). "
                f"Если контекст неполный - предложи запросить полный текст статьи."
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            log.info("Generating answer with LLM...")
            response = self.llm.invoke(messages)
            answer = response.content.strip()

            # НОВОЕ: Валидация ответа
            validator = AnswerValidator(self.db)
            is_valid, error_msg = validator.validate(question, answer, docs)

            if not is_valid:
                # DEF-014: внутреннюю причину — в лог, человеку — что делать.
                log.error("answer_validation_failed", reason=validator.reason_code,
                          detail=error_msg)
                return self._refusal_message(validator.reason_code)

            # DEF-013: ответ без ссылки отдаём, но честно говорим, чего в нём нет
            if "no_citation" in validator.warnings:
                answer += (
                    "\n\nℹ️ <i>В ответе нет ссылки на конкретную статью. "
                    "Если нужна точная норма — спросите прямо: "
                    "«что говорит статья 454».</i>"
                )

            # DEF-016: оговорки в ответе помечаем видимо для человека
            if "uncertain_wording" in validator.warnings:
                answer = (
                    "⚠️ <i>В ответе есть слова вроде «возможно» или «вероятно»: "
                    "однозначной нормы под вашу ситуацию найти не удалось.</i>\n\n"
                ) + answer

            # Добавляем дисклеймер
            answer += (
                "\n\n⚠️ <i>Информация носит справочный характер. "
                "Для точной консультации обратитесь к юристу.</i>"
            )

            # Кешируем
            self.cache.set(question, answer)
            
            log.info("Answer generated successfully")
            return answer

        except Exception as e:
            log.error(f"Error generating answer: {e}", exc_info=True)
            return (
                "❌ Произошла ошибка при генерации ответа.\n"
                "Попробуйте переформулировать вопрос или повторите попытку."
            )

    # ================= STATS =================

    def get_cache_stats(self) -> dict:
        """Получить статистику кеша"""
        return self.cache.stats()

    def get_db_stats(self) -> dict:
        """Получить статистику БД"""
        return self.db.get_stats()


# ================= SINGLETON =================

_rag_instance = None
_rag_lock = Lock()  # Потокобезопасная инициализация singleton


def get_rag_engine() -> LegalRAG:
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance
    with _rag_lock:
        if _rag_instance is None:
            _rag_instance = LegalRAG()
    return _rag_instance



# Удобные функции для импорта
def generate_answer(question: str) -> str:
    """Генерирует ответ на вопрос"""
    engine = get_rag_engine()
    return engine.generate_answer(question)


def get_cache_stats() -> dict:
    """Получить статистику кеша"""
    engine = get_rag_engine()
    return engine.get_cache_stats()


def get_db_stats() -> dict:
    """Получить статистику БД"""
    engine = get_rag_engine()
    return engine.get_db_stats()
