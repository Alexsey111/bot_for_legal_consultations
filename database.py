"""
database.py
Оптимизированная векторная база данных для ГК РФ
Разбиение по пунктам статей с сохранением контекста
"""

import shutil
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import re

import structlog

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Импортируем статичный маппинг структуры ГК РФ
from gk_structure import get_chapter_for_article, determine_gk_part

log = structlog.get_logger()

# ================= CONFIG =================
PERSIST_DIRECTORY = "./chroma_legal_db"
COLLECTION_NAME = "gk_rf_articles"

# Используем более дешевую модель для embeddings
EMBEDDING_MODEL = "text-embedding-3-small"

# Параметры разбиения
MIN_CHUNK_SIZE = 200
IDEAL_CHUNK_SIZE = 1200
MAX_CHUNK_SIZE = 2500

# Для экстремально длинных пунктов
FORCE_SPLIT_AT = 2500
FORCE_SPLIT_OVERLAP = 400


# ================= UTILS =================


def extract_article_info(text: str) -> Optional[Tuple[str, str]]:
    """
    Извлекает номер и название статьи
    Returns: (номер, название) или None
    """
    match = re.search(r"Статья\s+(\d+(?:\.\d+)?)\.\s*([^\n]+)", text)
    if match:
        return match.group(1), match.group(2).strip()
    return None


def extract_chapter_info(text: str) -> Optional[Tuple[str, str]]:
    """
    Извлекает номер и название главы
    Returns: (номер, название) или None
    """
    match = re.search(r"Глава\s+(\d+)\.\s*([^\n]+)", text)
    if match:
        return match.group(1), match.group(2).strip()
    return None


def split_article_by_points(article_text: str, article_num: str, article_title: str) -> List[Dict]:
    """
    ИСПРАВЛЕННАЯ ФУНКЦИЯ: Разбивает статью на пункты с использованием lookahead split
    Returns: список словарей с информацией о каждом пункте
    """
    # Убираем заголовок статьи из текста
    text_without_header = re.sub(r"Статья\s+\d+(?:\.\d+)?\.\s*[^\n]+\n*", "", article_text, count=1).strip()
    
    # ИСПРАВЛЕНО: Используем lookahead split для надежного разбиения по пунктам
    # (?=\n?\d+\.\s) означает "раздели перед каждым номером пункта"
    # \n? делает перенос строки опциональным - работает с любым форматированием
    points_split = re.split(r'(?=\n?\d+\.\s)', text_without_header)

    result = []
    
    # Обрабатываем каждый блок
    for block in points_split:
        block = block.strip()
        if not block:
            continue
        
        # Извлекаем номер пункта и текст
        match = re.match(r'\s*(\d+)\.\s+(.*)', block, re.DOTALL)

        if match:
            point_num = match.group(1)
            point_text = match.group(2).strip()

            # Проверяем наличие подпунктов (а, б, в) или скобочных (1, 2, 3)
            has_letter_subpoints = bool(re.search(r'\n\s*[а-яё]\)', point_text, re.IGNORECASE))
            has_digit_subpoints = bool(re.search(r'\n\s*\d+\)', point_text))
            has_abzac = bool(re.search(r'абзац', point_text, re.IGNORECASE))

            has_subpoints = has_letter_subpoints or has_digit_subpoints or has_abzac

            result.append({
                "point_num": point_num,
                "text": point_text,
                "has_subpoints": has_subpoints,
                "is_full_article": False
            })
        else:
            # Блок без номера - это преамбула или текст без нумерации
            # Добавляем к следующему пункту или сохраняем как отдельный блок
            if result:
                # Добавляем к последнему пункту
                result[-1]["text"] += "\n\n" + block
            else:
                # Если это первый блок без нумерации - сохраняем как полный текст
                log.debug("article_no_points", article_num=article_num)
                return [{
                    "point_num": None,
                    "text": block,
                    "has_subpoints": False,
                    "is_full_article": True
                }]

    # Если не нашли пунктов - сохраняем целиком
    if not result:
        log.debug("article_no_points_found", article_num=article_num)
        return [{
            "point_num": None,
            "text": text_without_header,
            "has_subpoints": False,
            "is_full_article": True
        }]

    log.debug("article_points_found", article_num=article_num, count=len(result))
    return result


def extract_keywords(text: str) -> List[str]:
    """Извлекает ключевые юридические термины"""
    keywords = []
    
    legal_terms = [
        "договор", "обязательство", "право", "обязанность", "ответственность",
        "сторона", "лицо", "имущество", "собственность", "владение",
        "сделка", "соглашение", "продавец", "покупатель", "арендатор",
        "исковая давность", "возмещение", "убытки", "проценты", "неустойка",
        "залог", "поручительство", "гарантия", "расторжение", "недействительность"
    ]
    
    text_lower = text.lower()
    for term in legal_terms:
        if term in text_lower:
            keywords.append(term)
    
    return keywords[:10]


def extract_article_references(text: str) -> List[str]:
    """Извлекает ссылки на другие статьи"""
    patterns = [
        r"статьи?\s+(\d+)",
        r"статье\s+(\d+)",
        r"статью\s+(\d+)",
        r"ст\.\s*(\d+)",
    ]
    
    references = set()
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            references.add(match.group(1))
    
    return sorted(list(references))


def chunk_long_text(text: str, max_size: int, overlap: int) -> List[str]:
    """Разбивает очень длинный текст на чанки"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )
    
    return splitter.split_text(text)


# ================= MAIN PROCESSING =================

def process_gk_text(text: str, law_name: str = "Гражданский кодекс РФ") -> List[Document]:
    """
    Обрабатывает текст ГК РФ и создает документы

    КРИТИЧНО: Сохраняет глобальный порядок документов через article_order_index
    для обеспечения правильного цитирования и последовательности
    """
    log.info("processing_start", law_name=law_name)
    
    documents = []
    # Глобальный счетчик порядка документов для строгого цитирования
    article_order_index = 0
    
    # ИСПРАВЛЕНО: улучшенное разбиение по статьям с использованием split по lookahead
    # Это более надежный подход, чем findall, т.к. не зависит от форматирования внутри статьи
    articles = re.split(r'(?=Статья\s+\d+(?:\.\d+)?\.)', text)
    articles = [a.strip() for a in articles if a.strip().startswith("Статья")]

    log.info("articles_found", count=len(articles))
    
    # ИСПРАВЛЕНО: Используем статичный маппинг для определения глав
    # Это надежнее, чем эвристика на основе количества статей
    log.info("📋 Using static GK structure mapping for chapter detection")

    for article_block in articles:
        if not article_block.strip():
            continue
        
        article_info = extract_article_info(article_block)
        if not article_info:
            continue
        
        article_num, article_title = article_info
        
        # ИСПРАВЛЕНО: Используем статичный маппинг для определения части и главы
        chapter_num, chapter_title, part_num = get_chapter_for_article(article_num)

        log.info(
            "processing_article",
            article_num=article_num,
            article_title=article_title[:50],
            part_num=part_num,
            chapter_num=chapter_num or "N/A",
            chapter_title=chapter_title
        )
        
        # Извлекаем ключевые слова и ссылки
        keywords = extract_keywords(article_block)
        references = extract_article_references(article_block)
        
        # ГЛАВНОЕ ИСПРАВЛЕНИЕ: Разбиваем статью на пункты
        points = split_article_by_points(article_block, article_num, article_title)
        
        # Создаем документы для каждого пункта
        for point_data in points:
            point_num = point_data["point_num"]
            point_text = point_data["text"]
            
            # Формируем полный текст для чанка
            if point_num:
                full_content = f"Статья {article_num}. {article_title}\n\n{point_num}. {point_text}"
            else:
                full_content = f"Статья {article_num}. {article_title}\n\n{point_text}"
            
            # Проверяем длину
            content_length = len(full_content)
            
            # Если текст слишком длинный - разбиваем
            if content_length > FORCE_SPLIT_AT:
                log.info("article_point_too_long", article_num=article_num, point_num=point_num, length=content_length)
                
                chunks = chunk_long_text(point_text, MAX_CHUNK_SIZE, FORCE_SPLIT_OVERLAP)
                
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_content = f"Статья {article_num}. {article_title}\n\n{point_num}. {chunk}"
                    
                    doc = create_document(
                        content=chunk_content,
                        law_name=law_name,
                        article_num=article_num,
                        article_title=article_title,
                        point_num=point_num,
                        part_num=part_num,
                        chapter_num=chapter_num,  # Используем статичный маппинг
                        chapter_title=chapter_title,  # Используем статичный маппинг
                        keywords=keywords,
                        references=references,
                        has_subpoints=point_data["has_subpoints"],
                        is_full_article=point_data["is_full_article"],
                        total_points=len(points),
                        chunk_index=chunk_idx,
                        total_chunks=len(chunks),
                        article_order_index=article_order_index
                    )
                    documents.append(doc)
                    article_order_index += 1  # Увеличиваем счетчик для каждого документа
            else:
                # Создаем обычный документ
                doc = create_document(
                    content=full_content,
                    law_name=law_name,
                    article_num=article_num,
                    article_title=article_title,
                    point_num=point_num,
                    part_num=part_num,
                    chapter_num=chapter_num,  # Используем статичный маппинг
                    chapter_title=chapter_title,  # Используем статичный маппинг
                    keywords=keywords,
                    references=references,
                    has_subpoints=point_data["has_subpoints"],
                    is_full_article=point_data["is_full_article"],
                    total_points=len(points),
                    article_order_index=article_order_index
                )
                documents.append(doc)
                article_order_index += 1  # Увеличиваем счетчик для каждого документа
    
    log.info("documents_created", count=len(documents), last_order_index=article_order_index - 1)
    
    return documents


def create_document(
    content: str,
    law_name: str,
    article_num: str,
    article_title: str,
    point_num: Optional[str],
    part_num: int,
    chapter_num: Optional[str],
    chapter_title: Optional[str],
    keywords: List[str],
    references: List[str],
    has_subpoints: bool,
    is_full_article: bool,
    total_points: int,
    chunk_index: int = 0,
    total_chunks: int = 1,
    article_order_index: int = 0
) -> Document:
    """
    Создает документ с богатыми метаданными

    Args:
        article_order_index: Глобальный индекс порядка документа для сохранения
                            правильной последовательности при цитировании
    """
    
    # Формируем ссылки
    if point_num and not is_full_article:
        reference = f"ст. {article_num} п. {point_num} ГК РФ"
        full_reference = f"Статья {article_num} пункт {point_num} Гражданского кодекса РФ (часть {part_num})"
    else:
        reference = f"ст. {article_num} ГК РФ"
        full_reference = f"Статья {article_num} Гражданского кодекса РФ (часть {part_num})"
    
    # Создаем уникальный ID документа
    # Формат: article_num_pointNum_chunkIndex (например: "454_1_0", "454_full_0")
    point_id = point_num if point_num else "full"
    doc_id = f"{article_num}_{point_id}_{chunk_index}"

    # Создаем метаданные
    metadata = {
        # Основная идентификация
        "doc_id": doc_id,  # Уникальный ID документа
        "law_name": law_name,
        "article_num": article_num,
        "article_title": article_title,
        "point_num": point_num if point_num else "full",
        
        # Порядок в документе (критично для строгого цитирования)
        "article_order_index": article_order_index,  # Глобальный порядок документа

        # Иерархия
        "part": part_num,
        "chapter": chapter_num if chapter_num else "unknown",
        "chapter_title": chapter_title if chapter_title else "unknown",
        
        # Для цитирования
        "reference": reference,
        "full_reference": full_reference,
        
        # Структура статьи
        "has_subpoints": has_subpoints,
        "is_full_article": is_full_article,
        "total_points": total_points,
        
        # Семантика
        "keywords": ",".join(keywords) if keywords else "",
        "references": ",".join(references) if references else "",
        
        # Техническое
        "chunk_type": "full_article" if is_full_article else "point",
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "char_length": len(content)
    }
    
    return Document(page_content=content, metadata=metadata)


# ================= MAIN CLASS =================

class LegalVectorDB:
    """Оптимизированная векторная БД для ГК РФ"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            chunk_size=1000
        )
        self.vector_db = None
        log.info("embedding_model_initialized", model=EMBEDDING_MODEL)
    
    def rebuild_from_file(self, file_path: str, law_name: str = "Гражданский кодекс РФ"):
        """Загружает ГК РФ из файла и создает векторную БД"""
        
        log.info("database_rebuild_start")
        
        # Читаем файл
        log.info("reading_file", path=file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        file_size_mb = len(text) / (1024 * 1024)
        log.info("file_size", mb=f"{file_size_mb:.2f}", chars=len(text))
        
        # Обрабатываем текст
        documents = process_gk_text(text, law_name)
        
        if not documents:
            raise ValueError("❌ No documents created!")
        
        log.info("documents_created", count=len(documents))
        
        # Удаляем старую БД
        if Path(PERSIST_DIRECTORY).exists():
            log.info("removing_old_database")
            shutil.rmtree(PERSIST_DIRECTORY)
        
        # Создаем БД пакетами
        log.info("creating_vector_database")
        
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            if i == 0:
                self.vector_db = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=PERSIST_DIRECTORY,
                    collection_name=COLLECTION_NAME,
                )
            else:
                self.vector_db.add_documents(batch)
            
            progress = min(i + batch_size, len(documents))
            percentage = (progress / len(documents)) * 100
            log.info("progress", current=progress, total=len(documents), percent=f"{percentage:.1f}%")
        
        log.info("database_rebuild_complete", chunks=len(documents))
    
    def _get_collection_count(self) -> int:
        """
        Безопасное получение количества документов в коллекции
        Избегает использования private API _collection когда это возможно
        """
        try:
            # Пытаемся получить все ID и посчитать их
            # Это использует публичный API, но может быть затратно для больших БД
            result = self.vector_db.get(include=["ids"])
            if result and "ids" in result:
                return len(result["ids"])
        except Exception as e:
            log.warning("count_via_public_api_failed", error=str(e)[:100])

        # Fallback: используем _collection с явным предупреждением
        try:
            if hasattr(self.vector_db, "_collection"):
                log.warning("using_private_api_fallback", method="_collection.count()")
                return self.vector_db._collection.count()
        except Exception as e:
            log.error("count_via_private_api_failed", error=str(e)[:100])

        return 0

    def load(self):
        """Загружает существующую БД"""
        if not Path(PERSIST_DIRECTORY).exists():
            raise FileNotFoundError(
                f"❌ Database not found at {PERSIST_DIRECTORY}\n"
                "Please run: python ingest_data.py first!"
            )
        
        log.info("loading_database", path=PERSIST_DIRECTORY)
        self.vector_db = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
        )
        
        count = self._get_collection_count()
        log.info("database_loaded", chunks=count)
    
    def similarity_search(self, query: str, k: int = 10, use_mmr: bool = True, fetch_k: int = 20) -> List[Document]:
        """
        Поиск похожих документов

        Args:
            query: Запрос
            k: Количество документов для возврата
            use_mmr: Использовать MMR (Maximal Marginal Relevance) для разнообразия
            fetch_k: Количество кандидатов для MMR (должно быть >= k)

        Returns:
            Список документов
        """
        if not self.vector_db:
            raise RuntimeError("❌ Database not loaded!")
        
        if use_mmr:
            # MMR: баланс между релевантностью и разнообразием
            # lambda_mult=0.5: 50% релевантность, 50% разнообразие
            return self.vector_db.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=0.5
            )
        else:
            # Обычный семантический поиск
            return self.vector_db.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Поиск с оценкой релевантности

        Примечание: MMR (max_marginal_relevance_search) не возвращает scores,
        поэтому для получения оценок используется обычный similarity_search_with_score.
        Для разнообразия результатов используйте similarity_search(use_mmr=True).
        """
        if not self.vector_db:
            raise RuntimeError("❌ Database not loaded!")
        
        return self.vector_db.similarity_search_with_score(query, k=k)
    
    def get_article_by_number(self, article_num: str) -> List[Document]:
        """Получить все пункты конкретной статьи"""
        if not self.vector_db:
            raise RuntimeError("❌ Database not loaded!")
        
        results = self.vector_db.get(
            where={"article_num": article_num}
        )
        
        return [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(results["documents"], results["metadatas"])]
        
    
    def get_all_documents(self) -> List[Document]:
        """Получить все документы"""
        if not self.vector_db:
            raise RuntimeError("❌ Database not loaded!")
        
        data = self.vector_db.get()
        return [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(data["documents"], data["metadatas"])]
        
    
    def get_stats(self) -> Dict:
        """Получить статистику БД"""
        if not self.vector_db:
            raise RuntimeError("❌ Database not loaded!")
        
        all_docs = self.get_all_documents()
        
        articles = set()
        points = 0
        full_articles = 0
        doc_ids = set()
        order_indices = []
        
        for doc in all_docs:
            articles.add(doc.metadata.get("article_num"))
            doc_ids.add(doc.metadata.get("doc_id"))
            order_idx = doc.metadata.get("article_order_index")
            if order_idx is not None:
                order_indices.append(order_idx)

            if doc.metadata.get("chunk_type") == "point":
                points += 1
            else:
                full_articles += 1
        
        # Проверяем целостность порядка
        order_integrity = "unknown"
        if order_indices:
            expected_indices = sorted(set(order_indices))
            actual_indices = sorted(order_indices)
            if expected_indices == actual_indices:
                order_integrity = "valid"
            else:
                order_integrity = "invalid"

        return {
            "total_chunks": len(all_docs),
            "unique_articles": len(articles),
            "unique_doc_ids": len(doc_ids),
            "point_chunks": points,
            "full_article_chunks": full_articles,
            "order_integrity": order_integrity,
            "max_order_index": max(order_indices) if order_indices else 0,
        }
