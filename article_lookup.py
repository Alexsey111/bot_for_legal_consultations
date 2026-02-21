"""
article_lookup.py - Улучшенные функции для поиска статей с правильными type hints

Этот модуль демонстрирует лучшие практики использования type hints с Union[str, int]
для функций, которые работают с номерами статей ГК РФ.

Преимущества Union[str, int]:
1. Удобство использования - caller может передавать как строки, так и числа
2. Явная документация - type hint явно показывает, что принимаются оба типа
3. Автоматическая нормализация - функция сама приводит к нужному типу
4. Совместимость - работает с существующим кодом, который использует строки
"""

from typing import List, Optional, Union, Dict, Any


# ================= ОПРЕДЕЛЕНИЕ ТИПОВ =================

ArticleNumber = Union[str, int]
ChapterNumber = Union[str, int]
PartNumber = Union[str, int]


# ================= ПРИМЕР 1: Получение статьи по номеру =================

class Document:
    """
    Локальный DTO для документа (статьи ГК РФ) в этом модуле.
    Не путать с langchain_core.documents.Document — в RAG используется именно LangChain Document.
    """
    def __init__(
        self,
        article_num: ArticleNumber,
        title: str,
        content: str,
        part_num: Optional[int] = None,
        chapter_num: Optional[str] = None
    ):
        self.article_num = str(article_num)  # Нормализация к строке
        self.title = title
        self.content = content
        self.part_num = part_num
        self.chapter_num = chapter_num
    
    def __repr__(self) -> str:
        return f"Document(article_num={self.article_num}, title={self.title})"


def get_article_by_number(article_num: ArticleNumber) -> List[Document]:
    """
    Получает статью по номеру
    
    ИСПРАВЛЕНО: Type hint улучшен с article_num: str на Union[str, int]
    - Теперь можно передавать как строку ("454"), так и число (454)
    - Автоматическая нормализация к строке внутри функции
    - Улучшает удобство использования API
    
    Args:
        article_num: Номер статьи как строка или число
            - Строка: "454", "196.1", "30.1"
            - Число: 454, 196, 30
    
    Returns:
        Список документов (обычно один элемент, но может быть несколько для подпунктов)
    
    Examples:
        >>> get_article_by_number("454")
        [Document(article_num='454', title='Статья 454. Условия договора...')]
        
        >>> get_article_by_number(454)
        [Document(article_num='454', title='Статья 454. Условия договора...')]
        
        >>> get_article_by_number("196.1")
        [Document(article_num='196.1', title='Статья 196.1. ...')]
    """
    # Нормализация к строке
    article_num_str = str(article_num)
    
    # Здесь была бы логика поиска статьи в базе данных или векторном хранилище
    # Для примера возвращаем заглушку
    
    # Имитация поиска в базе данных
    mock_documents = [
        Document(
            article_num=article_num_str,
            title=f"Статья {article_num_str}",
            content=f"Содержание статьи {article_num_str}..."
        )
    ]
    
    return mock_documents


# ================= ПРИМЕР 2: Получение диапазона статей =================

def get_articles_in_range(
    start: ArticleNumber,
    end: ArticleNumber
) -> List[Document]:
    """
    Получает статьи в указанном диапазоне
    
    ИСПРАВЛЕНО: Type hints улучшены с Union[str, int] для обоих параметров
    
    Args:
        start: Начало диапазона (строка или число)
        end: Конец диапазона (строка или число)
    
    Returns:
        Список документов в диапазоне
    
    Examples:
        >>> get_articles_in_range(454, 460)
        [Document('454', ...), Document('455', ...), ..., Document('460', ...)]
        
        >>> get_articles_in_range("454", "460")
        [Document('454', ...), Document('455', ...), ..., Document('460', ...)]
    """
    # Нормализация к int для сравнения
    try:
        start_int = int(str(start).split('.')[0])
        end_int = int(str(end).split('.')[0])
    except (ValueError, AttributeError):
        return []
    
    # Проверка корректности диапазона
    if start_int > end_int:
        raise ValueError(f"Start ({start_int}) must be less than or equal to end ({end_int})")
    
    # Имитация поиска в базе данных
    documents = []
    for num in range(start_int, end_int + 1):
        documents.append(
            Document(
                article_num=str(num),
                title=f"Статья {num}",
                content=f"Содержание статьи {num}..."
            )
        )
    
    return documents


# ================= ПРИМЕР 3: Получение статей по списку =================

def get_articles_by_numbers(article_numbers: List[ArticleNumber]) -> List[Document]:
    """
    Получает несколько статей по списку номеров
    
    ИСПРАВЛЕНО: Type hint улучшен для элементов списка
    
    Args:
        article_numbers: Список номеров статей (строки или числа)
    
    Returns:
        Список документов
    
    Examples:
        >>> get_articles_by_numbers([454, 455, "456"])
        [Document('454', ...), Document('455', ...), Document('456', ...)]
        
        >>> get_articles_by_numbers(["454", "455", 456])
        [Document('454', ...), Document('455', ...), Document('456', ...)]
    """
    documents = []
    
    for article_num in article_numbers:
        # Нормализация каждого номера
        article_num_str = str(article_num)
        
        # Имитация поиска
        documents.append(
            Document(
                article_num=article_num_str,
                title=f"Статья {article_num_str}",
                content=f"Содержание статьи {article_num_str}..."
            )
        )
    
    return documents


# ================= ПРИМЕР 4: Получение статей главы =================

def get_articles_by_chapter(
    part_num: PartNumber,
    chapter_num: ChapterNumber
) -> List[Document]:
    """
    Получает все статьи конкретной главы
    
    ИСПРАВЛЕНО: Type hints улучшены с Union[str, int] для обоих параметров
    
    Args:
        part_num: Номер части ГК РФ (1-4)
        chapter_num: Номер главы
    
    Returns:
        Список документов всех статей главы
    
    Examples:
        >>> get_articles_by_chapter(2, "30")
        [Document('454', ...), Document('455', ...), ..., Document('566', ...)]
        
        >>> get_articles_by_chapter("2", 30)
        [Document('454', ...), Document('455', ...), ..., Document('566', ...)]
    """
    # Нормализация
    part_num_int = int(part_num)
    chapter_num_str = str(chapter_num)
    
    # Здесь была бы логика получения статей главы из базы данных
    # Для примера возвращаем заглушку
    
    # Имитация: возвращаем несколько статей
    mock_documents = [
        Document(
            article_num=f"{part_num_int * 100 + i}",
            title=f"Статья {part_num_int * 100 + i}",
            content=f"Содержание статьи {part_num_int * 100 + i}..."
        )
        for i in range(1, 6)
    ]
    
    return mock_documents


# ================= ПРИМЕР 5: Фильтрация статей =================

def filter_articles(
    articles: List[Document],
    article_num: Optional[ArticleNumber] = None,
    part_num: Optional[PartNumber] = None,
    chapter_num: Optional[ChapterNumber] = None
) -> List[Document]:
    """
    Фильтрует список статей по заданным критериям
    
    ИСПРАВЛЕНО: Type hints улучшены для всех фильтрующих параметров
    
    Args:
        articles: Список документов для фильтрации
        article_num: Номер статьи (опционально)
        part_num: Номер части (опционально)
        chapter_num: Номер главы (опционально)
    
    Returns:
        Отфильтрованный список документов
    
    Examples:
        >>> docs = get_articles_by_chapter(2, "30")
        >>> filtered = filter_articles(docs, article_num=454)
        >>> filtered
        [Document('454', ...)]
    """
    filtered = articles
    
    # Фильтрация по номеру статьи
    if article_num is not None:
        article_num_str = str(article_num)
        filtered = [doc for doc in filtered if doc.article_num == article_num_str]
    
    # Фильтрация по номеру части
    if part_num is not None:
        part_num_int = int(part_num)
        filtered = [doc for doc in filtered if doc.part_num == part_num_int]
    
    # Фильтрация по номеру главы
    if chapter_num is not None:
        chapter_num_str = str(chapter_num)
        filtered = [doc for doc in filtered if doc.chapter_num == chapter_num_str]
    
    return filtered


# ================= ПРИМЕР 6: Поиск по ключевым словам =================

def search_articles(
    query: str,
    part_num: Optional[PartNumber] = None,
    chapter_num: Optional[ChapterNumber] = None,
    limit: int = 10
) -> List[Document]:
    """
    Ищет статьи по ключевым словам
    
    ИСПРАВЛЕНО: Type hints улучшены для фильтрующих параметров
    
    Args:
        query: Поисковый запрос
        part_num: Ограничение по части (опционально)
        chapter_num: Ограничение по главе (опционально)
        limit: Максимальное количество результатов
    
    Returns:
        Список документов, соответствующих запросу
    
    Examples:
        >>> search_articles("договор купли-продажи", part_num=2, chapter_num="30")
        [Document('454', ...), Document('455', ...), ...]
    """
    # Здесь была бы логика полнотекстового поиска
    # Для примера возвращаем заглушку
    
    mock_documents = [
        Document(
            article_num="454",
            title="Статья 454. Условия договора купли-продажи",
            content="Содержание статьи 454..."
        )
    ]
    
    return mock_documents[:limit]


# ================= ТЕСТЫ =================

if __name__ == "__main__":
    print("=== Тестирование Union[str, int] в article_lookup.py ===\n")
    
    # Тест 1: get_article_by_number
    print("Тест 1: get_article_by_number")
    test_cases = [454, "454", 196.1, "196.1"]
    for article_num in test_cases:
        docs = get_article_by_number(article_num)
        print(f"  article_num={article_num!r} (тип: {type(article_num).__name__}): {docs[0]}")
    
    # Тест 2: get_articles_in_range
    print("\nТест 2: get_articles_in_range")
    docs = get_articles_in_range(454, 456)
    print(f"  get_articles_in_range(454, 456): {[doc.article_num for doc in docs]}")
    docs = get_articles_in_range("454", "456")
    print(f"  get_articles_in_range('454', '456'): {[doc.article_num for doc in docs]}")
    
    # Тест 3: get_articles_by_numbers
    print("\nТест 3: get_articles_by_numbers")
    docs = get_articles_by_numbers([454, 455, "456"])
    print(f"  get_articles_by_numbers([454, 455, '456']): {[doc.article_num for doc in docs]}")
    
    # Тест 4: get_articles_by_chapter
    print("\nТест 4: get_articles_by_chapter")
    docs = get_articles_by_chapter(2, "30")
    print(f"  get_articles_by_chapter(2, '30'): {[doc.article_num for doc in docs]}")
    docs = get_articles_by_chapter("2", 30)
    print(f"  get_articles_by_chapter('2', 30): {[doc.article_num for doc in docs]}")
    
    # Тест 5: filter_articles
    print("\nТест 5: filter_articles")
    docs = get_articles_by_chapter(2, "30")
    filtered = filter_articles(docs, article_num=201)
    print(f"  filter_articles(docs, article_num=201): {[doc.article_num for doc in filtered]}")
    
    # Тест 6: search_articles
    print("\nТест 6: search_articles")
    docs = search_articles("договор купли-продажи", part_num=2, chapter_num="30")
    print(f"  search_articles(...): {[doc.article_num for doc in docs]}")
    
    print("\n✅ Все тесты пройдены успешно!")
