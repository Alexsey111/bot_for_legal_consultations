"""
Тесты для модуля article_lookup.py

Проверяем реальные функции и классы из модуля
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from article_lookup import Document, get_article_by_number, get_articles_in_range, get_articles_by_numbers


class TestDocument:
    """Тесты для Document класса"""
    
    def test_document_creation(self):
        """Тест создания документа"""
        doc = Document(
            article_num=1,
            title="Тестовая статья",
            content="Текст статьи"
        )
        
        assert doc.article_num == "1"
        assert doc.title == "Тестовая статья"
        assert doc.content == "Текст статьи"
    
    def test_document_with_int_article_num(self):
        """Тест создания документа с числовым номером статьи"""
        doc = Document(
            article_num=454,
            title="Статья 454",
            content="Текст"
        )
        
        assert doc.article_num == "454"  # Должен быть преобразован в строку
    
    def test_document_repr(self):
        """Тест строкового представления документа"""
        doc = Document(
            article_num=196,
            title="Исковая давность",
            content="Текст"
        )
        
        repr_str = repr(doc)
        assert "196" in repr_str
        assert "Исковая давность" in repr_str


class TestGetArticleByNumber:
    """Тесты для функции get_article_by_number"""
    
    def test_get_article_by_number_string(self):
        """Тест получения статьи по строковому номеру"""
        docs = get_article_by_number("454")
        
        assert len(docs) == 1
        assert docs[0].article_num == "454"
        assert "454" in docs[0].title
    
    def test_get_article_by_number_int(self):
        """Тест получения статьи по числовому номеру"""
        docs = get_article_by_number(454)
        
        assert len(docs) == 1
        assert docs[0].article_num == "454"
    
    def test_get_article_by_number_float(self):
        """Тест получения статьи с дробным номером"""
        docs = get_article_by_number(196.1)
        
        assert len(docs) == 1
        assert docs[0].article_num == "196.1"


class TestGetArticlesInRange:
    """Тесты для функции get_articles_in_range"""
    
    def test_get_articles_in_range_int(self):
        """Тест получения статей в диапазоне (int)"""
        docs = get_articles_in_range(454, 456)
        
        assert len(docs) == 3
        assert docs[0].article_num == "454"
        assert docs[1].article_num == "455"
        assert docs[2].article_num == "456"
    
    def test_get_articles_in_range_string(self):
        """Тест получения статей в диапазоне (string)"""
        docs = get_articles_in_range("454", "456")
        
        assert len(docs) == 3
        assert docs[0].article_num == "454"
        assert docs[1].article_num == "455"
        assert docs[2].article_num == "456"
    
    def test_get_articles_in_range_invalid(self):
        """Тест получения статей с невалидным диапазоном"""
        with pytest.raises(ValueError):
            get_articles_in_range(456, 454)


class TestGetArticlesByNumbers:
    """Тесты для функции get_articles_by_numbers"""
    
    def test_get_articles_by_numbers_list(self):
        """Тест получения статей по списку номеров"""
        docs = get_articles_by_numbers([454, 455, 456])
        
        assert len(docs) == 3
        assert docs[0].article_num == "454"
        assert docs[1].article_num == "455"
        assert docs[2].article_num == "456"
    
    def test_get_articles_by_numbers_mixed_types(self):
        """Тест получения статей со смешанными типами номеров"""
        docs = get_articles_by_numbers([454, "455", 456])
        
        assert len(docs) == 3
        assert docs[0].article_num == "454"
        assert docs[1].article_num == "455"
        assert docs[2].article_num == "456"
