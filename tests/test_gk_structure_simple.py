"""
Тесты для модуля gk_structure.py

Проверяем реальные функции из модуля
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gk_structure import (
    get_chapter_for_article,
    get_all_articles_for_chapter,
    get_chapter_title,
    determine_gk_part
)


class TestGetChapterForArticle:
    """Тесты для функции get_chapter_for_article"""
    
    def test_get_chapter_for_article_string(self):
        """Тест получения главы по строковому номеру статьи"""
        chapter_num, chapter_title, part_num = get_chapter_for_article("454")
        
        assert chapter_num == "30"
        assert chapter_title == "Купля-продажа"
        assert part_num == 2
    
    def test_get_chapter_for_article_int(self):
        """Тест получения главы по числовому номеру статьи"""
        chapter_num, chapter_title, part_num = get_chapter_for_article(454)
        
        assert chapter_num == "30"
        assert chapter_title == "Купля-продажа"
        assert part_num == 2
    
    def test_get_chapter_for_article_part1(self):
        """Тест получения главы для статьи из части 1"""
        chapter_num, chapter_title, part_num = get_chapter_for_article(196)
        
        assert part_num == 1
        assert chapter_num is not None
    
    def test_get_chapter_for_article_part3(self):
        """Тест получения главы для статьи из части 3"""
        chapter_num, chapter_title, part_num = get_chapter_for_article(1110)
        
        assert part_num == 3
        assert chapter_num == "61"
    
    def test_get_chapter_for_article_invalid(self):
        """Тест получения главы для невалидной статьи"""
        chapter_num, chapter_title, part_num = get_chapter_for_article(99999)
        
        assert chapter_num is None
        assert chapter_title == "Неизвестно"


class TestGetAllArticlesForChapter:
    """Тесты для функции get_all_articles_for_chapter"""
    
    def test_get_all_articles_for_chapter_int(self):
        """Тест получения всех статей главы (int)"""
        articles = get_all_articles_for_chapter(2, 30)
        
        assert len(articles) > 0
        assert articles[0] == 454
        assert articles[-1] == 566
    
    def test_get_all_articles_for_chapter_string(self):
        """Тест получения всех статей главы (string)"""
        articles = get_all_articles_for_chapter("2", "30")
        
        assert len(articles) > 0
        assert articles[0] == 454
        assert articles[-1] == 566
    
    def test_get_all_articles_for_chapter_invalid_chapter(self):
        """Тест получения статей для несуществующей главы"""
        articles = get_all_articles_for_chapter(2, 999)
        
        assert len(articles) == 0
    
    def test_get_all_articles_for_chapter_invalid_part(self):
        """Тест получения статей для несуществующей части"""
        articles = get_all_articles_for_chapter(9, 30)
        
        assert len(articles) == 0


class TestGetChapterTitle:
    """Тесты для функции get_chapter_title"""
    
    def test_get_chapter_title_int(self):
        """Тест получения названия главы (int)"""
        title = get_chapter_title(2, 30)
        
        assert title == "Купля-продажа"
    
    def test_get_chapter_title_string(self):
        """Тест получения названия главы (string)"""
        title = get_chapter_title("2", "30")
        
        assert title == "Купля-продажа"
    
    def test_get_chapter_title_invalid_chapter(self):
        """Тест получения названия для несуществующей главы"""
        title = get_chapter_title(2, 999)
        
        assert title is None
    
    def test_get_chapter_title_invalid_part(self):
        """Тест получения названия для несуществующей части"""
        title = get_chapter_title(9, 30)
        
        assert title is None


class TestDetermineGKPart:
    """Тесты для функции determine_gk_part"""
    
    def test_determine_gk_part_part1_string(self):
        """Тест определения части 1 (string)"""
        part = determine_gk_part("1")
        
        assert part == 1
    
    def test_determine_gk_part_part1_int(self):
        """Тест определения части 1 (int)"""
        part = determine_gk_part(1)
        
        assert part == 1
    
    def test_determine_gk_part_part2(self):
        """Тест определения части 2"""
        part = determine_gk_part(454)
        
        assert part == 2
    
    def test_determine_gk_part_part3(self):
        """Тест определения части 3"""
        part = determine_gk_part(1110)
        
        assert part == 3
    
    def test_determine_gk_part_part4(self):
        """Тест определения части 4"""
        part = determine_gk_part(1225)
        
        assert part == 4
    
    def test_determine_gk_part_invalid(self):
        """Тест определения части для невалидной статьи"""
        part = determine_gk_part(99999)
        
        assert part == 1  # Default fallback
    
    def test_determine_gk_part_float(self):
        """Тест определения части для дробного номера"""
        part = determine_gk_part(454.1)
        
        assert part == 2
