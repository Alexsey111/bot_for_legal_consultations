# Тесты для проекта bot_for_legal_consultations

## Структура тестов

```
tests/
├── __init__.py              # Пакет тестов
├── conftest.py              # Фикстуры pytest
├── test_security.py         # Тесты модуля безопасности
├── test_database.py         # Тесты векторной БД
├── test_rag_engine.py       # Тесты RAG движка
├── test_sql_logger.py       # Тесты SQLite БД
├── test_article_lookup.py   # Тесты поиска по статьям
├── test_gk_structure.py     # Тесты структуры ГК РФ
└── test_integration.py      # Интеграционные тесты
```

## Запуск тестов

### Все тесты
```bash
pytest
```

### С подробным выводом
```bash
pytest -v
```

### Запуск определённого файла
```bash
pytest tests/test_security.py
```

### Запуск определённого теста
```bash
pytest tests/test_security.py::TestEncryptionManager::test_encrypt_decrypt_success
```

### Запуск по маркерам
```bash
pytest -m unit          # Только unit-тесты
pytest -m integration   # Только интеграционные тесты
pytest -m "not slow"    # Все тесты кроме медленных
```

### С покрытием кода
```bash
pytest --cov=. --cov-report=html --cov-report=term-missing
```

### Параллельный запуск (если установлен pytest-xdist)
```bash
pytest -n auto
```

## Маркеры тестов

- `@pytest.mark.unit` — Unit-тесты
- `@pytest.mark.integration` — Интеграционные тесты
- `@pytest.mark.slow` — Медленные тесты
- `@pytest.mark.database` — Тесты с базой данных
- `@pytest.mark.encryption` — Тесты шифрования
- `@pytest.mark.async` — Асинхронные тесты

## Покрытие кода

Целевое покрытие: **80%+**

```bash
# Генерация отчёта в HTML
pytest --cov=. --cov-report=html

# Просмотр отчёта
# Откройте htmlcov/index.html в браузере
```

## Фикстуры

Основные фикстуры в `conftest.py`:

- `mock_openai_client` — Mock для OpenAI API
- `mock_embeddings` — Mock для embeddings
- `sample_documents` — Примеры документов
- `temp_db_path` — Временный путь к БД
- `encryption_manager` — Менеджер шифрования
- `sample_gk_text` — Пример текста ГК РФ
- `mock_chroma_db` — Mock для ChromaDB
- `mock_redis` — Mock для Redis

## Написание новых тестов

### Unit-тест
```python
import pytest
from unittest.mock import Mock

def test_function():
    result = your_function()
    assert result == expected
```

### Асинхронный тест
```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await your_async_function()
    assert result == expected
```

### С фикстурой
```python
def test_with_fixture(mock_openai_client):
    result = your_function(mock_openai_client)
    assert result is not None
```

## CI/CD

Тесты автоматически запускаются в CI/CD пайплайне:
- При каждом push в ветку
- При создании Pull Request
- Перед слиянием в main
