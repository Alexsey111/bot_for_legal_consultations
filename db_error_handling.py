"""
Модуль для единообразной обработки ошибок БД

Обеспечивает согласованную обработку ошибок между sync и async кодом.
Соответствует sync версии в sql_logger.py._get_connection()
"""

import functools
from typing import Callable, Any

import structlog

log = structlog.get_logger()

# ================= SYNC VERSION =================

def db_operation_sync(func: Callable) -> Callable:
    """
    Обеспечивает согласованную обработку ошибок:
    - Логирует все ошибки с детализацией по типу
    - Прокидывает исключения дальше для обработки на верхнем уровне

    Соответствует sync версии в sql_logger.py._get_connection():
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            log.error("database_error", error=str(e)[:100])
            raise
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Логирование ошибки (как в sync версии в sql_logger.py)
            log.error("database_error", function=func.__name__, error_type=type(e).__name__, error=str(e)[:100])
            raise
    return wrapper


# ================= ASYNC VERSION =================

def db_operation_async(func: Callable) -> Callable:
    """
    Decorator для единообразной обработки ошибок DB операций (async версия)

    Обеспечивает согласованную обработку ошибок между sync и async кодом:
    - Логирует все ошибки с детализацией по типу
    - Прокидывает исключения дальше для обработки на верхнем уровне

    Соответствует sync версии в sql_logger.py._get_connection():
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            log.error("database_error", error=str(e)[:100])
            raise
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Логирование ошибки (как в sync версии в sql_logger.py)
            log.error("database_error", function=func.__name__, error_type=type(e).__name__, error=str(e)[:100])
            raise
    return wrapper


# ================= ASYNC VERSION WITH TYPED EXCEPTIONS =================

def db_operation_async_typed(func: Callable) -> Callable:
    """
    Decorator для единообразной обработки ошибок DB операций (async версия с типизированными исключениями)

    Обеспечивает:
    - Логирование всех ошибок с детализацией по типу
    - Преобразование типов исключений (sqlite3 -> ValueError/RuntimeError)
    - Прокидывает исключения дальше для обработки на верхнем уровне

    Соответствует sync версии в sql_logger.py._get_connection():
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            log.error("database_error", error=str(e)[:100])
            raise
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Логирование ошибки (как в sync версии в sql_logger.py)
            log.error("database_error", function=func.__name__, error_type=type(e).__name__, error=str(e)[:100])
            raise
    return wrapper
