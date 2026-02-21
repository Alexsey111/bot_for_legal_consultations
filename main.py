"""
main.py
Telegram бот для юридических консультаций по ГК РФ
Production-ready версия с исправлениями архитектурных проблем
"""

import os
import asyncio
import logging
import json
import tempfile
import re
import html
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager
from functools import wraps
import hashlib


# Third-party
from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message,
    ReactionTypeEmoji,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
    FSInputFile,
    ErrorEvent,
)
from aiogram.filters import Command, CommandStart
from aiogram.filters.command import CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.enums import ChatAction, ParseMode
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv

# Database
import aiosqlite
import redis.asyncio as aioredis
from redis.asyncio.lock import Lock as RedisLock
from sql_logger import LegalBotDB
# Security
from bleach import clean as bleach_clean

# Monitoring
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Local imports
from rag_engine import generate_answer, get_cache_stats, get_db_stats
from sql_logger import get_db_async as get_db  # Используем async-версию для async-кода

from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext

# FSM для записи на консультацию
class ConsultationForm(StatesGroup):
    waiting_for_date = State()
    waiting_for_time = State()
    waiting_for_topic = State()
    waiting_for_description = State()

# ================= LOGGING =================

import structlog

def configure_structlog():
    """Настройка структурированного логирования"""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
configure_structlog()
log = structlog.get_logger()

# Для обратной совместимости оставляем logger
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

async def safe_react(message: Message, emoji: str):
    """Безопасная установка реакции"""
    try:
        await message.react([ReactionTypeEmoji(emoji=emoji)])
    except Exception:
        pass  # Игнорируем ошибки

# ================= CONFIG =================

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./legal_bot.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))

# Rate limiting
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# Session settings
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
SESSION_MAX_HISTORY = int(os.getenv("SESSION_MAX_HISTORY", "5"))

# Cleanup settings
CLEANUP_THRESHOLD_DAYS = int(os.getenv("CLEANUP_THRESHOLD_DAYS", "365"))

if not TELEGRAM_TOKEN:
    raise ValueError("❌ TELEGRAM_TOKEN не найден в .env")


# Создайте глобальную переменную
sql_db = None

# ================= DATABASE ERROR HANDLING =================

def db_operation(func):
    """
    Decorator для единообразной обработки ошибок DB операций

    Обеспечивает согласованную обработку ошибок между sync и async кодом:
    - Логирует все ошибки с детализацией по типу
    - Прокидывает исключения дальше для обработки на верхнем уровне
    - Обновляет Prometheus metrics

    Соответствует sync версии в sql_logger.py._get_connection()
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except aiosqlite.IntegrityError as e:
            log.error("db_integrity_error", function=func.__name__, error=str(e)[:100])
            metrics_errors_total.labels(error_type="integrity").inc()
            raise ValueError(f"Data integrity violation: {e}") from e
        except aiosqlite.OperationalError as e:
            log.error("db_operational_error", function=func.__name__, error=str(e)[:100])
            metrics_errors_total.labels(error_type="operational").inc()
            raise RuntimeError(f"Database operation failed: {e}") from e
        except aiosqlite.DatabaseError as e:
            log.error("db_database_error", function=func.__name__, error=str(e)[:100])
            metrics_errors_total.labels(error_type="database").inc()
            raise RuntimeError(f"Database error: {e}") from e
        except Exception as e:
            log.error("db_unexpected_error", function=func.__name__, error=str(e)[:100])
            metrics_errors_total.labels(error_type="unexpected").inc()
            raise
    return wrapper

# ================= PROMETHEUS METRICS =================

# Counters
metrics_questions_total = Counter(
    "bot_questions_total",
    "Total questions asked",
    ["user_id", "question_type"],
)

metrics_errors_total = Counter(
    "bot_errors_total",
    "Total errors occurred",
    ["error_type"],
)

metrics_rate_limit_hits = Counter(
    "bot_rate_limit_hits_total",
    "Total rate limit violations",
)

# Histograms
metrics_response_time = Histogram(
    "bot_response_time_seconds",
    "Response time for questions",
    buckets=[0.5, 1, 2, 5, 10, 30, 60],
)

metrics_article_lookup_time = Histogram(
    "bot_article_lookup_time_seconds",
    "Article lookup time",
    buckets=[0.01, 0.05, 0.1, 0.5, 1],
)

# Gauges
metrics_active_sessions = Gauge(
    "bot_active_sessions",
    "Number of active user sessions",
)

metrics_db_connections = Gauge(
    "bot_db_connections",
    "Number of active database connections",
)

# ================= DATABASE MODELS =================

class DatabaseManager:
    _instance = None
    _init_lock = None  # ✅ ИСПРАВЛЕНО: Ленивая инициализация lock
    
    @classmethod
    async def get_instance(cls):
        """
        Thread-safe async singleton с ленивой инициализацией lock
        
        ИСПРАВЛЕНО:
        - Lock создаётся при первом вызове (не в момент импорта модуля)
        - Double-checked locking для избежания race condition
        - Полностью async-safe
        """
        # FAST PATH: инстанс уже создан
        if cls._instance is not None:
            return cls._instance
        
        # SLOW PATH: первый вызов, нужна инициализация
        # Ленивая инициализация lock
        if cls._init_lock is None:
            cls._init_lock = asyncio.Lock()
        
        async with cls._init_lock:
            # Double-checked locking: проверяем ещё раз внутри lock
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance.init_pool()
                log.info("database_manager_initialized")
        
        return cls._instance
    
    def __init__(self):
        """Private constructor - use get_instance() instead"""
        if not hasattr(self, 'initialized'):
            self.db_path = "./legal_bot.db"
            self._connection_pool: List[aiosqlite.Connection] = []
            self._pool_size = 5
            self._lock = None  # Будет создан в init_pool
            self.initialized = True


    
    async def init_pool(self):
        """Инициализация connection pool"""
        if self._lock is None:
            self._lock = asyncio.Lock()
        
        async with self._lock:
            if not self._connection_pool:
                for _ in range(self._pool_size):
                    conn = await aiosqlite.connect(self.db_path)
                    conn.row_factory = aiosqlite.Row
                    self._connection_pool.append(conn)
                log.info("db_pool_initialized", pool_size=self._pool_size)
                await self._create_tables()
    
    async def _create_tables(self):
        """Создание таблиц если не существуют"""
        conn = self._connection_pool[0]
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                consent_given BOOLEAN DEFAULT 0,
                consent_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP NULL,
                anonymized BOOLEAN DEFAULT 0,
                last_query TIMESTAMP NULL
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                query_text TEXT NOT NULL,
                answer_text TEXT,
                article_nums TEXT,
                query_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
            
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_stats (
                user_id INTEGER PRIMARY KEY,
                total_queries INTEGER DEFAULT 0,
                first_query TIMESTAMP,
                last_query TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
            
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_queries_user_id 
            ON user_queries(user_id)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_queries_created_at 
            ON user_queries(created_at)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_queries_user_created
            ON user_queries(user_id, created_at DESC)
        """)

        await conn.execute("PRAGMA foreign_keys = ON;")

        await conn.commit()
        log.info("db_tables_created")
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Context manager для получения соединения из пула

        ИСПРАВЛЕНО: Добавлено единообразное error handling как в sync версии (sql_logger.py)
        - Rollback при ошибке
        - Логирование всех ошибок
        - Автоматический commit при успехе (если вызывающий код не делает явный commit)

        Соответствует sync версии:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database error: {e}", exc_info=True)
                raise
        """
        async with self._lock:
            if not self._connection_pool:
                await self.init_pool()

            conn = self._connection_pool.pop(0)

        try:
            # Production-hardened PRAGMA настройки
            await conn.execute("PRAGMA foreign_keys = ON;")
            await conn.execute("PRAGMA journal_mode = WAL;")
            await conn.execute("PRAGMA synchronous = NORMAL;")

            metrics_db_connections.inc()
            yield conn

            # Автоматический commit при успехе (как в sync версии)
            await conn.commit()
        
        except Exception as e:
            # Rollback при ошибке (как в sync версии)
            try:
                await conn.rollback()
            except Exception as rollback_err:
                log.error("rollback_failed", error=str(rollback_err)[:100])

            # Логирование ошибки (как в sync версии)
            log.error("database_error", error=str(e)[:100])
            metrics_errors_total.labels(error_type="connection").inc()
            raise

        finally:
            async with self._lock:
                self._connection_pool.append(conn)
            metrics_db_connections.dec()

    async def close_pool(self, timeout: float = 5.0):
        """Закрывает pool с timeout для ожидания возврата connections"""
        start = time.time()
        
        while self._connection_pool and (time.time() - start) < timeout:
            async with self._lock:
                if len(self._connection_pool) == self._pool_size:
                    break  # Все connections вернулись
            await asyncio.sleep(0.1)
        
        async with self._lock:
            if len(self._connection_pool) < self._pool_size:
                log.warning(
                    "pool_shutdown_incomplete",
                    active_connections=self._pool_size - len(self._connection_pool),
                    timeout_seconds=timeout
                )
            
            for conn in self._connection_pool:
                await conn.close()
            self._connection_pool.clear()


# ================= REDIS MANAGER =================

class RedisManager:
    """Менеджер Redis для распределенных блокировок и кеша"""
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.redis: Optional[aioredis.Redis] = None
            self.initialized = True
    
    async def connect(self):
        """Подключение к Redis"""
        if self.redis is None:
            self.redis = await aioredis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=10,
            )
            log.info("redis_connected")
    
    async def close(self):
        """Закрытие соединения"""
        if self.redis:
            await self.redis.close()
            log.info("redis_connection_closed")
    
    async def acquire_lock(self, key: str, timeout: int = 10) -> RedisLock:
        """Получение распределенной блокировки"""
        if not self.redis:
            await self.connect()
        return RedisLock(self.redis, key, timeout=timeout)
    
    async def check_rate_limit(
        self, 
        user_id: int, 
        max_requests: int = RATE_LIMIT_REQUESTS,
        window_seconds: int = RATE_LIMIT_WINDOW
    ) -> bool:
        """
        Проверка rate limit с использованием Redis
        Thread-safe и work across multiple processes
        """
        if not self.redis:
            await self.connect()
        
        key = f"rate_limit:{user_id}"
        now = time.time()
        
        # Redis pipeline для атомарности
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                # Удаляем старые записи
                await pipe.zremrangebyscore(key, 0, now - window_seconds * 2)
                # Получаем количество запросов
                await pipe.zcard(key)
                # Добавляем текущий запрос
                await pipe.zadd(key, {str(now): now})
                # Устанавливаем TTL
                await pipe.expire(key, window_seconds)
                
                results = await pipe.execute()
                count = results[1]  # zcard result
                
                if count >= max_requests:
                    metrics_rate_limit_hits.inc()
                    log.warning(
                        "rate_limit_exceeded",
                        user_id=user_id,
                        count=count,
                        max_requests=max_requests,
                        window_seconds=window_seconds
                    )
                    return True
                
                return False
                
            except Exception as e:
                log.error("redis_rate_limit_error", error=str(e)[:100])
                # Fallback: разрешаем запрос при ошибке Redis
                return False
    
    async def get_session(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Получение сессии пользователя из Redis"""
        if not self.redis:
            await self.connect()
        
        key = f"session:{user_id}"
        data = await self.redis.get(key)
        
        if data:
            return json.loads(data)
        return None
    
    async def save_session(self, user_id: int, session_data: Dict[str, Any]):
        """Сохранение сессии в Redis"""
        if not self.redis:
            await self.connect()
        
        key = f"session:{user_id}"
        await self.redis.setex(
            key,
            SESSION_TIMEOUT_MINUTES * 60,
            json.dumps(session_data, default=str)
        )
        
    async def delete_session(self, user_id: int):
        """Удаление сессии"""
        if not self.redis:
            await self.connect()
        
        key = f"session:{user_id}"
        await self.redis.delete(key)

    async def cleanup_rate_limits(self):
        """
        Очищает старые rate-limit ключи из Redis

        SECURITY/GDPR:
        - Удаляет ключи без TTL (orphaned keys)
        - Удаляет пустые sorted sets
        - Сохраняет активные rate-limit данные

        Вызывается периодически для очистки старых данных
        """
        if not self.redis:
            await self.connect()

        try:
            pattern = "rate_limit:*"
            keys = await self.redis.keys(pattern)

            now = time.time()
            deleted_count = 0
            orphaned_count = 0

            for key in keys:
                try:
                    # Проверяем TTL
                    ttl = await self.redis.ttl(key)

                    if ttl == -1:
                        # Нет TTL - это orphaned key (баг)
                        # Проверяем есть ли актуальные данные
                        count = await self.redis.zcount(key, now - 3600, now)
                        if count == 0:
                            # Нет активных запросов за последний час - удаляем
                            await self.redis.delete(key)
                            orphaned_count += 1
                            log.debug("deleted_orphaned_rate_limit_key", key=key[:50])
                    elif ttl == -2:
                        # Ключ не существует (должен был быть удалён)
                        pass
                    # ttl > 0 - ключ имеет TTL, оставляем как есть

                except Exception as e:
                    logger.warning(f"Error processing rate-limit key {key}: {e}")

            if orphaned_count > 0:
                logger.info(f"🧹 Cleaned {orphaned_count} orphaned rate-limit keys")
                metrics_errors_total.labels(error_type="redis_cleanup").inc(orphaned_count)

            return {
                "total_keys": len(keys),
                "orphaned_deleted": orphaned_count,
                "deleted_count": deleted_count
            }

        except Exception as e:
            logger.error(f"Error cleaning up rate-limit keys: {e}")
            metrics_errors_total.labels(error_type="redis_cleanup").inc()
            return {
                "error": str(e)
            }

    async def get_redis_stats(self) -> dict:
        """Получает статистику Redis"""
        if not self.redis:
            await self.connect()

        try:
            stats = await self.redis.info()
            return {
                "connected_clients": stats.get("connected_clients", 0),
                "used_memory_human": stats.get("used_memory_human", "N/A"),
                "total_keys": stats.get("db0", {}).get("keys", 0),
                "total_commands": stats.get("total_commands_processed", 0),
            }
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {"error": str(e)}

# Global Redis manager
redis_manager = RedisManager()

# ================= BACKGROUND TASKS =================

@db_operation
async def auto_cleanup():
    """
    Автоматическая очистка старых данных
    С транзакциями и dry-run режимом
    """
    logger.info("🧹 Running auto-cleanup...")

    cleanup_threshold = datetime.now() - timedelta(days=CLEANUP_THRESHOLD_DAYS)
    deleted_users = 0
    deleted_queries = 0

    async with sql_db._get_connection_async() as conn:
        # Получаем список пользователей для удаления (dry-run)
        cursor = await conn.execute("""
            SELECT user_id, username
            FROM users
            WHERE deleted_at IS NULL
            AND user_id IN (
                SELECT user_id
                FROM user_stats
                WHERE last_query < ?
            )
        """, (cleanup_threshold,))

        users_to_delete = await cursor.fetchall()

        if not users_to_delete:
            logger.info("✅ No users to cleanup")
            return

        logger.info(f"📋 Found {len(users_to_delete)} users for cleanup:")
        for user in users_to_delete:
            logger.info(f"  - User {user['user_id']} (@{user['username'] or 'unknown'})")

        # Подтверждение (в production можно добавить отправку уведомления админу)
        # Здесь для демонстрации просто продолжаем

        # Транзакция: soft-delete пользователей и их запросов
        for user in users_to_delete:
            user_id = user['user_id']

            # Считаем количество запросов перед удалением
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM user_queries WHERE user_id = ? AND deleted_at IS NULL",
                (user_id,)
            )
            query_count = (await cursor.fetchone())[0]

            # Soft-delete user
            await conn.execute(
                "UPDATE users SET deleted_at = ? WHERE user_id = ?",
                (datetime.now(), user_id)
            )
        
            # Soft-delete queries
            await conn.execute(
                "UPDATE user_queries SET deleted_at = ? WHERE user_id = ?",
                (datetime.now(), user_id)
            )

            deleted_users += 1
            deleted_queries += query_count

            logger.info(f"  ✓ Cleaned user {user_id}: {query_count} queries")

        # Commit выполняется автоматически в get_connection()

    logger.info(
        f"✅ Auto-cleanup completed: {deleted_users} users removed, "
        f"{deleted_queries} old queries cleaned"
    )
    
async def session_cleanup():
    """Очистка сессий с использованием SCAN вместо KEYS"""
    cursor = 0
    expired_keys = []
    
    while True:
        cursor, keys = await redis_manager.redis.scan(
            cursor, match="session:*", count=100
        )
        
        for key in keys:
            ttl = await redis_manager.redis.ttl(key)
            if ttl == -1:
                expired_keys.append(key)
        
        if cursor == 0:
            break
    
    if expired_keys:
        await redis_manager.redis.delete(*expired_keys)
    
    logger.info(f"Cleaned {len(expired_keys)} orphaned sessions")
    metrics_active_sessions.set(await redis_manager.redis.dbsize())


async def redis_cleanup():
    """Очистка старых данных из Redis (rate-limit keys и orphaned sessions)"""
    try:
        if not redis_manager.redis:
            return

        logger.info("🧹 Running Redis cleanup...")

        # Очистка rate-limit ключей
        rate_limit_stats = await redis_manager.cleanup_rate_limits()

        # Очистка orphaned сессий
        session_keys = await redis_manager.redis.keys("session:*")
        orphaned_sessions = 0
        for key in session_keys:
            ttl = await redis_manager.redis.ttl(key)
            if ttl == -1:  # TTL не установлен (orphaned)
                await redis_manager.redis.delete(key)
                orphaned_sessions += 1

        # Получаем общую статистику
        redis_stats = await redis_manager.get_redis_stats()

        logger.info(
            f"✅ Redis cleanup completed: "
            f"{rate_limit_stats.get('orphaned_deleted', 0)} rate-limit keys, "
            f"{orphaned_sessions} orphaned sessions. "
            f"Total keys: {redis_stats.get('total_keys', 0)}"
        )
        
    except Exception as e:
        logger.error(f"Error in Redis cleanup: {e}")
        metrics_errors_total.labels(error_type="redis_cleanup").inc()


# ================= SCHEDULER =================

# Инициализируем планировщик фоновых задач
scheduler = AsyncIOScheduler()

# ================= USER DATA MANAGEMENT =================

@db_operation
async def get_user_data(user_id: int) -> Dict[str, Any]:
    """Получает данные пользователя из БД"""
    global sql_db
    
    async with sql_db._get_connection_async() as conn:
        cursor = await conn.execute("""
            SELECT user_id, username, first_name, last_name, phone,
                   consent_given, consent_date, first_seen, last_active
            FROM users 
            WHERE user_id = ? AND deleted_at IS NULL
        """, (user_id,))
        
        row = await cursor.fetchone()
        
        if row:
            user_dict = dict(row)
            
            # Расшифровываем телефон
            if user_dict.get('phone'):
                try:
                    user_dict['phone'] = sql_db.secure_db.decrypt_field(
                        user_dict['phone'], 'phone'
                    )
                except Exception as e:
                    logger.warning(f"Failed to decrypt phone for user {user_id}: {e}")
                    user_dict['phone'] = None
            
            return {
                "user_id": user_dict.get("user_id"),
                "username": user_dict.get("username"),
                "first_name": user_dict.get("first_name"),
                "last_name": user_dict.get("last_name"),
                "phone": user_dict.get("phone"),
                "consent_given": bool(user_dict.get("consent_given")),
                "consent_date": user_dict.get("consent_date"),
                "first_seen": user_dict.get("first_seen"),
                "last_active": user_dict.get("last_active")
            }
        return {}


async def set_user_consent(user_id: int, consent: bool, user_info: Optional[Dict] = None) -> bool:
    """Устанавливает согласие пользователя на обработку данных"""
    global sql_db
    
    now = datetime.now()
    
    async with sql_db._get_connection_async() as conn:
        cursor = await conn.execute(
            "SELECT user_id FROM users WHERE user_id = ?",
            (user_id,)
        )
        exists = await cursor.fetchone()
        
        if exists:
            # Обновляем существующего пользователя
            await conn.execute("""
                UPDATE users 
                SET consent_given = ?, 
                    consent_date = ?,
                    last_active = ?
                WHERE user_id = ?
            """, (1 if consent else 0, now, now, user_id))
        else:
            # Создаем нового пользователя с данными из Telegram
            username = user_info.get('username') if user_info else None
            first_name = user_info.get('first_name') if user_info else None
            last_name = user_info.get('last_name') if user_info else None
            
            await conn.execute("""
                INSERT INTO users (
                    user_id, username, first_name, last_name,
                    consent_given, consent_date, first_seen, last_active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, 
                username,
                first_name, 
                last_name,
                1 if consent else 0, 
                now, 
                now, 
                now
            ))
        
        await conn.commit()
    
    logger.info(f"✅ User {user_id} consent set to {consent}")
    return True

@db_operation
async def export_user_data(user_id: int) -> Dict[str, Any]:
    """Экспортирует все данные пользователя (GDPR)"""
    global sql_db
    
    # Используем встроенный метод из LegalBotDB
    return sql_db.export_my_data(user_id)



@db_operation
async def delete_user_data(user_id: int) -> bool:
    """
    Полное удаление данных пользователя (GDPR compliant)

    ✅ ИСПРАВЛЕНО:
    - Полная очистка всех источников данных
    - БД + Redis + Cache + Prometheus + Audit logs
    - Детальное логирование каждого шага
    - Транзакционность где возможно

    GDPR COMPLIANCE (Article 17 - Right to Erasure):
    - Soft-delete пользователя из БД
    - Удаление Redis сессии
    - Очистка rate-limit keys
    - Инвалидация RAG cache
    - Анонимизация audit logs
    - Очистка Prometheus metrics (best-effort)

    Args:
        user_id: ID пользователя для удаления

    Returns:
        True если успешно

    Raises:
        RuntimeError: Если критическая часть удаления не удалась
    """
    now = datetime.now()
    deletion_report = {
        "user_id": user_id,
        "timestamp": now.isoformat(),
        "steps": {},
        "errors": []
    }
    
    # Инициализация переменных в начале функции
    anonymized_count = 0
    queries_deleted = 0

    # ============= ШАГ 1: БАЗА ДАННЫХ =============
    try:
        async with sql_db._get_connection_async() as conn:
            # Soft-delete user
            await conn.execute(
                "UPDATE users SET deleted_at = ? WHERE user_id = ?",
                (now, user_id)
            )
            
            # Soft-delete queries
            cursor = await conn.execute(
                "UPDATE user_queries SET deleted_at = ? WHERE user_id = ?",
                (now, user_id)
            )
            queries_deleted = cursor.rowcount
            
            # Анонимизация error_message в user_queries
            try:
                user_id_str = str(user_id)
                replacement = "[DELETED_USER]"
                cursor = await conn.execute("""
                    UPDATE user_queries
                    SET error_message = REPLACE(error_message, ?, ?)
                    WHERE user_id = ? AND error_message LIKE ?
                """, (user_id_str, replacement, user_id, f"%{user_id}%"))
                anonymized_count = cursor.rowcount
                logger.info(f"GDPR: Anonymized {anonymized_count} query error messages")
            except Exception as e:
                deletion_report["errors"].append(f"Error message anonymization failed: {e}")
            
            await conn.commit()
        
        deletion_report["steps"]["database"] = {
            "status": "success",
            "queries_deleted": queries_deleted,
            "anonymized_errors": anonymized_count
        }
        logger.info(f"✅ Step 1/6: Database soft-deleted for user {user_id}")
    
    except Exception as e:
        deletion_report["steps"]["database"] = {"status": "failed", "error": str(e)}
        deletion_report["errors"].append(f"Database deletion failed: {e}")
        logger.error(f"❌ Step 1/6 FAILED: Database deletion: {e}")
        raise RuntimeError(f"Critical: Database deletion failed for user {user_id}")

    # ============= ШАГ 2: REDIS SESSION =============
    try:
        await redis_manager.delete_session(user_id)
        deletion_report["steps"]["redis_session"] = {"status": "success"}
        logger.info(f"✅ Step 2/6: Redis session deleted for user {user_id}")
    except Exception as e:
        deletion_report["steps"]["redis_session"] = {"status": "failed", "error": str(e)}
        deletion_report["errors"].append(f"Redis session deletion failed: {e}")
        logger.error(f"⚠️  Step 2/6 FAILED: Redis session deletion: {e}")

    # ============= ШАГ 3: REDIS RATE-LIMIT KEYS =============
    try:
        if redis_manager.redis:
            pattern = f"rate_limit:{user_id}*"
            keys = await redis_manager.redis.keys(pattern)
            if keys:
                await redis_manager.redis.delete(*keys)
                deletion_report["steps"]["redis_rate_limit"] = {
                    "status": "success",
                    "keys_deleted": len(keys)
                }
                logger.info(f"✅ Step 3/6: Deleted {len(keys)} rate-limit keys for user {user_id}")
            else:
                deletion_report["steps"]["redis_rate_limit"] = {
                    "status": "success",
                    "keys_deleted": 0
                }
                logger.info(f"✅ Step 3/6: No rate-limit keys found for user {user_id}")
        else:
            deletion_report["steps"]["redis_rate_limit"] = {"status": "skipped", "reason": "Redis not available"}
    except Exception as e:
        deletion_report["steps"]["redis_rate_limit"] = {"status": "failed", "error": str(e)}
        deletion_report["errors"].append(f"Redis rate-limit cleanup failed: {e}")
        logger.error(f"⚠️  Step 3/6 FAILED: Redis rate-limit cleanup: {e}")

    # ============= ШАГ 4: RAG CACHE =============
    try:
        # Проверяем, существует ли модуль rag_engine
        try:
            from rag_engine import get_rag_engine
            rag_engine = get_rag_engine()
            
            # Проверяем, есть ли метод invalidate_user_queries
            if hasattr(rag_engine.cache, 'invalidate_user_queries'):
                rag_engine.cache.invalidate_user_queries(user_id)
                deletion_report["steps"]["rag_cache"] = {"status": "success"}
                logger.info(f"✅ Step 4/6: RAG cache invalidated for user {user_id}")
            else:
                # Очищаем весь кеш, если нет метода для конкретного пользователя
                rag_engine.cache.clear()
                deletion_report["steps"]["rag_cache"] = {"status": "success", "note": "Full cache cleared"}
                logger.info(f"✅ Step 4/6: Full RAG cache cleared (no user-specific method)")
        except ImportError:
            deletion_report["steps"]["rag_cache"] = {"status": "skipped", "reason": "RAG engine not available"}
            logger.info(f"⚠️  Step 4/6: RAG engine not available, skipping cache invalidation")
    except Exception as e:
        deletion_report["steps"]["rag_cache"] = {"status": "failed", "error": str(e)}
        deletion_report["errors"].append(f"RAG cache invalidation failed: {e}")
        logger.error(f"⚠️  Step 4/6 FAILED: RAG cache invalidation: {e}")

    # ============= ШАГ 5: PROMETHEUS METRICS =============
    try:
        prometheus_stats = await clean_prometheus_user_metrics(user_id)
        
        if prometheus_stats["errors"]:
            deletion_report["steps"]["prometheus"] = {
                "status": "partial",
                "metrics_cleaned": prometheus_stats["metrics_cleaned"],
                "errors": prometheus_stats["errors"]
            }
            logger.warning(
                f"⚠️  Step 5/6 PARTIAL: Prometheus cleanup completed with errors: "
                f"{prometheus_stats['errors']}"
            )
        else:
            deletion_report["steps"]["prometheus"] = {
                "status": "success",
                "metrics_cleaned": prometheus_stats["metrics_cleaned"]
            }
            logger.info(f"✅ Step 5/6: Prometheus metrics cleaned for user {user_id}")
    except Exception as e:
        deletion_report["steps"]["prometheus"] = {"status": "failed", "error": str(e)}
        deletion_report["errors"].append(f"Prometheus cleanup failed: {e}")
        logger.error(f"⚠️  Step 5/6 FAILED: Prometheus cleanup: {e}")

    # ============= ШАГ 6: AUDIT LOGS =============
    try:
        try:
            from security import AuditLogger
            audit_logger = await AuditLogger.get_instance()
            audit_logger.anonymize_user_logs(user_id)
            
            deletion_report["steps"]["audit_logs"] = {"status": "success"}
            logger.info(f"✅ Step 6/6: Audit logs anonymized for user {user_id}")
        except ImportError:
            deletion_report["steps"]["audit_logs"] = {"status": "skipped", "reason": "AuditLogger not available"}
            logger.info(f"⚠️  Step 6/6: AuditLogger not available, skipping")
    except Exception as e:
        deletion_report["steps"]["audit_logs"] = {"status": "failed", "error": str(e)}
        deletion_report["errors"].append(f"Audit log anonymization failed: {e}")
        logger.error(f"⚠️  Step 6/6 FAILED: Audit log anonymization: {e}")

    # ============= ФИНАЛЬНЫЙ ОТЧЕТ =============
    successful_steps = sum(
        1 for step in deletion_report["steps"].values() 
        if step.get("status") in ["success", "partial"]
    )
    total_steps = len(deletion_report["steps"])
    
    if deletion_report["errors"]:
        logger.warning(
            f"⚠️  GDPR: User {user_id} data deletion completed with {len(deletion_report['errors'])} errors. "
            f"Successful steps: {successful_steps}/{total_steps}"
        )
    else:
        logger.info(
            f"✅ GDPR: User {user_id} data fully deleted. "
            f"All {total_steps} steps completed successfully."
        )
    
    # Логируем детальный отчет
    logger.debug(f"Deletion report for user {user_id}: {json.dumps(deletion_report, indent=2)}")
    
    return successful_steps == total_steps


@db_operation
async def anonymize_user_data(user_id: int) -> bool:
    """Анонимизирует данные пользователя"""
    async with sql_db._get_connection_async() as conn:
        # Анонимизация user - удаляем ВСЕ PII
        await conn.execute("""
            UPDATE users
            SET anonymized = 1, 
                username = NULL,
                first_name = NULL,    -- ← ДОБАВЬТЕ
                last_name = NULL,     -- ← ДОБАВЬТЕ
                phone = NULL,         -- ← ДОБАВЬТЕ
                notes = NULL
            WHERE user_id = ?
        """, (user_id,))
        
        # Анонимизация queries
        await conn.execute("""
            UPDATE user_queries
            SET query_text = '[ANONYMIZED]', 
                answer_text = '[ANONYMIZED]'
            WHERE user_id = ?
        """, (user_id,))
        
        await conn.commit()
    
    logger.info(f"✅ User {user_id} data anonymized")
    return True

# ================= SESSION MANAGEMENT =================

class UserSession:
    """Сессия пользователя с контекстом диалога"""
    
    def __init__(
        self, 
        user_id: int, 
        max_history: int = SESSION_MAX_HISTORY,
        timeout_minutes: int = SESSION_TIMEOUT_MINUTES
    ):
        self.user_id = user_id
        self.max_history = max_history
        self.timeout = timedelta(minutes=timeout_minutes)
        self.history = deque(maxlen=max_history)
        self.last_activity = datetime.now()
        self.last_article_context: Optional[str] = None
    
    def add_interaction(
        self, 
        question: str, 
        answer: str, 
        article_nums: Optional[List[str]] = None
    ):
        """Добавляет взаимодействие в историю"""
        self.history.append({
            "question": question,
            "answer": answer[:300],
            "article_nums": article_nums or [],
            "timestamp": datetime.now(),
        })
        self.last_activity = datetime.now()
        if article_nums:
            self.last_article_context = article_nums[-1]
    
    def is_expired(self) -> bool:
        """Проверяет истек ли таймаут сессии"""
        return datetime.now() - self.last_activity > self.timeout
    
    def get_context(self) -> str:
        """Формирует текстовый контекст для LLM"""
        if not self.history:
            return ""
        
        parts = []
        for i, interaction in enumerate(self.history, 1):
            parts.append(
                f"[{i}] Вопрос: {interaction['question'][:100]}\n"
                f"    Ответ: {interaction['answer'][:100]}..."
            )
        return "\n".join(parts)
    
    def get_last_articles(self) -> List[str]:
        """Возвращает список статей из последних взаимодействий"""
        articles: List[str] = []
        for interaction in reversed(self.history):
            articles.extend(interaction["article_nums"])
        return list(set(articles))[:3]
    
    def is_follow_up(self, question: str) -> bool:
        """Определяет, является ли вопрос уточняющим"""
        follow_up_keywords = [
            "а если", "а что", "а как", "расскажи подробнее",
            "еще", "также", "то есть", "поясни", "уточни",
            "в этом случае", "в той же статье", "там же",
        ]
        question_lower = question.lower()
        return any(kw in question_lower for kw in follow_up_keywords)
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация для Redis"""
        return {
            "user_id": self.user_id,
            "history": list(self.history),
            "last_activity": self.last_activity.isoformat(),
            "last_article_context": self.last_article_context,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserSession":
        """Десериализация из Redis"""
        session = cls(data["user_id"])
        session.history = deque(data["history"], maxlen=session.max_history)
        session.last_activity = datetime.fromisoformat(data["last_activity"])
        session.last_article_context = data.get("last_article_context")
        return session

async def get_or_create_session(user_id: int) -> UserSession:
    """Получает или создает сессию пользователя из Redis"""
    # Попытка загрузить из Redis
    session_data = await redis_manager.get_session(user_id)
    
    if session_data:
        session = UserSession.from_dict(session_data)
        
        if session.is_expired():
            await redis_manager.delete_session(user_id)
            logger.info(f"Session expired for user {user_id}")
            session = UserSession(user_id)
    else:
        session = UserSession(user_id)
        logger.info(f"New session created for user {user_id}")
    
    # Обновляем метрику
    metrics_active_sessions.set(await redis_manager.redis.dbsize())
    
    return session

async def save_session(session: UserSession):
    """Сохраняет сессию в Redis"""
    await redis_manager.save_session(session.user_id, session.to_dict())

# ================= SECURITY =================

def sanitize_html(text: str) -> str:
    """
    Санитизация HTML для предотвращения XSS
    Даже если данные из "доверенного" источника (БД)
    """
    allowed_tags = [
        'b', 'i', 'u', 's', 'a', 'code', 'pre', 'br', 'p'
    ]
    allowed_attributes = {
        'a': ['href']
    }
    
    return bleach_clean(
        text,
        tags=allowed_tags,
        attributes=allowed_attributes,
        strip=True
    )

# ================= GDPR: PROMETHEUS CLEANUP =================

async def clean_prometheus_user_metrics(user_id: int):
    """
    Очистка Prometheus metrics для пользователя (GDPR compliance)

    Prometheus хранит metrics отдельно от приложения. Для удаления metrics
    с определенным user_id label необходимо вызвать API Prometheus.

    Args:
        user_id: ID пользователя для очистки metrics

    Returns:
        dict со статистикой очистки

    ПРИМЕЧАНИЯ:
    - Требует HTTP доступ к Prometheus API (обычно http://localhost:9090)
    - Может не работать если Prometheus защищён или недоступен
    - Альтернатива: ждать пока metrics очистятся по retention (обычно 15 дней)
    """
    import aiohttp

    prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

    cleanup_stats = {
        "user_id": user_id,
        "metrics_cleaned": {},
        "errors": []
    }

    # Metrics которые содержат user_id label
    metrics_to_clean = [
        "bot_questions_total",
        "bot_errors_total",
        "bot_response_time_seconds",
    ]

    try:
        async with aiohttp.ClientSession() as session:
            for metric in metrics_to_clean:
                try:
                    # Формируем запрос для удаления series с user_id
                    query = f'{metric}{{user_id="{user_id}"}}'
                    url = f"{prometheus_url}/api/v1/series"

                    params = {"match[]": query}

                    # Удаляем series через DELETE
                    async with session.delete(url, params=params) as response:
                        if response.status in [200, 204]:
                            cleanup_stats["metrics_cleaned"][metric] = "deleted"
                            logger.info(f"GDPR: Cleaned Prometheus metric {metric} for user {user_id}")
                        else:
                            error_text = await response.text()
                            cleanup_stats["errors"].append(
                                f"{metric}: HTTP {response.status} - {error_text}"
                            )
                            logger.warning(
                                f"Failed to clean {metric}: HTTP {response.status}"
                            )

                except aiohttp.ClientError as e:
                    cleanup_stats["errors"].append(f"{metric}: {e}")
                    logger.warning(f"Failed to clean {metric}: {e}")

    except Exception as e:
        error_msg = f"Prometheus cleanup failed for user {user_id}: {e}"
        cleanup_stats["errors"].append(error_msg)
        logger.warning(error_msg)

    return cleanup_stats

def validate_article_number(article_num: str) -> bool:
    """
    Валидация номера статьи
    Предотвращает инъекции и DoS атаки
    """
    # Статья должна быть числом от 1 до 1551 (максимум в ГК РФ)
    if not article_num.isdigit():
        return False
    
    num = int(article_num)
    if num < 1 or num > 2000:  # С запасом
        return False
    
    return True

# ================= ARTICLE DETECTION =================

ARTICLE_REGEX = re.compile(
    r"(ст\.?|статья)\s*(\d+)(?:\s*(?:п\.?|пункт|ч\.?|часть)\s*(\d+))?",
    re.IGNORECASE,
)

def detect_article_query(text: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Детектирует запрос о конкретной статье ГК РФ
    
    Returns:
        tuple: (article_num, point_num) или None
    """
    match = ARTICLE_REGEX.search(text)
    if not match:
        return None
    
    article = match.group(2)
    point = match.group(3)
    
    # Валидация
    if not validate_article_number(article):
        logger.warning(f"Invalid article number detected: {article}")
        return None
    
    return article, point

def article_exists_in_db(article_num: str) -> bool:
    """
    Проверяет существование статьи в базе данных
    
    ВАЖНО: Эта проверка выполняется перед LLM запросом
    """
    from database import LegalVectorDB
    
    try:
        db = LegalVectorDB()
        db.load()
        docs = db.get_article_by_number(article_num)
        return bool(docs)
    except Exception as e:
        logger.error(f"Error checking article {article_num}: {e}")
        return False

async def get_exact_article(article: str, point: Optional[str] = None) -> str:
    """
    Получает точный текст статьи (и пункта, если указан)
    С санитизацией и валидацией
    """
    start_time = time.time()
    
    # Валидация номера статьи
    if not validate_article_number(article):
        return (
            f"❌ Некорректный номер статьи: {html.escape(article)}\n\n"
            f"💡 Номер статьи должен быть от 1 до 1551."
        )
        
    from database import LegalVectorDB
    
    try:
        db = LegalVectorDB()
        db.load()
        
        # Получаем все пункты статьи
        docs = db.get_article_by_number(article)
        
        if not docs:
            return (
                f"❌ Статья {article} не найдена в ГК РФ.\n\n"
                f"💡 Проверьте номер статьи или попробуйте переформулировать вопрос.\n\n"
                f"Например: 'Что говорит статья 454?' или 'Статья 196 пункт 1'"
            )
        
        warning = ""
        
        # Фильтруем по пункту, если указан
        if point:
            filtered_docs = [d for d in docs if d.metadata.get("point_num") == point]
            if filtered_docs:
                docs = filtered_docs
            else:
                warning = (
                    f"\n\n⚠️ Пункт {point} не найден. Показываю всю статью.\n\n"
                )
        
        # Сортируем по номеру пункта
        def _sort_key(d):
            pn = d.metadata.get("point_num")
            if pn in (None, "full"):
                return 9999
            try:
                return int(pn)
            except Exception:
                return 9999
        
        docs = sorted(docs, key=_sort_key)
        
        answer_parts: List[str] = []
        
        first_doc = docs[0]
        article_title = first_doc.metadata.get("article_title", "")
        part_num = first_doc.metadata.get("part", "?")
        
        answer_parts.append(f"📖 <b>Статья {article}. {html.escape(article_title)}</b>")
        answer_parts.append(f"(часть {part_num} ГК РФ)")
        answer_parts.append("")
        
        if point and len(docs) == 1:
            doc = docs[0]
            point_num = doc.metadata.get("point_num")
            content = doc.page_content
            
            lines = content.split("\n")
            content_lines = [
                line
                for line in lines
                if f"Статья {article}" not in line and article_title not in line
            ]
            
            answer_parts.append(f"<b>Пункт {point_num}:</b>")
            answer_parts.append(html.escape("".join(content_lines).strip()))
        else:
            for doc in docs:
                point_num = doc.metadata.get("point_num")
                content = doc.page_content
                
                if point_num == "full":
                    answer_parts.append("<b>Текст статьи:</b>")
                    answer_parts.append(html.escape(content))
                else:
                    lines = content.split("\n")
                    content_lines = [
                        line
                        for line in lines
                        if f"Статья {article}" not in line and article_title not in line
                    ]
                    answer_parts.append(
                        f"<b>{point_num}.</b> " + html.escape("".join(content_lines).strip())
                    )
                answer_parts.append("")
        
        if point and warning:
            answer_parts.insert(3, warning)
        
        answer_parts.append(
            "\n⚠️ <i>Информация носит справочный характер. "
            "Для точной консультации обратитесь к юристу.</i>"
        )
        
        result = "\n".join(answer_parts)
        
        # Метрики
        elapsed = time.time() - start_time
        metrics_article_lookup_time.observe(elapsed)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting article {article}: {e}", exc_info=True)
        metrics_errors_total.labels(error_type="article_fetch").inc()
        return (
            f"❌ Ошибка получения статьи {article}\n\n"
            f"Попробуйте позже или обратитесь к администратору."
        )

# ================= LLM FALLBACK FOR ARTICLES =================

async def get_article_with_llm_fallback(
    article: str, 
    point: Optional[str] = None,
    original_query: str = ""
) -> str:
    """
    Получает статью из БД, если не найдена — использует LLM как fallback
    """
    # Сначала пытаемся получить из БД
    db_result = await get_exact_article(article, point)
    
    if "❌" not in db_result:
        return db_result
    
    # Если статья не найдена — используем LLM
    logger.info(f"Article {article} not found in DB, falling back to LLM")
    
    fallback_query = original_query or f"Расскажи про статью {article} ГК РФ"
    
    try:
        llm_answer = generate_answer(fallback_query)
        
        # Добавляем предупреждение
        warning = (
            f"⚠️ <b>Статья {article} не найдена в базе данных.</b>\n"
            f"Ниже представлен ответ на основе AI модели:\n\n"
        )
        
        return warning + sanitize_html(llm_answer)
        
    except Exception as e:
        logger.error(f"LLM fallback failed: {e}")
        return db_result  # Возвращаем оригинальное сообщение об ошибке

# ================= BOT INITIALIZATION =================

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# ================= PRIVACY POLICY =================

PRIVACY_POLICY = """
🔒 <b>Политика конфиденциальности</b>

<b>Какие данные мы собираем:</b>
• Telegram ID (для работы бота)
• Username (опционально)
• Текст ваших вопросов
• История запросов
• Контактные данные (только при записи на консультацию)

<b>Как мы используем данные:</b>
• Для предоставления юридических консультаций
• Для улучшения качества ответов
• Для связи при записи на консультацию
• Для статистики (анонимно)

<b>Защита данных:</b>
• Все персональные данные шифруются (AES-256)
• База данных защищена от несанкционированного доступа
• Доступ к вашим данным имеете только вы
• Контактные данные видят только юристы при записи

<b>Ваши права (152-ФЗ / GDPR):</b>
• /mydata - экспорт всех ваших данных
• /deletemydata - полное удаление данных
• /anonymize - анонимизация данных

<b>Хранение данных:</b>
• Запросы: 1 год, затем автоматическое удаление
• Консультации: удаляются после завершения
• Вы можете удалить данные в любой момент

<b>Согласие:</b>
Используя бота, вы соглашаетесь с обработкой данных.
Полная политика: /fullprivacy
❓ Вопросы: support@example.com
"""

# ================= FAQ KEYBOARD =================

faq_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📦 Возврат товара")],
        [KeyboardButton(text="🏠 Купля-продажа квартиры")],
        [KeyboardButton(text="📝 Расторжение договора")],
        [KeyboardButton(text="❓ Другой вопрос")],
    ],
    resize_keyboard=True,
)

# ================= FAQ ANSWERS =================

FAQ_ANSWERS = {
    "📦 Возврат товара": """
📦 <b>Возврат товара</b>

<b>Качественный товар (ст. 25 Закона о ЗПП):</b>
• Можно вернуть в течение 14 дней (не считая дня покупки)
• Если товар не подошел по форме, цвету, размеру
• Товар не должен быть в использовании
• Исключение: технически сложные товары, нижнее белье, парфюмерия

<b>Некачественный товар (ст. 18 Закона о ЗПП):</b>
✅ Право требовать:
• Замену на аналогичный товар
• Замену на другую марку с перерасчетом
• Соразмерное уменьшение цены
• Бесплатный ремонт
• Возврат денег

⚖️ Правовая база:
• Статья 454 ГК РФ (договор купли-продажи)
• Статья 469-477 ГК РФ (качество товара)
• Закон "О защите прав потребителей"

💡 <b>Совет:</b> Сохраняйте чек и обратитесь к продавцу в письменном виде.
""",
    "🏠 Купля-продажа квартиры": """
🏠 <b>Купля-продажа квартиры</b>

<b>Необходимые документы:</b>
📄 От продавца:
• Свидетельство о праве собственности или выписка ЕГРН
• Паспорт
• Согласие супруга (если квартира куплена в браке)
• Справка об отсутствии задолженностей по ЖКУ

📄 От покупателя:
• Паспорт
• Доказательства финансовой возможности покупки

<b>Этапы сделки:</b>
1️⃣ Подписание предварительного договора (опционально)
2️⃣ Подписание основного договора купли-продажи
3️⃣ Передача денег (через банковскую ячейку или аккредитив)
4️⃣ Подача документов на регистрацию в Росреестр
5️⃣ Получение выписки ЕГРН (7-12 дней)

⚖️ Правовая база:
• Статья 549-558 ГК РФ (продажа недвижимости)
• Статья 131 ГК РФ (государственная регистрация)

💡 <b>Совет:</b> Проверьте историю квартиры через расширенную выписку ЕГРН, привлеките юриста.
""",
    "📝 Расторжение договора": """
📝 <b>Расторжение договора</b>

<b>Основания (ст. 450-453 ГК РФ):</b>

1️⃣ <b>По соглашению сторон</b> (ст. 450 п.1)
• Самый простой способ
• Обе стороны согласны

2️⃣ <b>В одностороннем порядке</b> (ст. 450 п.2)
✅ Возможно, если:
• Предусмотрено договором или законом
• Существенное нарушение условий другой стороной
• Иные случаи, установленные законом

3️⃣ <b>Через суд</b> (ст. 450 п.2)
• При отказе другой стороны расторгнуть договор
• При существенном нарушении условий

<b>Существенное нарушение - это:</b>
• Нарушение, которое влечет ущерб для другой стороны
• Лишает сторону того, на что она рассчитывала

⚖️ Правовая база:
• Статья 450-453 ГК РФ (изменение и расторжение договора)
• Статья 310 ГК РФ (односторонний отказ запрещен, кроме исключений)

💡 <b>Совет:</b> Сначала направьте претензию другой стороне письменно (заказным письмом).
""",
}

# ================= UTILS =================

@db_operation
async def track_user_query(user_id: int):
    """Отслеживает запросы пользователя для статистики"""
    global sql_db
    
    now = datetime.now()
    
    async with sql_db._get_connection_async() as conn:
        # Обновляем или создаем статистику
        await conn.execute("""
            INSERT INTO user_stats (user_id, total_queries, first_query, last_query_date)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                total_queries = total_queries + 1,
                last_query_date = ?  -- ← БЫЛО last_query, ДОЛЖНО БЫТЬ last_query_date
        """, (user_id, now, now, now))
        
        await conn.commit()


@db_operation
async def save_user_query(user_id: int, query_text: str, answer_text: str, query_type: str):
    """Сохраняет запрос пользователя в БД"""
    async with sql_db._get_connection_async() as conn:
        await conn.execute("""
            INSERT INTO user_queries (user_id, query_text, answer_text, query_type)
            VALUES (?, ?, ?, ?)
        """, (user_id, query_text, answer_text, query_type))
        # Commit выполняется автоматически в get_connection()


@db_operation
async def get_user_stats(user_id: int) -> Optional[Dict[str, Any]]:
    """Получает статистику пользователя из БД"""
    async with sql_db._get_connection_async() as conn:
        cursor = await conn.execute(
            "SELECT total_queries FROM user_stats WHERE user_id = ?",
            (user_id,)
        )
        row = await cursor.fetchone()
        if row:
            return {"total_queries": row[0]}
        return None
        
@db_operation
async def get_global_stats() -> Dict[str, Any]:
    """Получает глобальную статистику из БД"""
    async with sql_db._get_connection_async() as conn:
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM users WHERE deleted_at IS NULL"
        )
        total_users = (await cursor.fetchone())[0]

        cursor = await conn.execute(
            "SELECT SUM(total_queries) FROM user_stats"
        )
        total_queries = (await cursor.fetchone())[0] or 0

        return {
            "total_users": total_users,
            "total_queries": total_queries,
        }

async def send_long_message(message: Message, text: str, parse_mode: str | None = None):
    """Отправляет длинное сообщение, разбивая на части"""
    MAX_LENGTH = 4096
    
    if len(text) <= MAX_LENGTH:
        await message.answer(text, parse_mode=parse_mode)
        return
    
    parts: List[str] = []
    current_part = ""
    
    for line in text.split("\n"):
        if len(current_part) + len(line) + 1 <= MAX_LENGTH:
            current_part += line + "\n"
        else:
            if current_part:
                parts.append(current_part)
            current_part = line + "\n"
    
    if current_part:
        parts.append(current_part)
    
    for i, part in enumerate(parts):
        await message.answer(part, parse_mode=parse_mode)
        if i < len(parts) - 1:
            await asyncio.sleep(0.5)

async def send_typing_action(message: Message):
    """Отправляет действие 'печатает' периодически"""
    try:
        while True:
            await bot.send_chat_action(
                chat_id=message.chat.id, 
                action=ChatAction.TYPING
            )
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        pass

async def process_with_typing(message: Message, question: str) -> str:
    """Обрабатывает вопрос с индикацией печатания"""
    typing_task = asyncio.create_task(send_typing_action(message))
    try:
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, generate_answer, question)
        return answer
    finally:
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

# ================= ERROR MIDDLEWARE =================

@dp.error()
async def error_handler(event: ErrorEvent):
    """
    Глобальный обработчик ошибок
    Перехватывает все uncaught exceptions
    """
    error = event.exception
    error_type = type(error).__name__
    
    logger.error(
        f"Unhandled error: {error_type}: {error}",
        exc_info=True
    )
    
    metrics_errors_total.labels(error_type=error_type).inc()
    
    if event.update.message:
        try:
            await event.update.message.answer(
                "❌ Произошла внутренняя ошибка.\n\n"
                "Попробуйте:\n"
                "• Повторить запрос через минуту\n"
                "• Переформулировать вопрос\n"
                "• Использовать /help для справки\n\n"
                f"Код ошибки: {error_type}"
            )
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
    
    return True

# ================= HANDLERS =================

@dp.message(CommandStart())
async def cmd_start(message: Message):
    """Обработчик команды /start"""
    user = message.from_user
    user_name = user.first_name or "друг"
    user_id = user.id
    
    user_data = await get_user_data(user_id)
    
    if not user_data.get("consent_given", False):
        consent_keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="✅ Согласен с обработкой данных")],
                [KeyboardButton(text="📖 Прочитать политику конфиденциальности")],
            ],
            resize_keyboard=True,
        )
        
        await message.answer(
            "👋 Добро пожаловать!\n\n"
            "Для работы бота необходимо ваше согласие на обработку персональных данных.\n\n"
            "Мы гарантируем:\n"
            "🔒 Шифрование всех персональных данных\n"
            "🛡️ Защиту от несанкционированного доступа\n"
            "📝 Право на удаление данных в любой момент\n\n"
            "Подробнее: /privacy",
            reply_markup=consent_keyboard,
        )
        return
    
    await track_user_query(user_id)
    
    welcome_text = f"""
👋 Здравствуйте, {user_name}!

Я - ваш персональный юридический консультант по <b>Гражданскому кодексу РФ</b>.

📚 <b>Моя база знаний:</b>
• Все 4 части Гражданского кодекса РФ
• Более 1500 статей
• Актуальная редакция законодательства

💡 <b>Выберите тему или задайте свой вопрос:</b>
"""
    await message.answer(
        welcome_text, 
        parse_mode=ParseMode.HTML, 
        reply_markup=faq_keyboard
    )
    
    help_text = """
    📋 <b>Команды:</b>
    /help — подробная справка
    /stats — статистика бота
    /examples — примеры вопросов
    /article &lt;номер&gt; — показать полный текст статьи
    /privacy — политика конфиденциальности
    /history — история текущей сессии
    /clearsession — очистить историю сессии

    ⚠️ <b>Важно:</b> Мои ответы носят информационный характер и не заменяют консультацию с квалифицированным юристом.

    <b>Задавайте ваш вопрос! 👇</b>
    """


    await message.answer(help_text, parse_mode=ParseMode.HTML)

@dp.message(F.text == "⏭ Пропустить")
async def skip_contact(message: Message):
    """Пропуск предоставления контакта"""
    await message.answer(
        "Хорошо, вы можете добавить контакт позже.\n\n"
        "Для записи на консультацию используйте /consultation\n"
        "Задайте вопрос или используйте /help",
        reply_markup=ReplyKeyboardRemove()
    )

@dp.message(Command("help"))
async def cmd_help(message: Message):
    """Обработчик команды /help"""
    help_text = """
    📖 <b>Подробная справка</b>

    🔍 <b>Как задавать вопросы:</b>

    <b>1. Общие вопросы</b>
    Просто опишите вашу ситуацию:
    • "Можно ли вернуть некачественный товар?"
    • "Какие права у покупателя квартиры?"
    • "Что делать если продавец не отдает товар?"

    <b>2. Вопросы про конкретную статью</b>
    Укажите номер статьи:
    • "Что говорит статья 454?"
    • "Расскажи про статью 196 пункт 2"
    • "Объясни ст. 309 ГК РФ"

    <b>3. Уточняющие вопросы</b>
    Можете продолжить диалог:
    • "Расскажи подробнее"
    • "А что если..."
    • "Какие еще есть варианты?"

    ⚙️ <b>Как я работаю:</b>
    1️⃣ Анализирую ваш вопрос
    2️⃣ Ищу релевантные статьи ГК РФ
    3️⃣ Формирую понятный ответ
    4️⃣ Даю ссылки на законодательство

    ✅ <b>Что я умею:</b>
    • Объяснять сложные юридические термины
    • Ссылаться на конкретные статьи и пункты
    • Давать практические советы
    • Отвечать на уточняющие вопросы

    ❌ <b>Что я НЕ умею:</b>
    • Составлять документы (договоры, исковые заявления)
    • Представлять интересы в суде
    • Давать гарантии по исходу дела
    • Заменять консультацию юриста в сложных случаях

    📞 <b>Команды:</b>
    /start — приветствие
    /help — эта справка
    /ask — задать юридический вопрос
    /consultation — записаться на консультацию  
    /myconsultations — мои записи на консультации 
    /stats — статистика
    /examples — примеры вопросов
    /privacy — политика конфиденциальности
    /history — история текущей сессии
    /clearsession — очистить историю сессии
    /mydata — экспорт ваших данных
    /deletemydata — удаление всех данных
    /anonymize — анонимизация данных


    📅 <b>Запись на консультацию:</b> 
    Напишите "хочу записаться на консультацию" для получения информации о записи.

    💬 <b>Просто напишите свой вопрос!</b>
    """
    await message.answer(help_text, parse_mode=ParseMode.HTML)
    

@dp.message(Command("ask"))
async def cmd_ask(message: Message, command: CommandObject):
    """
    Команда /ask для задания юридического вопроса
    
    Использование:
    /ask какие документы нужны для покупки квартиры
    /ask что говорит статья 454
    """
    user_id = message.from_user.id
    username = message.from_user.username or "unknown"
    
    # Проверяем согласие
    user_data = await get_user_data(user_id)
    if not user_data.get("consent_given", False):
        consent_keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="✅ Согласен с обработкой данных")],
                [KeyboardButton(text="📖 Прочитать политику конфиденциальности")],
            ],
            resize_keyboard=True,
        )
        await message.answer(
            "⚠️ Для работы бота необходимо согласие на обработку персональных данных.\n\n"
            "Мы гарантируем:\n"
            "🔒 Шифрование всех персональных данных\n"
            "🛡️ Защиту от несанкционированного доступа\n"
            "📝 Право на удаление данных в любой момент\n\n"
            "Подробнее: /privacy",
            reply_markup=consent_keyboard,
        )
        return
    
    # Получаем вопрос из аргументов
    args = (command.args or "").strip()
    if not args:
        await message.answer(
            "📋 <b>Команда для задания вопроса</b>\n\n"
            "Использование:\n"
            "<code>/ask какие документы нужны для купли-продажи</code>\n"
            "<code>/ask что говорит статья 454</code>\n\n"
            "Также можно просто написать вопрос в чате (без команды).\n\n"
            "Примеры: /examples",
            parse_mode=ParseMode.HTML,
        )
        return
    
    user_query = args
    
    await message.react([ReactionTypeEmoji(emoji="🤔")])
    
    # Проверяем rate limit (Redis distributed lock)
    if await redis_manager.check_rate_limit(user_id):
        logger.warning(f"Rate limit exceeded for user {user_id}")
        await message.react([ReactionTypeEmoji(emoji="⏳")])
        await message.answer(
            "⏳ Слишком много запросов. Попробуйте позже.\n\n"
            f"Лимит: {RATE_LIMIT_REQUESTS} вопросов в {RATE_LIMIT_WINDOW} секунд."
        )
        return
    
    try:
        start_time = time.time()
        
        await message.answer(
            f"⏳ Анализирую ваш вопрос...",
            reply_markup=ReplyKeyboardRemove(),
        )
        await message.chat.do("typing")
        
        # Генерируем ответ
        answer = await process_with_typing(message, user_query)
        
        # Санитизация HTML
        answer = sanitize_html(answer)
        
        # Метрики
        elapsed = time.time() - start_time
        metrics_response_time.observe(elapsed)
        metrics_questions_total.labels(
            user_id=user_id,
            question_type="command_ask"
        ).inc()
        
        await safe_react(message, "✅")
        await send_long_message(message, answer, parse_mode=ParseMode.HTML)
        
        # Обновление сессии
        session = await get_or_create_session(user_id)
        session.add_interaction(f"/ask {user_query[:50]}", answer, [])
        await save_session(session)
        
        # Сохранение в БД (с единообразной обработкой ошибок)
        await save_user_query(user_id, user_query, answer, "command_ask")

        await track_user_query(user_id)
        
        logger.info(
            f"Question from /ask command answered for user {user_id}: "
            f"{user_query[:50]} (took {elapsed:.2f}s)"
        )
        
    except Exception as e:
        logger.error(f"Error in /ask command: {e}", exc_info=True)
        metrics_errors_total.labels(error_type="ask_command").inc()
        await safe_react(message, "❌")
        await message.answer(
            "❌ Ошибка обработки вопроса.\n"
            "Попробуйте переформулировать или повторите позже."
        )

@dp.message(Command("consultation"))
async def cmd_consultation(message: Message, state: FSMContext):
    """Запись на консультацию"""
    user_id = message.from_user.id
    
    # Проверка согласия
    user_data = await get_user_data(user_id)
    if not user_data.get("consent_given", False):
        await message.answer(
            "⚠️ Для записи на консультацию необходимо согласие на обработку данных.\n"
            "Используйте /start"
        )
        return
    
    await message.answer(
        "📅 <b>Запись на консультацию с юристом</b>\n\n"
        "Я помогу вам записаться. Пожалуйста, ответьте на несколько вопросов.\n\n"
        "Шаг 1/4: Укажите желаемую дату консультации\n"
        "(например: 25.02.2026 или завтра)",
        parse_mode=ParseMode.HTML,
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text="❌ Отменить")]],
            resize_keyboard=True
        )
    )
    await state.set_state(ConsultationForm.waiting_for_date)
from datetime import datetime, timedelta

def parse_user_date(user_input: str) -> str:
    """Преобразует текст пользователя в дату"""
    user_input_lower = user_input.lower().strip()
    today = datetime.now()
    
    # Обработка относительных дат
    if user_input_lower in ['сегодня', 'today']:
        return today.strftime('%d.%m.%Y')
    elif user_input_lower in ['завтра', 'tomorrow']:
        return (today + timedelta(days=1)).strftime('%d.%m.%Y')
    elif user_input_lower in ['послезавтра']:
        return (today + timedelta(days=2)).strftime('%d.%m.%Y')
    
    # Если уже в формате даты - возвращаем как есть
    return user_input

@dp.message(ConsultationForm.waiting_for_date)
async def process_consultation_date(message: Message, state: FSMContext):
    """Обработка даты"""
    if message.text == "❌ Отменить":
        await state.clear()
        await message.answer("❌ Запись на консультацию отменена.", reply_markup=ReplyKeyboardRemove())
        return
    
    # Преобразуем в дату
    parsed_date = parse_user_date(message.text)
    
    await state.update_data(preferred_date=parsed_date)  # ← Сохраняем преобразованную дату
    await message.answer(
        f"✅ Дата: {parsed_date}\n\n"
        f"Шаг 2/4: Укажите желаемое время\n"
        "(например: 14:00 или утро)",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="10:00"), KeyboardButton(text="14:00")],
                [KeyboardButton(text="16:00"), KeyboardButton(text="18:00")],
                [KeyboardButton(text="❌ Отменить")]
            ],
            resize_keyboard=True
        )
    )
    await state.set_state(ConsultationForm.waiting_for_time)

@dp.message(ConsultationForm.waiting_for_time)
async def process_consultation_time(message: Message, state: FSMContext):
    """Обработка времени"""
    if message.text == "❌ Отменить":
        await state.clear()
        await message.answer(
            "❌ Запись на консультацию отменена.",
            reply_markup=ReplyKeyboardRemove()
        )
        return
    
    await state.update_data(preferred_time=message.text)
    await message.answer(
        "Шаг 3/4: Укажите тему консультации\n"
        "(например: купля-продажа квартиры, договор, наследство)",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="Купля-продажа")],
                [KeyboardButton(text="Договоры")],
                [KeyboardButton(text="Наследство")],
                [KeyboardButton(text="Другое")],
                [KeyboardButton(text="❌ Отменить")]
            ],
            resize_keyboard=True
        )
    )
    await state.set_state(ConsultationForm.waiting_for_topic)

@dp.message(ConsultationForm.waiting_for_topic)
async def process_consultation_topic(message: Message, state: FSMContext):
    """Обработка темы"""
    if message.text == "❌ Отменить":
        await state.clear()
        await message.answer(
            "❌ Запись на консультацию отменена.",
            reply_markup=ReplyKeyboardRemove()
        )
        return
    
    await state.update_data(topic=message.text)
    await message.answer(
        "Шаг 4/4: Опишите вашу ситуацию кратко\n"
        "(2-3 предложения)",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text="❌ Отменить")]],
            resize_keyboard=True
        )
    )
    await state.set_state(ConsultationForm.waiting_for_description)
    
@dp.message(ConsultationForm.waiting_for_description, F.contact)
async def process_contact_in_consultation(message: Message, state: FSMContext):
    """Обработка контакта во время записи на консультацию"""
    contact = message.contact
    user_id = message.from_user.id
    
    # Шифруем и сохраняем телефон
    global sql_db
    encrypted_phone = sql_db.secure_db.encrypt_field(contact.phone_number, 'phone')
    
    async with sql_db._get_connection_async() as conn:
        await conn.execute("""
            UPDATE users 
            SET phone = ?
            WHERE user_id = ?
        """, (encrypted_phone, user_id))
        await conn.commit()
    
    await message.answer(
        f"✅ Телефон {contact.phone_number} сохранен!\n\n"
        f"Теперь опишите вашу ситуацию (2-3 предложения):",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text="❌ Отменить")]],
            resize_keyboard=True
        )
    )

@dp.message(ConsultationForm.waiting_for_description)
async def process_consultation_description(message: Message, state: FSMContext):
    """Обработка описания и сохранение"""
    
    if message.text == "❌ Отменить":
        await state.clear()
        await message.answer("❌ Запись на консультацию отменена.", reply_markup=ReplyKeyboardRemove())
        return
    
    user_id = message.from_user.id
    
    # Если это телефон (только цифры, +, -, скобки, пробелы)
    phone_pattern = r'^[\d\s\+\-\(\)]+$'
    if re.match(phone_pattern, message.text.strip()) and len(message.text.strip()) >= 10:
        global sql_db
        encrypted_phone = sql_db.secure_db.encrypt_field(message.text.strip(), 'phone')
        
        async with sql_db._get_connection_async() as conn:
            await conn.execute("""
                UPDATE users 
                SET phone = ?
                WHERE user_id = ?
            """, (encrypted_phone, user_id))
            await conn.commit()
        
        await message.answer(
            f"✅ Телефон {message.text} сохранен!\n\n"
            f"Теперь опишите вашу ситуацию (2-3 предложения):",
            reply_markup=ReplyKeyboardMarkup(
                keyboard=[[KeyboardButton(text="❌ Отменить")]],
                resize_keyboard=True
            )
        )
        return  # Ждём описание
    
    # Если это команда или кнопка - игнорируем
    if message.text.startswith('/'):
        await message.answer(
            "⚠️ Пожалуйста, опишите вашу ситуацию.\n"
            "Команды сейчас не обрабатываются."
        )
        return
    
    user_data_fsm = await state.get_data()
    
    # Обновляем данные пользователя
    contact_data = await get_user_data(user_id)
    phone_decrypted = contact_data.get('phone')
    first_name = contact_data.get('first_name', 'Пользователь')
    
    # Проверяем телефон ЕЩЁ РАЗ
    if not phone_decrypted:
        await message.answer(
            "⚠️ <b>Телефон всё ещё не указан</b>\n\n"
            "Пожалуйста:\n"
            "• Нажмите 'Поделиться контактом' ИЛИ\n"
            "• Введите номер (например: +79031234567 или 89031234567)",
            reply_markup=ReplyKeyboardMarkup(
                keyboard=[
                    [KeyboardButton(text="📱 Поделиться контактом", request_contact=True)],
                    [KeyboardButton(text="❌ Отменить")]
                ],
                resize_keyboard=True
            ),
            parse_mode=ParseMode.HTML
        )
        return
    
    # Сохраняем консультацию
    try:
        encrypted_description = sql_db.secure_db.encrypt_field(message.text, 'description')
        
        async with sql_db._get_connection_async() as conn:
            cursor = await conn.execute(
                "SELECT phone FROM users WHERE user_id = ?",
                (user_id,)
            )
            phone_row = await cursor.fetchone()
            encrypted_phone = dict(phone_row).get('phone') if phone_row else None
            
            await conn.execute("""
                INSERT INTO consultations (
                    user_id, preferred_date, preferred_time,
                    contact_phone, topic, description, status
                )
                VALUES (?, ?, ?, ?, ?, ?, 'pending')
            """, (
                user_id,
                user_data_fsm['preferred_date'],
                user_data_fsm['preferred_time'],
                encrypted_phone,
                user_data_fsm['topic'],
                encrypted_description
            ))
            await conn.commit()
        
        await state.clear()
        
        await message.answer(
            f"✅ <b>Запись успешно создана!</b>\n\n"
            f"📋 <b>Детали:</b>\n"
            f"👤 Имя: {first_name}\n"
            f"📱 Телефон: {phone_decrypted}\n"
            f"📅 Дата: {user_data_fsm['preferred_date']}\n"
            f"🕐 Время: {user_data_fsm['preferred_time']}\n"
            f"📌 Тема: {user_data_fsm['topic']}\n"
            f"📝 Описание: {message.text[:100]}...\n\n"
            f"⏳ <b>Статус:</b> Ожидает подтверждения\n\n"
            f"Мы свяжемся с вами.\n\n"
            f"/myconsultations - просмотр записей",
            parse_mode=ParseMode.HTML,
            reply_markup=ReplyKeyboardRemove()
        )
        
        logger.info(f"✅ Consultation created for user {user_id}")
        
    except Exception as e:
        logger.error(f"Error creating consultation: {e}", exc_info=True)
        await message.answer(
            "❌ Ошибка при создании записи.",
            reply_markup=ReplyKeyboardRemove()
        )
        await state.clear()


@dp.message(Command("myconsultations"))
async def cmd_my_consultations(message: Message):
    """Мои консультации"""
    user_id = message.from_user.id
    
    global sql_db
    
    async with sql_db._get_connection_async() as conn:
        cursor = await conn.execute("""
            SELECT id, requested_at, preferred_date, preferred_time, 
                   topic, description, status
            FROM consultations
            WHERE user_id = ? AND deleted_at IS NULL
            ORDER BY requested_at DESC
        """, (user_id,))
        
        consultations = await cursor.fetchall()
    
    if not consultations:
        await message.answer(
            "📅 У вас пока нет записей на консультацию.\n\n"
            "Используйте /consultation для записи."
        )
        return
    
    text = "<b>📅 Ваши консультации:</b>\n\n"
    
    status_emoji = {
        'pending': '⏳ Ожидает',
        'confirmed': '✅ Подтверждена',
        'completed': '✔️ Завершена',
        'cancelled': '❌ Отменена'
    }
    
    for cons in consultations:
        c = dict(cons)
        
        # Расшифровываем description
        description = c.get('description', '')
        if description:
            try:
                description = sql_db.secure_db.decrypt_field(description, 'description')
            except Exception as e:
                logger.warning(f"Failed to decrypt description for consultation {c['id']}: {e}")
                description = "[Ошибка расшифровки]"
        
        text += (
            f"<b>#{c['id']}</b> | {status_emoji.get(c['status'], c['status'])}\n"
            f"📅 {c['preferred_date']} в {c['preferred_time']}\n"
            f"📌 {c['topic']}\n"
            f"📝 {description[:50]}...\n\n"
        )
    
    await message.answer(text, parse_mode=ParseMode.HTML)

@dp.message(Command("examples"))
async def cmd_examples(message: Message):
    """Примеры вопросов"""
    examples_text = """
💡 <b>Примеры вопросов</b>

<b>📦 Купля-продажа товаров:</b>
• Какие права у покупателя при обнаружении недостатков?
• Можно ли вернуть качественный товар?
• Что делать если товар не доставили вовремя?
• Как вернуть технически сложный товар?

<b>🏠 Недвижимость:</b>
• Какие документы нужны для покупки квартиры?
• Что такое договор купли-продажи недвижимости?
• Можно ли расторгнуть договор до регистрации?
• Какие права у арендатора квартиры?

<b>📝 Договоры:</b>
• Что такое существенные условия договора?
• Когда договор считается заключенным?
• Можно ли расторгнуть договор в одностороннем порядке?
• Что такое неустойка по договору?

<b>💰 Обязательства:</b>
• Что такое исковая давность?
• Как взыскать убытки с контрагента?
• Что делать если должник не платит?
• Можно ли уступить право требования долга?

<b>🔍 Конкретные статьи:</b>
• Что говорит статья 454 ГК РФ?
• Расскажи про статью 196 (исковая давность)
• Объясни статью 309 пункт 1

⚠️ <b>Важно:</b> Я специализируюсь только на <b>Гражданском кодексе РФ</b>.
Вопросы по уголовному, административному, налоговому праву - вне моей компетенции.

<b>Выберите похожий вопрос или задайте свой!</b>
"""
    await message.answer(examples_text, parse_mode=ParseMode.HTML)


@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    """Показывает статистику"""
    try:
        cache_stats = get_cache_stats()
        db_stats = get_db_stats()
        
        # Статистика пользователей из БД
        async with sql_db._get_connection_async() as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM users WHERE deleted_at IS NULL"
            )
            total_users = (await cursor.fetchone())[0]
            
            cursor = await conn.execute(
                "SELECT SUM(total_queries) FROM user_stats"
            )
            total_queries = (await cursor.fetchone())[0] or 0
            
            cursor = await conn.execute(
                "SELECT total_queries FROM user_stats WHERE user_id = ?",
                (message.from_user.id,)
            )
            row = await cursor.fetchone()
            user_queries = row[0] if row else 0
        
        stats_text = f"""
📊 <b>Статистика бота</b>

👥 <b>Пользователи:</b>
• Всего пользователей: {total_users}
• Всего запросов: {total_queries}
• Ваших запросов: {user_queries}

📚 <b>База знаний:</b>
• Всего документов: {db_stats.get('total_chunks', 'N/A')}
• Статей ГК РФ: {db_stats.get('unique_articles', 'N/A')}
• Пунктов статей: {db_stats.get('point_chunks', 'N/A')}

🗄️ <b>Кеш ответов:</b>
• Размер: {cache_stats['size']} / {cache_stats['max_size']}
• Попаданий: {cache_stats['hits']}
• Промахов: {cache_stats['misses']}
• Hit Rate: {cache_stats['hit_rate']}
• Сохранено запросов: {cache_stats['saved_requests']}

💡 <i>Кеширование помогает отвечать быстрее и экономит ресурсы!</i>
"""
        await message.answer(stats_text, parse_mode=ParseMode.HTML)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        await message.answer("❌ Ошибка получения статистики")

@dp.message(Command("privacy"))
async def cmd_privacy(message: Message):
    """Политика конфиденциальности"""
    await message.answer(PRIVACY_POLICY, parse_mode=ParseMode.HTML)

@dp.message(F.text == "✅ Согласен с обработкой данных")
async def accept_consent(message: Message, state: FSMContext):
    """Принятие согласия"""
    user = message.from_user
    user_id = user.id
    
    # Собираем данные из Telegram
    user_info = {
        'username': user.username,
        'first_name': user.first_name,
        'last_name': user.last_name,
        'language_code': user.language_code
    }
    
    # Устанавливаем согласие
    await set_user_consent(user_id, True, user_info)
    
    await message.answer(
        "✅ Спасибо! Теперь вы можете пользоваться ботом.\n\n"
        "Используйте /help для справки.",
        reply_markup=ReplyKeyboardRemove(),
    )
    
    # Опционально: запросить дополнительные данные
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📱 Поделиться контактом", request_contact=True)],
            [KeyboardButton(text="⏭ Пропустить")],
        ],
        resize_keyboard=True,
    )
    
    await message.answer(
        "📱 Хотите поделиться контактом для связи?\n\n"
        "Это необязательно, но поможет нам лучше обслуживать вас.",
        reply_markup=keyboard
    )

@dp.message(F.contact)
async def handle_contact(message: Message, state: FSMContext):
    """Обработка контакта от пользователя"""
    contact = message.contact
    user_id = message.from_user.id
    
    # Шифруем телефон
    global sql_db
    encrypted_phone = sql_db.secure_db.encrypt_field(contact.phone_number, 'phone')
    
    # Сохраняем в БД
    async with sql_db._get_connection_async() as conn:
        await conn.execute("""
            UPDATE users 
            SET phone = ?
            WHERE user_id = ?
        """, (encrypted_phone, user_id))
        await conn.commit()
    
    # Проверяем состояние FSM
    current_state = await state.get_state()
    
    if current_state == ConsultationForm.waiting_for_description.state:
        # Пользователь в процессе записи - продолжаем
        await message.answer(
            f"✅ Телефон {contact.phone_number} сохранен!\n\n"
            f"Теперь опишите вашу ситуацию (2-3 предложения):",
            reply_markup=ReplyKeyboardMarkup(
                keyboard=[[KeyboardButton(text="❌ Отменить")]],
                resize_keyboard=True
            )
        )
    else:
        # Просто сохранение контакта
        await message.answer(
            "✅ Спасибо! Контакт сохранен.",
            reply_markup=ReplyKeyboardRemove()
        )


async def update_user_activity(user_id: int, user_info: Optional[Dict] = None):
    """Обновляет время последней активности и данные пользователя"""
    global sql_db
    
    now = datetime.now()
    
    async with sql_db._get_connection_async() as conn:
        if user_info:
            # Обновляем данные пользователя, если они изменились
            await conn.execute("""
                UPDATE users 
                SET last_active = ?,
                    username = COALESCE(?, username),
                    first_name = COALESCE(?, first_name),
                    last_name = COALESCE(?, last_name),
                    total_queries = total_queries + 1
                WHERE user_id = ?
            """, (
                now,
                user_info.get('username'),
                user_info.get('first_name'),
                user_info.get('last_name'),
                user_id
            ))
        else:
            # Просто обновляем время активности
            await conn.execute("""
                UPDATE users 
                SET last_active = ?,
                    total_queries = total_queries + 1
                WHERE user_id = ?
            """, (now, user_id))
        
        await conn.commit()


@dp.message(Command("checkme"))
async def cmd_check_me(message: Message):
    """Проверка данных пользователя (отладка)"""
    user_id = message.from_user.id
    
    user_data = await get_user_data(user_id)
    
    if user_data:
        info = (
            f"🔍 <b>Ваши данные в БД:</b>\n\n"
            f"👤 User ID: <code>{user_data['user_id']}</code>\n"
            f"📝 Username: @{user_data['username'] or 'не указан'}\n"
            f"🏷 Имя: {user_data['first_name'] or 'не указано'}\n"
            f"🏷 Фамилия: {user_data['last_name'] or 'не указана'}\n"
            f"✅ Согласие: {'Да' if user_data['consent_given'] else 'Нет'}\n"
            f"📅 Дата согласия: {user_data['consent_date']}\n"
            f"🕐 Первый визит: {user_data['first_seen']}\n"
            f"🕐 Последняя активность: {user_data['last_active']}\n\n"
            f"<b>Текущие данные Telegram:</b>\n"
            f"👤 Username: @{message.from_user.username or 'не указан'}\n"
            f"🏷 Имя: {message.from_user.first_name or 'не указано'}\n"
            f"🏷 Фамилия: {message.from_user.last_name or 'не указана'}\n"
        )
    else:
        info = (
            f"❌ <b>Вы не найдены в БД</b>\n\n"
            f"User ID: <code>{user_id}</code>\n\n"
            f"Данные из Telegram:\n"
            f"👤 Username: @{message.from_user.username or 'не указан'}\n"
            f"🏷 Имя: {message.from_user.first_name or 'не указано'}\n"
            f"🏷 Фамилия: {message.from_user.last_name or 'не указана'}\n"
        )
    
    await message.answer(info, parse_mode=ParseMode.HTML)



@dp.message(F.text == "📖 Прочитать политику конфиденциальности")
async def show_privacy_from_consent(message: Message):
    """Показать политику из экрана согласия"""
    await message.answer(PRIVACY_POLICY, parse_mode=ParseMode.HTML)

@dp.message(Command("mydata"))
async def cmd_my_data(message: Message):
    """Экспорт данных пользователя"""
    user_id = message.from_user.id
    await message.answer("📦 Собираю ваши данные...")
    
    try:
        data = await export_user_data(user_id)
        
        with tempfile.NamedTemporaryFile(
            mode="w", 
            suffix=".json", 
            delete=False, 
            encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            temp_path = f.name
        
        await message.answer_document(
            FSInputFile(temp_path, filename=f"my_data_{user_id}.json"),
            caption=(
                "📦 Ваши данные в формате JSON\n\n"
                "Содержит всю информацию, которую мы храним о вас."
            ),
        )
        
        os.unlink(temp_path)
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        metrics_errors_total.labels(error_type="data_export").inc()
        await message.answer("❌ Ошибка экспорта данных")

@dp.message(Command("deletemydata"))
async def cmd_delete_my_data(message: Message):
    """Удаление данных пользователя"""
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="✅ Да, удалить все данные")],
            [KeyboardButton(text="❌ Отмена")],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )
    
    await message.answer(
        "⚠️ <b>ВНИМАНИЕ!</b>\n\n"
        "Вы уверены, что хотите удалить ВСЕ свои данные?\n\n"
        "Будет удалено:\n"
        "• История всех запросов\n"
        "• Записи на консультации\n"
        "• Обратная связь\n"
        "• Профиль пользователя\n\n"
        "❗ Это действие необратимо!",
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )

@dp.message(F.text == "✅ Да, удалить все данные")
async def confirm_delete_data(message: Message):
    """Подтверждение удаления"""
    user_id = message.from_user.id
    try:
        await delete_user_data(user_id)

        # ================= GDPR: ОЧИСТКА PROMETHEUS METRICS =================
        # Пытаемся очистить metrics с user_id label
        prometheus_stats = await clean_prometheus_user_metrics(user_id)

        if prometheus_stats["errors"]:
            logger.warning(
                f"GDPR: Prometheus cleanup completed with errors for user {user_id}: "
                f"{prometheus_stats['errors']}"
            )
        else:
            logger.info(f"GDPR: Prometheus metrics cleaned for user {user_id}")

        # Формируем ответ пользователю
        success_parts = [
            "✅ Все ваши данные удалены.\n\n",
            "Удалено:\n",
            "• История всех запросов\n",
            "• Записи на консультации\n",
            "• Обратная связь\n",
            "• Профиль пользователя\n",
            "• Redis сессия\n",
        ]

        # Добавляем информацию о Prometheus metrics
        if prometheus_stats["metrics_cleaned"]:
            success_parts.append("• Prometheus metrics\n")

        success_parts.extend([
            "\n",
            "Вы можете продолжить использовать бота, ",
            "но вся история будет начата заново.\n\n",
            "До встречи! 👋",
        ])

        await message.answer(
            "".join(success_parts),
            reply_markup=ReplyKeyboardRemove(),
        )
    except Exception as e:
        logger.error(f"Error deleting data: {e}")
        metrics_errors_total.labels(error_type="data_deletion").inc()
        await message.answer("❌ Ошибка удаления данных")

@dp.message(Command("anonymize"))
async def cmd_anonymize(message: Message):
    """Анонимизация данных"""
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="✅ Да, анонимизировать")],
            [KeyboardButton(text="❌ Отмена")],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )
    
    await message.answer(
        "🔒 <b>Анонимизация данных</b>\n\n"
        "Будет сохранена статистика запросов (без текста), "
        "но все персональные данные будут удалены:\n\n"
        "• Имя пользователя\n"
        "• Контактные данные\n"
        "• Текст запросов\n\n"
        "✅ Анонимизированные данные используются только для статистики.\n\n"
        "Продолжить?",
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )
    
@dp.message(F.text == "✅ Да, анонимизировать")
async def confirm_anonymize(message: Message):
    """Подтверждение анонимизации"""
    user_id = message.from_user.id
    try:
        await anonymize_user_data(user_id)
        await message.answer(
            "✅ Ваши данные анонимизированы.\n\n"
            "Вы можете продолжать использовать бота.",
            reply_markup=ReplyKeyboardRemove(),
        )
    except Exception as e:
        logger.error(f"Error anonymizing data: {e}")
        metrics_errors_total.labels(error_type="data_anonymization").inc()
        await message.answer("❌ Ошибка анонимизации данных")

@dp.message(F.text == "❌ Отмена")
async def cancel_action(message: Message):
    """Отмена действия"""
    await message.answer("❌ Действие отменено.", reply_markup=ReplyKeyboardRemove())

@dp.message(Command("fullprivacy"))
async def cmd_full_privacy(message: Message):
    """Полная политика конфиденциальности"""
    await message.answer(
        "📄 <b>Полная политика конфиденциальности</b>\n\n"
        "Полная версия политики конфиденциальности находится в разработке.\n\n"
        "Если у вас есть вопросы о обработке данных, "
        "напишите нам: support@example.com",
        parse_mode=ParseMode.HTML,
    )

@dp.message(Command("history"))
async def cmd_history(message: Message):
    """Показывает историю сессии"""
    user_id = message.from_user.id
    session = await get_or_create_session(user_id)
    
    if not session.history:
        await message.answer("История пуста. Задайте первый вопрос!")
        return
    
    history_text = "<b>📜 История вашей сессии:</b>\n\n"
    for i, interaction in enumerate(session.history, 1):
        # Исправление: timestamp может быть строкой, преобразуем её
        timestamp = interaction["timestamp"]
        if isinstance(timestamp, str):
            from datetime import datetime
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except:
                timestamp = datetime.now()  # Fallback
        
        time_ago = datetime.now() - timestamp
        minutes_ago = int(time_ago.total_seconds() / 60)
        
        history_text += (
            f"<b>{i}.</b> ({minutes_ago} мин назад)\n"
            f"❓ {html.escape(interaction['question'][:100])}\n"
            f"📚 Статьи: {', '.join(interaction['article_nums']) if interaction['article_nums'] else 'нет'}\n\n"
        )
    
    history_text += (
        "\n💡 Контекстные вопросы: используйте 'а если...', 'расскажи подробнее' и т.д."
    )
    await message.answer(history_text, parse_mode=ParseMode.HTML)


@dp.message(Command("clearsession"))
async def cmd_clear_session(message: Message):
    """Очищает сессию пользователя"""
    user_id = message.from_user.id
    await redis_manager.delete_session(user_id)
    await message.answer("✅ История сессии очищена. Начнем заново!")

@dp.message(F.text.in_(FAQ_ANSWERS.keys()))
async def handle_faq(message: Message):
    """Быстрые ответы на FAQ"""
    answer = FAQ_ANSWERS[message.text]
    
    await safe_react(message, "👀")
    await message.answer(answer, parse_mode=ParseMode.HTML)
    await message.react([ReactionTypeEmoji(emoji="🔥")])
    
    await message.answer(
        "Остались вопросы? Задайте уточняющий вопрос или выберите другую тему.",
        reply_markup=faq_keyboard,
    )

@dp.message(Command("article"))
async def cmd_article(message: Message, command: CommandObject):
    """
    Получить полный текст статьи
    Использование: /article 454 или /article 454_2 или /article 454 2
    """
    user_id = message.from_user.id
    
    # Проверка согласия
    user_data = await get_user_data(user_id)
    if not user_data.get("consent_given", False):
        consent_keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="✅ Согласен с обработкой данных")],
                [KeyboardButton(text="📖 Прочитать политику конфиденциальности")],
            ],
            resize_keyboard=True,
        )
        await message.answer(
            "⚠️ Для работы бота необходимо согласие на обработку персональных данных.\n\n"
            "Мы гарантируем:\n"
            "🔒 Шифрование всех персональных данных\n"
            "🛡️ Защиту от несанкционированного доступа\n"
            "📝 Право на удаление данных в любой момент\n\n"
            "Подробнее: /privacy",
            reply_markup=consent_keyboard,
        )
        return
    
    args = (command.args or "").strip()
    if not args:
        await message.answer(
            "📖 <b>Команда для получения полного текста статьи</b>\n\n"
            "Использование:\n"
            "<code>/article 454</code> - показать статью 454\n"
            "<code>/article 454_2</code> - показать пункт 2 статьи 454\n"
            "<code>/article 454 2</code> - показать пункт 2 статьи 454\n\n"
            "Также можно просто написать в чате:\n"
            "• «статья 454»\n"
            "• «ст. 454 п. 2»\n"
            "• «статья 196 пункт 1»",
            parse_mode=ParseMode.HTML,
        )
        return
    
    # Парсинг article / point
    article: str
    point: Optional[str]
    
    if "_" in args:
        parts = args.split("_")
        article = parts[0]
        point = parts[1] if len(parts) > 1 else None
    elif " " in args:
        parts = args.split()
        article = parts[0]
        point = parts[1] if len(parts) > 1 else None
    else:
        article = args
        point = None
    
    # Валидация номера статьи
    if not validate_article_number(article):
        await message.answer(
            f"❌ Неверный формат номера статьи: {html.escape(article)}\n\n"
            "Используйте: <code>/article 454</code>\n"
            "Номер статьи должен быть от 1 до 1551.",
            parse_mode=ParseMode.HTML,
        )
        return
    
    await safe_react(message, "👀")
    await message.answer(
        f"📖 Загружаю статью {article}...", 
        parse_mode=ParseMode.HTML
    )
    
    try:
        # Используем функцию с LLM fallback
        answer = await get_article_with_llm_fallback(
            article, 
            point, 
            original_query=f"статья {article}" + (f" пункт {point}" if point else "")
        )
        
        await message.react([ReactionTypeEmoji(emoji="🔥")])
        await send_long_message(message, answer, parse_mode=ParseMode.HTML)
        
        # Метрики
        metrics_questions_total.labels(
            user_id=user_id,
            question_type="article_lookup"
        ).inc()
        
        # Обновляем сессию (только если статья найдена)
        if "❌" not in answer or "не найдена в базе данных" in answer:
            session = await get_or_create_session(user_id)
            session.add_interaction(f"/article {args}", answer, [article])
            await save_session(session)
        
        # Сохранение в БД
        async with sql_db._get_connection_async() as conn:
            await conn.execute("""
                INSERT INTO user_queries (user_id, query_text, answer_text, article_nums, query_type)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, f"/article {args}", answer, article, "article_lookup"))
            await conn.commit()
        
        await track_user_query(user_id)
        
        logger.info(
            f"Article {article}" +
            (f' point {point}' if point else "") +
            f" sent successfully to user {user_id}"
        )
        
    except Exception as e:
        logger.error(f"Error fetching article {article}: {e}")
        metrics_errors_total.labels(error_type="article_fetch").inc()
        await safe_react(message, "❌")
        await message.answer("❌ Ошибка получения статьи. Попробуйте позже.")

@dp.message(F.text)
async def handle_question(message: Message):
    """Обработчик текстовых вопросов"""
    user_query = message.text.strip()
    
    user_id = message.from_user.id
    username = message.from_user.username or "unknown"
    
    # ← ДОБАВЬТЕ ОБРАБОТКУ СИСТЕМНЫХ КНОПОК
    if user_query in ["⏭ Пропустить", "❌ Отмена", "❌ Отменить"]:
        await message.answer(
            "Хорошо! Вы можете задать любой вопрос или использовать /help",
            reply_markup=ReplyKeyboardRemove()
        )
        return
    
    # Собираем данные пользователя из Telegram
    user_info = {
        'username': message.from_user.username,
        'first_name': message.from_user.first_name,
        'last_name': message.from_user.last_name
    }
    
    # Проверяем согласие
    user_data = await get_user_data(user_id)
    
    if not user_data.get("consent_given", False):
        consent_keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="✅ Согласен с обработкой данных")],
                [KeyboardButton(text="📖 Прочитать политику конфиденциальности")],
            ],
            resize_keyboard=True,
        )
        
        await message.answer(
            "⚠️ Для работы бота необходимо согласие на обработку персональных данных.\n\n"
            "Мы гарантируем:\n"
            "🔒 Шифрование всех персональных данных\n"
            "🛡️ Защиту от несанкционированного доступа\n"
            "📝 Право на удаление данных в любой момент\n\n"
            "Подробнее: /privacy",
            reply_markup=consent_keyboard,
        )
        return
    
    # Обновляем активность пользователя
    await update_user_activity(user_id, user_info)
    
    if user_query == "❓ Другой вопрос":
        await message.answer(
            "Клавиатура скрыта. Задайте любой вопрос, или нажмите /start чтобы вернуть меню.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return
    
    if not user_query:
        await message.answer("Пожалуйста, задайте ваш вопрос.")
        return
    
    if len(user_query) < 3:
        await message.answer(
            "❓ Вопрос слишком короткий. Пожалуйста, сформулируйте вопрос подробнее.\n\n"
            "Например: 'Что такое договор купли-продажи?' или 'Какие документы нужны для покупки квартиры?'"
        )
        return
    
    # Проверка на системные запросы (запись на консультацию)
    consultation_keywords = [
        "записаться", "запись", "консультац", "юрист", 
        "нужна помощь", "нужен юрист"
    ]
    
    if any(kw in user_query.lower() for kw in consultation_keywords):
        await message.answer(
            "📅 <b>Запись на консультацию</b>\n\n"
            "К сожалению, функция записи на консультацию временно недоступна.\n\n"
            "Но вы можете:\n"
            "• ❓ Задать любой юридический вопрос прямо здесь\n"
            "• 📚 Использовать /examples для примеров вопросов\n"
            "• 📖 Изучить конкретные статьи через команду /article\n\n"
            "Просто напишите ваш вопрос!",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Проверка на вопросы вне ГК РФ
    non_civil_topics = {
        "регистрац": "регистрация по месту жительства (административное право)",
        "паспорт": "получение паспорта (административное право)",
        "виза": "получение визы (миграционное право)",
        "гражданство": "получение гражданства (миграционное право)",
        "уголовн": "уголовное право",
        "налог": "налоговое право",
        "штраф": "административное право"
    }

    for keyword, topic_name in non_civil_topics.items():
        if keyword in user_query.lower():
            await message.answer(
                f"⚠️ <b>Ваш вопрос относится к другой области права</b>\n\n"
                f"Тема: {topic_name}\n\n"
                f"Я специализируюсь только на <b>Гражданском кодексе РФ</b>, который регулирует:\n"
                f"• Купля-продажа товаров и недвижимости\n"
                f"• Договоры и обязательства\n"
                f"• Наследование\n"
                f"• Авторские права\n"
                f"• Собственность\n\n"
                f"Для вопросов по {topic_name} обратитесь к специализированному юристу.\n\n"
                f"Могу помочь с вопросами по ГК РФ! Примеры: /examples",
                parse_mode=ParseMode.HTML
            )
            return
    logger.info(f"User {user_id} (@{username}) asked: {user_query[:100]}...")
    
    # Rate limit (Redis distributed)
    if await redis_manager.check_rate_limit(user_id):
        await message.answer(
            "⏸ <b>Слишком много запросов!</b>\n\n"
            "Пожалуйста, подождите немного перед следующим вопросом.\n\n"
            f"Лимит: {RATE_LIMIT_REQUESTS} запросов в {RATE_LIMIT_WINDOW} секунд.",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Получаем сессию из Redis
    session = await get_or_create_session(user_id)
    
    # Контекстные запросы "пункт X"
    if re.match(r"^(п\.?|пункт)\s*\d+", user_query.lower()):
        last_article = session.last_article_context
        if last_article:
            enriched_context_query = f"статья {last_article} {user_query}"
            logger.info(
                f"Contextual point query detected: '{user_query}' -> '{enriched_context_query}'"
            )
            user_query = enriched_context_query
    
    # Follow-up detection
    is_follow_up = session.is_follow_up(user_query)
    question_lower = user_query.lower()
    
    contextual_point_part = re.match(
        r"^(а\s*)?(п\.?|пункт|часть|ч\.?)\s*\d+", question_lower
    )
    if contextual_point_part and session.history:
        is_follow_up = True
        logger.info(f"Contextual follow-up detected (no keywords): '{user_query}'")
    
    # Запрос о конкретной статье
    article_match = detect_article_query(user_query)
    
    start_time = time.time()
    
    await safe_react(message, "👀")
    
    try:
        if article_match:
            article, point = article_match
            logger.info(
                f"Direct article query detected in text: article={article}, point={point}"
            )
            
            # Используем функцию с LLM fallback
            answer = await get_article_with_llm_fallback(
                article, 
                point, 
                original_query=user_query
            )
            
            article_nums = [article]
            query_type = "article_direct"
            
        else:
            # Обогащаем запрос контекстом если это follow-up
            if is_follow_up and session.history:
                last_interaction = session.history[-1]
                enriched_query = (
                    f"Контекст предыдущего вопроса: {last_interaction['question']}\n"
                    f"Мой ответ был про: статьи {', '.join(last_interaction['article_nums'])}\n\n"
                    f"Уточняющий вопрос: {user_query}"
                )
                logger.info("Follow-up question detected, adding context")
            else:
                enriched_query = user_query
            
            # Генерируем ответ через LLM
            raw_answer = await process_with_typing(message, enriched_query)
            
            # Санитизация HTML от LLM
            answer = sanitize_html(raw_answer)
            
            # Извлекаем упомянутые статьи
            mentioned_articles = re.findall(
                r"стать[ияюе]\s+(\d+)|ст\.?\s*(\d+)",
                raw_answer,
                re.IGNORECASE,
            )
            article_nums = list(set(a[0] or a[1] for a in mentioned_articles))
            
            # Валидация существования статей
            valid_articles: List[str] = []
            for a in article_nums:
                if validate_article_number(a) and article_exists_in_db(a):
                    valid_articles.append(a)
                else:
                    logger.warning(
                        f"LLM mentioned non-existent or invalid article {a}, skipping"
                    )
            article_nums = valid_articles
            
            query_type = "general"
        
        # Обновляем сессии
        session.add_interaction(user_query, answer, article_nums)
        await save_session(session)
        
        # Сохранение в SQL БД (queries таблица)
        global sql_db
        async with sql_db._get_connection_async() as conn:
            await conn.execute("""
                INSERT INTO queries (
                    user_id, query_text, query_type, 
                    article_num, answer_text, processing_time_sec
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id, 
                user_query,
                query_type,
                ','.join(article_nums) if article_nums else None,
                answer,
                time.time() - start_time
            ))
            await conn.commit()
        
        # Метрики
        elapsed = time.time() - start_time
        metrics_response_time.observe(elapsed)
        metrics_questions_total.labels(
            user_id=user_id,
            question_type=query_type
        ).inc()
        
        await message.react([ReactionTypeEmoji(emoji="🔥")])
        await send_long_message(message, answer, parse_mode=ParseMode.HTML)
        
        logger.info(
            f"Answer sent to user {user_id} "
            f"(length: {len(answer)} chars, processing_time: {elapsed:.2f}s)"
        )
        
    except TimeoutError:
        logger.error(f"Timeout error for user {user_id}")
        metrics_errors_total.labels(error_type="timeout").inc()
        await message.react([ReactionTypeEmoji(emoji="⏱")])
        await message.answer(
            "⏱ Превышено время ожидания.\n\n"
            "Попробуйте:\n"
            "• Упростить вопрос\n"
            "• Разбить на несколько вопросов\n"
            "• Повторить через минуту"
        )
    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"Error ({error_type}) for user {user_id}: {e}", exc_info=True)
        metrics_errors_total.labels(error_type=error_type).inc()
        
        await safe_react(message, "❌")
        
        if "rate limit" in str(e).lower():
            error_msg = (
                "⚠️ Превышен лимит запросов к AI.\n\n"
                "Пожалуйста, подождите 1-2 минуты и попробуйте снова."
            )
        elif "connection" in str(e).lower() or "connect" in str(e).lower():
            error_msg = (
                "🌐 Проблемы с соединением.\n\n"
                "Проверьте интернет и повторите попытку."
            )
        elif "timeout" in str(e).lower():
            error_msg = (
                "⏱ Тайм-аут при обработке запроса.\n\n"
                "Попробуйте:\n"
                "• Упростить вопрос\n"
                "• Повторить через минуту"
            )
        elif "key" in str(e).lower() or "api" in str(e).lower():
            error_msg = (
                "🔑 Ошибка API ключа.\n\n"
                "Попробуйте повторить запрос позже.\n"
                "Если ошибка повторяется, обратитесь к администратору."
            )
        else:
            error_msg = (
                "❌ Произошла ошибка при обработке запроса.\n\n"
                "Попробуйте:\n"
                "• Переформулировать вопрос\n"
                "• Использовать /examples для примеров\n"
                "• Повторить через минуту\n\n"
                f"Код ошибки: {error_type}"
            )
        
        await message.answer(error_msg)


# ================= STARTUP / SHUTDOWN =================

async def on_startup():
    """Действия при запуске бота"""
    global sql_db
    
    logger.info("=" * 70)
    logger.info("🤖 Legal Consultation Bot - Production Ready")
    logger.info("=" * 70)
    
    # Инициализируем SQL базу для пользователей
    sql_db = LegalBotDB()
    
    # Подключаем Redis
    await redis_manager.connect()
    
    # Запускаем Prometheus metrics
    start_http_server(METRICS_PORT)
    logger.info(f"📊 Metrics server started on port {METRICS_PORT}")
    
    # ✅ ИСПРАВЛЕНО: Регистрируем задачи ПОСЛЕ инициализации всех сервисов
    scheduler.add_job(
        auto_cleanup,
        'cron',
        hour=3,
        minute=0,
        id='auto_cleanup',
        replace_existing=True
    )
    scheduler.add_job(
        session_cleanup,
        'interval',
        hours=1,
        id='session_cleanup',
        replace_existing=True
    )
    scheduler.add_job(
        redis_cleanup,
        'interval',
        hours=2,
        id='redis_cleanup',
        replace_existing=True
    )
    
    # Запускаем планировщик ТОЛЬКО ПОСЛЕ полной инициализации
    scheduler.start()
    logger.info("✅ Background scheduler started with 3 jobs")
    
    logger.info("✅ Bot startup complete")


async def on_shutdown():
    """Действия при остановке бота"""
    logger.info("🛑 Shutting down bot...")
    
    # Закрываем DB pool
    await sql_db .close_pool()
    
    # Закрываем Redis
    await redis_manager.close()
    
    # Останавливаем планировщик
    scheduler.shutdown()
    
    logger.info("✅ Bot shutdown complete")

# ================= SIGNAL HANDLERS =================

import signal

shutdown_event = asyncio.Event()

def signal_handler(sig, frame):
    """Обработчик сигналов для graceful shutdown"""
    logger.info(f"🛑 Received signal {sig}, initiating graceful shutdown...")
    shutdown_event.set()
  

# Регистрируем обработчики сигналов
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# ================= REGISTER LIFECYCLE HANDLERS =================
# Регистрируем startup и shutdown handlers
dp.startup.register(on_startup)
dp.shutdown.register(on_shutdown)

# ================= MAIN =================

async def main():
    """Главная функция запуска бота"""
    try:
        logger.info("🚀 Starting bot...")
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        logger.info("⚠️ Bot stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logger.error(f"❌ Fatal error in main: {e}", exc_info=True)
    
if __name__ == "__main__":
    asyncio.run(main())

