"""sql_logger.py
SQLite база данных для логирования и управления пользователями
PRODUCTION-READY ВЕРСИЯ С ШИФРОВАНИЕМ И ПОЛНЫМ GDPR COMPLIANCE
"""

import os
import sqlite3
import aiosqlite
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Literal
from contextlib import contextmanager, asynccontextmanager
from threading import Lock
from collections import defaultdict
import csv
import hashlib

import structlog

from security import FieldLevelEncryptionWrapper, EncryptionManager, GDPRCompliance, AuditLogger
from gdpr_exceptions import ConsentRequiredError, ConsentAlreadyGivenError
from security import EncryptionError, DecryptionError

log = structlog.get_logger()

# ================= CONFIGURATION =================

DATABASE_PATH = "./data/legal_bot.db"
ALLOWED_TABLES = {"users", "queries", "consultations", "feedback"}
EXPORTS_DIR = Path("./exports").resolve()

# Лимиты длины полей (защита от DoS)
MAX_QUERY_LENGTH = 5000
MAX_ANSWER_LENGTH = 20000
MAX_ERROR_LENGTH = 2000
MAX_COMMENT_LENGTH = 1000
MAX_DESCRIPTION_LENGTH = 5000
MAX_TOPIC_LENGTH = 200

# Rate limiting (для Redis в будущем)
RATE_LIMIT_WINDOW = 60  # секунды
RATE_LIMIT_MAX_REQUESTS = 10

ALLOWED_CONSULTATION_STATUSES = {"pending", "confirmed", "completed", "cancelled"}
# ================= DATABASE SCHEMA =================

SCHEMA = """
-- Таблица пользователей
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    username TEXT,  -- ENCRYPTED
    first_name TEXT,  -- ENCRYPTED
    last_name TEXT,  -- ENCRYPTED
    phone TEXT,  -- ENCRYPTED
    language_code TEXT,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_queries INTEGER DEFAULT 0,
    is_blocked INTEGER CHECK(is_blocked IN (0,1)) DEFAULT 0,
    consent_given INTEGER CHECK(consent_given IN (0,1)) DEFAULT 0,
    consent_date TIMESTAMP,
    deleted_at TIMESTAMP,
    anonymized INTEGER CHECK(anonymized IN (0,1)) DEFAULT 0,
    notes TEXT  -- ENCRYPTED (может содержать PII)
);


CREATE TABLE IF NOT EXISTS user_stats (
    user_id INTEGER PRIMARY KEY,
    total_queries INTEGER DEFAULT 0,
    first_query TIMESTAMP,
    last_query_date TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Таблица запросов
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    query_text TEXT NOT NULL,  -- ENCRYPTED
    query_type TEXT,
    article_num TEXT,
    point_num TEXT,
    answer_text TEXT,  -- ENCRYPTED
    answer_length INTEGER,
    from_cache INTEGER CHECK(from_cache IN (0,1)) DEFAULT 0,
    processing_time_sec REAL,
    tokens_used INTEGER,
    error_occurred INTEGER CHECK(error_occurred IN (0,1)) DEFAULT 0,
    error_message TEXT,
    deleted_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    query_text TEXT NOT NULL,  -- ENCRYPTED
    answer_text TEXT,  -- ENCRYPTED
    article_nums TEXT,
    query_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Таблица записей на консультации
CREATE TABLE IF NOT EXISTS consultations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    preferred_date TEXT,
    preferred_time TEXT,
    contact_phone TEXT,  -- ENCRYPTED
    contact_email TEXT,  -- ENCRYPTED
    topic TEXT,
    description TEXT,  -- ENCRYPTED
    status TEXT DEFAULT 'pending',
    consultation_date TIMESTAMP,
    lawyer_notes TEXT,
    deleted_at TIMESTAMP,  
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);


-- Таблица обратной связи
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    query_id INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rating INTEGER CHECK(rating >= 1 AND rating <= 5),
    comment TEXT,  -- ENCRYPTED (может содержать PII)
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE
);

-- Таблица для отслеживания ротации ключей шифрования
CREATE TABLE IF NOT EXISTS encryption_key_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_version INTEGER NOT NULL,
    action TEXT NOT NULL,  -- 'CREATED', 'ROTATED', 'DEACTIVATED'
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    details TEXT,
    affected_records INTEGER,
    UNIQUE(key_version, action)
);

-- Таблица для хранения версий ключей шифрования в encrypted data
CREATE TABLE IF NOT EXISTS encrypted_data_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    record_id INTEGER NOT NULL,
    field_name TEXT NOT NULL,
    key_version INTEGER NOT NULL,
    encrypted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(table_name, record_id, field_name)
);

-- Индексы для ускорения запросов
CREATE INDEX IF NOT EXISTS idx_queries_user_id ON queries(user_id);
CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp);
CREATE INDEX IF NOT EXISTS idx_queries_article ON queries(article_num);
CREATE INDEX IF NOT EXISTS idx_user_queries_user_id ON user_queries(user_id);
CREATE INDEX IF NOT EXISTS idx_user_queries_created_at ON user_queries(created_at);
CREATE INDEX IF NOT EXISTS idx_user_stats_user_id ON user_stats(user_id);  -- ← ДОБАВЬТЕ ИНДЕКС
CREATE INDEX IF NOT EXISTS idx_consultations_user_id ON consultations(user_id);
CREATE INDEX IF NOT EXISTS idx_consultations_status ON consultations(status);
CREATE INDEX IF NOT EXISTS idx_users_last_active ON users(last_active);
CREATE INDEX IF NOT EXISTS idx_encryption_key_audit_version ON encryption_key_audit(key_version);
CREATE INDEX IF NOT EXISTS idx_encrypted_data_versions_lookup ON encrypted_data_versions(table_name, record_id, field_name);
CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback(user_id); 
CREATE INDEX IF NOT EXISTS idx_feedback_query_id ON feedback(query_id);
"""

# ================= GLOBAL SINGLETON LOCKS =================
# ✅ ИСПРАВЛЕНО: Lock создаётся один раз на уровне модуля

_db_instance = None
_db_lock = Lock()  # Threading Lock для sync контекста
_db_async_lock = None  # Asyncio Lock (lazy init)

# ================= DATABASE MANAGER =================

class LegalBotDB:
    """
    Production-ready база данных с полным GDPR compliance
    
    ⚠️ ВАЖНО:
    - SQLite НЕ рекомендуется для high-load продакшена
    - Для продакшена используйте PostgreSQL с connection pooling
    - Rate limiting требует Redis для multi-instance окружения
    """
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем системы безопасности
        self.encryption = EncryptionManager()
        self.secure_db = FieldLevelEncryptionWrapper(str(db_path))
        self._audit = None
        
        # Rate-limiting (thread-safe, но in-memory)
        # ⚠️ TODO: Переместить в Redis для продакшена
        self._rate_limits = defaultdict(list)
        self._rate_limit_lock = Lock()
        
        self._init_database()
        
        # Устанавливаем права доступа
        if self.db_path.exists():
            os.chmod(self.db_path, 0o600)
            log.info("✅ Database file permissions set to 0600")
    
    def _init_database(self):
        """
        Инициализирует схему БД с оптимизациями для многопоточности
        
        ✅ ИСПРАВЛЕНО: Добавлены PRAGMA для WAL mode
        """
        with self._get_connection() as conn:
            # ✅ КРИТИЧНО: Включаем WAL mode для многопоточной работы
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")
            
            conn.executescript(SCHEMA)
        
        log.info("✅ Database schema initialized with WAL mode")
    
    @contextmanager
    def _get_connection(self):
        """Context manager для получения соединения с БД"""
        conn = sqlite3.connect(
            self.db_path, 
            check_same_thread=False,
            timeout=10.0  # ✅ Добавлен timeout для избежания дедлоков
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            log.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    @asynccontextmanager
    async def _get_connection_async(self):
        """
        Async context manager для aiosqlite
        
        ⚠️ ВАЖНО: SQLite + async + многопоточность = проблемы
        Для продакшена используйте asyncpg (PostgreSQL)
        """
        async with aiosqlite.connect(
            self.db_path,
            timeout=10.0
        ) as conn:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
                await conn.commit()
            except Exception as e:
                await conn.rollback()
                log.error(f"Database error: {e}")
                raise
    
    async def _get_audit_logger(self):
        """Ленивое получение audit logger (async)"""
        if self._audit is None:
            self._audit = await AuditLogger.get_instance()
        return self._audit
    
    def _get_audit_logger_sync(self):
        """Sync версия (упрощенная - только для критических случаев)"""
        if self._audit is None:
            log.warning("⚠️  Audit logger not initialized in sync context")
            return None
        return self._audit
    
    # ================= FIELD LENGTH VALIDATION =================
    
    def _validate_field_length(self, value: Optional[str], max_length: int, field_name: str):
        """
        Проверяет длину поля для защиты от DoS
        
        ✅ ИСПРАВЛЕНО: Добавлена валидация длины всех TEXT полей
        """
        if value and len(value) > max_length:
            raise ValueError(
                f"{field_name} exceeds maximum length of {max_length} characters. "
                f"Got {len(value)} characters."
            )
    
    # ================= CONSENT CHECKS =================
    
    def has_user_consent(self, user_id: int) -> bool:
        """Sync версия проверки согласия"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT consent_given FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        
        has_consent = bool(row["consent_given"]) if row else False
        
        if not has_consent:
            log.warning(f"⚠️  GDPR: User {user_id} attempted operation without consent")
            audit = self._get_audit_logger_sync()
            if audit:
                audit.log_access(
                    user_id=user_id,
                    action="OPERATION_BLOCKED_NO_CONSENT",
                    data_type="PERSONAL_DATA",
                    details="User attempted operation without consent"
                )
        
        return has_consent
    
    async def has_user_consent_async(self, user_id: int) -> bool:
        """Async версия проверки согласия"""
        async with self._get_connection_async() as conn:
            cursor = await conn.execute(
                "SELECT consent_given FROM users WHERE user_id = ?", (user_id,)
            )
            row = await cursor.fetchone()
        
        has_consent = bool(row["consent_given"]) if row else False
        
        if not has_consent:
            log.warning(f"⚠️  GDPR: User {user_id} attempted operation without consent")
            audit = await self._get_audit_logger()
            await audit.log_access(
                user_id=user_id,
                action="OPERATION_BLOCKED_NO_CONSENT",
                data_type="PERSONAL_DATA",
                details="User attempted operation without consent"
            )
        
        return has_consent
    
    # ================= RATE LIMITING =================
    
    def _check_rate_limit(self, key: str, max_requests: int = RATE_LIMIT_MAX_REQUESTS) -> bool:
        """
        Проверяет rate limit для операции
        
        ⚠️ TODO: Переместить в Redis для multi-instance окружения
        
        Returns:
            True если лимит не превышен, False иначе
        """
        with self._rate_limit_lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=RATE_LIMIT_WINDOW)
            
            # Очистка старых записей
            self._rate_limits[key] = [
                ts for ts in self._rate_limits[key] if ts > cutoff
            ]
            
            # Проверка лимита
            if len(self._rate_limits[key]) >= max_requests:
                return False
            
            self._rate_limits[key].append(now)
            return True
    
    # ================= USERS =================
    
    def register_user(
        self,
        user_id: int,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        language_code: Optional[str] = None
    ):
        """
        Регистрирует нового пользователя или обновляет существующего
        
        ✅ ИСПРАВЛЕНО: PII данные теперь шифруются
        """
        # ✅ Валидация длины
        self._validate_field_length(username, 100, "username")
        self._validate_field_length(first_name, 100, "first_name")
        self._validate_field_length(last_name, 100, "last_name")
        
        # ✅ Шифрование PII
        encrypted_username = self.secure_db.encrypt_field(username, 'username') if username else None
        encrypted_first_name = self.secure_db.encrypt_field(first_name, 'first_name') if first_name else None
        encrypted_last_name = self.secure_db.encrypt_field(last_name, 'last_name') if last_name else None
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO users (user_id, username, first_name, last_name, language_code)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    username = excluded.username,
                    first_name = excluded.first_name,
                    last_name = excluded.last_name,
                    language_code = excluded.language_code,
                    last_active = CURRENT_TIMESTAMP
            """, (user_id, encrypted_username, encrypted_first_name, encrypted_last_name, language_code))
    
    def update_user_activity(self, user_id: int):
        """Обновляет время последней активности"""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE users 
                SET last_active = CURRENT_TIMESTAMP,
                    total_queries = total_queries + 1
                WHERE user_id = ?
            """, (user_id,))
    
    def get_user(self, user_id: int, decrypt: bool = True) -> Optional[Dict]:
        """
        Получает информацию о пользователе
        
        Args:
            user_id: ID пользователя
            decrypt: Расшифровывать ли PII данные (требует согласие)
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            
            if not row:
                return None
            
            user = dict(row)
            
            # Расшифровка если запрошено
            if decrypt:
                try:
                    if user['username']:
                        user['username'] = self.secure_db.decrypt_field(user['username'], 'username')
                    if user['first_name']:
                        user['first_name'] = self.secure_db.decrypt_field(user['first_name'], 'first_name')
                    if user['last_name']:
                        user['last_name'] = self.secure_db.decrypt_field(user['last_name'], 'last_name')
                    if user['notes']:
                        user['notes'] = self.secure_db.decrypt_field(user['notes'], 'notes')
                except DecryptionError as e:
                    log.error(f"Decryption failed for user {user_id}: {e.message}", exc_info=False)
            
            return user
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Получает статистику пользователя"""
        with self._get_connection() as conn:
            user = conn.execute(
                "SELECT total_queries, first_seen, last_active FROM users WHERE user_id = ?",
                (user_id,)
            ).fetchone()
            
            recent_queries = conn.execute("""
                SELECT COUNT(*) as count 
                FROM queries
                WHERE user_id = ? AND timestamp >= datetime('now', '-7 days')
            """, (user_id,)).fetchone()
            
            consultations_count = conn.execute("""
                SELECT COUNT(*) as count, status
                FROM consultations 
                WHERE user_id = ?
                GROUP BY status
            """, (user_id,)).fetchall()
            
            return {
                "total_queries": user["total_queries"] if user else 0,
                "first_seen": user["first_seen"] if user else None,
                "last_active": user["last_active"] if user else None,
                "recent_queries": recent_queries["count"] if recent_queries else 0,
                "consultations": {row["status"]: row["count"] for row in consultations_count}
            }
    
    def block_user(self, user_id: int, reason: str = None):
        """Блокирует пользователя"""
        # ✅ Шифруем reason (может содержать PII)
        encrypted_reason = self.secure_db.encrypt_field(reason, 'notes') if reason else None
        
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE users SET is_blocked = 1, notes = ? WHERE user_id = ?",
                (encrypted_reason, user_id)
            )
        log.info(f"User {user_id} blocked")
    
    def is_user_blocked(self, user_id: int) -> bool:
        """Проверяет заблокирован ли пользователь"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT is_blocked FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            return bool(row["is_blocked"]) if row else False
    
    def set_user_consent(self, user_id: int, consent: bool = True):
        """Устанавливает согласие пользователя на обработку данных"""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE users 
                SET consent_given = ?, consent_date = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (consent, user_id))
        log.info(f"User {user_id} consent set to {consent}")
    
    # ================= QUERIES =================
    
    def log_query(
        self,
        user_id: int,
        query_text: str,
        query_type: str = "general",
        article_num: Optional[str] = None,
        point_num: Optional[str] = None,
        answer_text: Optional[str] = None,
        from_cache: bool = False,
        processing_time_sec: float = 0.0,
        tokens_used: int = 0,
        error_occurred: bool = False,
        error_message: Optional[str] = None
    ) -> int:
        """
        Логирует запрос пользователя
        
        ✅ ИСПРАВЛЕНО:
        - Проверка согласия
        - Валидация длины полей
        - Шифрование PII (query_text, answer_text)
        
        Raises:
            ConsentRequiredError: Если пользователь не дал согласие
            ValueError: Если поля превышают максимальную длину
        """
        # ✅ CONSENT CHECK
        if not self.has_user_consent(user_id):
            log.error(f"❌ GDPR: Query logging blocked for user {user_id}")
            raise ConsentRequiredError(
                f"Query logging requires user consent for user {user_id}",
                user_id=user_id
            )
        
        # ✅ VALIDATION
        self._validate_field_length(query_text, MAX_QUERY_LENGTH, "query_text")
        self._validate_field_length(answer_text, MAX_ANSWER_LENGTH, "answer_text")
        self._validate_field_length(error_message, MAX_ERROR_LENGTH, "error_message")
        
        # ✅ ENCRYPTION (query может содержать PII)
        encrypted_query = self.secure_db.encrypt_field(query_text, 'query_text')
        encrypted_answer = self.secure_db.encrypt_field(answer_text, 'answer_text') if answer_text else None

        answer_length = len(answer_text) if answer_text else 0
        
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO queries (
                    user_id, query_text, query_type, article_num, point_num,
                    answer_text, answer_length, from_cache, processing_time_sec,
                    tokens_used, error_occurred, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, encrypted_query, query_type, article_num, point_num,
                encrypted_answer, answer_length, from_cache, processing_time_sec,
                tokens_used, error_occurred, error_message
            ))
            return cursor.lastrowid
    
    def get_user_queries(self, user_id: int, requesting_user_id: int, limit: int = 10) -> List[Dict]:
        """
        Получает последние запросы пользователя
        
        ✅ ИСПРАВЛЕНО: Добавлена проверка согласия и access control
        """
        # ✅ ACCESS CONTROL
        if user_id != requesting_user_id:
            log.warning(f"🚫 ACCESS DENIED: user {requesting_user_id} tried to view queries of user {user_id}")
            return []
        
        # ✅ CONSENT CHECK
        if not self.has_user_consent(user_id):
            raise ConsentRequiredError(
                f"Viewing queries requires user consent for user {user_id}",
                user_id=user_id
            )
        
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT id, timestamp, query_text, query_type, article_num, 
                       answer_length, from_cache, processing_time_sec
                FROM queries
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit)).fetchall()
            
            result = []
            for row in rows:
                query = dict(row)
                
                # Расшифровка
                try:
                    if query['query_text']:
                        query['query_text'] = self.secure_db.decrypt_field(query['query_text'], 'query_text')
                except DecryptionError as e:
                    log.error(f"Decryption failed for query {query['id']}: {e.message}", exc_info=False)
                    query['query_text'] = "[Decryption failed]"
                
                result.append(query)
            
            return result
    
    def get_popular_queries(self, limit: int = 10) -> List[Dict]:
        """
        Получает популярные запросы
        
        ⚠️ ВАЖНО: Не возвращает расшифрованные тексты (GDPR)
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT article_num, COUNT(*) as count
                FROM queries
                WHERE article_num IS NOT NULL
                GROUP BY article_num
                ORDER BY count DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(row) for row in rows]
    
    def get_query_statistics(self, days: int = 7) -> Dict:
        """Получает общую статистику запросов"""
        ALLOWED_PERIODS = {1: '-1 days', 7: '-7 days', 30: '-30 days', 90: '-90 days', 365: '-365 days'}
        
        if days not in ALLOWED_PERIODS:
            raise ValueError(f"Invalid period: {days}. Allowed: {sorted(ALLOWED_PERIODS.keys())}")
        
        time_filter = ALLOWED_PERIODS[days]

        with self._get_connection() as conn:
            stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(processing_time_sec) as avg_processing_time,
                    SUM(from_cache) as cache_hits,
                    SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END) as errors,
                    SUM(tokens_used) as total_tokens
                FROM queries
                WHERE timestamp >= datetime('now', ?)
            """, (time_filter,)).fetchone()

            type_distribution = conn.execute("""
                SELECT query_type, COUNT(*) as count
                FROM queries
                WHERE timestamp >= datetime('now', ?)
                GROUP BY query_type
            """, (time_filter,)).fetchall()

            return {
                **dict(stats),
                "type_distribution": {row["query_type"]: row["count"] for row in type_distribution},
                "period_days": days
            }
    
    # ================= CONSULTATIONS =================
    
    def create_consultation_request(
        self,
        user_id: int,
        preferred_date: str,
        preferred_time: str,
        contact_phone: Optional[str] = None,
        contact_email: Optional[str] = None,
        topic: Optional[str] = None,
        description: Optional[str] = None
    ) -> int:
        """
        Создает запрос на консультацию
        
        ✅ ИСПРАВЛЕНО: Добавлена проверка блокировки
        """
        # ✅ BLOCK CHECK
        if self.is_user_blocked(user_id):
            log.warning(f"🚫 BLOCKED USER: User {user_id} attempted to create consultation")
            raise PermissionError(f"User {user_id} is blocked")
        
        # ✅ CONSENT CHECK
        if not self.has_user_consent(user_id):
            log.error(f"❌ GDPR: Consultation creation blocked for user {user_id}")
            raise ConsentRequiredError(
                f"Consultation creation requires user consent for user {user_id}",
                user_id=user_id
            )
        
        # ✅ VALIDATION
        self._validate_field_length(topic, MAX_TOPIC_LENGTH, "topic")
        self._validate_field_length(description, MAX_DESCRIPTION_LENGTH, "description")
        
        # ✅ ENCRYPTION
        try:
            encrypted_phone = self.secure_db.encrypt_field(contact_phone, 'phone') if contact_phone else None
            encrypted_email = self.secure_db.encrypt_field(contact_email, 'email') if contact_email else None
            encrypted_desc = self.secure_db.encrypt_field(description, 'description') if description else None
        except EncryptionError as e:
            log.error(f"Encryption failed for user {user_id}: {e.message}", exc_info=False)
            raise
        except Exception as e:
            log.error(f"Unexpected encryption error for user {user_id}: {type(e).__name__}", exc_info=False)
            raise RuntimeError(f"Unexpected encryption error for user {user_id}") from e
        
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO consultations (
                    user_id, preferred_date, preferred_time,
                    contact_phone, contact_email, topic, description
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, preferred_date, preferred_time,
                encrypted_phone, encrypted_email, topic, encrypted_desc
            ))
            consultation_id = cursor.lastrowid
        
        log.info(f"✅ Consultation request created: #{consultation_id} for user {user_id}")
        return consultation_id

    
    def get_user_consultations(
        self, 
        user_id: int, 
        requesting_user_id: int, 
        user_role: str = "user"
    ) -> List[Dict]:
        """
        Получает консультации с разграничением доступа
        
        ✅ ИСПРАВЛЕНО: Использует централизованный rate limiting
        """
        # ✅ RATE-LIMITING
        rate_limit_key = f"consultations_read_{requesting_user_id}"
        if not self._check_rate_limit(rate_limit_key):
            log.warning(f"⚠️ RATE-LIMIT exceeded for user {requesting_user_id}")
            audit = self._get_audit_logger_sync()
            if audit:
                audit.log_access(
                    user_id=requesting_user_id,
                    action="RATE_LIMIT_EXCEEDED",
                    data_type="CONSULTATIONS_LIST",
                    details="User exceeded 10 requests/minute"
                )
            raise PermissionError("Rate limit exceeded: maximum 10 requests per minute")
        
        # ✅ ACCESS CONTROL
        if user_role != "admin" and user_id != requesting_user_id:
            log.warning(f"🚫 ACCESS DENIED: user {requesting_user_id} tried to view consultations of user {user_id}")
            audit = self._get_audit_logger_sync()
            if audit:
                audit.log_access(
                    user_id=requesting_user_id,
                    action="ACCESS_DENIED",
                    data_type="CONSULTATIONS_LIST",
                    details=f"Attempted to view consultations of user #{user_id}"
                )
            return []
        
        # ✅ FETCH & DECRYPT
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM consultations
                WHERE user_id = ?
                ORDER BY requested_at DESC
            """, (user_id,)).fetchall()
            
            result = []
            for row in rows:
                consultation = dict(row)
                
                try:
                    if consultation['contact_phone']:
                        consultation['contact_phone'] = self.secure_db.decrypt_field(
                            consultation['contact_phone'], 'phone'
                        )
                    if consultation['contact_email']:
                        consultation['contact_email'] = self.secure_db.decrypt_field(
                            consultation['contact_email'], 'email'
                        )
                    if consultation['description']:
                        consultation['description'] = self.secure_db.decrypt_field(
                            consultation['description'], 'description'
                        )
                except DecryptionError as e:
                    log.error(f"Decryption error for consultation {consultation['id']}: {e.message}", exc_info=False)
                except Exception as e:
                    log.error(f"Unexpected decryption error: {type(e).__name__}", exc_info=False)
                
                if user_role != "admin":
                    consultation.pop('lawyer_notes', None)
                
                result.append(consultation)
                
                audit = self._get_audit_logger_sync()
                if audit:
                    audit.log_access(
                        user_id=requesting_user_id,
                        action="READ",
                        data_type="CONSULTATION",
                        details=f"Consultation #{consultation['id']} (status: {consultation['status']})"
                    )
            
            return result
    
    def get_consultation_with_decryption(self, consultation_id: int, requesting_user_id: int) -> Optional[Dict]:
        """Получает консультацию с расшифровкой (только для владельца)"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM consultations WHERE id = ?", (consultation_id,)
            ).fetchone()
            
            if not row:
                return None
            
            consultation = dict(row)
            
            if consultation['user_id'] != requesting_user_id:
                log.warning(f"Unauthorized access attempt: user {requesting_user_id} to consultation {consultation_id}")
                audit = self._get_audit_logger_sync()
                if audit:
                    audit.log_access(
                        user_id=requesting_user_id,
                        action="UNAUTHORIZED_ACCESS_ATTEMPT",
                        data_type="CONSULTATION",
                        details=f"Attempted to access consultation #{consultation_id}"
                    )
                return None
            
            try:
                if consultation['contact_phone']:
                    consultation['contact_phone'] = self.secure_db.decrypt_field(
                        consultation['contact_phone'], 'phone'
                    )
                if consultation['contact_email']:
                    consultation['contact_email'] = self.secure_db.decrypt_field(
                        consultation['contact_email'], 'email'
                    )
                if consultation['description']:
                    consultation['description'] = self.secure_db.decrypt_field(
                        consultation['description'], 'description'
                    )
            except DecryptionError as e:
                log.error(f"Decryption failed for consultation #{consultation_id}: {e.message}", exc_info=False)
                consultation['contact_phone'] = "[Decryption failed]"
                consultation['contact_email'] = "[Decryption failed]"
                consultation['description'] = "[Decryption failed]"
            except Exception as e:
                log.error(f"Unexpected decryption error for consultation #{consultation_id}: {type(e).__name__}", exc_info=False)
                return None
            
            return consultation
    
    def get_pending_consultations(self, requesting_user_role: str) -> List[Dict]:
        """
        Получает все ожидающие консультации для админа
        
        ✅ ИСПРАВЛЕНО для Telegram:
        - Расшифровывает username, first_name, last_name
        - Админ видит читаемые данные
        - Обрабатывает ошибки расшифровки
        """
        if requesting_user_role != "admin":
            log.warning("🚫 ACCESS DENIED: Non-admin tried to view pending consultations")
            return []
        
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT c.*, u.username, u.first_name, u.last_name
                FROM consultations c
                JOIN users u ON c.user_id = u.user_id
                WHERE c.status = 'pending'
                ORDER BY c.requested_at ASC
            """).fetchall()
            
            result = []
            for row in rows:
                consultation = dict(row)
                
                # ✅ РАСШИФРОВКА PII для админа
                try:
                    if consultation.get('username'):
                        consultation['username'] = self.secure_db.decrypt_field(
                            consultation['username'], 'username'
                        )
                    if consultation.get('first_name'):
                        consultation['first_name'] = self.secure_db.decrypt_field(
                            consultation['first_name'], 'first_name'
                        )
                    if consultation.get('last_name'):
                        consultation['last_name'] = self.secure_db.decrypt_field(
                            consultation['last_name'], 'last_name'
                        )
                    
                    # ✅ Расшифровка контактных данных консультации
                    if consultation.get('contact_phone'):
                        consultation['contact_phone'] = self.secure_db.decrypt_field(
                            consultation['contact_phone'], 'phone'
                        )
                    if consultation.get('contact_email'):
                        consultation['contact_email'] = self.secure_db.decrypt_field(
                            consultation['contact_email'], 'email'
                        )
                    if consultation.get('description'):
                        consultation['description'] = self.secure_db.decrypt_field(
                            consultation['description'], 'description'
                        )
                        
                except DecryptionError as e:
                    log.error(
                        f"Decryption failed for consultation {consultation['id']}: {e.message}",
                        exc_info=False
                    )
                    # ✅ Fallback: показываем что расшифровка не удалась
                    consultation['username'] = "[Decryption failed]"
                    consultation['first_name'] = "[Decryption failed]"
                    consultation['last_name'] = "[Decryption failed]"
                    consultation['contact_phone'] = "[Decryption failed]"
                    consultation['contact_email'] = "[Decryption failed]"
                    consultation['description'] = "[Decryption failed]"
                
                except Exception as e:
                    log.error(
                        f"Unexpected decryption error for consultation {consultation['id']}: {type(e).__name__}",
                        exc_info=False
                    )
                    # ✅ Пропускаем эту консультацию при критической ошибке
                    continue
                
                result.append(consultation)
            
            return result

           

    def update_consultation_status(
        self,
        consultation_id: int,
        status: str,
        requesting_user_role: str,
        consultation_date: Optional[str] = None,
        lawyer_notes: Optional[str] = None
    ):
        """
        Обновляет статус консультации
        
        ✅ ИСПРАВЛЕНО: Добавлена валидация статусов
        """
        if requesting_user_role != "admin":
            log.warning("🚫 ACCESS DENIED: Non-admin tried to update consultation status")
            raise PermissionError("Only admins can update consultation status")
        
        # ✅ ВАЛИДАЦИЯ СТАТУСА
        if status not in ALLOWED_CONSULTATION_STATUSES:
            raise ValueError(
                f"Invalid consultation status: '{status}'. "
                f"Allowed statuses: {', '.join(sorted(ALLOWED_CONSULTATION_STATUSES))}"
            )
        
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE consultations
                SET status = ?, consultation_date = ?, lawyer_notes = ?
                WHERE id = ?
            """, (status, consultation_date, lawyer_notes, consultation_id))
        log.info(f"Consultation #{consultation_id} status updated to {status}")

    def cancel_consultation(self, consultation_id: int, user_id: int) -> bool:
        """Отменяет консультацию (только pending)"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                UPDATE consultations
                SET status = 'cancelled'
                WHERE id = ? AND user_id = ? AND status = 'pending'
            """, (consultation_id, user_id))
            success = cursor.rowcount > 0
        
        if success:
            log.info(f"Consultation #{consultation_id} cancelled by user {user_id}")
        return success
    
    # ================= FEEDBACK =================
    
    def add_feedback(
        self,
        user_id: int,
        rating: int,
        comment: Optional[str] = None,
        query_id: Optional[int] = None
    ):
        """
        Добавляет отзыв пользователя
        
        ✅ ИСПРАВЛЕНО: Добавлена проверка блокировки
        """
        # ✅ BLOCK CHECK
        if self.is_user_blocked(user_id):
            log.warning(f"🚫 BLOCKED USER: User {user_id} attempted to add feedback")
            raise PermissionError(f"User {user_id} is blocked")
        
        # ✅ CONSENT CHECK
        if not self.has_user_consent(user_id):
            log.error(f"❌ GDPR: Feedback blocked for user {user_id}")
            raise ConsentRequiredError(
                f"Adding feedback requires user consent for user {user_id}",
                user_id=user_id
            )
        
        # ✅ VALIDATION
        if rating < 1 or rating > 5:
            raise ValueError(f"Rating must be between 1 and 5, got {rating}")
        
        self._validate_field_length(comment, MAX_COMMENT_LENGTH, "comment")
        
        # ✅ ENCRYPTION
        encrypted_comment = self.secure_db.encrypt_field(comment, 'comment') if comment else None
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO feedback (user_id, query_id, rating, comment)
                VALUES (?, ?, ?, ?)
            """, (user_id, query_id, rating, encrypted_comment))
        log.info(f"Feedback added: user {user_id}, rating {rating}")

    
    def get_average_rating(self) -> float:
        """Получает средний рейтинг бота"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT AVG(rating) as avg_rating FROM feedback"
            ).fetchone()
            return round(row["avg_rating"], 2) if row and row["avg_rating"] else 0.0
    
    # ================= KEY ROTATION =================
    
    def rotate_encryption_keys(self, admin_user_id: int, admin_role: str) -> Dict:
        """
        Создаёт новый ключ шифрования (БЕЗ re-encryption старых данных)
        
        ⚠️ ОГРАНИЧЕНИЕ: Старые данные остаются зашифрованными старым ключом
        
        Новые данные будут использовать новый ключ.
        Для полной ротации требуется background re-encryption.
        
        Returns:
            dict со статистикой ротации
        """
        if admin_role != "admin":
            raise PermissionError("Only admins can rotate encryption keys")
        
        # Получаем текущую версию
        current_version = self.encryption.get_current_key_version()
        
        # ✅ СОЗДАЁМ НОВЫЙ КЛЮЧ
        try:
            new_key = self.encryption.generate_new_key()
            new_version = self.encryption.activate_new_key(new_key)
        except Exception as e:
            log.error(f"Failed to generate new encryption key: {e}")
            raise RuntimeError("Key rotation failed") from e
        
        # ✅ ЛОГИРУЕМ РОТАЦИЮ
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO encryption_key_audit (key_version, action, details, affected_records)
                VALUES (?, 'ROTATED', ?, 0)
            """, (new_version, f"Rotated by admin user {admin_user_id}. Old data NOT re-encrypted."))
        
        log.warning(
            f"⚠️ Encryption key rotated: v{current_version} → v{new_version}. "
            f"OLD DATA STILL ENCRYPTED WITH OLD KEY. Re-encryption required."
        )
        
        return {
            "old_version": current_version,
            "new_version": new_version,
            "status": "partial",
            "warning": "Old data NOT re-encrypted. Background job required for full rotation."
        }

    # ================= GDPR METHODS =================
    
    def export_my_data(self, user_id: int) -> dict:
        """Экспорт всех данных пользователя (GDPR право на доступ)"""
        with self._get_connection() as conn:
            gdpr = GDPRCompliance(conn)
            data = gdpr.export_user_data(user_id)

        audit = self._get_audit_logger_sync()
        if audit:
            audit.log_export(user_id=user_id, exported_by=user_id)
        
        log.info(
            f"User {user_id} exported their data: "
            f"{len(data.get('queries', []))} queries, "
            f"{len(data.get('consultations', []))} consultations, "
            f"{len(data.get('feedback', []))} feedback"
        )

        return data

    def delete_my_data(self, user_id: int, soft_delete: bool = False) -> dict:
        """Удаление всех данных пользователя (GDPR право на забвение)"""
        with self._get_connection() as conn:
            gdpr = GDPRCompliance(conn)
            stats = gdpr.delete_user_data(user_id, soft_delete=soft_delete)

        audit = self._get_audit_logger_sync()
        if audit:
            audit.log_deletion(user_id=user_id, deleted_by=user_id)

        if stats.get("errors"):
            log.error(f"GDPR deletion completed with errors for user {user_id}: {stats['errors']}")
        else:
            log.info(
                f"✅ User {user_id} data {'soft-deleted' if soft_delete else 'deleted'} per GDPR request. "
                f"Total records: {stats.get('total_deleted', 0)}"
            )

        return stats

    def anonymize_my_data(self, user_id: int) -> dict:
        """Анонимизация данных (альтернатива удалению)"""
        with self._get_connection() as conn:
            gdpr = GDPRCompliance(conn)
            stats = gdpr.anonymize_user_data(user_id)

        log.info(f"✅ User {user_id} data anonymized. Total records: {stats.get('total_anonymized', 0)}")
        return stats
    
    # ================= ANALYTICS =================
    
    def get_daily_activity(self, days: int = 30) -> List[Dict]:
        """Ежедневная статистика запросов"""
        ALLOWED_PERIODS = {7: '-7 days', 14: '-14 days', 30: '-30 days', 60: '-60 days', 
                          90: '-90 days', 180: '-180 days', 365: '-365 days'}
        
        if days not in ALLOWED_PERIODS:
            raise ValueError(f"Invalid period: {days}. Allowed: {sorted(ALLOWED_PERIODS.keys())}")
        
        time_filter = ALLOWED_PERIODS[days]
        
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as queries
                FROM queries
                WHERE timestamp >= datetime('now', ?)
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, (time_filter,)).fetchall()
            return [dict(row) for row in rows]
    
    def get_peak_hours(self) -> List[Dict]:
        """Определяет часы пиковой активности"""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT 
                    CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                    COUNT(*) as queries
                FROM queries
                WHERE timestamp >= datetime('now', '-30 days')
                GROUP BY hour
                ORDER BY queries DESC
            """).fetchall()
            return [dict(row) for row in rows]
    
    def export_to_csv(
        self, 
        table: str, 
        output_path: str, 
        requesting_user_role: str,
        decrypt_pii: bool = False
    ):
        """
        Экспортирует таблицу в CSV
        
        ⚠️ ВАЖНО: По умолчанию экспортирует ЗАШИФРОВАННЫЕ данные (encrypted blobs)
        
        Для GDPR export используйте export_my_data() вместо этого метода.
        
        Args:
            table: Имя таблицы
            output_path: Путь к файлу
            requesting_user_role: Роль (admin only)
            decrypt_pii: Если True, расшифровывает PII (МЕДЛЕННО, не рекомендуется для больших таблиц)
        
        ⚠️ SECURITY WARNING:
        - decrypt_pii=False: экспортирует зашифрованные blob'ы (нечитаемо)
        - decrypt_pii=True: расшифровывает (читаемо, но МЕДЛЕННО и опасно)
        """
        # ✅ ROLE CHECK
        if requesting_user_role != "admin":
            raise PermissionError("Only admins can export tables to CSV")
        
        # ✅ SQL INJECTION PROTECTION
        if table not in ALLOWED_TABLES:
            raise ValueError(f"Invalid table name: '{table}'. Allowed tables: {', '.join(sorted(ALLOWED_TABLES))}")

        # ✅ PATH TRAVERSAL PROTECTION
        allowed_dir = EXPORTS_DIR
        safe_path = Path(output_path).resolve()

        if not str(safe_path).startswith(str(allowed_dir)):
            log.warning(f"🚫 PATH TRAVERSAL ATTEMPT: '{output_path}'")
            audit = self._get_audit_logger_sync()
            if audit:
                audit.log_access(
                    user_id=0,
                    action="PATH_TRAVERSAL_ATTEMPT",
                    data_type="CSV_EXPORT",
                    details=f"Attempted export to '{output_path}'"
                )
            raise ValueError(f"Invalid export path. Export is only allowed in '{allowed_dir}'.")

        safe_path.parent.mkdir(parents=True, exist_ok=True)

        # ✅ WARNING при экспорте без расшифровки
        if not decrypt_pii:
            log.warning(
                f"⚠️ CSV EXPORT WARNING: Exporting '{table}' with ENCRYPTED data. "
                f"File will contain unreadable encrypted blobs. "
                f"Use decrypt_pii=True for readable export (not recommended for large tables)."
            )

        # ✅ EXPORT
        with self._get_connection() as conn:
            cursor = conn.execute(f"SELECT * FROM {table}")
            
            with open(safe_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                
                # Заголовки
                headers = [description[0] for description in cursor.description]
                
                # ✅ Добавляем префикс [ENCRYPTED] к зашифрованным колонкам
                if not decrypt_pii:
                    encrypted_fields = {
                        'users': ['username', 'first_name', 'last_name', 'notes'],
                        'queries': ['query_text', 'answer_text'],
                        'consultations': ['contact_phone', 'contact_email', 'description'],
                        'feedback': ['comment']
                    }
                    if table in encrypted_fields:
                        headers = [
                            f"[ENCRYPTED] {h}" if h in encrypted_fields[table] else h
                            for h in headers
                        ]
                
                writer.writerow(headers)
                
                # Данные
                if decrypt_pii and table in ['users', 'queries', 'consultations', 'feedback']:
                    # ⚠️ МЕДЛЕННАЯ РАСШИФРОВКА
                    log.warning(f"⚠️ Decrypting {table} for CSV export (this may take a while)...")
                    
                    rows = cursor.fetchall()
                    for row in rows:
                        decrypted_row = self._decrypt_row_for_export(table, dict(row))
                        writer.writerow(decrypted_row.values())
                else:
                    # Быстрый экспорт без расшифровки
                    writer.writerows(cursor)
        
        audit = self._get_audit_logger_sync()
        if audit:
            audit.log_access(
                user_id=0,
                action="CSV_EXPORT",
                data_type=table,
                details=f"Table '{table}' exported (decrypt_pii={decrypt_pii})"
            )

        log.info(f"✅ Exported {table} to {safe_path} (decrypt_pii={decrypt_pii})")


    def _decrypt_row_for_export(self, table: str, row: Dict) -> Dict:
        """
        Вспомогательный метод: расшифровывает строку для CSV export
        
        ⚠️ ВНУТРЕННИЙ МЕТОД: Используется только в export_to_csv
        """
        encrypted_fields_map = {
            'users': {
                'username': 'username',
                'first_name': 'first_name',
                'last_name': 'last_name',
                'notes': 'notes'
            },
            'queries': {
                'query_text': 'query_text',
                'answer_text': 'answer_text'
            },
            'consultations': {
                'contact_phone': 'phone',
                'contact_email': 'email',
                'description': 'description'
            },
            'feedback': {
                'comment': 'comment'
            }
        }
        
        if table not in encrypted_fields_map:
            return row
        
        for field, field_type in encrypted_fields_map[table].items():
            if field in row and row[field]:
                try:
                    row[field] = self.secure_db.decrypt_field(row[field], field_type)
                except Exception as e:
                    row[field] = f"[DECRYPTION_FAILED: {type(e).__name__}]"
        
        return row



# ================= SINGLETON FUNCTIONS =================
# ✅ ИСПРАВЛЕНО: Глобальные Lock'и создаются один раз

def get_db() -> LegalBotDB:
    """
    Получить singleton экземпляр БД (thread-safe)
    
    ✅ ИСПРАВЛЕНО: Использует глобальный _db_lock
    """
    global _db_instance

    if _db_instance is not None:
        return _db_instance

    with _db_lock:  # ✅ Используем глобальный Lock
        if _db_instance is None:
            _db_instance = LegalBotDB()
            log.info("✅ Database singleton initialized (thread-safe)")
    
    return _db_instance


async def get_db_async() -> LegalBotDB:
    """
    Получить singleton экземпляр БД (async-safe)
    
    ✅ ИСПРАВЛЕНО: Использует глобальный asyncio.Lock
    """
    global _db_instance, _db_async_lock

    if _db_instance is not None:
        return _db_instance

    # Ленивая инициализация asyncio Lock
    if _db_async_lock is None:
        import asyncio
        _db_async_lock = asyncio.Lock()

    async with _db_async_lock:  # ✅ Используем глобальный async Lock
        if _db_instance is None:
            _db_instance = LegalBotDB()
            log.info("✅ Database singleton initialized (async-safe)")
    
    return _db_instance
