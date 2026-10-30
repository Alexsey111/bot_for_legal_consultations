"""
security.py
Система безопасности и шифрования персональных данных

АРХИТЕКТУРА ШИФРОВАНИЯ:
========================

ВАЖНО: В данном проекте используется FIELD-LEVEL ENCRYPTION (шифрование на уровне полей),
а не полная прозрачная шифрование базы данных (как SQLCipher).

Что шифруется:
- consultations.contact_phone - телефон для связи
- consultations.contact_email - email для связи
- consultations.description - описание проблемы

Что НЕ шифруется:
- Все остальные поля (username, query_text, answer_text и т.д.)
- Структура базы данных
- Индексы

Безопасность на диске:
- Файл БД (.db) НЕ зашифрован
- Злоумышленник с доступом к файлу БД сможет прочитать незащищенные данные
- Только персональные данные в consultations защищены шифрованием

Для полной прозрачной шифрации БД на диске рассмотрите SQLCipher:
https://www.zetetic.net/sqlcipher/
"""

import os
import hashlib
import sqlite3
import json
import hmac
import time
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from dotenv import load_dotenv

import structlog

log = structlog.get_logger()

load_dotenv()

logger = logging.getLogger(__name__)

# ================= CUSTOM EXCEPTIONS =================

class EncryptionError(Exception):
    """
    Кастомное исключение для ошибок шифрования

    SECURITY:
    - Позволяет отличить ошибки шифрования от других исключений
    - Caller может явно обрабатывать ошибки шифрования
    - Содержит безопасное сообщение об ошибке (без PII)

    Пример использования:
        try:
            encrypted = encryption_manager.encrypt(data)
        except EncryptionError as e:
            logger.error(f"Encryption failed: {e}")
            # Обработка ошибки шифрования
    """

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        """
        Args:
            message: Безопасное сообщение об ошибке (без PII)
            original_exception: Оригинальное исключение (для отладки)
        """
        self.message = message
        self.original_exception = original_exception
        super().__init__(self.message)


class DecryptionError(Exception):
    """
    Кастомное исключение для ошибок расшифровки

    SECURITY:
    - Позволяет отличить ошибки расшифровки от других исключений
    - Caller может явно обрабатывать ошибки расшифровки
    - Содержит безопасное сообщение об ошибке (без PII)

    Пример использования:
        try:
            decrypted = encryption_manager.decrypt(encrypted_data)
        except DecryptionError as e:
            logger.error(f"Decryption failed: {e}")
            # Обработка ошибки расшифровки
    """

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        """
        Args:
            message: Безопасное сообщение об ошибке (без PII)
            original_exception: Оригинальное исключение (для отладки)
        """
        self.message = message
        self.original_exception = original_exception
        super().__init__(self.message)


# ================= ENCRYPTION KEY MANAGEMENT =================

class EncryptionManager:
    """
    Управление шифрованием данных с поддержкой ротации ключей
    
    АРХИТЕКТУРА KEY VERSIONING:
    ============================
    Каждый зашифрованный блок содержит:
    - Версия ключа (1 байт)
    - Зашифрованные данные (Fernet формат)
    
    Формат: b'[KEY_ID]:[ENCRYPTED_DATA]'
    Пример: b'1:gAAAAABl...'
    
    COMPLIANCE:
    - GDPR Article 32: Encryption key rotation
    - 152-ФЗ: Ротация ключей шифрования
    - Audit trail: Логирование ротаций
    - Brute-force protection: Tracking failed decryption attempts
    """
    
    # Brute-force protection: track failed decryption attempts
    _failed_decrypts = {}  # {timestamp: count} для rate limiting
    _max_failed_attempts = 10  # Failed attempts per minute
    _lockout_duration_sec = 60
    
    def __init__(self):
        self.keys_config_file = Path("./data/.encryption_keys.json")
        self.keys = self._load_or_create_keys()
        self.current_key_version = self._get_current_key_version()
        # MultiFernet позволяет расшифровать с любым из старых ключей
        self.multi_cipher = MultiFernet([Fernet(k) for k in self.keys])
        log.info("encryption_initialized", keys_count=len(self.keys), current_version=self.current_key_version)

    
    def _load_or_create_keys(self) -> list:
        """Загружает или создает ключи с версионированием"""
        is_production = os.getenv("PRODUCTION", "").lower() in ("true", "1", "yes")
        
        # Приоритет 1: Переменная окружения (для production)
        env_key = os.getenv("ENCRYPTION_KEY")
        if env_key:
            log.info("encryption_key_from_environment")
            # В env может быть один ключ или JSON с массивом ключей
            try:
                keys_data = json.loads(env_key)
                if isinstance(keys_data, list):
                    return [k.encode() if isinstance(k, str) else k for k in keys_data]
                elif isinstance(keys_data, dict):
                    return [keys_data.get("current_key", env_key).encode()]
            except (json.JSONDecodeError, ValueError):
                return [env_key.encode()]
        
        # Приоритет 2: Файл конфигурации с ключами
        if self.keys_config_file.exists():
            try:
                with open(self.keys_config_file, 'r') as f:
                    keys_config = json.load(f)
                keys_list = [k.encode() if isinstance(k, str) else k for k in keys_config.get("keys", [])]
                if keys_list:
                    log.info("encryption_keys_loaded_from_config", count=len(keys_list))
                    return keys_list
            except (json.JSONDecodeError, IOError) as e:
                log.warning("keys_config_load_failed", error=str(e)[:100])
        
        # Приоритет 3: Старый файл с одним ключом (.encryption_key)
        legacy_key_file = Path("./data/.encryption_key")
        if legacy_key_file.exists():
            with open(legacy_key_file, 'rb') as f:
                log.info("encryption_key_from_legacy_file")
                return [f.read()]
        
        # Production - требуем явное задание ключей
        if is_production:
            error_msg = (
                "🔴 CRITICAL: No encryption keys found in production!\n"
                "Set ENCRYPTION_KEY environment variable with key(s) before starting.\n"
                "This prevents catastrophic data loss from key regeneration.\n"
                "For key rotation, set ENCRYPTION_KEY with JSON: "
                '{"keys": ["current_key_base64", "old_key_base64"]}'
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Development - автогенерируем новый ключ
        logger.warning("⚠️  DEVELOPMENT MODE: Auto-generating new encryption key")
        logger.warning("⚠️  For production, export and set ENCRYPTION_KEY environment variable!")
        
        new_key = Fernet.generate_key()
        
        # Сохраняем в JSON конфиг с versioning info
        self.keys_config_file.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "keys": [new_key.decode()],
            "current_key_version": 1,
            "created_at": str(datetime.now()),
            "description": "Encryption keys with versioning support"
        }
        
        with open(self.keys_config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Защищаем файл
        os.chmod(self.keys_config_file, 0o600)
        
        logger.warning("🔐 IMPORTANT: Backup your encryption keys!")
        logger.warning(f"Keys location: {self.keys_config_file.absolute()}")
        logger.warning(
            "For production deployment, export current key:\n"
            f"  export ENCRYPTION_KEY=$(cat {self.keys_config_file})\n"
            "Or individual key:\n"
            f"  export ENCRYPTION_KEY=$(jq -r '.keys[0]' {self.keys_config_file})"
        )

        return [new_key]
    
    def _get_current_key_version(self) -> int:
        """Получает версию текущего ключа из конфига"""
        if self.keys_config_file.exists():
            try:
                with open(self.keys_config_file, 'r') as f:
                    config = json.load(f)
                    return config.get("current_key_version", 1)
            except (json.JSONDecodeError, IOError):
                pass
        return 1
    
    def _sanitize_error_for_logging(self, error: Exception, sensitive_data: Optional[str] = None) -> str:
        """
        Санитизация ошибки для безопасного логирования

        SECURITY:
        - НЕ логирует сообщение об ошибке (может содержать PII)
        - Логирует только тип исключения
        - Заменяет чувствительные данные на [REDACTED]

        Args:
            error: Исключение
            sensitive_data: Данные которые нужно скрыть из сообщения

        Returns:
            Безопасное строковое представление ошибки
        """
        # НЕ логируем str(error) - может содержать зашифрованные данные!
        # Некоторые криптографические библиотеки включают данные в сообщение об ошибке

        error_type = type(error).__name__

        # Если есть чувствительные данные, не показываем их
        if sensitive_data:
            # Показываем только первые 10 символов для отладки (если они не чувствительны)
            preview = sensitive_data[:10] if len(sensitive_data) > 10 else sensitive_data
            return f"{error_type} (data preview: {preview}...)"


        return error_type

    def encrypt(self, data: str, retry_count: int = 3) -> str:
        """
        Шифрует данные с retry механизмом

        SECURITY:
        - Использует кастомное EncryptionError для явной обработки ошибок
        - Retry с exponential backoff для transient errors
        - Safe logging (без PII)

        Args:
            data: Данные для шифрования
            retry_count: Количество попыток (default: 3)

        Returns:
            Зашифрованные данные в формате '1:gAAAAABl...'

        Raises:
            EncryptionError: Если шифрование не удалось после всех попыток
        """
        # DEF-006: контракт для пустой строки. Шифровать нечего — возвращаем
        # пустую строку, как и decrypt(""). Иначе в базе оказывается шифротекст
        # от пустого значения, и отличить «поля нет» от «поле пустое» нельзя.
        if data is None or data == "":
            return ""

        for attempt in range(retry_count):
            try:
                encrypted = Fernet(self.keys[0]).encrypt(data.encode())
                return f"{self.current_key_version}:{encrypted.decode()}"
            except Exception as e:
                if attempt == retry_count - 1:
                    # Санитизация ошибки перед логированием
                    safe_error = self._sanitize_error_for_logging(e, data)
                    logger.error(f"Encryption failed after {retry_count} attempts: {safe_error}")
                    # Прокидываем кастомное исключение
                    raise EncryptionError(
                        f"Encryption failed after {retry_count} attempts",
                        original_exception=e
                    ) from e
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff

    def decrypt(self, encrypted_data: str) -> str:
        """
        Расшифровывает данные с поддержкой старых ключей и brute-force protection

        SECURITY:
        - Отслеживает failed decryption attempts
        - Если > 10 попыток за минуту → блокирует дальнейшие попытки
        - Логирует попытки взлома для audit trail
        - Санитизация ошибок для предотвращения утечки PII в логи
        - Использует кастомное DecryptionError для явной обработки ошибок

        Ожидает формат: '1:gAAAAABlZrx...'
        MultiFernet автоматически пробует все ключи

        Args:
            encrypted_data: Зашифрованные данные

        Returns:
            Расшифрованные данные

        Raises:
            DecryptionError: Если расшифровка не удалась
            RuntimeError: Если превышен лимит попыток (brute-force protection)
        """
        if not encrypted_data:
            return encrypted_data

        # BRUTE-FORCE PROTECTION: Check failed attempts
        now = __import__('time').time()
        recent_attempts = [ts for ts in self._failed_decrypts.keys()
                          if now - ts < self._lockout_duration_sec]

        if len(recent_attempts) >= self._max_failed_attempts:
            error_msg = f"🔒 SECURITY: Too many failed decryption attempts. Locked for {self._lockout_duration_sec}s"
            logger.warning(error_msg)
            raise RuntimeError(error_msg)

        try:
            # Парсим версию ключа
            if ':' in encrypted_data:
                key_version_str, encrypted_payload = encrypted_data.split(':', 1)
                try:
                    key_version = int(key_version_str)
                    log.debug("decrypting_with_key_version", version=key_version)
                except ValueError:
                    # Старый формат без версии, используем первый ключ
                    encrypted_payload = encrypted_data
                    key_version = self.current_key_version
            else:
                # Старый формат без версии
                encrypted_payload = encrypted_data
                key_version = self.current_key_version

            # MultiFernet пробует все ключи в порядке до первого успеха
            decrypted = self.multi_cipher.decrypt(encrypted_payload.encode())

            # Очищаем failed attempts на успешной расшифровке
            self._failed_decrypts.clear()
            return decrypted.decode()
        except Exception as e:
            # BRUTE-FORCE PROTECTION: Track failed attempt
            self._failed_decrypts[now] = self._failed_decrypts.get(now, 0) + 1
            failed_count = len(recent_attempts)

            # ⚠️ SECURITY: Санитизация ошибки перед логированием
            # НЕ логируем str(e) - может содержать зашифрованные данные!
            safe_error = self._sanitize_error_for_logging(e, encrypted_data)

            logger.warning(
                f"Failed decryption attempt ({failed_count}/{self._max_failed_attempts}). "
                f"Error: {safe_error}",
                exc_info=False  # ← CRITICAL: БЕЗ stack trace!
            )

            # Прокидываем кастомное исключение
            raise DecryptionError(
                f"Decryption failed (attempt {failed_count}/{self._max_failed_attempts})",
                original_exception=e
            ) from e
    
    def rotate_key(self) -> dict:
        """
        Ротирует ключ шифрования
        
        Возвращает:
            dict с информацией о ротации (old_version, new_version, timestamp)
        
        ВАЖНО: После ротации нужно:
        1. Экспортировать новый ключ: export ENCRYPTION_KEY=...
        2. Переобновить все encrypted данные через re_encrypt_all_data()
        3. Проверить что все данные читаются корректно
        """
        new_key = Fernet.generate_key()
        old_version = self.current_key_version
        new_version = old_version + 1
        
        # Добавляем новый ключ в начало списка (текущий)
        self.keys.insert(0, new_key)
        
        # Сохраняем конфиг
        config = {
            "keys": [k.decode() if isinstance(k, bytes) else k for k in self.keys],
            "current_key_version": new_version,
            "rotated_at": str(datetime.now()),
            "old_version": old_version,
            "description": "Encryption keys with versioning support"
        }
        
        self.keys_config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.keys_config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        os.chmod(self.keys_config_file, 0o600)
        
        # Обновляем инстанс
        self.current_key_version = new_version
        self.multi_cipher = MultiFernet([Fernet(k) for k in self.keys])
        
        rotation_info = {
            "old_version": old_version,
            "new_version": new_version,
            "timestamp": config["rotated_at"],
            "total_keys_stored": len(self.keys),
            "message": f"Key rotated from v{old_version} to v{new_version}. "
                      f"Re-encrypt data by calling re_encrypt_all_data()."
        }
        
        logger.warning(f"🔑 KEY ROTATION: {rotation_info['message']}")
        logger.warning(f"Export new key: export ENCRYPTION_KEY=$(jq -r '.keys[0]' {self.keys_config_file})")
        
        return rotation_info
    
    def re_encrypt_all_data(self, db_connection, tables_fields: dict) -> dict:
        """
        Re-шифрует все данные новым ключом
        
        Args:
            db_connection: Соединение с БД
            tables_fields: {'table_name': ['field1', 'field2']}
            Пример: {'consultations': ['contact_phone', 'contact_email', 'description']}
        
        Returns:
            dict с статистикой re-encryption
        """
        encryption = EncryptionManager()
        cursor = db_connection.cursor()
        stats = {
            "total_records": 0,
            "re_encrypted_records": 0,
            "errors": []
        }
        
        for table_name, fields in tables_fields.items():
            for field_name in fields:
                try:
                    # Получаем все encrypted данные
                    cursor.execute(f"SELECT id, {field_name} FROM {table_name} WHERE {field_name} IS NOT NULL")
                    rows = cursor.fetchall()
                    stats["total_records"] += len(rows)
                    
                    # Re-шифруем с новым ключом
                    for row_id, encrypted_value in rows:
                        try:
                            # Расшифровываем старым ключом
                            decrypted = self.decrypt(encrypted_value)
                            # Шифруем новым ключом
                            re_encrypted = encryption.encrypt(decrypted)
                            # Обновляем в БД
                            cursor.execute(
                                f"UPDATE {table_name} SET {field_name} = ? WHERE id = ?",
                                (re_encrypted, row_id)
                            )
                            stats["re_encrypted_records"] += 1
                        except Exception as e:
                            stats["errors"].append(f"Failed to re-encrypt {table_name}.{field_name}[id={row_id}]: {e}")
                            logger.error(f"Re-encryption error: {e}", exc_info=False)
                    
                except Exception as e:
                    stats["errors"].append(f"Failed to process {table_name}.{field_name}: {e}")
                    logger.error(f"Table processing error: {e}", exc_info=False)
        
        db_connection.commit()
        logger.info(f"Re-encryption complete: {stats['re_encrypted_records']}/{stats['total_records']} records updated")
        return stats
    
    def hash_data(self, data: str) -> str:
        """
        Создает безопасный хеш для PII (с HMAC)
        
        SECURITY: Использует HMAC-SHA256 с secret key, не простой SHA256
        
        Почему не unsalted SHA256:
        - Unsalted SHA256 можно перебрать словарём
        - email/phone имеют ограниченное пространство значений
        - Можно построить rainbow table за < 1 часа
        
        HMAC (Hash-based Message Authentication Code):
        - Требует знание secret key
        - Защищает от dictionary/rainbow table атак
        - GDPR compliant
        
        Использование:
        - Хеш email для поиска без расшифровки
        - Хеш phone для DLP (data loss prevention) checks
        - Индексы для быстрого поиска зашифрованных данных
        
        ВАЖНО: Secret key производится из encryption key, поэтому:
        - Тот же email всегда даст тот же хеш
        - Без knowledge of encryption key нельзя построить радугу таблицу
        """
        if not data:
            return data
        
        # Используем первый ключ шифрования как secret key для HMAC
        # Это гарантирует что хеш совпадает при использовании того же ключа
        secret_key = self.keys[0]
        
        # HMAC-SHA256 с secret key
        return hmac.new(
            secret_key,
            data.encode(),
            hashlib.sha256
        ).hexdigest()


# ================= DATA ANONYMIZATION =================

class DataAnonymizer:
    """Анонимизация персональных данных"""
    
    @staticmethod
    def anonymize_user_id(user_id: int) -> str:
        """Создает анонимный идентификатор"""
        return hashlib.sha256(str(user_id).encode()).hexdigest()[:16]
    
    @staticmethod
    def mask_phone(phone: str) -> str:
        """Маскирует телефон: +7 (XXX) XXX-12-34"""
        if not phone or len(phone) < 4:
            return phone
        return phone[:-4] + "****"
    
    @staticmethod
    def mask_email(email: str) -> str:
        """
        Маскирует email: показывает только первый символ
        
        SECURITY: Защита от информационной утечки при маскировании
        
        Старая логика:
        - "ab@x.com" → "a*b@x.com" (reveals first + last letter + full domain)
        - "alice.smith@example.com" → "a***h@example.com" (reveals pattern)
        
        Новая логика:
        - Любой email → "a***@example.com" (only first letter visible)
        - Скрывает длину local part, последний символ, структуру
        - Domain остаётся видим (нужен для отправки писем)
        
        COMPLIANCE:
        - GDPR Article 5: Minimization of personal data in logs
        - Sufficient for logging/debugging (see who accessed)
        - Insufficient for reconstruction (защита от pattern analysis)
        """
        if not email or '@' not in email:
            return email
        
        local, domain = email.split('@', 1)
        
        if not local:
            return email
        
        # Показываем только первый символ + три звёздочки
        # Это минимизирует информацию но сохраняет читаемость
        masked_local = local[0] + '***'
        return f"{masked_local}@{domain}"
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100) -> str:
        """Обрезает длинный текст для логов"""
        if not text or len(text) <= max_length:
            return text
        return text[:max_length] + "..."


# ================= GDPR COMPLIANCE =================

class GDPRCompliance:
    """Соответствие GDPR/152-ФЗ"""
    
    def __init__(self, db_connection):
        self.conn = db_connection
        self.encryption = EncryptionManager()
    
    def export_user_data(self, user_id: int) -> dict:
        """Экспорт данных пользователя (GDPR)"""
        cursor = self.conn.cursor()
        
        # Информация о пользователе
        user = cursor.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        
        user_dict = dict(user) if user else {}
        
        # ✅ ИСПОЛЬЗУЙТЕ FieldLevelEncryptionWrapper для расшифровки
        from sql_logger import get_db
        db = get_db()
        
        # Расшифровываем PII
        if user_dict.get('username'):
            try:
                user_dict['username'] = db.secure_db.decrypt_field(user_dict['username'], 'username')
            except Exception as e:
                logger.warning(f"Failed to decrypt username: {e}")
                user_dict['username'] = None
        
        if user_dict.get('first_name'):
            try:
                user_dict['first_name'] = db.secure_db.decrypt_field(user_dict['first_name'], 'first_name')
            except Exception as e:
                logger.warning(f"Failed to decrypt first_name: {e}")
                user_dict['first_name'] = None
        
        if user_dict.get('last_name'):
            try:
                user_dict['last_name'] = db.secure_db.decrypt_field(user_dict['last_name'], 'last_name')
            except Exception as e:
                logger.warning(f"Failed to decrypt last_name: {e}")
                user_dict['last_name'] = None
        
        if user_dict.get('phone'):
            try:
                user_dict['phone'] = db.secure_db.decrypt_field(user_dict['phone'], 'phone')
            except Exception as e:
                logger.warning(f"Failed to decrypt phone: {e}")
                user_dict['phone'] = None
        
        # Запросы
        queries_raw = cursor.execute(
            "SELECT timestamp, query_text, query_type, answer_text FROM queries WHERE user_id = ?",
            (user_id,)
        ).fetchall()
        
        queries = []
        for q in queries_raw:
            q_dict = dict(q)
            
            if q_dict.get('query_text'):
                try:
                    q_dict['query_text'] = db.secure_db.decrypt_field(q_dict['query_text'], 'query_text')
                except:
                    q_dict['query_text'] = None
            
            if q_dict.get('answer_text'):
                try:
                    q_dict['answer_text'] = db.secure_db.decrypt_field(q_dict['answer_text'], 'answer_text')
                except:
                    q_dict['answer_text'] = None
            
            queries.append(q_dict)
        
        # Консультации
        consultations_raw = cursor.execute(
            "SELECT * FROM consultations WHERE user_id = ?", (user_id,)
        ).fetchall()
        
        consultations_decrypted = []
        for consultation in consultations_raw:
            consultation_dict = dict(consultation)
            
            if consultation_dict.get('contact_phone'):
                try:
                    consultation_dict['contact_phone'] = db.secure_db.decrypt_field(
                        consultation_dict['contact_phone'], 'phone'
                    )
                except:
                    consultation_dict['contact_phone'] = None

            if consultation_dict.get('contact_email'):
                try:
                    consultation_dict['contact_email'] = db.secure_db.decrypt_field(
                        consultation_dict['contact_email'], 'email'
                    )
                except:
                    consultation_dict['contact_email'] = None

            if consultation_dict.get('description'):
                try:
                    consultation_dict['description'] = db.secure_db.decrypt_field(
                        consultation_dict['description'], 'description'
                    )
                except:
                    consultation_dict['description'] = None
            
            consultations_decrypted.append(consultation_dict)
        
        # Отзывы
        feedback_raw = cursor.execute(
            "SELECT timestamp, rating, comment FROM feedback WHERE user_id = ?",
            (user_id,)
        ).fetchall()
        
        feedback = []
        for f in feedback_raw:
            f_dict = dict(f)
            if f_dict.get('comment'):
                try:
                    f_dict['comment'] = db.secure_db.decrypt_field(f_dict['comment'], 'comment')
                except:
                    f_dict['comment'] = None
            feedback.append(f_dict)
        
        logger.info(
            f"GDPR: User {user_id} data exported "
            f"({len(consultations_decrypted)} consultations)"
        )
        return {
            "user": user_dict,
            "queries": queries,
            "consultations": consultations_decrypted,
            "feedback": feedback,
            "export_timestamp": str(datetime.now()),
            "gdpr_article": "Article 20 (Right to data portability)"
        }

    
    def anonymize_user_data(self, user_id: int) -> dict:
        """
        Анонимизация данных пользователя (вместо удаления)

        Returns:
            dict со статистикой анонимизации

        ИСПРАВЛЕНО: Возвращает статистику, использует _soft_delete_user_data
        """
        cursor = self.conn.cursor()
        return self._soft_delete_user_data(user_id, cursor)
        
    def delete_user_data(self, user_id: int, soft_delete: bool = False) -> dict:
        """
        Полное удаление данных пользователя (право на забвение)

        GDPR COMPLIANCE (Article 17 - Right to Erasure):
        - Удаляет все персональные данные пользователя из БД
        - Анонимизирует audit logs (для сохранения audit trail)
        - Требует дополнительного удаления Redis сессий (caller responsibility)

        Args:
            user_id: ID пользователя для удаления
            soft_delete: Если True, делает мягкое удаление (анонимизация)

        Returns:
            dict со статистикой удаления

        ИСПРАВЛЕНО:
        - Проверка что foreign_keys включен (иначе orphan записи)
        - Детальное логирование удалений из каждой таблицы
        - Проверка успешности удаления
        - Опция soft-delete для альтернативы полному удалению
        - Анонимизация audit logs с user_id (для сохранения audit trail)

        ЗАМЕТКА:
        - Redis сессии должны быть удалены отдельно (redis_manager.delete_session)
        - Prometheus metrics должны быть очищены отдельно (через API Prometheus)
        """
        cursor = self.conn.cursor()
        
        # ИСПРАВЛЕНО: Проверка что foreign_keys включен
        fk_status = cursor.execute("PRAGMA foreign_keys;").fetchone()[0]
        if fk_status != 1:
            logger.error(
                f"CRITICAL: Cannot delete user {user_id} - foreign_keys is OFF! "
                "Orphan records will remain."
            )
            raise RuntimeError(
                "foreign_keys is OFF. Data deletion will leave orphan records. "
                "Enable PRAGMA foreign_keys = ON before deletion."
            )

        # ИСПРАВЛЕНО: Soft-delete опция
        if soft_delete:
            return self._soft_delete_user_data(user_id, cursor)

        # Статистика удаления
        deletion_stats = {
            "user_id": user_id,
            "deleted_records": {},
            "total_deleted": 0,
            "errors": []
        }

        try:
            # Порядок важен: сначала дочерние таблицы, потом родительская

            # 1. Удаляем feedback
            cursor.execute("DELETE FROM feedback WHERE user_id = ?", (user_id,))
            feedback_deleted = cursor.rowcount
            deletion_stats["deleted_records"]["feedback"] = feedback_deleted
            logger.info(f"GDPR: Deleted {feedback_deleted} feedback records for user {user_id}")

            # 2. Удаляем consultations (зависит от users)
            cursor.execute("DELETE FROM consultations WHERE user_id = ?", (user_id,))
            consultations_deleted = cursor.rowcount
            deletion_stats["deleted_records"]["consultations"] = consultations_deleted
            logger.info(f"GDPR: Deleted {consultations_deleted} consultation records for user {user_id}")

            # 3. Удаляем queries (зависит от users)
            cursor.execute("DELETE FROM queries WHERE user_id = ?", (user_id,))
            queries_deleted = cursor.rowcount
            deletion_stats["deleted_records"]["queries"] = queries_deleted
            logger.info(f"GDPR: Deleted {queries_deleted} query records for user {user_id}")

            # 4. Удаляем пользователя (родительская таблица)
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            users_deleted = cursor.rowcount
            deletion_stats["deleted_records"]["users"] = users_deleted

            if users_deleted == 0:
                error_msg = f"User {user_id} not found or already deleted"
                deletion_stats["errors"].append(error_msg)
                logger.warning(f"GDPR: {error_msg}")
            else:
                logger.info(f"GDPR: Deleted user record {user_id}")

            # ================= GDPR: АНОНИМИЗАЦИЯ AUDIT LOGS =================
            # GDPR требует сохранения audit trail, но user_id должен быть анонимизирован
            # Заменяем все вхождения user_id на [DELETED_USER] в audit logs
            try:
                user_id_str = str(user_id)
                replacement = "[DELETED_USER]"

                # Анонимизируем audit logs в encryption_key_audit (если таблица существует)
                try:
                    cursor.execute("""
                        UPDATE encryption_key_audit
                        SET details = REPLACE(details, ?, ?)
                        WHERE details LIKE ?
                    """, (user_id_str, replacement, f"%{user_id}%"))
                    audit_anonymized = cursor.rowcount
                    deletion_stats["deleted_records"]["audit_logs_anonymized"] = audit_anonymized
                    logger.info(f"GDPR: Anonymized {audit_anonymized} audit log entries for user {user_id}")
                except Exception as e:
                    # Таблица может не существовать - это нормально
                    logger.debug(f"Audit log anonymization skipped: {e}")

                # Анонимизируем любые другие таблицы с user_id в текстовых полях
                # Белый список таблиц для защиты от SQL injection
                ANONYMIZATION_TABLES = ["queries", "consultations"]

                for table in ANONYMIZATION_TABLES:
                    # ✅ Дополнительная проверка белого списка
                    if table not in ANONYMIZATION_TABLES:
                        log.error("invalid_anonymization_table", table=table)
                        continue

                    try:
                        cursor.execute(f"""
                            UPDATE {table}
                            SET error_message = REPLACE(error_message, ?, ?)
                            WHERE error_message LIKE ?
                        """, (user_id_str, replacement, f"%{user_id}%"))
                    except Exception:
                        pass  # Таблица может не иметь error_message

            except Exception as e:
                error_msg = f"Failed to anonymize audit logs for user {user_id}: {e}"
                deletion_stats["errors"].append(error_msg)
                logger.warning(f"GDPR: {error_msg}")

            # Подсчитываем общее количество
            deletion_stats["total_deleted"] = sum(deletion_stats["deleted_records"].values())

            self.conn.commit()
            logger.info(
                f"GDPR: User {user_id} data deleted successfully. "
                f"Total records: {deletion_stats['total_deleted']}"
            )

            # ================= GDPR: REMINDER =================
            logger.warning(
                f"⚠️ GDPR: Remember to delete Redis session for user {user_id}: "
                f"await redis_manager.delete_session({user_id})"
            )
            logger.warning(
                f"⚠️ GDPR: Remember to clean Prometheus metrics: "
                f"metrics may still contain user_id labels"
            )

            return deletion_stats

        except Exception as e:
            self.conn.rollback()
            error_msg = f"Failed to delete user {user_id}: {e}"
            deletion_stats["errors"].append(error_msg)
            logger.error(f"GDPR: {error_msg}", exc_info=True)
            raise

    def _soft_delete_user_data(self, user_id: int, cursor) -> dict:
        """
        Мягкое удаление данных (анонимизация вместо удаления)
        
        GDPR COMPLIANCE (Article 17 - Right to Erasure):
        Полная анонимизация для re-identification prevention
        
        Args:
            user_id: ID пользователя
            cursor: Курсор БД

        Returns:
            dict со статистикой анонимизации
        """
        anonymization_stats = {
            "user_id": user_id,
            "anonymized_records": {},
            "total_anonymized": 0
        }

        # 1. Анонимизируем пользователя (ВСЕ идентифицирующие поля)
        # ВАЖНО: Обнуляем ВСЕ поля которые могут привести к re-identification
        cursor.execute("""
            UPDATE users
            SET username = NULL,
                first_name = '[Anonymized]',
                last_name = '[Anonymized]',
                language_code = NULL,
                consent_given = 0,
                consent_date = NULL,
                total_queries = 0,
                is_blocked = 0,
                notes = 'Anonymized per GDPR Article 17 on ' || datetime('now')
            WHERE user_id = ?
        """, (user_id,))
        users_anonymized = cursor.rowcount
        anonymization_stats["anonymized_records"]["users"] = users_anonymized

        # 2. Анонимизируем консультации (шифруем = удаляем PII)
        cursor.execute("""
            UPDATE consultations
            SET contact_phone = NULL,
                contact_email = NULL,
                description = '[Anonymized]',
                topic = '[Anonymized]',
                lawyer_notes = NULL
            WHERE user_id = ?
        """, (user_id,))
        consultations_anonymized = cursor.rowcount
        anonymization_stats["anonymized_records"]["consultations"] = consultations_anonymized

        # 3. Анонимизируем запросы (удаляем текст запроса и ответа)
        cursor.execute("""
            UPDATE queries
            SET query_text = '[Anonymized]',
                answer_text = '[Anonymized]',
                error_message = NULL
            WHERE user_id = ?
        """, (user_id,))
        queries_anonymized = cursor.rowcount
        anonymization_stats["anonymized_records"]["queries"] = queries_anonymized

        # 4. Удаляем feedback (личные отзывы не нужны после анонимизации)
        cursor.execute("DELETE FROM feedback WHERE user_id = ?", (user_id,))
        feedback_deleted = cursor.rowcount
        anonymization_stats["anonymized_records"]["feedback"] = feedback_deleted

        anonymization_stats["total_anonymized"] = sum(
            anonymization_stats["anonymized_records"].values()
        )

        self.conn.commit()
        logger.info(
            f"GDPR: User {user_id} data anonymized (soft-deleted). "
            f"Total records: {anonymization_stats['total_anonymized']}"
        )
    
        return anonymization_stats
    
    def auto_cleanup_old_data(self, days: int = 365):
        """
        Автоматическая очистка старых данных
        
        SECURITY: Использует parameterized queries (защита от SQL injection)
        
        Args:
            days: Количество дней для хранения данных (default: 365)
        
        Удаляет:
        - Запросы старше N дней (query_text = NULL)
        - Завершенные консультации старше N дней
        """
        cursor = self.conn.cursor()
        
        # Параметр для datetime: '-365 days'
        days_param = f"-{days} days"
        
        # Удаляем старые запросы (оставляем только статистику)
        cursor.execute("""
            DELETE FROM queries 
            WHERE timestamp < datetime('now', ?)
            AND query_text IS NOT NULL
        """, (days_param,))
        
        queries_deleted = cursor.rowcount
        logger.info(f"Auto-cleanup: Deleted {queries_deleted} old query records")
        
        # Удаляем завершенные консультации старше года
        cursor.execute("""
            DELETE FROM consultations
            WHERE status = 'completed'
            AND requested_at < datetime('now', ?)
        """, (days_param,))
        
        consultations_deleted = cursor.rowcount
        logger.info(f"Auto-cleanup: Deleted {consultations_deleted} completed consultation records")
        
        total_deleted = queries_deleted + consultations_deleted
        self.conn.commit()
        
        logger.info(f"Auto-cleanup: Total {total_deleted} old records removed")
        return total_deleted


# ================= FIELD-LEVEL ENCRYPTION WRAPPER =================

class FieldLevelEncryptionWrapper:
    """
    Wrapper для field-level encryption отдельных PII полей
    
    ⚠️  АРХИТЕКТУРА: FIELD-LEVEL ENCRYPTION ТОЛЬКО
    
    ЧТО ЗАЩИЩЕНО:
    - consultations.contact_phone (Fernet encryption)
    - consultations.contact_email (Fernet encryption)
    - consultations.description (Fernet encryption)
    
    ЧТО НЕ ЗАЩИЩЕНО:
    - Файл БД на диске (NOT encrypted) - используется обычный sqlite3
    - Все остальные поля (query_text, answer_text, username...)
    - Структура БД, индексы, логи
    - Данные в памяти процесса (в RAM незащищено)
    - Данные при передаче (требуется TLS/HTTPS отдельно)
    
    ОГРАНИЧЕНИЯ:
    - Нельзя искать по зашифрованным полям без расшифровки
    - Нельзя создавать индексы по зашифрованным полям
    - Если процесс скомпрометирован, ключ украден из памяти
    - Не соответствует требованиям полной прозрачной шифрации
    
    КОГДА ИСПОЛЬЗОВАТЬ:
    ✓ Защита PII на диске от casual access
    ✓ Compliance требование для шифрования sensitive data
    ✓ Защита от theft бэкапов БД файла
    
    КОГДА НЕ ИСПОЛЬЗОВАТЬ:
    ✗ Требуется полная file-level encryption - используйте SQLCipher
    ✗ Требуется защита данных в памяти - нет solution
    ✗ Требуется поиск по encrypted данным - невозможно
    ✗ Требуется full-disk encryption - используйте ОС уровень
    
    ДЛЯ ПОЛНОЙ ШИФРАЦИИ БД:
    https://www.zetetic.net/sqlcipher/
    
    Args:
        db_path: Путь к файлу базы данных (будет обычный sqlite3, не зашифрован)
    """
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.encryption = EncryptionManager()
        self.anonymizer = DataAnonymizer()
        
        # Создаем защищенную директорию
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Устанавливаем права доступа
        if self.db_path.exists():
            os.chmod(self.db_path, 0o600)
    
    def encrypt_field(self, value: Any, field_type: str) -> str:
        """
        Шифрует поле в зависимости от типа

        SECURITY:
        - Safe logging - никаких чувствительных данных в логах
        - Прокидывает EncryptionError для явной обработки ошибок

        Args:
            value: Значение для шифрования
            field_type: Тип поля (phone, email, description, notes)

        Returns:
            Зашифрованное значение или исходное (если не требует шифрования)

        Raises:
            EncryptionError: Если шифрование не удалось
        """
        if value is None:
            return None

        str_value = str(value)

        # Поля, требующие шифрования (PII и чувствительные тексты)
        encrypted_field_types = [
            'phone', 'email', 'description', 'notes',
            'username', 'first_name', 'last_name',
            'query_text', 'answer_text', 'comment',
        ]
        if field_type in encrypted_field_types:
            try:
                return self.encryption.encrypt(str_value)
            except Exception as e:
                # ⚠️ SECURITY: Санитизация ошибки перед логированием
                safe_error = self.encryption._sanitize_error_for_logging(e, str_value)
                logger.error(
                    f"Failed to encrypt field '{field_type}': {safe_error}",
                    exc_info=False  # ← CRITICAL: БЕЗ stack trace!
                )
                # Прокидываем кастомное исключение
                raise EncryptionError(
                    f"Failed to encrypt field '{field_type}'",
                    original_exception=e
                ) from e

        return str_value

    def decrypt_field(self, encrypted_value: str, field_type: str) -> str:
        """
        Расшифровывает поле

        SECURITY:
        - Safe logging - никаких чувствительных данных в логах
        - Прокидывает DecryptionError для явной обработки ошибок

        Args:
            encrypted_value: Зашифрованное значение
            field_type: Тип поля (phone, email, description, notes)

        Returns:
            Расшифрованное значение или исходное (если не требует расшифровки)

        Raises:
            DecryptionError: Если расшифровка не удалась
        """
        if encrypted_value is None:
            return None

        encrypted_field_types = [
            'phone', 'email', 'description', 'notes',
            'username', 'first_name', 'last_name',
            'query_text', 'answer_text', 'comment',
        ]
        if field_type in encrypted_field_types:
            try:
                return self.encryption.decrypt(encrypted_value)
            except Exception as e:
                # ⚠️ SECURITY: Санитизация ошибки перед логированием
                safe_error = self.encryption._sanitize_error_for_logging(e, encrypted_value)
                logger.error(
                    f"Failed to decrypt field '{field_type}': {safe_error}",
                    exc_info=False  # ← CRITICAL: БЕЗ stack trace!
                )
                # Прокидываем кастомное исключение
                raise DecryptionError(
                    f"Failed to decrypt field '{field_type}'",
                    original_exception=e
                ) from e

        return encrypted_value
    
    def mask_for_logs(self, value: str, field_type: str) -> str:
        """Маскирует данные для логов"""
        if not value:
            return value
        
        if field_type == 'phone':
            return self.anonymizer.mask_phone(value)
        elif field_type == 'email':
            return self.anonymizer.mask_email(value)
        elif field_type == 'description':
            return self.anonymizer.truncate_text(value, 50)
        
        return value


# ================= AUDIT LOG =================

class AuditLogger:
    """
    Логирование доступа к персональным данным
    
    ✅ ИСПРАВЛЕНО:
    - Полностью async-safe singleton
    - Убраны threading.Lock (вызывали deadlock в async)
    - Ленивая инициализация через async factory method
    """
    
    _instance = None
    _lock = None  # ✅ ИСПРАВЛЕНО: Ленивая инициализация
    
    @classmethod
    async def get_instance(cls, log_file: str = "./data/audit.log"):
        """
        Async-safe singleton initialization
        
        ✅ ИСПРАВЛЕНО:
        - Используется asyncio.Lock вместо threading.Lock
        - Ленивая инициализация lock
        - Полностью async-safe
        
        Использование:
            audit_logger = await AuditLogger.get_instance()
            audit_logger.log_access(...)
        """
        # FAST PATH: уже инициализирован
        if cls._instance is not None:
            return cls._instance
        
        # SLOW PATH: первый вызов
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        
        async with cls._lock:
            # Double-checked locking
            if cls._instance is None:
                cls._instance = super(AuditLogger, cls).__new__(cls)
                cls._instance._init_logger(log_file)
                log.info("audit_logger_initialized", log_file=log_file)
        
        return cls._instance
    
    def _init_logger(self, log_file: str):
        """
        ✅ ИСПРАВЛЕНО: Приватный метод инициализации (вызывается ОДИН раз)
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Настраиваем отдельный логгер
        self.logger = logging.getLogger('audit')
        
        # Проверяем что handlers ещё не добавлены
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            logger.debug(f"✅ AuditLogger handler added")
        
        # SECURITY: Защищаем audit.log файл
        try:
            os.chmod(self.log_file, 0o600)
            logger.debug(f"✅ Audit log permissions set to 0o600")
        except (OSError, PermissionError) as e:
            logger.warning(f"⚠️  Could not set audit log permissions: {e}", exc_info=False)
    
    def log_access(self, user_id: int, action: str, data_type: str, details: str = ""):
        """Логирует доступ к персональным данным"""
        self.logger.info(
            f"USER={user_id} | ACTION={action} | TYPE={data_type} | DETAILS={details}"
        )

    def log_export(self, user_id: int, exported_by: int):
        """Логирует экспорт данных"""
        self.log_access(
            user_id=user_id,
            action="EXPORT",
            data_type="ALL",
            details=f"Exported by user {exported_by}"
        )
    
    def log_deletion(self, user_id: int, deleted_by: int):
        """Логирует удаление данных"""
        self.log_access(
            user_id=user_id,
            action="DELETE",
            data_type="ALL",
            details=f"Deleted by user {deleted_by}"
        )
    
    def anonymize_user_logs(self, user_id: int):
        """
        ✅ НОВОЕ: Анонимизирует логи пользователя (GDPR compliance)
        
        Заменяет все вхождения user_id на [DELETED_USER] в audit.log
        Сохраняет audit trail но удаляет PII
        """
        if not self.log_file.exists():
            return
        
        try:
            # Читаем файл
            with open(self.log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Заменяем user_id
            user_id_str = f"USER={user_id}"
            replacement = "USER=[DELETED_USER]"
            
            anonymized_content = content.replace(user_id_str, replacement)
            
            # Подсчитываем количество замен
            replacements = content.count(user_id_str)
            
            if replacements > 0:
                # Записываем обратно
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    f.write(anonymized_content)
                
                logger.info(
                    f"✅ Anonymized {replacements} audit log entries for user {user_id}"
                )
                
                self.log_access(
                    user_id=0,  # System
                    action="ANONYMIZE_LOGS",
                    data_type="AUDIT_LOG",
                    details=f"Anonymized {replacements} entries for user [DELETED_USER]"
                )
            else:
                logger.info(f"No audit log entries found for user {user_id}")
        
        except Exception as e:
            logger.error(f"Failed to anonymize audit logs for user {user_id}: {e}")