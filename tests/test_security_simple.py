"""
Тесты для модуля security.py

Проверяем реальные классы и функции из модуля
"""

import pytest
import os
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from unittest.mock import patch
from security import EncryptionManager, DataAnonymizer


class TestEncryptionManager:
    """Тесты для EncryptionManager"""
    
    @pytest.fixture
    def encryption_manager(self):
        """Создаёт менеджер шифрования в development mode"""
        with patch.dict(os.environ, {"PRODUCTION": "false"}):
            return EncryptionManager()
    
    def test_encrypt_decrypt_success(self, encryption_manager):
        """Тест успешного шифрования и расшифровки"""
        original_data = "Тестовые данные для шифрования"
        
        encrypted = encryption_manager.encrypt(original_data)
        decrypted = encryption_manager.decrypt(encrypted)
        
        assert decrypted == original_data
    
    def test_encrypt_empty_string(self, encryption_manager):
        """Тест шифрования пустой строки"""
        assert encryption_manager.encrypt("") == ""
    
    def test_decrypt_empty_string(self, encryption_manager):
        """Тест расшифровки пустой строки"""
        assert encryption_manager.decrypt("") == ""
    
    def test_decrypt_invalid_data(self, encryption_manager):
        """Тест расшифровки невалидных данных"""
        with pytest.raises(Exception):  # DecryptionError
            encryption_manager.decrypt("invalid:encrypted_data")
    
    def test_hash_data(self, encryption_manager):
        """Тест хеширования данных"""
        data = "test@example.com"
        
        hash1 = encryption_manager.hash_data(data)
        hash2 = encryption_manager.hash_data(data)
        
        assert hash1 == hash2
        assert hash1 != data


class TestDataAnonymizer:
    """Тесты для DataAnonymizer"""
    
    def test_anonymize_user_id(self):
        """Тест анонимизации user_id"""
        user_id = 12345
        
        anonymized = DataAnonymizer.anonymize_user_id(user_id)
        
        assert anonymized != str(user_id)
        assert len(anonymized) == 16
    
    def test_anonymize_user_id_consistent(self):
        """Тест что анонимизация одного ID всегда даёт одинаковый результат"""
        user_id = 12345
        
        anonymized1 = DataAnonymizer.anonymize_user_id(user_id)
        anonymized2 = DataAnonymizer.anonymize_user_id(user_id)
        
        assert anonymized1 == anonymized2
    
    def test_mask_phone(self):
        """Тест маскирования телефона"""
        phone = "+79001234567"
        
        masked = DataAnonymizer.mask_phone(phone)
        
        assert masked == "+7900123****"
    
    def test_mask_phone_short(self):
        """Тест маскирования короткого телефона"""
        short_phone = "123"
        
        masked = DataAnonymizer.mask_phone(short_phone)
        
        assert masked == short_phone
    
    def test_mask_phone_none(self):
        """Тест маскирования None телефона"""
        assert DataAnonymizer.mask_phone(None) is None
    
    def test_mask_email(self):
        """Тест маскирования email"""
        email = "alice.smith@example.com"
        
        masked = DataAnonymizer.mask_email(email)
        
        assert masked == "a***@example.com"
    
    def test_mask_email_short_local(self):
        """Тест маскирования email с короткой локальной частью"""
        email = "a@example.com"
        
        masked = DataAnonymizer.mask_email(email)
        
        assert masked == "a***@example.com"
    
    def test_mask_email_invalid(self):
        """Тест маскирования невалидного email"""
        invalid_email = "not_an_email"
        
        masked = DataAnonymizer.mask_email(invalid_email)
        
        assert masked == invalid_email
    
    def test_mask_email_none(self):
        """Тест маскирования None email"""
        assert DataAnonymizer.mask_email(None) is None
    
    def test_truncate_text_short(self):
        """Тест обрезки короткого текста"""
        short_text = "Короткий текст"
        
        truncated = DataAnonymizer.truncate_text(short_text, max_length=100)
        
        assert truncated == short_text
    
    def test_truncate_text_long(self):
        """Тест обрезки длинного текста"""
        long_text = "А" * 200
        
        truncated = DataAnonymizer.truncate_text(long_text, max_length=100)
        
        assert len(truncated) == 103  # 100 + "..."
        assert truncated.endswith("...")
