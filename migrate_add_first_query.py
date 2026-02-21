"""Скрипт для добавления колонки first_query в таблицу user_stats"""
import sqlite3
from pathlib import Path

DB_PATH = "./legal_bot.db"

def migrate():
    """Выполняет миграцию базы данных"""
    if not Path(DB_PATH).exists():
        print(f"[ERROR] Database not found: {DB_PATH}")
        return False
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Проверяем, существует ли колонка first_query
        cursor.execute("PRAGMA table_info(user_stats);")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "first_query" in columns:
            print("[OK] Column first_query already exists in table user_stats")
        else:
            # Добавляем колонку
            print("[INFO] Adding column first_query...")
            cursor.execute("ALTER TABLE user_stats ADD COLUMN first_query TIMESTAMP;")
            conn.commit()
            print("[OK] Column first_query added successfully")
        
        # Показываем структуру таблицы
        print("\n[INFO] Table user_stats structure:")
        cursor.execute("PRAGMA table_info(user_stats);")
        for row in cursor.fetchall():
            print(f"  - {row[1]} ({row[2]})")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Migration error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
