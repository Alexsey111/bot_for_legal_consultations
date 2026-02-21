"""Проверка структуры базы данных"""
import sqlite3
from pathlib import Path

DB_PATHS = ["./legal_bot.db", "./data/legal_bot.db"]

for db_path in DB_PATHS:
    if not Path(db_path).exists():
        print(f"[SKIP] {db_path} - not found")
        continue
    
    print(f"\n[INFO] Database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Получаем список таблиц
    cursor.execute('SELECT name FROM sqlite_master WHERE type="table";')
    tables = [row[0] for row in cursor.fetchall()]
    
    if not tables:
        print("  [INFO] No tables found")
    else:
        print(f"  [INFO] Tables: {', '.join(tables)}")
        
        # Проверяем user_stats
        if "user_stats" in tables:
            print("\n  [INFO] Table user_stats structure:")
            cursor.execute("PRAGMA table_info(user_stats);")
            for row in cursor.fetchall():
                print(f"    - {row[1]} ({row[2]})")
        else:
            print("  [INFO] Table user_stats not found")
    
    conn.close()
