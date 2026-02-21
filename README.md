# Legal Consultation Bot

Telegram бот для юридических консультаций по Гражданскому кодексу РФ.

## Установка

1. Клонируйте репозиторий
2. Создайте виртуальное окружение: `python -m venv .venv`
3. Активируйте: `.venv\Scripts\activate` (Windows) или `source .venv/bin/activate` (Linux/Mac)
4. Установите зависимости: `pip install -r requirements.txt`
5. Скопируйте `.env.example` в `.env` и заполните секреты
6. Запустите Redis: `docker run -d -p 6379:6379 redis`
7. Запустите бота: `python main.py`

## ВАЖНО - Безопасность

**НЕ коммитьте:**
- `.env` файлы
- `encryption_key.key`
- `*.db` файлы
- Логи с персональными данными
