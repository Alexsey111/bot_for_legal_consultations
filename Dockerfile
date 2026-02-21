# Telegram-бот для юридических консультаций по ГК РФ
FROM python:3.11-slim

WORKDIR /app

# Системные зависимости для sentence-transformers / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Данные и БД монтируются через volumes в docker-compose
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "main.py"]
