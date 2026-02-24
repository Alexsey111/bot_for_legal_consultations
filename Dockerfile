FROM python:3.12-slim

WORKDIR /app

# Минимальные системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Используем облегчённые requirements
COPY requirements-light.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data

ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=2

CMD ["python", "-u", "main.py"]
