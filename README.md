````markdown
# ⚖️ Legal Consultation Bot

### RAG-powered Telegram Bot for Legal Information

AI Telegram-бот для информационных консультаций по
Гражданскому кодексу Российской Федерации.

Система использует Retrieval-Augmented Generation (RAG):
перед генерацией ответа выполняется поиск релевантных
статей в собственной базе знаний.

Вместо того чтобы полагаться только на знания LLM,
система сначала находит подходящий контекст, а затем
передаёт его языковой модели для формирования ответа.

> ⚠️ Проект предназначен исключительно для информационных
> и образовательных целей и не заменяет консультацию
> профессионального юриста.

---

# 🎯 Problem

При использовании LLM напрямую юридический вопрос
передаётся модели без гарантии, что ответ будет основан
на конкретных положениях используемой нормативной базы.

Например:

```text
Пользователь
     ↓
     LLM
     ↓
Ответ
````

Такой подход может приводить к неточным или
неподтверждённым ответам.

---

# 💡 Solution

В проекте используется RAG-подход:

```text
Пользователь
     ↓
Telegram Bot
     ↓
Поиск в базе знаний
     ↓
Релевантные статьи ГК РФ
     ↓
Формирование контекста
     ↓
LLM
     ↓
Ответ пользователю
```

Таким образом, генерация ответа происходит с учётом
найденного нормативного контекста.

---

# ✨ Key Features

* 📖 поиск по Гражданскому кодексу РФ;
* 🔎 гибридный поиск;
* 🧠 Vector Search;
* 🔤 BM25 full-text search;
* 📌 прямой поиск по номеру статьи;
* 🤖 генерация ответа через OpenAI API;
* 💬 Telegram-интерфейс;
* 📚 локальная база знаний;
* ⚡ быстрый локальный запуск.

---

# 🔎 Hybrid Retrieval

Одна из ключевых особенностей проекта —
использование нескольких механизмов поиска.

```text
                       User Query
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
      Article Search      BM25      Vector Search
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    Candidate Documents
                           ↓
                    Context Selection
                           ↓
                         LLM
                           ↓
                      Final Answer
```

### 1. Direct Article Search

Используется для запросов, содержащих номер статьи.

Например:

```text
статья 196 ГК РФ
```

Такой запрос целесообразно обрабатывать напрямую,
без зависимости от семантического поиска.

### 2. BM25

Полнотекстовый поиск хорошо подходит для точных
терминов и формулировок.

Например:

```text
срок исковой давности
```

### 3. Vector Search

Семантический поиск позволяет находить документы,
которые близки к запросу по смыслу, даже если
формулировки отличаются.

Например:

```text
Через сколько лет можно взыскать долг?
```

может быть сопоставлен с материалами,
содержащими терминологию об исковой давности.

---

# 🧠 RAG Pipeline

После получения пользовательского вопроса
система выполняет последовательность операций.

```text
Question
   ↓
Query Processing
   ↓
┌─────────────────────────────┐
│ Article Search              │
│ BM25 Search                 │
│ Vector Search               │
└─────────────────────────────┘
   ↓
Retrieved Documents
   ↓
Context Construction
   ↓
LLM Prompt
   ↓
Generated Answer
```

Найденные документы используются как контекст
для генерации ответа.

RAG не гарантирует отсутствие ошибок LLM,
но позволяет дополнить генерацию конкретными
документами из используемой базы знаний.

---

# 📚 Knowledge Base

В качестве базы знаний используется:

```text
Гражданский кодекс Российской Федерации
```

Исходный документ:

```text
knowledge_base/
└── gk_rf.txt
```

Перед запуском данные необходимо проиндексировать.

```bash
python ingest_data.py
```

Во время индексации:

```text
Source Document
      ↓
Text Processing
      ↓
Logical Chunks
      ↓
Embeddings
      ↓
ChromaDB
```

После индексации векторная база готова
к поиску.

---

# 🏗️ Architecture

Общая схема работы:

```text
┌──────────────────────┐
│       User           │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    Telegram Bot      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Query Processing     │
└──────────┬───────────┘
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
  Article BM25 Vector
  Search  Search Search
     │     │     │
     └─────┼─────┘
           ▼
┌──────────────────────┐
│ Context Construction │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│     OpenAI LLM       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Telegram Answer    │
└──────────────────────┘
```

---

# 🧠 Prompt Engineering

Prompt Engineering используется на этапе
формирования запроса к LLM.

В prompt передаются:

* пользовательский вопрос;
* найденный контекст;
* инструкции по формированию ответа.

Цель — заставить модель использовать найденные
материалы как основной источник контекста,
а не отвечать только на основе собственных знаний.

Prompt-related логика проекта вынесена в:

```text
prompts.py
```

---

# 🗄️ Why ChromaDB?

Для MVP была выбрана **ChromaDB**.

Основные причины:

* простой локальный запуск;
* не требует отдельного сервера;
* хранение embeddings и metadata;
* удобная интеграция с Python;
* подходит для небольших локальных RAG-систем.

Для дальнейшего production-развития возможна замена
на специализированные решения, например:

```text
Qdrant
pgvector
Milvus
```

При этом замена vector store не меняет саму
концепцию retrieval pipeline.

---

# 🔤 Why BM25 + Vector Search?

Vector Search хорошо работает с семантическим
сходством.

Но некоторые запросы требуют точного
совпадения терминов или идентификаторов.

Например:

```text
статья 196
```

или:

```text
срок исковой давности
```

Поэтому используется комбинация:

```text
BM25
   +
Vector Search
   +
Direct Article Search
```

Это позволяет сочетать:

* точное совпадение;
* полнотекстовый поиск;
* семантический поиск.

---

# 🛠️ Tech Stack

| Technology             | Purpose             |
| ---------------------- | ------------------- |
| Python                 | Backend             |
| aiogram                | Telegram Bot        |
| OpenAI API             | LLM generation      |
| ChromaDB               | Vector database     |
| BM25                   | Full-text retrieval |
| text-embedding-3-small | Embeddings          |
| structlog              | Logging             |
| python-dotenv          | Configuration       |

---

# 📁 Project Structure

```text
bot_for_legal_consultations/
│
├── main.py                # Application entry point
├── bot.py                 # Telegram bot
├── database.py            # ChromaDB and search
├── ingest_data.py         # Knowledge base indexing
├── prompts.py             # LLM prompts
├── config.py              # Configuration
│
├── knowledge_base/
│   └── gk_rf.txt
│
├── chroma_db/             # Vector database
│
└── requirements.txt
```

---

# 🚀 Local Setup

## 1. Clone

```bash
git clone https://github.com/Alexsey111/bot_for_legal_consultations.git
cd bot_for_legal_consultations
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure environment

Create `.env`:

```env
OPENAI_API_KEY=...
BOT_TOKEN=...
```

## 4. Prepare knowledge base

Place the source document here:

```text
knowledge_base/
└── gk_rf.txt
```

## 5. Build the index

```bash
python ingest_data.py
```

## 6. Start the bot

```bash
python main.py
```

---

# 💬 Example

Example query:

```text
Какой срок исковой давности?
```

Processing:

```text
User Question
      ↓
Retrieval
      ↓
Relevant Articles
      ↓
Context
      ↓
OpenAI
      ↓
Answer
```

The generated response is based on
the retrieved legal context.

---

# 🔮 Possible Improvements

Possible directions for further development:

* поддержка нескольких кодексов РФ;
* расширение базы нормативных документов;
* reranking retrieved documents;
* автоматическая оценка качества retrieval;
* evaluation dataset для RAG;
* web interface;
* голосовые сообщения;
* streaming responses;
* переход с ChromaDB на Qdrant или pgvector;
* расширенная история диалогов.

---

# 🎯 What This Project Demonstrates

Проект демонстрирует практический опыт разработки
LLM-приложений:

* 🧠 проектирование RAG-систем;
* 🔎 hybrid retrieval;
* 📚 работа с vector databases;
* 🔤 BM25 full-text search;
* 🧩 работа с embeddings;
* 🤖 интеграция LLM API;
* ✍️ Prompt Engineering;
* 💬 разработка Telegram AI-ботов;
* 📄 индексация документов;
* 🐍 Python backend development.

---

# ⚠️ Disclaimer

Проект создан в образовательных и демонстрационных целях.

Ответы системы не являются юридической консультацией,
юридическим заключением или заменой профессиональной
помощи квалифицированного специалиста.

Пользователь самостоятельно принимает решения,
основанные на полученной информации.

---

# 👨‍💻 Author

**Alexsey**

AI Developer · Prompt Engineer · AI Automation · Vibe Coder

GitHub: [@Alexsey111](https://github.com/Alexsey111)

````
