# Деплой бота на Render

Пошаговая инструкция по развёртыванию Telegram-бота для юридических консультаций на [Render](https://dashboard.render.com).

## Что нужно заранее

- Аккаунт на [Render](https://dashboard.render.com) (можно через GitHub).
- Репозиторий проекта на **GitHub** (или GitLab/Bitbucket).
- **TELEGRAM_TOKEN** и **OPENAI_API_KEY** — те же значения, что у вас в локальном файле `.env`.

---

## Где ввести TELEGRAM_TOKEN и OPENAI_API_KEY (у меня они в .env)

Файл **`.env`** используется только у вас на компьютере. На Render переменные окружения задаются **в веб-интерфейсе** — туда нужно вставить те же значения, что в `.env`.

**Откуда взять значения:** откройте свой `.env` и скопируйте строки после `=`:
- `TELEGRAM_TOKEN=123456:ABC...` → значение `123456:ABC...`
- `OPENAI_API_KEY=sk-...` → значение `sk-...`

**Куда вставить на Render:**

- **Если создаёте сервис через Blueprint** (New → Blueprint):
  1. После выбора репозитория Render покажет экран «Blueprint spec» со списком переменных.
  2. У переменных **TELEGRAM_TOKEN** и **OPENAI_API_KEY** будет кнопка или поле **Enter value** / **Add secret** (или иконка замка).
  3. Вставьте туда значения из `.env` (без слова `TELEGRAM_TOKEN=` и кавычек — только сам токен/ключ).

- **Если создаёте Background Worker вручную** (New → Background Worker):
  1. На странице создания найдите блок **Environment** (Environment Variables).
  2. Нажмите **Add Environment Variable**.
  3. Key: `TELEGRAM_TOKEN`, Value: вставьте токен из `.env`. Добавьте ещё одну переменную: Key: `OPENAI_API_KEY`, Value: ключ из `.env`.

- **Если сервис уже создан:**
  1. В [Dashboard](https://dashboard.render.com) откройте свой сервис (воркер бота).
  2. Слева выберите **Environment**.
  3. **Add Environment Variable** → Key: `TELEGRAM_TOKEN`, Value: из `.env`. Повторите для `OPENAI_API_KEY`.

Значения хранятся зашифрованно и не попадают в репозиторий.

---

## Вариант 1: Деплой через Blueprint (рекомендуется)

1. Откройте [Render Dashboard](https://dashboard.render.com) и войдите в аккаунт.

2. Нажмите **New** → **Blueprint**.

3. Подключите репозиторий:
   - Выберите **GitHub** (или другой хостинг).
   - Укажите репозиторий `bot_for_legal_consultations`.
   - Render подхватит `render.yaml` из корня репозитория.

4. На экране настройки Blueprint введите значения для переменных с замком (см. раздел выше):
   - **TELEGRAM_TOKEN** — скопируйте из вашего `.env`.
   - **OPENAI_API_KEY** — скопируйте из вашего `.env`.
   - **ENCRYPTION_KEY** можно сгенерировать автоматически (уже настроено в Blueprint).

5. Нажмите **Apply** и дождитесь первого деплоя.

6. После деплоя в настройках сервиса добавьте **REDIS_URL** (если нужны сессии и rate-limit):
   - **New** → **Redis** → создайте Redis (например, free plan).
   - В сервисе бота: **Environment** → **Add Environment Variable**:
     - Key: `REDIS_URL`
     - Value: скопируйте **Internal Connection String** из созданного Redis.

---

## Вариант 2: Ручное создание сервиса

1. [Dashboard](https://dashboard.render.com) → **New** → **Background Worker**.

2. Подключите репозиторий (GitHub/GitLab/Bitbucket) и выберите репозиторий проекта.

3. Настройки:
   - **Name**: `legal-consultation-bot` (или любое).
   - **Runtime**: **Python 3**.
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`
   - **Plan**: Free или Starter — по необходимости.

4. **Environment** — добавьте переменные:

   | Key              | Описание                    | Обязательно |
   |------------------|-----------------------------|-------------|
   | TELEGRAM_TOKEN   | Токен Telegram-бота         | Да          |
   | OPENAI_API_KEY   | Ключ OpenAI API             | Да          |
   | ENCRYPTION_KEY   | Ключ шифрования (32+ симв.) | Для продакшена |
   | PRODUCTION       | `true` для продакшена       | Рекомендуется |
   | REDIS_URL        | URL Redis (если создали)    | Нет (по умолчанию локальный) |
   | DATABASE_URL     | По умолчанию SQLite в `./data` | Нет     |
   | CHROMA_PERSIST_DIR | Путь к Chroma (по умолч. `./chroma_legal_db`) | Нет |

5. Сохраните и нажмите **Create Background Worker**. Дождитесь окончания деплоя.

---

## Постоянные данные (SQLite и Chroma)

На бесплатном плане диск эфемерный: при перезапуске сервиса данные SQLite и векторной БД Chroma теряются.

Чтобы сохранять данные между деплоями:

1. В Render перейдите в сервис бота → **Settings** → **Disks**.
2. Добавьте диск, например:
   - **Name**: `legal-bot-data`
   - **Mount Path**: `/data`
   - **Size**: 1 GB (достаточно для старта).
3. В **Environment** задайте:
   - `DATABASE_URL` = `sqlite+aiosqlite:////data/legal_bot.db`
   - `CHROMA_PERSIST_DIR` = `/data/chroma_legal_db`
4. После первого деплоя с диском нужно один раз заполнить векторную БД (см. раздел «Первичная загрузка данных» ниже).

---

## Первичная загрузка данных (Chroma)

Локально или в одноразовом запуске на Render нужно выполнить скрипт загрузки статей ГК в Chroma:

```bash
# Локально (с теми же OPENAI_API_KEY и путями, что на Render):
python ingest_data.py
```

На Render с диском можно один раз запустить **Cron Job** или временный **Background Worker** с командой `python ingest_data.py` и теми же переменными окружения (включая `CHROMA_PERSIST_DIR` и `DATABASE_URL`), чтобы заполнить `/data/chroma_legal_db`. После этого основной воркер бота будет использовать уже созданную БД.

---

## Проверка работы

- В логах сервиса в Render должно быть сообщение вида: `Starting bot...` и отсутствие ошибок при старте.
- В Telegram найдите бота по имени и отправьте команду `/start`.

---

## Полезные ссылки

- [Render — Background Workers](https://render.com/docs/background-workers)
- [Render — Blueprint Spec](https://docs.render.com/blueprint-spec)
- [Render — Environment Variables](https://docs.render.com/configure-environment-variables)
