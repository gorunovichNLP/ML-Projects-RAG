# Hybrid RAG: шаг 1 — загрузка raw-документов в MinIO

На первом шаге у нас всего четыре рабочих элемента:

- `raw_data_local/` — 10 исходных DOCX-файлов;
- `compose.yaml` — локальный MinIO;
- `requirements.txt` — Python SDK для MinIO;
- `upload_raw_docs.py` — загрузка документов.

Документы сохраняются в bucket `rag-documents` с ключами
`raw/<имя документа>.docx`. Скрипт записывает SHA-256 в метаданные объекта,
поэтому при повторном запуске неизменённые документы пропускаются.

## Запуск

Создаём окружение и устанавливаем одну зависимость:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Запускаем MinIO:

```powershell
docker compose up -d
```

Загружаем документы:

```powershell
python upload_raw_docs.py
```

MinIO Console будет доступна по адресу `http://localhost:9001`.

- логин: `rag_admin`
- пароль: `rag_local_password`

Когда дойдём до Airflow, вынесем параметры MinIO в Airflow Connection, а эту
функцию загрузки вызовем из отдельной задачи DAG.
