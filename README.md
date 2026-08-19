# Hybrid RAG

## Структура проекта

```text
ingestion/            загрузка, обработка, чанкинг и dense-индексация
retrieval/            dense, BM25, гибридный поиск и reranking
data/raw_docs/        10 исходных DOCX-файлов
data/chunks.jsonl     локальная контрольная выгрузка чанков
compose.yaml          MinIO и Qdrant
requirements.txt      зависимости Python
```

Все команды ниже выполняются из корня проекта.

Документы сохраняются в bucket `rag-documents` с ключами
`raw/<имя документа>.docx`. Скрипт записывает SHA-256 в метаданные объекта,
поэтому при повторном запуске неизменённые документы пропускаются.

## Запуск

Создаём окружение и устанавливаем зависимости:

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
python ingestion/upload_raw_docs.py
```

MinIO Console будет доступна по адресу `http://localhost:9001`.

- логин: `rag_admin`
- пароль: `rag_local_password`

Когда дойдём до Airflow, вынесем параметры MinIO в Airflow Connection, а эту
функцию загрузки вызовем из отдельной задачи DAG.

## Обработка документов

Скрипт `ingestion/process_documents.py` берёт все DOCX из MinIO, извлекает
текст, таблицы и изображения, распознаёт изображения через Tesseract
(`rus+eng`) и сохраняет результат обратно в MinIO:

```text
processed/<название документа>/
├── document.md
└── images/
```

Длинные вертикальные схемы сохраняются как изображения без OCR: распознанный
текст теряет связи между узлами схемы и только ухудшает будущий поиск.

Запуск:

```powershell
python -m pip install -r requirements.txt
python ingestion/process_documents.py
```

Если Tesseract установлен нестандартно, укажите путь явно:

```powershell
$env:TESSERACT_PATH = "C:\path\to\tesseract.exe"
python ingestion/process_documents.py
```

## Чанкинг

Скрипт `ingestion/chunk_documents.py` читает обработанные Markdown, строит путь
по заголовкам и объединяет соседние абзацы одного раздела в чанки до 512
токенов:

```powershell
python ingestion/chunk_documents.py
```

Результат сохраняется в MinIO:

```text
chunks/<название документа>/chunks.jsonl
```

Каждая строка JSONL содержит `chunk_id`, источник, `heading_path`, текст чанка и
точный `token_count`. В лимит входят префикс `passage:`, путь заголовков и
служебные токены `multilingual-e5-large`. Чанки не пересекают границы разделов.
Если отдельный абзац превышает лимит, он делится с перекрытием 64 токена.

## Qdrant

Qdrant запускается вместе с MinIO:

```powershell
docker compose up -d
```

- REST API: `http://localhost:6333`
- Web UI: `http://localhost:6333/dashboard`
- gRPC API: `localhost:6334`

Dense-векторы хранятся в Docker volume `rag-local_qdrant_data`. Локальная
коллекция `document_chunks` использует векторы размером 1024 и Cosine distance.
Qdrant запущен без аутентификации, поэтому его порты привязаны только к
`127.0.0.1`. Для внешнего или production-развёртывания нужна отдельная настройка
безопасности.

## Dense-индексация

Скрипт `ingestion/index_dense.py` читает все чанки из MinIO, строит
нормализованные векторы `multilingual-e5-large` и загружает их вместе с текстом
и метаданными в Qdrant:

```powershell
python -m pip install -r requirements.txt
python ingestion/index_dense.py
```

Перед текстом каждого чанка добавляется обязательный для E5 префикс
`passage:`. Повторный запуск обновляет те же точки по стабильным UUID и не
создаёт дубликаты.

Проверить dense-поиск отдельно:

```powershell
python retrieval/search_dense.py
```

## BM25-поиск

Скрипт `retrieval/search_bm25.py` читает чанки из MinIO и при запуске строит
небольшой BM25-индекс в оперативной памяти:

```powershell
python retrieval/search_bm25.py
```

Индекс не сохраняется на диск: для текущих 384 чанков он строится быстро. Razdel
разбивает русский текст на токены, после чего Snowball приводит формы русских
слов к общей основе. Английские термины, идентификаторы и числа сохраняются без
стемминга. BM25 дополняет dense-поиск точными лексическими совпадениями.

## Гибридный поиск

Скрипт `retrieval/search_hybrid.py` получает по 20 результатов из dense и BM25
поиска, отдельно нормализует их scores в диапазон от 0 до 1 и объединяет с
весами 85% и 15%:

```text
hybrid_score = 0.85 * dense_normalized + 0.15 * bm25_normalized
```

Запуск:

```powershell
python retrieval/search_hybrid.py
```

Результаты объединяются по `chunk_id` и сортируются по `hybrid_score`. Лучшие 20
кандидатов передаются в `BAAI/bge-reranker-v2-m3`, который совместно оценивает
вопрос и полный текст каждого чанка. После reranking выводятся итоговые top 5.
Исходные scores и позиции всех этапов сохраняются в выводе для ручной проверки.
Логика этого этапа вынесена в отдельный файл `retrieval/reranker.py`.

## Генерация ответа через OpenRouter

После reranker лучшие пять чанков передаются в Qwen через OpenRouter. Модель
получает пронумерованные источники и должна отвечать только по ним, указывая
ссылки `[1]`, `[2]` и так далее. Если контекста недостаточно, модель должна
отказаться от содержательного ответа.

Перед передачей контекста система удаляет полные текстовые дубли. Если два
соседних чанка одного раздела имеют подтверждённое перекрытие на границе,
остаётся чанк с более высоким reranker score. При этом reranker сначала
оценивает все 20 кандидатов, поэтому после удаления дублей top-5 дополняется
следующими результатами.

Для документов с несколькими версиями чанки должны содержать metadata:

```json
{
  "metadata": {
    "document_group": "mortgage_rules",
    "version": "2026-08-01",
    "is_active": true
  }
}
```

Кандидаты с `is_active: false` исключаются до reranking. Если metadata нет,
система не пытается определять версию по имени файла и сохраняет документ.

Создайте `.env` в корне проекта:

```env
OPENROUTER_API_KEY=ваш-ключ
OPENROUTER_MODEL=qwen/qwen3-14b
```

После этого запустите:

```powershell
python generation/answer.py
```

По умолчанию используется `qwen/qwen3-14b`. Другую модель OpenRouter можно
выбрать без изменения кода:

```powershell
$env:OPENROUTER_MODEL="qwen/qwen3-32b"
python generation/answer.py
```

Системные переменные окружения имеют приоритет над значениями из `.env`.
Файл `.env` исключён из Git и не попадёт в репозиторий.

## Оценка качества поиска

Размеченный набор находится в `data/evaluation.jsonl`. В нём 40 вопросов:
по четыре на каждый из десяти документов. Для каждого вопроса указаны тип
вопроса, split (`dev` или `test`) и допустимые `relevant_chunk_ids`.

В `dev` находятся 30 вопросов для подбора параметров поиска. Оставшиеся 10
вопросов образуют `test`: его следует использовать только для итоговой проверки,
а не для выбора весов hybrid search или других настроек. Из-за небольшого размера
test его метрики пока подходят для поиска крупных регрессий, а не различий в сотые.

Запуск полного сравнения:

```powershell
python retrieval/evaluate.py
```

Скрипт один раз строит BM25, загружает E5 и выполняет retrieval для всех
вопросов. Затем освобождает E5, загружает reranker и считает:

- `Recall@20` и `MRR` отдельно для dense, BM25 и hybrid;
- `Precision@1` и `MRR` после reranker.

Покейсовый вывод показывает позицию первого эталонного чанка на каждом этапе.
Если стоит `—`, правильный чанк не попал в соответствующий top-20. Итоговые
метрики считаются отдельно для `dev` и `test` и сохраняются в
`data/evaluation_metrics.json`; каждый новый запуск полностью обновляет файл.
