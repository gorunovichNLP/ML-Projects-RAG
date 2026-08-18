"""Простой лексический поиск BM25 по чанкам из MinIO."""

import json
import os
import re

from minio import Minio
from rank_bm25 import BM25Okapi


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "rag_admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "rag_local_password")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rag-documents")

CHUNKS_PREFIX = "chunks/"
RESULTS_LIMIT = 5
TOKEN_PATTERN = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Приводит текст к нижнему регистру и выделяет слова и числа."""

    return TOKEN_PATTERN.findall(text.casefold())


def load_chunks(client: Minio) -> list[dict]:
    """Читает все JSONL-файлы с чанками из MinIO."""

    object_names = sorted(
        (
            item.object_name
            for item in client.list_objects(
                MINIO_BUCKET,
                prefix=CHUNKS_PREFIX,
                recursive=True,
            )
            if item.object_name.endswith("/chunks.jsonl")
        ),
        key=str.casefold,
    )
    if not object_names:
        raise RuntimeError("В MinIO нет чанков для поиска")

    chunks = []

    for object_name in object_names:
        response = client.get_object(MINIO_BUCKET, object_name)
        try:
            lines = response.read().decode("utf-8").splitlines()
        finally:
            response.close()
            response.release_conn()

        chunks.extend(json.loads(line) for line in lines if line.strip())

    return chunks


def search_bm25() -> None:
    """Строит индекс в памяти и выводит наиболее подходящие чанки."""

    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    chunks = load_chunks(minio_client)
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"BM25-индекс построен: {len(chunks)} чанка")

    question = input("Введите вопрос: ").strip()
    query_tokens = tokenize(question)
    if not query_tokens:
        raise ValueError("Запрос не содержит слов или чисел")

    scores = bm25.get_scores(query_tokens)
    best_indices = scores.argsort()[::-1]
    results_found = 0

    for chunk_index in best_indices:
        score = scores[chunk_index]
        if score <= 0 or results_found == RESULTS_LIMIT:
            break

        chunk = chunks[chunk_index]
        results_found += 1

        print(f"\n{'=' * 80}")
        print(f"Результат: {results_found}")
        print(f"BM25 score: {score:.4f}")
        print(f"Документ: {chunk['document']}")
        print(f"Раздел: {' > '.join(chunk['heading_path'])}")
        print(f"\n{chunk['text']}")

    if not results_found:
        print("Совпадений по словам запроса не найдено")


if __name__ == "__main__":
    search_bm25()
