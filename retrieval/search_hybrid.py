"""Гибридный поиск: 85% dense и 15% BM25."""

import json
import os

import snowballstemmer
from minio import Minio
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from razdel import tokenize as razdel_tokenize
from sentence_transformers import SentenceTransformer


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "rag_admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "rag_local_password")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rag-documents")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "document_chunks"

CHUNKS_PREFIX = "chunks/"
MODEL_NAME = "intfloat/multilingual-e5-large"
RUSSIAN_STEMMER = snowballstemmer.stemmer("russian")

CANDIDATES_LIMIT = 20
RESULTS_LIMIT = 20
DENSE_WEIGHT = 0.85
BM25_WEIGHT = 0.15
TEXT_PREVIEW_LENGTH = 600


def tokenize(text: str) -> list[str]:
    """Токенизирует текст и приводит русские слова к общей основе."""

    terms = []

    for token in razdel_tokenize(text):
        # Дефисные слова и конструкции со слешем полезнее искать по частям.
        value = token.text.casefold().replace("ё", "е")
        parts = value.replace("-", " ").replace("/", " ").split()

        for part in parts:
            if not any(character.isalnum() for character in part):
                continue

            # Английские термины, числа и идентификаторы сохраняем как есть.
            is_russian_word = part.isalpha() and any(
                "а" <= character <= "я" for character in part
            )
            terms.append(
                RUSSIAN_STEMMER.stemWord(part)
                if is_russian_word
                else part
            )

    return terms


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


def min_max_normalize(scores_by_id: dict[str, float]) -> dict[str, float]:
    """Приводит scores одного поисковика к диапазону от 0 до 1."""

    if not scores_by_id:
        return {}

    minimum = min(scores_by_id.values())
    maximum = max(scores_by_id.values())

    if maximum == minimum:
        return {chunk_id: 1.0 for chunk_id in scores_by_id}

    score_range = maximum - minimum
    return {
        chunk_id: (score - minimum) / score_range
        for chunk_id, score in scores_by_id.items()
    }


def hybrid_search() -> None:
    """Выполняет dense и BM25 поиск и объединяет нормализованные scores."""

    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    chunks = load_chunks(minio_client)
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])
    print(f"BM25-индекс построен: {len(chunks)} чанка")

    question = input("Введите вопрос: ").strip()
    query_tokens = tokenize(question)
    if not query_tokens:
        raise ValueError("Запрос не содержит слов или чисел")

    model = SentenceTransformer(MODEL_NAME)
    query_vector = model.encode(
        f"query: {question}",
        normalize_embeddings=True,
    )

    qdrant_client = QdrantClient(url=QDRANT_URL)
    dense_results = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector.tolist(),
        limit=CANDIDATES_LIMIT,
        with_payload=True,
    ).points

    dense_scores = {
        result.payload["chunk_id"]: result.score
        for result in dense_results
    }
    dense_ranks = {
        result.payload["chunk_id"]: rank
        for rank, result in enumerate(dense_results, start=1)
    }

    bm25_scores_array = bm25.get_scores(query_tokens)
    bm25_scores = {}
    bm25_ranks = {}

    for chunk_index in bm25_scores_array.argsort()[::-1]:
        score = float(bm25_scores_array[chunk_index])
        if score <= 0 or len(bm25_scores) == CANDIDATES_LIMIT:
            break

        chunk_id = chunks[chunk_index]["chunk_id"]
        bm25_scores[chunk_id] = score
        bm25_ranks[chunk_id] = len(bm25_scores)

    dense_normalized = min_max_normalize(dense_scores)
    bm25_normalized = min_max_normalize(bm25_scores)
    candidate_ids = dense_scores.keys() | bm25_scores.keys()

    hybrid_results = []
    for chunk_id in candidate_ids:
        dense_score = dense_normalized.get(chunk_id, 0.0)
        bm25_score = bm25_normalized.get(chunk_id, 0.0)
        hybrid_score = (
            DENSE_WEIGHT * dense_score
            + BM25_WEIGHT * bm25_score
        )
        hybrid_results.append((hybrid_score, chunk_id))

    hybrid_results.sort(reverse=True)
    hybrid_results = hybrid_results[:RESULTS_LIMIT]

    for result_number, (hybrid_score, chunk_id) in enumerate(
        hybrid_results,
        start=1,
    ):
        chunk = chunks_by_id[chunk_id]
        dense_rank = dense_ranks.get(chunk_id, "—")
        bm25_rank = bm25_ranks.get(chunk_id, "—")
        dense_raw = dense_scores.get(chunk_id)
        bm25_raw = bm25_scores.get(chunk_id)
        text = chunk["text"]
        if len(text) > TEXT_PREVIEW_LENGTH:
            text = text[:TEXT_PREVIEW_LENGTH].rstrip() + "…"

        print(f"\n{'=' * 80}")
        print(f"Результат: {result_number}")
        print(f"Hybrid score: {hybrid_score:.4f}")
        print(
            f"Dense: rank={dense_rank}, "
            f"raw={dense_raw:.4f}, norm={dense_normalized[chunk_id]:.4f}"
            if dense_raw is not None
            else "Dense: не найден в top 20"
        )
        print(
            f"BM25: rank={bm25_rank}, "
            f"raw={bm25_raw:.4f}, norm={bm25_normalized[chunk_id]:.4f}"
            if bm25_raw is not None
            else "BM25: не найден в top 20"
        )
        print(f"Документ: {chunk['document']}")
        print(f"Раздел: {' > '.join(chunk['heading_path'])}")
        print(f"\n{text}")


if __name__ == "__main__":
    hybrid_search()
