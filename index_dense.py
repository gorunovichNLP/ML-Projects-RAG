"""Построение dense-векторов для чанков и их загрузка в Qdrant."""

import json
import os
import uuid

from minio import Minio
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "rag_admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "rag_local_password")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rag-documents")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "document_chunks"

CHUNKS_PREFIX = "chunks/"
MODEL_NAME = "intfloat/multilingual-e5-large"
VECTOR_SIZE = 1024
MAX_TOKENS = 512
BATCH_SIZE = 8


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
        raise RuntimeError("В MinIO нет чанков для индексации")

    chunks = []
    chunk_ids = set()

    for object_name in object_names:
        response = client.get_object(MINIO_BUCKET, object_name)
        try:
            lines = response.read().decode("utf-8").splitlines()
        finally:
            response.close()
            response.release_conn()

        for line in lines:
            if not line.strip():
                continue

            chunk = json.loads(line)
            chunk_id = chunk["chunk_id"]
            if chunk_id in chunk_ids:
                raise ValueError(f"Повторяющийся chunk_id: {chunk_id}")

            chunk_ids.add(chunk_id)
            chunks.append(chunk)

    return chunks


def prepare_collection(client: QdrantClient) -> None:
    """Создаёт коллекцию или проверяет параметры уже существующей."""

    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        return

    vectors = client.get_collection(QDRANT_COLLECTION).config.params.vectors
    if not isinstance(vectors, models.VectorParams):
        raise ValueError("Ожидалась коллекция с одним dense-вектором")

    if vectors.size != VECTOR_SIZE or vectors.distance != models.Distance.COSINE:
        raise ValueError(
            f"Коллекция {QDRANT_COLLECTION!r} имеет неожиданные параметры"
        )


def qdrant_id(chunk_id: str) -> str:
    """Преобразует строковый chunk_id в стабильный UUID для Qdrant."""

    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def index_dense() -> None:
    """Векторизует все чанки пакетами и загружает их в Qdrant."""

    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    chunks = load_chunks(minio_client)
    print(f"Загружено чанков из MinIO: {len(chunks)}", flush=True)

    qdrant_client = QdrantClient(url=QDRANT_URL)
    prepare_collection(qdrant_client)

    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_TOKENS

    actual_vector_size = model.get_embedding_dimension()
    if actual_vector_size != VECTOR_SIZE:
        raise ValueError(
            f"Модель вернула размерность {actual_vector_size}, ожидалось {VECTOR_SIZE}"
        )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        passages = [f"passage: {chunk['text']}" for chunk in batch]
        vectors = model.encode(
            passages,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        points = [
            models.PointStruct(
                id=qdrant_id(chunk["chunk_id"]),
                vector=vector.tolist(),
                payload={**chunk, "embedding_model": MODEL_NAME},
            )
            for chunk, vector in zip(batch, vectors, strict=True)
        ]
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
            wait=True,
        )

        indexed = min(start + BATCH_SIZE, len(chunks))
        print(f"Проиндексировано: {indexed}/{len(chunks)}", flush=True)

    print(f"Готово. В Qdrant загружено чанков: {len(chunks)}")


if __name__ == "__main__":
    index_dense()
