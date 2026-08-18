"""Загрузка исходных DOCX-документов из data/raw_docs в MinIO."""

import hashlib
import os
from pathlib import Path

from minio import Minio
from minio.error import S3Error


# Пока проект локальный, значения по умолчанию совпадают с compose.yaml.
# Позже Airflow будет передавать эти параметры через Airflow Connection.
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "rag_admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "rag_local_password")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rag-documents")

SOURCE_DIR = Path("data/raw_docs")
RAW_PREFIX = "raw"


def calculate_sha256(file_path: Path) -> str:
    """Считает хеш файла, не загружая весь документ в оперативную память."""

    file_hash = hashlib.sha256()

    with file_path.open("rb") as file:
        # Читаем по 1 МБ — этот способ подходит и для более крупных файлов.
        while chunk := file.read(1024 * 1024):
            file_hash.update(chunk)

    return file_hash.hexdigest()


def get_remote_sha256(
    client: Minio,
    object_name: str,
) -> str | None:
    """Возвращает хеш уже загруженного объекта или None, если его ещё нет."""

    try:
        object_info = client.stat_object(MINIO_BUCKET, object_name)
    except S3Error as error:
        if error.code in {"NoSuchKey", "NoSuchObject"}:
            return None
        raise

    return object_info.metadata.get("X-Amz-Meta-Sha256")


def upload_raw_documents() -> None:
    """Загружает новые и изменённые DOCX-файлы в префикс raw/."""

    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Не найдена директория: {SOURCE_DIR.resolve()}")

    # MinIO предоставляет S3-совместимый API. secure=False означает обычный
    # HTTP, что допустимо только для нашего локального окружения.
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)
        print(f"Создан bucket: {MINIO_BUCKET}")

    # Сортировка нужна для предсказуемого порядка обработки и логов.
    documents = sorted(
        (
            path
            for path in SOURCE_DIR.rglob("*.docx")
            if not path.name.startswith("~$")
        ),
        key=lambda path: path.as_posix().casefold(),
    )

    if not documents:
        raise FileNotFoundError(f"В {SOURCE_DIR.resolve()} нет DOCX-файлов")

    uploaded = 0
    skipped = 0

    for file_path in documents:
        # as_posix() гарантирует прямые слеши в object key даже на Windows.
        relative_path = file_path.relative_to(SOURCE_DIR).as_posix()
        object_name = f"{RAW_PREFIX}/{relative_path}"
        local_sha256 = calculate_sha256(file_path)

        # Повторный запуск не загружает неизменённый документ. Это пригодится
        # для retry и регулярных запусков будущего Airflow DAG.
        if get_remote_sha256(client, object_name) == local_sha256:
            skipped += 1
            print(f"Пропущен без изменений: {object_name}")
            continue

        client.fput_object(
            MINIO_BUCKET,
            object_name,
            str(file_path),
            content_type=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            metadata={"sha256": local_sha256},
        )
        uploaded += 1
        print(f"Загружен: {object_name}")

    print(
        f"Готово. Найдено: {len(documents)}, "
        f"загружено: {uploaded}, пропущено: {skipped}"
    )


if __name__ == "__main__":
    upload_raw_documents()
