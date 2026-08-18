"""Чанкинг обработанных Markdown-документов с сохранением пути заголовков."""

import json
import os
import tempfile
from pathlib import Path

from minio import Minio
from transformers import AutoTokenizer


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "rag_admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "rag_local_password")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rag-documents")

PROCESSED_PREFIX = "processed/"
CHUNKS_PREFIX = "chunks/"

TOKENIZER_NAME = "intfloat/multilingual-e5-large"
MAX_TOKENS = 512
OVERLAP_TOKENS = 64


def update_heading_path(
    heading_path: list[str],
    level: int,
    title: str,
) -> list[str]:
    """Обновляет путь при встрече очередного Markdown-заголовка."""

    # Если документ начинается с ##, отсутствующий уровень # не создаёт
    # пустой узел: такой заголовок просто становится корнем нашего пути.
    position = min(level - 1, len(heading_path))
    return [*heading_path[:position], title]


def markdown_blocks(markdown: str) -> list[tuple[list[str], str]]:
    """Возвращает текстовые блоки вместе с активным путём заголовков."""

    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
    heading_path: list[str] = []
    blocks: list[tuple[list[str], str]] = []
    current_lines: list[str] = []
    inside_code_block = False

    def flush_block() -> None:
        text = "\n".join(current_lines).strip()
        current_lines.clear()
        if text:
            blocks.append((heading_path.copy(), text))

    for line in markdown.splitlines():
        stripped = line.strip()

        # OCR заключён в ```text```. Сами маркеры не нужны в чанке, но внутри
        # блока символ # является обычным текстом, а не заголовком Markdown.
        if stripped.startswith("```"):
            flush_block()
            inside_code_block = not inside_code_block
            continue

        if not inside_code_block and stripped.startswith("#"):
            hashes, separator, title = stripped.partition(" ")
            if separator and 1 <= len(hashes) <= 6 and set(hashes) == {"#"}:
                flush_block()
                heading_path = update_heading_path(
                    heading_path,
                    level=len(hashes),
                    title=title.strip(),
                )
                continue

        if not inside_code_block and (
            stripped.startswith("<!--")
            or stripped.startswith("![")
            or stripped == "**Текст с изображения:**"
        ):
            flush_block()
            continue

        if not stripped:
            flush_block()
            continue

        current_lines.append(line)

    flush_block()
    return blocks


def chunk_text(heading_path: list[str], content: str) -> str:
    """Добавляет к содержимому полный путь заголовков."""

    breadcrumb = " > ".join(heading_path)
    return f"{breadcrumb}\n\n{content}" if breadcrumb else content


def model_input(heading_path: list[str], content: str) -> str:
    """Формирует точный текст, который позже получит E5."""

    return "passage: " + chunk_text(heading_path, content)


def token_count(tokenizer, heading_path: list[str], content: str) -> int:
    """Считает токены вместе с passage-префиксом и служебными токенами."""

    return len(
        tokenizer.encode(
            model_input(heading_path, content),
            add_special_tokens=True,
            verbose=False,
        )
    )


def split_long_block(
    tokenizer,
    heading_path: list[str],
    block_text: str,
) -> list[str]:
    """Режет один длинный абзац токенами с небольшим перекрытием."""

    prefix_token_count = token_count(tokenizer, heading_path, "")
    window_size = MAX_TOKENS - prefix_token_count
    if window_size <= OVERLAP_TOKENS:
        raise ValueError("Путь заголовков не оставляет места для текста чанка")

    block_tokens = tokenizer.encode(
        block_text,
        add_special_tokens=False,
        verbose=False,
    )
    parts = []
    start = 0

    while start < len(block_tokens):
        end = min(start + window_size, len(block_tokens))
        part = tokenizer.decode(block_tokens[start:end], skip_special_tokens=True).strip()

        # После decode границы токенов могут слегка измениться. Уменьшаем окно,
        # пока полный вход модели гарантированно не поместится в 512 токенов.
        while part and token_count(tokenizer, heading_path, part) > MAX_TOKENS:
            end -= 1
            part = tokenizer.decode(
                block_tokens[start:end],
                skip_special_tokens=True,
            ).strip()

        if not part:
            raise ValueError("Не удалось разделить длинный текстовый блок")

        parts.append(part)
        if end == len(block_tokens):
            break

        start = end - OVERLAP_TOKENS

    return parts


def build_chunks(
    tokenizer,
    blocks: list[tuple[list[str], str]],
    document_name: str,
    source_object: str,
) -> list[dict]:
    """Объединяет соседние абзацы одного раздела в чанки заданного размера."""

    chunks = []
    current_path: list[str] = []
    current_parts: list[str] = []

    def flush_chunk() -> None:
        if not current_parts:
            return

        content = "\n\n".join(current_parts)
        text = chunk_text(current_path, content)
        tokens = token_count(tokenizer, current_path, content)
        if tokens > MAX_TOKENS:
            raise ValueError(f"Чанк превышает лимит: {tokens} токенов")

        chunk_number = len(chunks) + 1

        chunks.append(
            {
                "chunk_id": f"{document_name}::{chunk_number:04d}",
                "document": document_name,
                "source": source_object,
                "heading_path": current_path.copy(),
                "text": text,
                "token_count": tokens,
            }
        )
        current_parts.clear()

    for heading_path, block_text in blocks:
        if token_count(tokenizer, heading_path, block_text) > MAX_TOKENS:
            block_parts = split_long_block(tokenizer, heading_path, block_text)
        else:
            block_parts = [block_text]

        for block_part in block_parts:
            candidate_parts = [*current_parts, block_part]
            candidate_content = "\n\n".join(candidate_parts)

            # Чанк никогда не пересекает границу раздела и никогда не
            # полагается на автоматическое усечение текста токенизатором.
            if current_parts and (
                heading_path != current_path
                or token_count(tokenizer, heading_path, candidate_content)
                > MAX_TOKENS
            ):
                flush_chunk()

            if not current_parts:
                current_path = heading_path.copy()

            current_parts.append(block_part)

    flush_chunk()
    return chunks


def chunk_documents() -> None:
    """Строит чанки для всех processed Markdown и сохраняет JSONL в MinIO."""

    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    markdown_objects = sorted(
        (
            item.object_name
            for item in client.list_objects(
                MINIO_BUCKET,
                prefix=PROCESSED_PREFIX,
                recursive=True,
            )
            if item.object_name.endswith("/document.md")
        ),
        key=str.casefold,
    )
    if not markdown_objects:
        raise RuntimeError("В MinIO нет обработанных Markdown-документов")

    total_chunks = 0

    for document_number, source_object in enumerate(markdown_objects, start=1):
        response = client.get_object(MINIO_BUCKET, source_object)
        markdown = response.read().decode("utf-8")
        response.close()
        response.release_conn()

        document_name = source_object.split("/")[-2]
        chunks = build_chunks(
            tokenizer,
            markdown_blocks(markdown),
            document_name=document_name,
            source_object=source_object,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "chunks.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
                for chunk in chunks:
                    jsonl_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            destination = f"{CHUNKS_PREFIX}{document_name}/chunks.jsonl"
            client.fput_object(
                MINIO_BUCKET,
                destination,
                str(jsonl_path),
                content_type="application/x-ndjson; charset=utf-8",
            )

        total_chunks += len(chunks)
        max_tokens = max(chunk["token_count"] for chunk in chunks)
        print(
            f"[{document_number}/{len(markdown_objects)}] "
            f"{len(chunks)} чанков, максимум {max_tokens} токенов"
        )

    print(f"Готово. Документов: {len(markdown_objects)}, чанков: {total_chunks}")


if __name__ == "__main__":
    chunk_documents()
