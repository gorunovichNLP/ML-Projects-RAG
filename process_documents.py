"""Преобразование всех DOCX из MinIO в Markdown с OCR изображений."""

import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.table import Table
from minio import Minio


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "rag_admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "rag_local_password")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rag-documents")

RAW_PREFIX = "raw/"


def find_tesseract() -> str:
    """Находит Tesseract из переменной окружения, PATH или стандартного пути."""

    configured_path = os.getenv("TESSERACT_PATH")
    if configured_path and Path(configured_path).is_file():
        return configured_path

    path_from_system = shutil.which("tesseract")
    if path_from_system:
        return path_from_system

    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        windows_path = (
            Path(local_app_data) / "Programs" / "Tesseract-OCR" / "tesseract.exe"
        )
        if windows_path.is_file():
            return str(windows_path)

    raise FileNotFoundError(
        "Tesseract не найден. Установите его или задайте TESSERACT_PATH."
    )


def is_long_diagram(image_path: Path) -> bool:
    """Определяет длинную вертикальную схему по соотношению сторон."""

    # Все изображения в наших DOCX имеют формат PNG. В PNG ширина и высота
    # записаны в байтах 16–24, поэтому отдельная библиотека изображений не нужна.
    with image_path.open("rb") as image_file:
        header = image_file.read(24)

    if header.startswith(b"\x89PNG") and len(header) == 24:
        width, height = struct.unpack(">II", header[16:24])
        return height > width * 1.2

    return False


def extract_text_from_image(tesseract: str, image_path: Path) -> str:
    """Распознаёт русский и английский текст на одном изображении."""

    result = subprocess.run(
        [
            tesseract,
            str(image_path),
            "stdout",
            "-l",
            "rus+eng",
            "--psm",
            "6",
        ],
        capture_output=True,
        check=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def paragraph_prefix(style_name: str) -> str:
    """Преобразует основные стили Word в простую Markdown-разметку."""

    if style_name == "Title":
        return "# "
    if style_name.startswith("Heading "):
        level = style_name.removeprefix("Heading ")
        if level.isdigit():
            return "#" * min(int(level), 6) + " "
    if style_name.startswith("List Bullet"):
        return "- "
    if style_name.startswith("List Number"):
        return "1. "
    return ""


def table_to_markdown(table: Table) -> str:
    """Преобразует таблицу Word в Markdown-таблицу."""

    rows = []
    for row in table.rows:
        cells = [cell.text.strip().replace("|", "\\|") for cell in row.cells]
        rows.append("| " + " | ".join(cells) + " |")

    if not rows:
        return ""

    column_count = len(table.rows[0].cells)
    separator = "| " + " | ".join(["---"] * column_count) + " |"
    return "\n".join([rows[0], separator, *rows[1:]])


def process_document(
    client: Minio,
    tesseract: str,
    source_object: str,
) -> tuple[int, int, int]:
    """Обрабатывает один DOCX и возвращает счётчики изображений и OCR."""

    document_name = Path(source_object).stem
    processed_prefix = f"processed/{document_name}"

    # Временная папка удалится автоматически после загрузки результата в MinIO.
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        docx_path = work_dir / "source.docx"
        images_dir = work_dir / "images"
        images_dir.mkdir()

        client.fget_object(MINIO_BUCKET, source_object, str(docx_path))
        document = Document(docx_path)

        markdown = [f"<!-- source: {source_object} -->", ""]
        image_number = 0
        ocr_number = 0
        skipped_ocr = 0

        # iter_inner_content() сохраняет порядок абзацев и таблиц в DOCX.
        for block in document.iter_inner_content():
            if isinstance(block, Table):
                table_markdown = table_to_markdown(block)
                if table_markdown:
                    markdown.extend([table_markdown, ""])
            else:
                text = block.text.strip()
                if text:
                    style_name = block.style.name if block.style else ""
                    markdown.extend([paragraph_prefix(style_name) + text, ""])

            # Изображения могут быть как встроенными, так и плавающими. Оба
            # варианта содержат ссылку a:blip на соответствующий image part.
            for blip in block._element.xpath(".//a:blip"):
                relationship_id = blip.get(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
                )
                image_part = document.part.related_parts[relationship_id]
                image_number += 1

                extension = Path(str(image_part.partname)).suffix or ".png"
                image_name = f"image_{image_number:03d}{extension}"
                image_path = images_dir / image_name
                image_path.write_bytes(image_part.blob)

                markdown.extend([f"![{image_name}](images/{image_name})", ""])

                # Длинные вертикальные изображения в наших документах — схемы.
                # OCR теряет связи между узлами и создаёт бессвязный текст,
                # поэтому картинку сохраняем, но в поисковый текст не добавляем.
                if is_long_diagram(image_path):
                    skipped_ocr += 1
                    print(f"OCR пропущен для длинной схемы: {image_name}")
                else:
                    ocr_text = extract_text_from_image(tesseract, image_path)
                    ocr_number += 1
                    markdown.extend(
                        [
                            "**Текст с изображения:**",
                            "",
                            "```text",
                            ocr_text or "_Текст не распознан._",
                            "```",
                            "",
                        ]
                    )

                client.fput_object(
                    MINIO_BUCKET,
                    f"{processed_prefix}/images/{image_name}",
                    str(image_path),
                    content_type=image_part.content_type,
                )

        markdown_path = work_dir / "document.md"
        markdown_path.write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")

        client.fput_object(
            MINIO_BUCKET,
            f"{processed_prefix}/document.md",
            str(markdown_path),
            content_type="text/markdown; charset=utf-8",
        )

    print(f"Источник: {source_object}")
    print(f"Изображений обработано: {image_number}")
    print(f"OCR выполнен: {ocr_number}, длинных схем пропущено: {skipped_ocr}")
    print(f"Результат: {processed_prefix}/document.md")
    return image_number, ocr_number, skipped_ocr


def process_documents() -> None:
    """Последовательно обрабатывает все raw DOCX-документы из MinIO."""

    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    tesseract = find_tesseract()

    source_objects = sorted(
        (
            item.object_name
            for item in client.list_objects(
                MINIO_BUCKET,
                prefix=RAW_PREFIX,
                recursive=True,
            )
            if item.object_name.lower().endswith(".docx")
        ),
        key=str.casefold,
    )
    if not source_objects:
        raise RuntimeError(f"В {MINIO_BUCKET}/{RAW_PREFIX} нет DOCX-документов")

    total_images = 0
    total_ocr = 0
    total_skipped_ocr = 0

    for document_number, source_object in enumerate(source_objects, start=1):
        print(f"\n[{document_number}/{len(source_objects)}]")
        images, ocr, skipped_ocr = process_document(
            client,
            tesseract,
            source_object,
        )
        total_images += images
        total_ocr += ocr
        total_skipped_ocr += skipped_ocr

    print("\nВсе документы обработаны.")
    print(f"Документов: {len(source_objects)}")
    print(f"Изображений: {total_images}")
    print(f"OCR выполнен: {total_ocr}")
    print(f"Длинных схем пропущено: {total_skipped_ocr}")


if __name__ == "__main__":
    process_documents()
