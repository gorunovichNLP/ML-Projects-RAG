"""Переупорядочивание найденных чанков multilingual BGE reranker-ом."""

import re

from sentence_transformers import CrossEncoder
from torch import nn


MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 1024
BATCH_SIZE = 1
MIN_OVERLAP_WORDS = 20


def filter_current_versions(candidates: list[dict]) -> list[dict]:
    """Оставляет только active-версию при наличии version metadata."""

    version_groups: dict[str, list[dict]] = {}
    candidates_without_version = []

    for candidate in candidates:
        metadata = candidate["chunk"].get("metadata", {})
        document_group = metadata.get("document_group")
        is_active = metadata.get("is_active")

        # Если metadata ещё не добавлены, ничего не угадываем по имени файла.
        if not document_group or not isinstance(is_active, bool):
            candidates_without_version.append(candidate)
            continue

        version_groups.setdefault(document_group, []).append(candidate)

    current_candidates = candidates_without_version.copy()
    for group_candidates in version_groups.values():
        current_candidates.extend(
            candidate
            for candidate in group_candidates
            if candidate["chunk"]["metadata"]["is_active"]
        )

    return current_candidates


def normalized_text(text: str) -> str:
    """Нормализует пробелы и регистр для поиска полных текстовых дублей."""

    return " ".join(text.casefold().split())


def chunk_number(chunk_id: str) -> int | None:
    """Извлекает порядковый номер из chunk_id формата document::0001."""

    _, separator, number = chunk_id.rpartition("::")
    if not separator or not number.isdigit():
        return None
    return int(number)


def content_words(chunk: dict) -> list[str]:
    """Возвращает слова контента без повторяющегося пути заголовков."""

    text = chunk["text"]
    _, separator, content = text.partition("\n\n")
    if not separator:
        content = text
    return re.findall(r"\w+", content.casefold())


def are_overlapping_neighbors(first: dict, second: dict) -> bool:
    """Проверяет точное перекрытие границ двух соседних чанков."""

    if (
        first["document"] != second["document"]
        or first["source"] != second["source"]
        or first["heading_path"] != second["heading_path"]
    ):
        return False

    first_number = chunk_number(first["chunk_id"])
    second_number = chunk_number(second["chunk_id"])
    if (
        first_number is None
        or second_number is None
        or abs(first_number - second_number) != 1
    ):
        return False

    if first_number < second_number:
        earlier_words = content_words(first)
        later_words = content_words(second)
    else:
        earlier_words = content_words(second)
        later_words = content_words(first)

    max_overlap = min(len(earlier_words), len(later_words))
    for overlap_size in range(max_overlap, MIN_OVERLAP_WORDS - 1, -1):
        if earlier_words[-overlap_size:] == later_words[:overlap_size]:
            return True

    return False


def select_context_results(
    reranked: list[dict],
    limit: int,
) -> list[dict]:
    """Удаляет дубли и выбирает разнообразные чанки для контекста LLM."""

    if limit <= 0:
        return []

    # Сортировка гарантирует, что из дублей останется лучший reranker score.
    candidates = sorted(
        reranked,
        key=lambda candidate: candidate["reranker_score"],
        reverse=True,
    )
    selected = []
    seen_texts = set()

    for candidate in candidates:
        chunk = candidate["chunk"]
        text_key = normalized_text(chunk["text"])
        if text_key in seen_texts:
            continue

        if any(
            are_overlapping_neighbors(chunk, selected_item["chunk"])
            for selected_item in selected
        ):
            continue

        selected.append(candidate)
        seen_texts.add(text_key)

        if len(selected) == limit:
            break

    return selected


def load_reranker() -> CrossEncoder:
    """Загружает модель один раз для одного поискового или evaluation-запуска."""

    return CrossEncoder(
        MODEL_NAME,
        max_length=MAX_LENGTH,
        activation_fn=nn.Sigmoid(),
    )


def rerank(
    question: str,
    candidates: list[dict],
    limit: int,
    model: CrossEncoder | None = None,
    show_progress_bar: bool = True,
) -> list[dict]:
    """Оценивает пары «вопрос + чанк» и возвращает лучшие кандидаты."""

    if not candidates:
        return []

    if model is None:
        model = load_reranker()

    pairs = [
        (question, candidate["chunk"]["text"])
        for candidate in candidates
    ]
    scores = model.predict(
        pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=show_progress_bar,
    )

    reranked = [
        {**candidate, "reranker_score": float(score)}
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    reranked.sort(
        key=lambda candidate: candidate["reranker_score"],
        reverse=True,
    )
    return reranked[:limit]
