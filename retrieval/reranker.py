"""Переупорядочивание найденных чанков multilingual BGE reranker-ом."""

from sentence_transformers import CrossEncoder
from torch import nn


MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 1024
BATCH_SIZE = 1


def rerank(
    question: str,
    candidates: list[dict],
    limit: int,
) -> list[dict]:
    """Оценивает пары «вопрос + чанк» и возвращает лучшие кандидаты."""

    if not candidates:
        return []

    model = CrossEncoder(
        MODEL_NAME,
        max_length=MAX_LENGTH,
        activation_fn=nn.Sigmoid(),
    )
    pairs = [
        (question, candidate["chunk"]["text"])
        for candidate in candidates
    ]
    scores = model.predict(
        pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
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
