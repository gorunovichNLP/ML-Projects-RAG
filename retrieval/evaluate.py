"""Офлайн-оценка dense, BM25, hybrid retrieval и reranker."""

import json
from pathlib import Path

from reranker import load_reranker, rerank
from search_hybrid import prepare_search, retrieve


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_PATH = PROJECT_ROOT / "data" / "evaluation.jsonl"
METRICS_PATH = PROJECT_ROOT / "data" / "evaluation_metrics.json"
RERANK_LIMIT = 20
SPLITS = ("dev", "test")


def load_evaluation_cases() -> list[dict]:
    """Читает и проверяет размеченные вопросы из JSONL."""

    cases = []

    with EVALUATION_PATH.open(encoding="utf-8") as evaluation_file:
        for line_number, line in enumerate(evaluation_file, start=1):
            if not line.strip():
                continue

            case = json.loads(line)
            question = case.get("question", "").strip()
            relevant_ids = case.get("relevant_chunk_ids", [])
            split = case.get("split", "").strip()

            if not question:
                raise ValueError(f"Строка {line_number}: пустой question")
            if not relevant_ids:
                raise ValueError(
                    f"Строка {line_number}: нет relevant_chunk_ids"
                )
            if split not in SPLITS:
                raise ValueError(
                    f"Строка {line_number}: split должен быть dev или test"
                )

            cases.append(case)

    if not cases:
        raise RuntimeError("Evaluation-набор пуст")

    missing_splits = set(SPLITS) - {case["split"] for case in cases}
    if missing_splits:
        raise ValueError(
            f"В evaluation-наборе отсутствуют split: {sorted(missing_splits)}"
        )

    return cases


def reciprocal_rank(ranking: list[str], relevant_ids: set[str]) -> float:
    """Возвращает 1 / позицию первого релевантного чанка или 0."""

    for rank, chunk_id in enumerate(ranking, start=1):
        if chunk_id in relevant_ids:
            return 1 / rank
    return 0.0


def recall_at_k(ranking: list[str], relevant_ids: set[str]) -> float:
    """Проверяет, найден ли хотя бы один эталонный чанк в top-k."""

    return float(bool(set(ranking) & relevant_ids))


def mean(values: list[float]) -> float:
    """Считает среднее для уже проверенного непустого набора."""

    return sum(values) / len(values)


def empty_metrics() -> dict[str, list[float]]:
    """Создаёт независимые накопители метрик для одного split."""

    return {
        "dense_recall": [],
        "dense_mrr": [],
        "bm25_recall": [],
        "bm25_mrr": [],
        "hybrid_recall": [],
        "hybrid_mrr": [],
        "reranker_precision": [],
        "reranker_mrr": [],
    }


def build_split_report(metrics: dict[str, list[float]]) -> dict:
    """Собирает итоговые метрики одного split."""

    return {
        "evaluation_cases": len(metrics["dense_recall"]),
        "dense": {
            "recall_at_20": round(mean(metrics["dense_recall"]), 6),
            "mrr": round(mean(metrics["dense_mrr"]), 6),
        },
        "bm25": {
            "recall_at_20": round(mean(metrics["bm25_recall"]), 6),
            "mrr": round(mean(metrics["bm25_mrr"]), 6),
        },
        "hybrid": {
            "recall_at_20": round(mean(metrics["hybrid_recall"]), 6),
            "mrr": round(mean(metrics["hybrid_mrr"]), 6),
        },
        "reranker": {
            "precision_at_1": round(
                mean(metrics["reranker_precision"]),
                6,
            ),
            "mrr": round(mean(metrics["reranker_mrr"]), 6),
        },
    }


def print_split_report(split: str, report: dict) -> None:
    """Печатает компактный отчёт для одного split."""

    print(f"\n{split.upper()}: {report['evaluation_cases']} вопросов")
    for name in ("dense", "bm25", "hybrid"):
        print(
            f"{name.capitalize():<10} "
            f"Recall@20={report[name]['recall_at_20']:.3f}  "
            f"MRR={report[name]['mrr']:.3f}"
        )
    print(
        f"Reranker   Precision@1={report['reranker']['precision_at_1']:.3f}  "
        f"MRR={report['reranker']['mrr']:.3f}"
    )


def evaluate() -> None:
    """Прогоняет весь набор и показывает итоговые и покейсовые метрики."""

    cases = load_evaluation_cases()
    chunks, bm25, dense_model, qdrant_client = prepare_search()
    known_chunk_ids = {chunk["chunk_id"] for chunk in chunks}

    for case in cases:
        unknown_ids = set(case["relevant_chunk_ids"]) - known_chunk_ids
        if unknown_ids:
            raise ValueError(
                f"Неизвестные chunk_id для вопроса {case['question']!r}: "
                f"{sorted(unknown_ids)}"
            )

    # Сначала выполняем retrieval для всех вопросов. После этого E5 можно
    # удалить из памяти и только затем загрузить тяжёлый cross-encoder.
    prepared_cases = []
    for number, case in enumerate(cases, start=1):
        print(f"Retrieval {number}/{len(cases)}")
        result = retrieve(
            case["question"],
            chunks,
            bm25,
            dense_model,
            qdrant_client,
        )
        prepared_cases.append((case, result))

    del dense_model
    reranker_model = load_reranker()

    metrics_by_split = {split: empty_metrics() for split in SPLITS}

    print("\nРезультаты по вопросам")
    print("=" * 80)

    for number, (case, result) in enumerate(prepared_cases, start=1):
        print(f"Reranking {number}/{len(prepared_cases)}")
        relevant_ids = set(case["relevant_chunk_ids"])
        dense_ids = result["dense_ids"]
        bm25_ids = result["bm25_ids"]
        hybrid_ids = [
            candidate["chunk_id"]
            for candidate in result["hybrid_candidates"]
        ]
        reranked = rerank(
            case["question"],
            result["hybrid_candidates"],
            limit=RERANK_LIMIT,
            model=reranker_model,
            show_progress_bar=False,
        )
        reranked_ids = [candidate["chunk_id"] for candidate in reranked]
        metrics = metrics_by_split[case["split"]]

        metrics["dense_recall"].append(recall_at_k(dense_ids, relevant_ids))
        metrics["dense_mrr"].append(reciprocal_rank(dense_ids, relevant_ids))
        metrics["bm25_recall"].append(recall_at_k(bm25_ids, relevant_ids))
        metrics["bm25_mrr"].append(reciprocal_rank(bm25_ids, relevant_ids))
        metrics["hybrid_recall"].append(recall_at_k(hybrid_ids, relevant_ids))
        metrics["hybrid_mrr"].append(reciprocal_rank(hybrid_ids, relevant_ids))
        metrics["reranker_precision"].append(
            float(bool(reranked_ids) and reranked_ids[0] in relevant_ids)
        )
        metrics["reranker_mrr"].append(
            reciprocal_rank(reranked_ids, relevant_ids)
        )

        print(f"{number:02d}. [{case['split']}] {case['question']}")
        print(
            "    ranks: "
            f"dense={rank_or_dash(dense_ids, relevant_ids)}, "
            f"bm25={rank_or_dash(bm25_ids, relevant_ids)}, "
            f"hybrid={rank_or_dash(hybrid_ids, relevant_ids)}, "
            f"reranker={rank_or_dash(reranked_ids, relevant_ids)}"
        )

    print("\nИтоговые метрики")
    print("=" * 80)

    report = {
        "evaluation_cases": len(cases),
        "splits": {
            split: build_split_report(metrics_by_split[split])
            for split in SPLITS
        },
    }

    for split in SPLITS:
        print_split_report(split, report["splits"][split])

    METRICS_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nМетрики сохранены: {METRICS_PATH}")


def rank_or_dash(ranking: list[str], relevant_ids: set[str]) -> str:
    """Возвращает позицию первого эталонного чанка для диагностики."""

    for rank, chunk_id in enumerate(ranking, start=1):
        if chunk_id in relevant_ids:
            return str(rank)
    return "—"


if __name__ == "__main__":
    evaluate()
