"""Простой семантический поиск по чанкам в Qdrant."""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "document_chunks"

model = SentenceTransformer(MODEL_NAME)
qdrant = QdrantClient(url="http://localhost:6333")

question = input("Введите вопрос: ")

# Для поисковых запросов E5 требует префикс query:.
query_vector = model.encode(
    f"query: {question}",
    normalize_embeddings=True,
)

results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector.tolist(),
    limit=5,
    with_payload=True,
).points

for number, result in enumerate(results, start=1):
    payload = result.payload

    print(f"\n{'=' * 80}")
    print(f"Результат: {number}")
    print(f"Score: {result.score:.4f}")
    print(f"Документ: {payload['document']}")
    print(f"Раздел: {' > '.join(payload['heading_path'])}")
    print(f"\n{payload['text']}")