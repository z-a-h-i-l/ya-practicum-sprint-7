import os
import json
import requests
import numpy as np
import faiss
import re
from typing import List

# ================= НАСТРОЙКИ =================
DATASET_DIR = "dataset"
INDEX_PATH = "faiss_index_v5.bin"
META_PATH = "faiss_metadata_v5.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# URL вашего llama-сервера (запускаем отдельно)
LLAMA_SERVER_URL = "http://localhost:8080/v1/embeddings"

# ================= ФУНКЦИЯ ЧАНКОВАНИЯ =================
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        start += (chunk_size - overlap)
    return chunks

# ================= ЧТЕНИЕ ДОКУМЕНТОВ =================
print("Чтение документов из папки dataset...")
documents = []
for filename in sorted(os.listdir(DATASET_DIR)):
    if not filename.endswith(".md"):
        continue
    filepath = os.path.join(DATASET_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    title_match = re.search(r"^#\s+(.*)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else filename
    content = re.sub(r"^#\s+.*$", "", text, 1, re.MULTILINE).strip()
    documents.append({"title": title, "content": content, "file": filename})

print(f"Найдено документов: {len(documents)}")

# ================= ФОРМИРОВАНИЕ ЧАНКОВ =================
all_chunks = []
all_metadata = []

for doc in documents:
    full_text = f"{doc['title']}\n{doc['content']}"
    chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_metadata.append({
            "title": doc["title"],
            "file": doc["file"],
            "chunk_id": i,
            "text": chunk
        })

print(f"Всего чанков: {len(all_chunks)}")

# ================= ЗАПРОС ЭМБЕДДИНГОВ У LLAMA.CPP =================
def get_embeddings(texts: List[str], prompt_prefix: str = "Document: ") -> np.ndarray:
    """
    Отправляет тексты на сервер llama.cpp для векторизации.
    Возвращает нормализованные эмбеддинги.
    """
    prefixed_texts = [f"{prompt_prefix}{text}" for text in texts]
    response = requests.post(
        LLAMA_SERVER_URL,
        json={"input": prefixed_texts},
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    data = response.json()
    
    # Извлекаем эмбеддинги из ответа
    embeddings = np.array([item["embedding"] for item in data["data"]], dtype=np.float32)
    
    # Нормализация для косинусной близости
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings

print("Вычисление эмбеддингов с помощью llama.cpp...")
print("Убедитесь, что llama-сервер запущен!")
embeddings = get_embeddings(all_chunks, prompt_prefix="Document: ")

# ================= СОЗДАНИЕ FAISS ИНДЕКСА =================
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # косинусное сходство
index.add(embeddings)
print(f"Индекс создан, векторов: {index.ntotal}")

# ================= СОХРАНЕНИЕ =================
faiss.write_index(index, INDEX_PATH)
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)

print(f"Индекс сохранён в {INDEX_PATH}")
print(f"Метаданные сохранены в {META_PATH}")

# ================= ПРИМЕР ПОИСКА =================
print("\n--- Тестовый поиск ---")
query = "результаты матчей НБА плей-офф"
query_emb = get_embeddings([query], prompt_prefix="Query: ")
k = 3
distances, indices = index.search(query_emb, k)

for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    meta = all_metadata[idx]
    print(f"{i+1}. (Сходство: {dist:.4f})")
    print(f"   Заголовок: {meta['title']}")
    print(f"   Текст: {meta['text'][:150]}...")
    print()