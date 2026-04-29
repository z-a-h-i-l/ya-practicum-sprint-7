# add_to_faiss.py
import os
import json
import requests
import numpy as np
import faiss
import re
from typing import List, Set

# ================= НАСТРОЙКИ =================
DATASET_DIR = "dataset"
INDEX_PATH = "faiss_index_v5.bin"
META_PATH = "faiss_metadata_v5.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

LLAMA_SERVER_URL = "http://localhost:8080/v1/embeddings"

# ================= ЗАГРУЗКА СУЩЕСТВУЮЩЕГО ИНДЕКСА И МЕТАДАННЫХ =================
print("Загрузка существующего индекса и метаданных...")
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Индекс загружен: {index.ntotal} векторов, {len(metadata)} чанков.")
else:
    # Если файлов нет — создаём пустые, чтобы скрипт мог работать с нуля
    print("Индекс не найден, будет создан новый после обработки документов.")
    index = None  # размерность определим по первому эмбеддингу
    metadata = []

# Собираем имена файлов, которые уже проиндексированы
indexed_files: Set[str] = {os.path.basename(entry["file"]) for entry in metadata}
print(f"Уже проиндексировано файлов: {len(indexed_files)}")

# ================= ПОИСК НОВЫХ ФАЙЛОВ =================
print("Поиск новых .md файлов в папке dataset...")
new_files = []
for filename in sorted(os.listdir(DATASET_DIR)):
    if not filename.endswith(".md"):
        continue
    if filename not in indexed_files:
        new_files.append(filename)

if not new_files:
    print("Новых файлов не обнаружено. Выход.")
    exit(0)

print(f"Найдено новых файлов: {len(new_files)}")

# ================= ЧТЕНИЕ НОВЫХ ДОКУМЕНТОВ =================
documents = []
for filename in new_files:
    filepath = os.path.join(DATASET_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    title_match = re.search(r"^#\s+(.*)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else filename
    content = re.sub(r"^#\s+.*$", "", text, 1, re.MULTILINE).strip()
    documents.append({"title": title, "content": content, "file": filename})

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

# ================= ФОРМИРОВАНИЕ НОВЫХ ЧАНКОВ =================
new_chunks = []
new_metadata = []

for doc in documents:
    full_text = f"{doc['title']}\n{doc['content']}"
    chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    for i, chunk in enumerate(chunks):
        new_chunks.append(chunk)
        new_metadata.append({
            "title": doc["title"],
            "file": doc["file"],
            "chunk_id": i,
            "text": chunk
        })

print(f"Новых чанков: {len(new_chunks)}")

# ================= ФУНКЦИЯ ЭМБЕДДИНГА =================
def get_embeddings(texts: List[str], prompt_prefix: str = "Document: ") -> np.ndarray:
    prefixed_texts = [f"{prompt_prefix}{text}" for text in texts]
    response = requests.post(
        LLAMA_SERVER_URL,
        json={"input": prefixed_texts},
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    data = response.json()
    embeddings = np.array([item["embedding"] for item in data["data"]], dtype=np.float32)
    # нормализация для косинусной близости
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings

# ================= ВЕКТОРИЗАЦИЯ НОВЫХ ЧАНКОВ =================
print("Вычисление эмбеддингов для новых чанков...")
new_embeddings = get_embeddings(new_chunks, prompt_prefix="Document: ")

# ================= ДОБАВЛЕНИЕ В ИНДЕКС =================
if index is None:
    # создаём новый индекс, если его не было
    dim = new_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    print(f"Создан новый индекс размерности {dim}")

index.add(new_embeddings)
metadata.extend(new_metadata)

print(f"Добавлено векторов: {len(new_chunks)}. Всего векторов в индексе: {index.ntotal}")

# ================= СОХРАНЕНИЕ =================
faiss.write_index(index, INDEX_PATH)
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"Индекс сохранён в {INDEX_PATH}")
print(f"Метаданные сохранены в {META_PATH}")

# ================= ПРОВЕРОЧНЫЙ ПОИСК =================
print("\n--- Проверочный поиск по обновлённому индексу ---")
query = "результаты матчей НБА"
query_emb = get_embeddings([query], prompt_prefix="Query: ")
k = 3
distances, indices = index.search(query_emb, k)
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    meta = metadata[idx]
    print(f"{i+1}. (Сходство: {dist:.4f}) {meta['title']} -> {meta['text'][:100]}...")
print("Готово.")