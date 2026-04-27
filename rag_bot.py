import json
import requests
import numpy as np
import faiss

# ================= НАСТРОЙКИ =================
INDEX_PATH = "faiss_index_v5.bin"
META_PATH = "faiss_metadata_v5.json"
EMBED_SERVER_URL = "http://localhost:8080/v1/embeddings"
LLM_SERVER_URL = "http://localhost:8081/v1/chat/completions"

EMBED_PREFIX_QUERY = "Query: "   # префикс для поискового запроса
EMBED_PREFIX_DOC = "Document: "  # (использовался при индексации)
TOP_K = 3                        # сколько ближайших чанков извлекать

# ================= ЗАГРУЗКА ИНДЕКСА И МЕТАДАННЫХ =================
print("Загрузка FAISS индекса...")
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
print(f"Готово. Чанков в индексе: {index.ntotal}")

# ================= ФУНКЦИЯ ПОЛУЧЕНИЯ ЭМБЕДДИНГА =================
def get_embedding(text: str, prefix: str = EMBED_PREFIX_QUERY) -> np.ndarray:
    """Отправляет текст на эмбеддинг-сервер и возвращает нормализованный вектор."""
    response = requests.post(
        EMBED_SERVER_URL,
        json={"input": [f"{prefix}{text}"]},
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    emb = np.array(response.json()["data"][0]["embedding"], dtype=np.float32)
    # нормализация для косинусного сходства
    emb = emb / np.linalg.norm(emb)
    return emb.reshape(1, -1)

# ================= ПОИСК ПО ИНДЕКСУ =================
def search(query: str):
    """Возвращает список кортежей (distance, metadata) для ближайших чанков."""
    query_emb = get_embedding(query, EMBED_PREFIX_QUERY)
    distances, indices = index.search(query_emb, TOP_K)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((dist, metadata[idx]))
    return results

# ================= СИСТЕМНЫЙ ПРОМПТ С CoT И FEW-SHOT =================
SYSTEM_PROMPT = """Ты — спортивный аналитик, который помогает пользователям находить информацию о матчах и событиях NBA. 
Ты должен отвечать ТОЛЬКО на основе предоставленных фрагментов новостей. 
Если в документах нет ответа или информация не определена так и скажи.

Твои ответы должны следовать такому стилю: 
сначала ты объясняешь свои шаги (анализ документов), затем даёшь окончательный ответ. 
Это называется "Цепочка размышлений" (Chain-of-Thought). 
Всегда начинай с анализа предоставленных фрагментов.

Примеры твоих ответов (Few-shot):

Вопрос: Какая команда выиграла у «Лейкерс» в последнем матче?
Фрагменты новостей:
1. «Лейкерс» уступили «Хьюстону» со счётом 98:105 в матче, прошедшем 27 апреля.
2. «Бостон» разгромил «Филадельфию» 122:90.
Ответ: 
1. Анализ фрагментов: В первом фрагменте указано, что «Лейкерс» проиграли «Хьюстону». Во втором фрагменте упоминается другая игра, которая не относится к вопросу.
2. Следовательно, команда, обыгравшая «Лейкерс» — «Хьюстон».

Вопрос: Кто набрал трипл-дабл в матче «Атланта» — «Нью-Йорк»?
Фрагменты новостей:
1. «Нью-Йорк» обыграл «Атланту» (114:98). Трипл-дабл оформил Карл-Энтони Таунс.
2. В другом матче Йокич отметился дабл-даблом.
Ответ: 
1. Анализ: Первый фрагмент прямо говорит, что трипл-дабл у Таунса. Второй фрагмент нерелевантен.
2. Игрок с трипл-даблом — Карл-Энтони Таунс.

Теперь отвечай на следующий запрос пользователя, используя предоставленные фрагменты. 
Начинай с анализа, затем дай ответ."""

# ================= СБОРКА ПРОМПТА =================
def build_user_prompt(query: str, retrieved_chunks):
    """Формирует сообщение пользователя с контекстом."""
    context_str = ""
    for i, (_, meta) in enumerate(retrieved_chunks):
        context_str += f"{i+1}. {meta['text']}\n"
    return f"Вопрос: {query}\n\nФрагменты новостей:\n{context_str}\nОтвет:"

# ================= ЗАПРОС К LLM =================
def ask_llm(messages):
    """Отправляет сообщения в чат-сервер llama.cpp и возвращает ответ."""
    response = requests.post(
        LLM_SERVER_URL,
        json={
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 512,
            "stop": ["<|im_end|>", "<|endoftext|>"]
        },
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ================= ИНТЕРАКТИВНЫЙ ЦИКЛ =================
def main():
    print("\n🏀 RAG-бот НБА-новостей готов. Введите вопрос (или 'выход' для завершения).")
    while True:
        try:
            query = input("\n❓ Вопрос: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if query.lower() in ("выход", "exit", "quit"):
            print("До свидания!")
            break
        if not query:
            continue

        # 1. Поиск в FAISS
        results = search(query)

        # 2. Формируем диалог
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query, results)}
        ]

        # 3. Запрос к модели
        print("⏳ Бот размышляет...")
        try:
            answer = ask_llm(messages)
        except Exception as e:
            answer = f"Ошибка обращения к LLM: {str(e)}"

        # 4. Вывод ответа
        print("\n" + "="*60)
        print(answer)
        print("="*60)

if __name__ == "__main__":
    main()