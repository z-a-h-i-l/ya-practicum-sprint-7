import json
import sys
from rag_bot import search, ask_llm, build_user_prompt, SYSTEM_PROMPT  # импортируем из основного бота
from logger import log_request

THRESHOLD = 0.5
GOLDEN_FILE = "golden_questions.json"

def evaluate():
    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for item in questions:
        qid = item["id"]
        query = item["question"]
        known = item["known"]
        expected_keywords = item.get("expected_keywords", [])

        print(f"Тестирую вопрос #{qid}: {query}")

        # 1. Поиск
        retrieved = search(query)  # list of (dist, meta)
        found = any(dist >= THRESHOLD for dist, _ in retrieved)
        sources = [{"title": m["title"], "text": m["text"][:200], "distance": float(d)} 
                   for d, m in retrieved if d >= THRESHOLD]

        # 2. Запрос к LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query, retrieved)}
        ]
        try:
            answer = ask_llm(messages)
        except Exception as e:
            answer = f"Ошибка: {e}"

        # 3. Оценка корректности
        # Эвристика: если тема известна, то ответ должен содержать хотя бы одно ключевое слово
        # и не содержать фраз "не найдено". Если тема неизвестна, то должно быть "не найдено".
        answer_lower = answer.lower()
        if known:
            # успех, если есть ключевые слова или хотя бы найден чанк
            success = any(kw.lower() in answer_lower for kw in expected_keywords) and \
                      not any(ph in answer_lower for ph in ["не найдено", "нет информации"])
        else:
            # успех, если бот сообщает об отсутствии информации
            success = any(ph in answer_lower for ph in ["не найдено", "нет информации", "не могу ответить", "неизвестно"])

        # 4. Логирование
        log_request(query, found, answer, sources, success)

        results.append({
            "id": qid,
            "query": query,
            "known": known,
            "success": success,
            "answer": answer[:200] + "..."  # для отчёта только начало
        })

    # Вывод сводки
    total = len(results)
    correct = sum(1 for r in results if r["success"])
    print(f"\nИтого правильных ответов: {correct}/{total}")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"{status} Вопрос {r['id']}: {r['query'][:50]}... | known={r['known']}")

if __name__ == "__main__":
    evaluate()