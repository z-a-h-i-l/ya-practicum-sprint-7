# logger.py
import json
import datetime
from typing import Optional, List, Dict

LOG_FILE = "logs.jsonl"

def log_request(
    query: str,
    found_chunks: bool,
    answer: str,
    sources: List[Dict],
    success: bool,
    error: Optional[str] = None
) -> None:
    """Записывает один запрос в JSONL."""
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "found_chunks": found_chunks,
        "answer_length": len(answer) if answer else 0,
        "success_flag": success,
        "sources": sources,
        "error": error
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")