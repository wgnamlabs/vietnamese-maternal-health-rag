import re
import requests
from typing import List, Dict

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"

SYSTEM = (
    "Bạn là trợ lý cung cấp thông tin về sức khỏe bà mẹ và trẻ sơ sinh dựa trên "
    "hướng dẫn của Bộ Y tế. Chỉ trả lời dựa trên các đoạn tài liệu được cung cấp. "
    "Nếu tài liệu không đủ thông tin, hãy nói rõ là không đủ thông tin. "
    "Giữ nguyên số liệu và đơn vị như trong tài liệu. "
    "Ghi nguồn bằng [1], [2] sau mỗi ý. Không chẩn đoán, không kê đơn. "
    "Trả lời ngắn gọn bằng tiếng Việt."
)

def build_prompt(question: str, chunks: List[Dict]) -> str:
    docs = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, 1))
    return f"Tài liệu:\n{docs}\n\nCâu hỏi: {question}\n\nTrả lời:"

def clean_output(text: str) -> str:
    """Loại bỏ khối <think>...</think> nếu có."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

def generate(question: str, chunks: List[Dict]) -> str:
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
        "options": {
            "temperature": 0.1, 
            "repeat_penalty": 1.15,
            "num_ctx": 4096,
            "num_predict": 400
        },
        "keep_alive": "30m",
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    raw_content = r.json()["message"]["content"]
    return clean_output(raw_content)
