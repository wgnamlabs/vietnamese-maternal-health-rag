import json
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path.cwd() / "src"))

from pipeline import RAGPipeline

def main():
    root = Path.cwd()
    data_path = root / "evaluation" / "datasets" / "eval_questions.json"
    with data_path.open(encoding="utf-8") as f:
        data = json.load(f)
    
    dev = [q for q in data if q.get("split") == "dev"]
    
    hard_ids = {"EVAL_QD_776_002", "EVAL_QD_1139_009", "EVAL_QD_1359_016"}
    ood = []
    hard = []
    easy = []
    
    for q in data:
        if q["id"] in hard_ids:
            hard.append(q)
        elif q["id"].startswith("OOD_"):
            if len(ood) < 2:
                ood.append(q)
        elif q.get("split") == "dev":
            if len(easy) < 5 and q["id"] not in hard_ids:
                easy.append(q)
                
    test_set = easy + ood + hard
    
    print("Khởi tạo RAG Pipeline và warm up...")
    pipeline = RAGPipeline(root)
    pipeline.warm_up()
    print("Warm up hoàn tất. Bắt đầu thử nghiệm sinh câu trả lời.\n")
    
    for i, q in enumerate(test_set, 1):
        print(f"\n{'='*50}")
        print(f"--- Câu {i} / 10 ---")
        print(f"ID: {q['id']}")
        print(f"Câu hỏi: {q['question']}")
        
        resp = pipeline.answer_query(q['question'])
        
        print(f"\nKết quả:")
        print(f"- Refused: {resp.refused} (Top Score: {resp.top_score:.4f})")
        print(f"- Chunks ({len(resp.chunks)}): {resp.chunks}")
        print(f"- Trả lời: {resp.answer}")

if __name__ == "__main__":
    main()
