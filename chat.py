import sys
import time
from pathlib import Path

# Thêm thư mục src vào sys.path để import được các module
sys.path.insert(0, str(Path.cwd() / "src"))

from pipeline import RAGPipeline

def main():
    root = Path.cwd()
    
    print("="*60)
    print("HỆ THỐNG HỎI ĐÁP Y TẾ - RAG PIPELINE v0")
    print("="*60)
    print("\nĐang khởi tạo mô hình và làm nóng (warm-up)... Vui lòng đợi.\n")
    
    try:
        pipeline = RAGPipeline(root)
        pipeline.warm_up()
    except Exception as e:
        print(f"Lỗi khi khởi tạo: {e}")
        return
        
    print("\n✅ Khởi tạo thành công! Sẵn sàng phục vụ.")
    print("Gõ 'exit', 'quit' hoặc nhấn Ctrl+C để thoát.")
    print("-"*60)
    
    while True:
        try:
            query = input("\nBạn: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "thoát"]:
                print("Tạm biệt!")
                break
                
            print("Bot đang xử lý...", end="\r")
            
            t0 = time.time()
            resp = pipeline.answer_query(query)
            t1 = time.time()
            
            print(" " * 30, end="\r") # Xóa dòng chữ đang xử lý
            
            print(f"Bot: {resp.answer}")
            
            # In thêm thông tin debug nhỏ
            print(f"\n(Thời gian xử lý: {t1-t0:.2f}s | Số đoạn trích: {len(resp.chunks)} | OOD: {resp.refused})")
            
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f"\n[Lỗi] Đã xảy ra lỗi trong quá trình xử lý: {e}")

if __name__ == "__main__":
    main()
