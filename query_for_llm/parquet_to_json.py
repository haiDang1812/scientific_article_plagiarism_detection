import pyarrow.parquet as pq
import json
import sys
import os

def convert_parquet_to_json(parquet_file_path="input.parquet", json_file_path="input.json"):
    try:
        # Kiểm tra file tồn tại
        if not os.path.exists(parquet_file_path):
            print(f"Lỗi: Không tìm thấy file '{parquet_file_path}'")
            return None

        # Đọc file parquet
        table = pq.read_table(parquet_file_path)
        records = table.to_pylist()
        
        # Ghi ra file JSON dạng danh sách (list of objects)
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=4)
                
        print(f"Chuyển đổi thành công: {parquet_file_path} -> {json_file_path}")
        return json.dumps(records, ensure_ascii=False, indent=4)
        
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        return None

if __name__ == "__main__":
    # Nếu chạy script không truyền tham số, sẽ tự động dùng "input.parquet" -> "input.json"
    if len(sys.argv) == 3:
        parquet_file = sys.argv[1]
        json_file = sys.argv[2]
        convert_parquet_to_json(parquet_file, json_file)
    else:
        print("Đang chạy với cấu hình mặc định: input.parquet -> input.json")
        convert_parquet_to_json()
