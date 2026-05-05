import os
import time
import json
import requests

SERVER_URL = "http://localhost:8000"
DATA_DIR = "data"
JSON_DIR = "json"

def save_metadata_to_json(paper_info):
    """
    Hàm này mở tệp JSON tương ứng với nguồn của bài báo, 
    thêm mới thông tin metadata rồi ghi đè lại file JSON đó.
    """
    source = paper_info["source"]
    # Xác định file json tương ứng
    json_filename = f"{source}_json.json" if source in ["acl", "cvf"] else "arxiv_papers.json"
    json_path = os.path.join(JSON_DIR, json_filename)
    
    # Đọc dữ liệu JSON hiện tại (nếu file tồn tại)
    data = []
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pass
                
    # Tạo object metadata lưu trữ như cấu trúc hiện hữu của bạn
    # Sửa "paper_path" format giống với các file có sẵn (ví dụ: pdf\arxiv_paper\...)
    # Ở đây chúng ta map link path folder thực tế trong máy
    new_record = {
        "paper_name": paper_info["paper_name"],
        "year": "2026", 
        "conference_name_or_source": source.upper() if source != "arxiv" else "arXiv",
        "workshop_or_main_conference": "main conference" if source == "acl" else ("workshop" if source == "cvf" else "null"),
        "link_web": f"http://localhost:8000/download/{paper_info['filename']}?paper_id={paper_info['id']}",
        "paper_path": f"pdf\\{source}_paper\\{paper_info['filename']}"
    }
    
    # Append dữ liệu mới vào file Json hiện tại
    data.append(new_record)
    
    # Ghi lại file json đầy đủ cấu trúc
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def poll_server():
    """
    Client liên tục chọc vào API của Server theo chu kỳ để tải về những file chưa fetch
    """
    print(f"[Client] Bắt đầu polling từ server {SERVER_URL} mỗi 5 giây...")
    while True:
        try:
            response = requests.get(f"{SERVER_URL}/poll")
            if response.status_code == 200:
                new_papers = response.json().get("new_papers", [])
                
                if new_papers:
                    print(f"\n[*] Tìm thấy {len(new_papers)} papers mới!")
                    
                    for paper in new_papers:
                        source = paper["source"]
                        filename = paper["filename"]
                        paper_id = paper["id"]
                        
                        print(f"  -> Đang tải file {filename}...")
                        download_res = requests.get(f"{SERVER_URL}/download/{filename}?paper_id={paper_id}")
                        
                        if download_res.status_code == 200:
                            # 1. Lưu file PDF vào đúng folder (data/acl, data/arxiv, data/cvf)
                            target_dir = os.path.join(DATA_DIR, source)
                            os.makedirs(target_dir, exist_ok=True)
                            
                            file_save_path = os.path.join(target_dir, filename)
                            with open(file_save_path, "wb") as f:
                                f.write(download_res.content)
                            
                            # 2. Lưu vào JSON
                            save_metadata_to_json(paper)
                            print(f"  -> Thành công: Đã lưu {filename} và metadata vào {source}_json")
            
        except requests.exceptions.ConnectionError:
             pass 
        except Exception as e:
            print(f"[Client Error]: {e}")
            
        # Nghỉ 5 giây trước khi poll tiếp
        time.sleep(5)

if __name__ == "__main__":
    poll_server()