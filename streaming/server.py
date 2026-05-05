import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse

app = FastAPI()

# Thư mục lưu dữ liệu do người dùng upload lên (đóng vai trò như Database của Server)
API_DATA_DIR = "api_data"
os.makedirs(API_DATA_DIR, exist_ok=True)

# Bộ nhớ tạm (mock database) lưu trữ thông tin các file đã được upload
papers_db = []

@app.post("/upload")
async def upload_paper(
    file: UploadFile = File(...),
    paper_name: str = Form(...),
    source: str = Form(...) # acl, arxiv, hoặc cvf
):
    """
    API để client / user thực hiện upload paper lên server.
    File PDF sẽ được lưu trong folder `api_data/`
    """
    file_path = os.path.join(API_DATA_DIR, file.filename)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    paper_info = {
        "id": len(papers_db) + 1,
        "paper_name": paper_name,
        "source": source.lower(),
        "filename": file.filename,
        "polled": False # false nghĩa là file này chưa được fetch về máy đích
    }
    papers_db.append(paper_info)
    
    return {"message": "Upload thành công!", "paper": paper_info}

@app.get("/poll")
def poll_new_papers():
    """
    API cho phép Client gọi theo chu kỳ (polling) để xem có paper mới chưa được fetch hay không.
    """
    new_papers = [p for p in papers_db if not p["polled"]]
    return {"new_papers": new_papers}

@app.get("/download/{filename}")
def download_paper(filename: str, paper_id: int):
    """
    API cho phép Client tải file PDF và đánh dấu là đã được lưu về máy (polled = True)
    """
    file_path = os.path.join(API_DATA_DIR, filename)
    if os.path.exists(file_path):
        # Đánh dấu đã poll
        for p in papers_db:
            if p["id"] == paper_id:
                p["polled"] = True
                break
        return FileResponse(file_path)
    return {"error": "File not found"}

if __name__ == "__main__":
    # Khởi chạy server ở cổng 8000
    print("[Server] Mock API đang khởi động. Lắng nghe ở http://localhost:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)