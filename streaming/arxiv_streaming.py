import os
import json
import requests
import feedparser  


JSON_PATH = "papers.json"
SAVE_DIR = "arxiv_papers"
ARXIV_API = "http://export.arxiv.org/api/query?search_query=all:machine+learning&start=0&max_results=50"

os.makedirs(SAVE_DIR, exist_ok=True)

#LOAD JSON
if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        existing_papers = json.load(f)
else:
    existing_papers = []

#BUILD SET
existing_ids = set()

for p in existing_papers:
    if "pdf_url" in p:
        arxiv_id = p["pdf_url"].split("/")[-1].replace(".pdf", "")
        existing_ids.add(arxiv_id)

#DOWNLOAD 
def download_pdf(url, path):
    try:
        r = requests.get(url, stream=True, timeout=10)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        print("error", e)
        return False

#STREAM FROM ARXIV
feed = feedparser.parse(ARXIV_API)

for entry in feed.entries:

    #EXTRACT ID
    # ví dụ: http://arxiv.org/abs/1234.5678v1
    raw_id = entry.id.split("/")[-1]
    arxiv_id = raw_id.split("v")[0]

    #CHECK TRÙNG NGAY TẠI STREAM 
    if arxiv_id in existing_ids:
        print(f"⚡ Skip (exists): {arxiv_id}")
        continue

    # BUILD PDF URL
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    file_name = f"{arxiv_id}.pdf"
    save_path = os.path.join(SAVE_DIR, file_name)

    print(f"⬇️ Downloading: {arxiv_id}")

    ok = download_pdf(pdf_url, save_path)

    if not ok:
        continue

    #UPDATE JSON NGAY
    new_entry = {
        "paper_name": entry.title,
        "year": entry.published[:4],
        "conference_name": "arXiv",
        "workshop_or_main_conference": "preprint",
        "paper_path": save_path,
        "pdf_url": pdf_url
    }

    existing_papers.append(new_entry)
    existing_ids.add(arxiv_id)

    #SAVE
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_papers, f, indent=4, ensure_ascii=False)

print("✅ Done streaming!")