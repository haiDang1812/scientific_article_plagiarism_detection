import requests
from bs4 import BeautifulSoup
import os
import json
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================
# CONFIG
# =====================
BASE_URL = "https://openaccess.thecvf.com"

ROOT_DIR = "CVF_MAIN_FIXED"
PDF_DIR = os.path.join(ROOT_DIR, "pdfs")
JSON_DIR = os.path.join(ROOT_DIR, "json")

MAX_WORKERS = 4

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# =====================
# 1. PARSE ROOT MENU (DD-BASED)
# =====================
print("Parsing CVF root menu...")

root_html = requests.get(BASE_URL, timeout=30).text
soup = BeautifulSoup(root_html, "html.parser")

conferences = []

for dd in soup.find_all("dd"):
    text = dd.get_text(" ", strip=True)

    if not any(k in text for k in ["CVPR", "ICCV", "WACV"]):
        continue

    head = text.split(",")[0]
    parts = head.split()
    if len(parts) != 2:
        continue

    conference, year = parts
    year = int(year)

    main_href = None
    for a in dd.find_all("a"):
        if a.text.strip() == "Main Conference":
            main_href = a["href"]
            break

    if not main_href:
        continue

    all_url = urljoin(BASE_URL + "/", main_href) + "?day=all"

    conferences.append({
        "conference": conference,
        "year": year,
        "all_url": all_url
    })

print(f"Discovered {len(conferences)} main conferences")

# =====================
# 2. COLLECT REMOTE PDF URLs
# =====================
tasks = []

for c in conferences:
    print(f"Scanning {c['conference']} {c['year']}")

    html = requests.get(c["all_url"], timeout=30).text
    soup = BeautifulSoup(html, "html.parser")

    for a in soup.find_all("a"):
        if a.text.strip().lower() == "pdf":
            remote_pdf_url = BASE_URL + a["href"]

            tasks.append({
                "conference": c["conference"],
                "year": c["year"],
                "remote_pdf_url": remote_pdf_url
            })

print(f"Total PDFs found: {len(tasks)}")

# =====================
# 3. DOWNLOAD PDF + WRITE JSON
# =====================
def download_and_write(task):
    remote_pdf_url = task["remote_pdf_url"]
    filename = remote_pdf_url.split("/")[-1]

    pdf_path = os.path.join(PDF_DIR, filename)
    json_path = os.path.join(JSON_DIR, filename.replace(".pdf", ".json"))

    if not os.path.exists(pdf_path):
        r = requests.get(remote_pdf_url, stream=True, timeout=60)
        r.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

    metadata = {
        "year": task["year"],
        "conference": task["conference"],
        "track": "main",
        "pdf_path": pdf_path.replace("\\", "/")
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

# =====================
# 4. PARALLEL EXECUTION
# =====================
print("Downloading PDFs...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download_and_write, t) for t in tasks]
    for _ in as_completed(futures):
        pass

print("DONE – PDFs downloaded, metadata uses LOCAL pdf_path only.")