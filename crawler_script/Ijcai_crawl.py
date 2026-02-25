import os
import re
import json
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================
# CONFIG
# =====================
BASE_URL   = "https://www.ijcai.org/proceedings/2025/"
CONFERENCE = "IJCAI"
YEAR       = 2025

ROOT_DIR = "IJCAI2025"
PDF_DIR  = os.path.join(ROOT_DIR, "pdfs")
JSON_DIR = os.path.join(ROOT_DIR, "json")

MAX_WORKERS = 6

os.makedirs(PDF_DIR,  exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# =====================
# UTILS
# =====================
def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|,]', "", name)
    name = re.sub(r'\s+', "_", name.strip())
    return name[:180]


def get_prev_text_siblings(tag, n=5):
    texts = []
    sib = tag.previous_sibling
    while sib and len(texts) < n:
        if isinstance(sib, NavigableString):
            t = str(sib).strip()
            if t:
                texts.append(t)
        elif isinstance(sib, Tag):
            t = sib.get_text(strip=True)
            if t:
                texts.append(t)
        sib = sib.previous_sibling
    return texts


def detect_track(a_tag):
    for anc in a_tag.parents:
        h2 = anc.find_previous_sibling("h2")
        if h2:
            t = h2.get_text().lower()
            if "workshop" in t: return "workshop"
            if "survey"   in t: return "survey"
            if "demo"     in t: return "demo"
            return "main"
    return "main"


# =====================
# 1. COLLECT TASKS
# =====================
print("Parsing IJCAI 2025 page...")

html = requests.get(BASE_URL, headers=HEADERS, timeout=30).text
soup = BeautifulSoup(html, "html.parser")

tasks = []

pdf_links = soup.find_all("a", href=re.compile(r'^\d+\.pdf$'))
print(f"Found {len(pdf_links)} PDF links")

for a in pdf_links:
    pdf_href = a["href"]
    pdf_url  = urljoin(BASE_URL, pdf_href)

    parent = a.parent
    prev_texts = get_prev_text_siblings(parent)

    # prev_texts[1] = title, prev_texts[0] = authors
    title = prev_texts[1] if len(prev_texts) >= 2 else pdf_href.replace(".pdf", "")
    title = title.strip()

    safe_name = sanitize_filename(title)

    tasks.append({
        "title": title,
        "safe_name": safe_name,
        "pdf_url": pdf_url,
        "track": detect_track(a)
    })

print(f"Collected {len(tasks)} tasks\n")

# =====================
# 2. DOWNLOAD + WRITE
# =====================
def download_and_write(task):
    pdf_filename  = task["safe_name"] + ".pdf"
    json_filename = task["safe_name"] + ".json"

    pdf_path  = os.path.join(PDF_DIR,  pdf_filename)
    json_path = os.path.join(JSON_DIR, json_filename)

    # download pdf
    if not os.path.exists(pdf_path):
        r = requests.get(task["pdf_url"], headers=HEADERS, stream=True, timeout=60)
        r.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

    metadata = {
        "year": YEAR,
        "conference": CONFERENCE,
        "track": task["track"],
        "pdf_path": f"IJCAI{YEAR}/pdfs/{pdf_filename}"
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


# =====================
# 3. PARALLEL EXECUTION
# =====================
print("Downloading PDFs (parallel)...")

all_metadata = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download_and_write, t) for t in tasks]
    for f in as_completed(futures):
        try:
            all_metadata.append(f.result())
        except Exception as e:
            print("Error:", e)

# =====================
# 4. WRITE SUMMARY
# =====================
summary_path = os.path.join(ROOT_DIR, "all_papers.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)

print("\n✅ DONE")
print(f"   Papers       : {len(all_metadata)}")
print(f"   PDFs dir     : {PDF_DIR}")
print(f"   JSON dir     : {JSON_DIR}")
print(f"   Summary file : {summary_path}")