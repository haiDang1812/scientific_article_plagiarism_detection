import os
import re
import json
import requests
from bs4 import BeautifulSoup, NavigableString
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================
# CONFIG
# =====================
BASE_URL = "https://proceedings.mlr.press/"
CONFERENCE = "PMLR"
TRACK = "main"

ROOT_DIR = "PMLR"
PDF_DIR = os.path.join(ROOT_DIR, "pdfs")
JSON_DIR = os.path.join(ROOT_DIR, "json")

MAX_WORKERS = 6
HEADERS = {"User-Agent": "Mozilla/5.0"}

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# =====================
# UTILS
# =====================
def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/*?\"<>|]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:200]

def detect_year(text: str):
    m = re.search(r"(19|20)\d{2}", text)
    return int(m.group()) if m else 0

def find_prev_text(tag):
    sib = tag.previous_sibling
    while sib:
        if isinstance(sib, NavigableString):
            t = sib.strip()
            if t:
                return t
        sib = sib.previous_sibling
    return None

# =====================
# 1. GET ALL VOLUMES
# =====================
print("🔍 Fetching PMLR index...")
html = requests.get(BASE_URL, headers=HEADERS, timeout=30).text
soup = BeautifulSoup(html, "html.parser")

volume_urls = sorted({
    urljoin(BASE_URL, a["href"] + "/")
    for a in soup.find_all("a", href=re.compile(r"^(v\d+|r\d+)$"))
})

print(f"📚 Found {len(volume_urls)} volumes\n")

# =====================
# 2. COLLECT PAPERS
# =====================
tasks = []

for vol_url in volume_urls:
    print(f"➡️  Scanning volume: {vol_url}")
    html = requests.get(vol_url, headers=HEADERS, timeout=30).text
    vsoup = BeautifulSoup(html, "html.parser")

    year = detect_year(vsoup.get_text(" ", strip=True))

    # ---- NEW STYLE: vXXX ----
    papers = vsoup.find_all("div", class_="paper")
    if papers:
        for p in papers:
            title_tag = p.find("p", class_="title")
            pdf_tag = p.find("a", href=re.compile(r"\.pdf$"))
            if not title_tag or not pdf_tag:
                continue

            title = title_tag.get_text(strip=True)
            tasks.append({
                "title": title,
                "pdf_url": urljoin(vol_url, pdf_tag["href"]),
                "year": year
            })
        continue

    # ---- OLD STYLE: rX ----
    for a in vsoup.find_all("a", string=re.compile("Download PDF", re.I)):
        title = find_prev_text(a)
        if not title:
            continue

        tasks.append({
            "title": title,
            "pdf_url": urljoin(vol_url, a["href"]),
            "year": year
        })

print(f"\n📄 Total papers collected: {len(tasks)}\n")

# =====================
# 3. DOWNLOAD + JSON
# =====================
def process(idx, task):
    title = task["title"]
    safe = sanitize_filename(title)
    pdf_url = task["pdf_url"]
    year = task["year"]

    print(f"[{idx}] ⬇️ {title}")

    pdf_path = os.path.join(PDF_DIR, f"{safe}.pdf")
    json_path = os.path.join(JSON_DIR, f"{safe}.json")

    if not os.path.exists(pdf_path):
        r = requests.get(pdf_url, headers=HEADERS, stream=True, timeout=60)
        r.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

    metadata = {
        "year": year,
        "conference": CONFERENCE,
        "track": TRACK,
        "pdf_path": pdf_path.replace("\\", "/")
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata

# =====================
# 4. PARALLEL EXECUTION
# =====================
all_metadata = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(process, i + 1, t): t
        for i, t in enumerate(tasks)
    }
    for f in as_completed(futures):
        all_metadata.append(f.result())

# =====================
# 5. SUMMARY
# =====================
summary_path = os.path.join(ROOT_DIR, "all_papers.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)

print("\n✅ DONE")
print(f"   Papers : {len(all_metadata)}")
print(f"   PDFs   : {PDF_DIR}")
print(f"   JSON   : {JSON_DIR}")
print(f"   Index  : {summary_path}")