import os
import re
import json
import time
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://aclanthology.org"
EVENTS_URL = "https://aclanthology.org/events/"

SAVE_DIR = "acl_paper"
JSON_FILE = "acl_json.json"

os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# Session + Retry setup
# ----------------------------

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

retry_strategy = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# ----------------------------
# Load existing JSON (resume)
# ----------------------------

if os.path.exists(JSON_FILE):
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    data = []

existing_titles = set(x["paper_name"] for x in data)

def save_json():
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ----------------------------
# Utilities
# ----------------------------

def sanitize(text):
    return "".join(c for c in text if c.isalnum() or c in " _-")[:150]

def safe_request(url):
    try:
        res = session.get(url, timeout=30)
        res.raise_for_status()
        return res
    except Exception as e:
        print("Request failed:", e)
        time.sleep(5)
        return None

def download_file(url, path):
    try:
        with session.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)

        time.sleep(random.uniform(1.5, 3.5))
        return True

    except Exception as e:
        print("Download failed:", e)
        time.sleep(5)
        return False

# ----------------------------
# Step 1: Get ACL event pages
# ----------------------------

def get_acl_events():
    print("Fetching ACL events...")

    res = safe_request(EVENTS_URL)
    if not res:
        return []

    soup = BeautifulSoup(res.text, "html.parser")

    events = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/events/acl-") and href.endswith("/"):
            events.append(urljoin(BASE, href))

    events = list(set(events))
    print("Total ACL events:", len(events))
    return events

# ----------------------------
# Step 2: Get volumes from event
# ----------------------------

def get_volumes(event_url):
    res = safe_request(event_url)
    if not res:
        return []

    soup = BeautifulSoup(res.text, "html.parser")

    volumes = []
    for a in soup.find_all("a", href=True):
        if a["href"].startswith("/volumes/"):
            volumes.append(urljoin(BASE, a["href"]))

    return list(set(volumes))

# ----------------------------
# Step 3: Get paper pages
# ----------------------------

def get_paper_links(volume_url):
    res = safe_request(volume_url)
    if not res:
        return []

    soup = BeautifulSoup(res.text, "html.parser")

    papers = []
    for a in soup.find_all("a", href=True):
        href = a["href"]

        # match paper id like /P23-1001/
        if re.match(r"^/[A-Z]\d+-\d+/?$", href):
            papers.append(urljoin(BASE, href))

    return list(set(papers))

# ----------------------------
# Step 4: Download paper
# ----------------------------

def process_paper(paper_url, year, conf_type):

    res = safe_request(paper_url)
    if not res:
        return

    soup = BeautifulSoup(res.text, "html.parser")

    title_tag = soup.find("h2", id="title")
    if not title_tag:
        return

    paper_name = title_tag.text.strip()

    if paper_name in existing_titles:
        return

    pdf_link = None
    for a in soup.find_all("a", href=True):
        if a["href"].endswith(".pdf"):
            pdf_link = urljoin(BASE, a["href"])
            break

    if not pdf_link:
        return

    safe_name = sanitize(paper_name)
    file_path = os.path.join(SAVE_DIR, f"{year}_{safe_name}.pdf")

    print("Downloading:", paper_name)

    success = download_file(pdf_link, file_path)
    if not success:
        return

    record = {
        "paper_name": paper_name,
        "year": year,
        "conference_name": "ACL",
        "workshop_or_main_conference": conf_type,
        "paper_path": file_path
    }

    data.append(record)
    existing_titles.add(paper_name)
    save_json()

# ----------------------------
# Main
# ----------------------------

def main():

    events = get_acl_events()

    for event in events:

        year = event.split("-")[-1].replace("/", "")
        print("\nProcessing event:", event)

        volumes = get_volumes(event)
        print("Found volumes:", len(volumes))

        for volume in volumes:

            if "workshop" in volume.lower():
                conf_type = "workshop"
            else:
                conf_type = "main conference"

            papers = get_paper_links(volume)
            print("Found papers:", len(papers))

            for paper in papers:
                process_paper(paper, year, conf_type)


if __name__ == "__main__":
    main()