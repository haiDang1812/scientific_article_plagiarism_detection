import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://www.ijcai.org"
ALL_PROC = f"{BASE}/all_proceedings"

OUTPUT_DIR = "ijcai_paper"
JSON_FILE = "ijcai_json.json"

HEADERS = {"User-Agent": "Mozilla/5.0"}


os.makedirs(OUTPUT_DIR, exist_ok=True)

if os.path.exists(JSON_FILE):
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)
else:
    all_data = []


def save_json():
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)


def sanitize(name):
    return "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()


def get_proceeding_pages():
    r = requests.get(ALL_PROC, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    pages = []

    container = soup.find("div", class_="field-item")
    for a in container.find_all("a"):
        if "proceedings" in a["href"]:
            pages.append(a["href"])

    return list(set(pages))


def extract_papers(page_url):
    r = requests.get(page_url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    papers = []

    year_match = re.search(r"(19|20)\d{2}", page_url)
    year = year_match.group() if year_match else "Unknown"
    conf_name = f"IJCAI {year}"

    paper_blocks = soup.find_all("div", class_="paper_wrapper")

    for block in paper_blocks:

        title_div = block.find("div", class_="title")
        if not title_div:
            continue

        title = title_div.get_text(strip=True)

        details_div = block.find("div", class_="details")
        if not details_div:
            continue

        pdf_link = details_div.find("a", href=re.compile(r"\.pdf$"))
        if not pdf_link:
            continue

        pdf_url = urljoin(page_url, pdf_link["href"])

        papers.append({
            "paper_name": title,
            "year": year,
            "conference_name": conf_name,
            "workshop_or_main": "main",
            "pdf_url": pdf_url
        })

    return papers


def download_pdf(url, path):
    r = requests.get(url, headers=HEADERS, stream=True)
    with open(path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)


def main():
    pages = get_proceeding_pages()
    print("Total proceedings pages:", len(pages))

    for relative_link in pages:

        page_url = urljoin(BASE, relative_link)
        print(f"\nProcessing: {page_url}")

        papers = extract_papers(page_url)
        print("Found papers:", len(papers))

        for paper in papers:

            safe_title = sanitize(paper["paper_name"])[:120]
            filename = f"{paper['year']}_{safe_title}.pdf"
            local_path = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(local_path):
                continue

            print("Downloading:", paper["paper_name"])

            try:
                download_pdf(paper["pdf_url"], local_path)

                all_data.append({
                    "paper_name": paper["paper_name"],
                    "year": paper["year"],
                    "conference_name": paper["conference_name"],
                    "workshop_or_main": paper["workshop_or_main"],
                    "paper_path": local_path
                })

                save_json()
                time.sleep(0.2)

            except Exception as e:
                print("Error:", e)


if __name__ == "__main__":
    main()
