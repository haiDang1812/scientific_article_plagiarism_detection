import os
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://arxiv.org/list/cs/recent"
BASE_DOMAIN = "https://arxiv.org"

OUTPUT_DIR = "recent_arxiv"
JSON_FILE = "papers.json"

HEADERS = {"User-Agent": "Mozilla/5.0"}


os.makedirs(OUTPUT_DIR, exist_ok=True)

if os.path.exists(JSON_FILE):
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        json_data = json.load(f)
else:
    json_data = []

existing_titles = set(item["paper_name"] for item in json_data)


def save_json():
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def sanitize(name):
    return "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()


def get_papers(skip):
    url = f"{BASE_URL}?skip={skip}&show=2000"
    print("Fetching:", url)

    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    papers = []

    dl = soup.find("dl")
    if not dl:
        return papers

    dt_tags = dl.find_all("dt")
    dd_tags = dl.find_all("dd")

    for dt, dd in zip(dt_tags, dd_tags):

        pdf_link = dt.find("a", title="Download PDF")
        if not pdf_link:
            continue

        pdf_url = urljoin(BASE_DOMAIN, pdf_link["href"])

        title_div = dd.find("div", class_="list-title")
        if not title_div:
            continue

        title = title_div.text.replace("Title:", "").strip()
        title = " ".join(title.split())

        papers.append({
            "paper_name": title,
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

    skip = 0

    while True:
        papers = get_papers(skip)

        if not papers:
            break

        print(f"Found {len(papers)} papers")

        for paper in papers:

            if paper["paper_name"] in existing_titles:
                continue

            safe = sanitize(paper["paper_name"])[:150]
            filename = f"{safe}.pdf"
            local_path = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(local_path):
                continue

            print("Downloading:", paper["paper_name"])

            try:
                download_pdf(paper["pdf_url"], local_path)

                record = {
                    "paper_name": paper["paper_name"],
                    "paper_path": local_path
                }

                json_data.append(record)
                existing_titles.add(paper["paper_name"])

                save_json() 

                time.sleep(0.3)

            except Exception as e:
                print("Error:", e)

        skip += len(papers)


if __name__ == "__main__":
    main()
