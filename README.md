# Scientific Article Plagiarism Detection

Hệ thống crawl, trích xuất, tóm tắt và (sẽ) phát hiện đạo văn cho các bài báo khoa học từ **arXiv**, **ACL Anthology**, **IJCAI** và các hội nghị CVF (CVPR/ICCV/WACV/ECCV).

> **Trạng thái hiện tại:** Pipeline *tiền xử lý* (crawl → parse → tóm tắt → JSON) đã hoạt động. Module *so sánh đạo văn* chưa được triển khai — xem [Roadmap](#roadmap) để biết kế hoạch hoàn thiện.

---

## Mục lục

- [Tính năng](#tính-năng)
- [Kiến trúc tổng quan](#kiến-trúc-tổng-quan)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Định dạng dữ liệu đầu ra](#định-dạng-dữ-liệu-đầu-ra)
- [Cấu hình](#cấu-hình)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)

---

## Tính năng

### Đã có
- **Crawl đa nguồn:** arXiv (CS recent), ACL Anthology, IJCAI proceedings — có resume, tránh tải trùng.
- **PDF parsing 2 backend:**
  - `PyMuPDF` — trích text theo block + toạ độ.
  - `PaddleOCR + PyMuPDF` — OCR cho hình ảnh/figure có chữ.
- **Pipeline chính** dùng [Docling](https://github.com/docling-project/docling) (GPU/CUDA) để convert PDF → Markdown có cấu trúc.
- **Phân tích:**
  - Tách section theo header `#`/`##` và số La Mã (`I.`, `II.` …).
  - Trích abstract, references, tổ chức/đơn vị, ngày xuất bản (xử lý được conference name như `CVPR 2023`).
  - Detect citation theo 3 pattern: `[1,2]`, `(Author, 2020)`, `(Author et al., 2020)`.
- **Tóm tắt extractive** mỗi section bằng **Sumy LexRank**.
- **Batch processing** với `ProcessPoolExecutor` + timeout 300s/file (kill được tiến trình treo).
- **Lưu JSON** có cấu trúc chuẩn, fail-safe (try/except IOError).

### Chưa có (xem [Roadmap](#roadmap))
- So sánh cặp (pairwise similarity) giữa các paper.
- Đánh chỉ mục (index) để tìm kiếm nhanh trên corpus lớn.
- API / UI để người dùng upload paper và nhận báo cáo đạo văn.
- Test suite.

---

## Kiến trúc tổng quan

```
┌──────────────┐     ┌──────────────┐     ┌────────────────────────┐
│  Crawlers    │ ──▶ │  PDFs trên   │ ──▶ │ plagiarism_detector.py │
│ (arxiv/acl/  │     │   đĩa local  │     │  (Docling + Sumy)      │
│   ijcai)     │     └──────────────┘     └──────────┬─────────────┘
└──────┬───────┘                                     │
       │ metadata                                    │ JSON có cấu trúc
       ▼                                             ▼
   json/*.json                                 json_output/*.json
                                                     │
                                                     ▼
                                   ┌──────────────────────────────┐
                                   │  [CHƯA CÓ] Similarity Engine │
                                   │  • TF-IDF / SBERT embeddings │
                                   │  • MinHash LSH cho n-gram    │
                                   │  • FAISS index               │
                                   └──────────────────────────────┘
```

---

## Cấu trúc thư mục

```
scientific_article_plagiarism_detection/
├── plagiarism_detector.py       # Pipeline chính (Docling + Sumy)
├── crawler_script/
│   ├── arxiv_crawler.py         # Crawl arXiv CS recent
│   ├── acl_crawler.py           # Crawl ACL Anthology (có retry)
│   └── ijcai_crawler.py         # Crawl IJCAI proceedings
├── pdf_parser/
│   ├── pdf_pymupdf.py           # Extract text block + bbox
│   ├── pdf_paddle.py            # + OCR cho figure có chữ
│   └── output_sample/           # Ví dụ output
├── json/                        # Metadata từ crawler
│   ├── arxiv_json.json
│   ├── acl_json.json
│   └── ijcai_json.json
├── json_output/                 # Paper đã xử lý xong
│   └── *_processed.json
├── pdfs/                        # (tự tạo) nơi đặt PDF cần xử lý
└── README.md
```

---

## Yêu cầu hệ thống

| Thành phần  | Khuyến nghị                                             |
|-------------|---------------------------------------------------------|
| Python      | 3.10 – 3.12                                             |
| GPU         | NVIDIA CUDA (cho Docling). Mac: đổi sang `mps`.         |
| RAM         | ≥ 8 GB (Docling load model)                             |
| Disk        | ≥ 5 GB nếu crawl corpus lớn                             |
| OS          | Windows / Linux / macOS                                 |

---

## Cài đặt

### 1. Tạo virtualenv

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Tạo `requirements.txt`

Dự án chưa có file này — tạo mới với nội dung sau:

```txt
# Core pipeline
docling>=2.0.0
sumy>=0.11.0
nltk>=3.8

# PDF parsing
pymupdf>=1.24.0
paddleocr>=2.7.0
paddlepaddle>=2.6.0     # hoặc paddlepaddle-gpu nếu có CUDA

# Crawlers
requests>=2.31.0
beautifulsoup4>=4.12.0
urllib3>=2.0.0

# (Tuỳ chọn cho roadmap)
# scikit-learn>=1.3.0
# sentence-transformers>=2.7.0
# faiss-cpu>=1.7.4
# datasketch>=1.6.0
```

```bash
pip install -r requirements.txt
```

### 3. Tải NLTK data (tự động chạy lần đầu)

Nếu muốn tải sẵn:
```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
```

### 4. Chọn thiết bị Docling

Mở [plagiarism_detector.py:88](plagiarism_detector.py#L88) và chỉnh:
```python
pipeline_options.accelerator_options.device = "cuda"   # NVIDIA
# hoặc "mps" cho Mac M1/M2, "cpu" nếu không có GPU
```

---

## Hướng dẫn sử dụng

### Bước 1 — Crawl PDF

Chạy crawler tương ứng với nguồn cần tải. Mỗi crawler tạo thư mục riêng + file JSON metadata.

```bash
# arXiv CS recent → thư mục recent_arxiv/, metadata papers.json
python crawler_script/arxiv_crawler.py

# ACL Anthology
python crawler_script/acl_crawler.py

# IJCAI
python crawler_script/ijcai_crawler.py
```

> Các crawler có **resume**: chạy lại sẽ bỏ qua paper đã tải.

### Bước 2 — Xử lý PDF

#### 2a. Xử lý một file

```bash
python plagiarism_detector.py path/to/paper.pdf
# Mặc định output vào ./json_output/
python plagiarism_detector.py path/to/paper.pdf ./my_output
```

#### 2b. Xử lý hàng loạt

```bash
# Mặc định: tất cả PDF trong ./pdfs/
python plagiarism_detector.py

# Chỉ định thư mục
python plagiarism_detector.py --batch ./recent_arxiv ./json_output
```

Batch mode có timeout **300 giây/file** ([`BATCH_FILE_TIMEOUT`](plagiarism_detector.py#L34)) — paper nào treo quá sẽ bị kill và skip.

#### 2c. Dùng trực tiếp PDF parser (không cần Docling)

```bash
python pdf_parser/pdf_pymupdf.py     # text block + bbox
python pdf_parser/pdf_paddle.py      # + OCR figures
```

---

## Định dạng dữ liệu đầu ra

### Metadata crawler (`json/*.json`)

```json
[
  {
    "paper_name": "Attention Is All You Need",
    "year": 2017,
    "conference_name": "NeurIPS",
    "workshop_or_main_conference": "main",
    "paper_path": "recent_arxiv/Attention_Is_All_You_Need.pdf"
  }
]
```

### Paper đã xử lý (`json_output/*_processed.json`)

```json
{
  "doc_id": "paper_Attention_Is_All_You_Need_20260420",
  "title": "Attention Is All You Need",
  "abstract": "The dominant sequence transduction models...",
  "sections": [
    {
      "section_id": "1",
      "title": "I. Introduction",
      "summary": "Recurrent neural networks... long established...",
      "has_citation": true,
      "citation_count": 7
    }
  ],
  "references": [
    { "ref_id": "1", "raw": "Bahdanau et al. 2015..." }
  ]
}
```

---

## Cấu hình

| Cấu hình                | Vị trí                                                                        | Ghi chú                          |
|-------------------------|-------------------------------------------------------------------------------|----------------------------------|
| Timeout/file            | [plagiarism_detector.py:34](plagiarism_detector.py#L34)                       | mặc định 300s                    |
| Thiết bị GPU            | [plagiarism_detector.py:88](plagiarism_detector.py#L88)                       | `cuda` / `mps` / `cpu`           |
| Số câu tóm tắt/section  | [plagiarism_detector.py:482](plagiarism_detector.py#L482) `sentences_count=2` |                                  |
| Citation patterns       | [plagiarism_detector.py:40-44](plagiarism_detector.py#L40-L44)                | thêm pattern mới nếu cần         |
| Output dir mặc định     | [plagiarism_detector.py:695](plagiarism_detector.py#L695) `./json_output`     |                                  |

---

## Roadmap

Đây là kế hoạch để biến repo thành một **hệ thống phát hiện đạo văn hoàn chỉnh**. Chia thành 5 phase, mỗi phase có mốc deliverable cụ thể.

### Phase 0 — Consolidation (1–2 ngày)
Mục tiêu: khoá base, dễ onboard.

- [ ] Thêm `requirements.txt` / `pyproject.toml` (đã có template ở mục [Cài đặt](#cài-đặt)).
- [ ] Thêm `.gitignore` (venv, PDF, `__pycache__`, `recent_arxiv/`, …).
- [ ] Thêm `config.py` gom các hằng số (timeout, device, paths).
- [ ] Cấu hình `logging` thay cho `print()`.
- [ ] `pytest` skeleton + unit test cho các hàm regex (`_extract_abstract`, `_extract_references`, citation detection).

### Phase 1 — Corpus Indexing (3–5 ngày)
Mục tiêu: từ `json_output/` xây index tìm kiếm nhanh.

- [ ] **Module `indexer/`:**
  - `indexer/tfidf_index.py` — scikit-learn `TfidfVectorizer` + `NearestNeighbors` cosine.
  - `indexer/minhash_index.py` — `datasketch` MinHash LSH trên n-gram (5-gram) để tìm trùng text literal.
  - `indexer/embedding_index.py` — `sentence-transformers` (model `all-MiniLM-L6-v2`) + FAISS cho semantic similarity.
- [ ] CLI: `python -m indexer build --input json_output/ --out index/`.
- [ ] Hỗ trợ index **incremental** (thêm paper mới không cần rebuild).

### Phase 2 — Plagiarism Detection Engine (5–7 ngày)
Mục tiêu: cho 1 paper input, trả về danh sách paper nghi đạo văn kèm bằng chứng.

- [ ] **`detector/` module:**
  - `detector/candidate_retrieval.py` — dùng TF-IDF/MinHash lọc top-K candidates (K=50).
  - `detector/pairwise_compare.py` — với mỗi candidate, so sánh **từng câu/đoạn**:
    - Exact match (n-gram overlap).
    - Semantic match (cosine similarity của sentence embeddings ≥ ngưỡng 0.85).
  - `detector/report.py` — sinh báo cáo JSON + HTML highlight các đoạn nghi ngờ.
- [ ] Metric đánh giá: precision/recall trên [PAN Plagiarism Corpus](https://pan.webis.de/data.html).
- [ ] Ngưỡng mặc định có thể cấu hình qua `config.py`.

### Phase 3 — API & CLI (2–3 ngày)

- [ ] **FastAPI service:**
  - `POST /upload` — nhận PDF, chạy pipeline, trả `doc_id`.
  - `POST /detect/{doc_id}` — chạy detection, trả báo cáo.
  - `GET /report/{doc_id}` — lấy báo cáo JSON/HTML.
- [ ] CLI thống nhất: `python -m plagiarism_detector detect paper.pdf`.
- [ ] Dockerfile + `docker-compose.yml`.

### Phase 4 — UI + Production (tuỳ chọn)

- [ ] Web UI (Streamlit hoặc Next.js) để upload + xem báo cáo.
- [ ] Persistence layer (PostgreSQL + pgvector, hoặc Qdrant/Weaviate).
- [ ] Auth + rate limit.
- [ ] Monitoring (Prometheus + Grafana).
- [ ] CI/CD (GitHub Actions).

### Phase 5 — Research Extensions

- [ ] Cross-lingual plagiarism detection (đa ngôn ngữ) bằng LaBSE/multilingual-SBERT.
- [ ] Paraphrase detection bằng mô hình fine-tune riêng.
- [ ] Citation context analysis — phân biệt "trích dẫn hợp lệ" vs "đạo văn trá hình".

---

## Troubleshooting

| Lỗi                                                   | Nguyên nhân & Khắc phục                                                  |
|-------------------------------------------------------|--------------------------------------------------------------------------|
| `ImportError: Please install docling`                 | `pip install docling`                                                    |
| `CUDA out of memory`                                  | Đổi `device = "cpu"` tại [plagiarism_detector.py:88](plagiarism_detector.py#L88) |
| `Timed out after 300s — skipping`                     | Tăng `BATCH_FILE_TIMEOUT`, hoặc PDF có vấn đề — kiểm tra riêng           |
| NLTK `punkt` not found                                | `python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"` |
| PaddleOCR lỗi lib trên Windows                        | Cài `Microsoft Visual C++ Redistributable 2019+`                         |
| Crawler bị chặn 429/403                               | Tăng `time.sleep(...)`, đổi User-Agent                                   |
| JSON output thiếu title                               | PDF có watermark CVF — đã xử lý ở [plagiarism_detector.py:274-316](plagiarism_detector.py#L274-L316), kiểm tra lại region 4000 ký tự đầu |

---

## Đóng góp

1. Fork repo, tạo nhánh từ `main`.
2. Viết test trước khi implement (TDD).
3. Format: `black . && ruff check .`.
4. PR có mô tả + test plan.

---

## License

(Chưa xác định — khuyến nghị MIT hoặc Apache-2.0 trước khi public.)
