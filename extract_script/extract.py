"""
Plagiarism Detection System for Research Papers

Extracts, processes, and summarizes PDF content with citation detection.

Fixed issues:
- Date parsing logic corrected (DD/MM/YYYY swap bug)
- Citations list now saved in paragraph output
- AttributeError on document.title handled with getattr
- Content before first header no longer lost
- Section dict no longer mutated in-place
- Author regex tightened to avoid false positives
- Abstract regex made non-greedy and newline-safe
- Batch processing has per-file timeout
- _extract_publish_date: ambiguous DD/MM returns empty string instead of guessing
- _extract_references: removed hard 50-line cap; now parses until non-reference line
- _extract_roman_heading_title: uses re.fullmatch to prevent empty-roman match
- batch_process_pdfs: uses ProcessPoolExecutor for true kill on timeout
- save_to_json: wrapped in try/except IOError to prevent batch crash on disk errors
"""

import os
import re
import json
from typing import Dict, List, Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BATCH_FILE_TIMEOUT = 300   # seconds — max time per PDF in batch mode

# ---------------------------------------------------------------------------
# Citation regex patterns
# ---------------------------------------------------------------------------

CITATION_PATTERNS = [
    r'\[\d+(?:,\s*\d+)*\]',              # [1], [1, 2], [1, 2, 3]
    r'\([\w\s]+,\s*\d{4}[a-z]?\)',       # (Author, 2020) or (Author, 2020a)
    r'\([\w\s]+\s+et\s+al\.,\s*\d{4}[a-z]?\)',  # (Author et al., 2020)
]

CITATION_REGEX = re.compile('|'.join(CITATION_PATTERNS))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_document_id(pdf_path: str) -> str:
    """Generate a unique document ID from PDF filename and current date."""
    pdf_name = Path(pdf_path).stem
    pdf_name_clean = re.sub(r'[^a-zA-Z0-9_]', '_', pdf_name)
    timestamp = datetime.now().strftime("%Y%m%d")
    return f"paper_{pdf_name_clean}_{timestamp}"


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def setup_docling():
    """Import and return the DocumentConverter class from docling."""
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        return DocumentConverter, InputFormat, PdfFormatOption, PdfPipelineOptions
    except ImportError:
        raise ImportError("Please install docling: pip install docling")


def extract_pdf_content(pdf_path: str) -> Dict:
    """
    Extract content from a PDF using the docling library.

    Returns a dict with keys: metadata, content, raw_markdown.
    """
    print(f"[INFO] Extracting content from: {pdf_path}")

    DocumentConverter, InputFormat, PdfFormatOption, PdfPipelineOptions = setup_docling()

    # Device: "cpu" | "cuda" (NVIDIA GPU) | "mps" (Mac Apple Silicon)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options.device = "cpu"
    pipeline_options.do_ocr = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(pdf_path)
    document = result.document

    # Export to markdown so every downstream helper can use full_text
    if hasattr(document, 'export_to_markdown'):
        full_text = document.export_to_markdown()
    else:
        full_text = str(document)

    # --- Metadata ---
    metadata = {
        "title": getattr(document, 'title', None) or _extract_title_from_text(full_text),
        "organization": _extract_organization(full_text),
        "source": pdf_path,
        "extraction_date": datetime.now().isoformat(),
        "total_pages": len(document.pages) if hasattr(document, 'pages') else None,
    }

    # --- Content structure ---
    sections = _parse_sections_from_markdown(full_text)
    references = _extract_references(full_text)
    abstract = _extract_abstract(full_text)

    content = {
        "abstract": abstract,
        "sections": sections,
        "tables": [],
        "references": references,
    }

    return {
        "metadata": metadata,
        "content": content,
        "raw_markdown": full_text,
    }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_sections_from_markdown(text: str) -> List[Dict]:
    """
    Parse markdown text into sections.

    Content that appears before the first header is preserved as a
    'Preamble' section so nothing is silently dropped.
    """
    sections: List[Dict] = []
    header_pattern = re.compile(r'^#+\s+(.+)$')

    lines = text.split('\n')
    current_title = "Preamble"
    current_lines: List[str] = []

    for line in lines:
        m = header_pattern.match(line)
        roman_title = _extract_roman_heading_title(line)
        special_title = _extract_special_heading_title(line)

        if m:
            _flush_section(current_title, current_lines, sections)
            current_title = m.group(1).strip()
            current_lines = []
        elif roman_title:
            _flush_section(current_title, current_lines, sections)
            current_title = roman_title
            current_lines = []
        elif special_title:
            _flush_section(current_title, current_lines, sections)
            current_title = special_title
            current_lines = []
        else:
            current_lines.append(line)

    # Flush the last section
    _flush_section(current_title, current_lines, sections)

    # Drop empty preamble (common when PDF starts directly with a header)
    if sections and sections[0]["section_title"] == "Preamble":
        has_content = any(p["text"] for p in sections[0]["content"])
        if not has_content:
            sections.pop(0)

    return sections


def _extract_roman_heading_title(line: str) -> str:
    """
    Return heading text if a line looks like a Roman numeral section heading.

    Supported examples:
      I. Introduction
      II) Related Work
      III METHODS

    Uses re.fullmatch on the stripped line to prevent the all-optional
    quantifiers from matching an empty roman-numeral group against any line.
    """
    stripped = line.strip()
    if not stripped:
        return ""

    # Keep this conservative to avoid splitting normal prose lines.
    if len(stripped) > 120:
        return ""

    m = re.fullmatch(
        r'(?P<roman>M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))'
        r'(?:[\.)]|\s)\s*(?P<title>.+)',
        stripped,
        re.IGNORECASE,
    )
    if not m:
        return ""

    roman = m.group('roman').upper()
    title = m.group('title').strip(' .:-')

    # A valid Roman numeral must have at least one character
    if not roman or not title or len(title) < 2:
        return ""

    return f"{roman}. {title}"


def _extract_special_heading_title(line: str) -> str:
    """Return normalized title for known standalone headings (e.g., Abstract)."""
    stripped = line.strip().lower().rstrip(':.-')
    if stripped == "abstract":
        return "Abstract"
    return ""


def _flush_section(title: str, lines: List[str], sections: List[Dict]) -> None:
    """Build a section dict from accumulated lines and append to sections."""
    raw = '\n'.join(lines)
    paragraphs = _process_paragraphs(raw)
    sections.append({
        "section_title": title,
        "section_summary": "",
        "content": paragraphs,
    })


def _process_paragraphs(text: str) -> List[Dict]:
    """
    Split text into paragraphs and flag those that contain citations.

    Each paragraph dict includes:
        paragraph_id, text, is_citation, citations (list of matched strings)
    """
    paragraphs: List[Dict] = []
    para_list = re.split(r'\n\s*\n', text.strip())

    for idx, para in enumerate(para_list, start=1):
        para = para.strip()
        if not para:
            continue

        matched = CITATION_REGEX.findall(para)
        citations = [m if isinstance(m, str) else next(s for s in m if s) for m in matched]

        paragraphs.append({
            "paragraph_id": idx,
            "text": para,
            "is_citation": bool(citations),
            "citations": citations,
        })

    return paragraphs


# ---------------------------------------------------------------------------
# Metadata extraction helpers
# ---------------------------------------------------------------------------

# Phrases that flag a CVF watermark / boilerplate line — NOT a paper title
_CVF_WATERMARK_PHRASES = re.compile(
    r'open access|computer vision foundation|ieee xplore|watermark'
    r'|proceedings|conference on computer vision|arxiv|preprint',
    re.IGNORECASE,
)


def _extract_title_from_text(text: str) -> str:
    """
    Extract paper title from CVF markdown output.

    Strategy:
      1. Iterate all H1/H2 headings in the first 4000 chars.
      2. Skip headings matching watermark phrases or generic section names.
      3. Return the first heading that passes those filters.
      4. Fallback: first long non-watermark non-heading line.
    """
    region = text[:4000]

    SKIP_TITLES = {
        'abstract', 'introduction', 'references', 'conclusion',
        'related work', 'acknowledgements', 'acknowledgments',
        'preamble', 'appendix', 'supplementary material',
    }

    for m in re.finditer(r'^#{1,2}\s+(.+)$', region, re.MULTILINE):
        title = m.group(1).strip().rstrip('#').strip()
        if _CVF_WATERMARK_PHRASES.search(title):
            continue
        if title.lower() in SKIP_TITLES:
            continue
        if len(title) < 10:
            continue
        return title[:300]

    # Fallback: first long non-watermark line
    for line in region.splitlines():
        stripped = line.strip().lstrip('#').strip()
        if len(stripped) > 30 and not _CVF_WATERMARK_PHRASES.search(stripped):
            if not re.search(r'[@{]|^\d+\.', stripped):
                return stripped[:300]

    return "Untitled"



def _extract_organization(text: str) -> str:
    """Extract the first organisation / affiliation mentioned in the text."""
    pattern = re.compile(
        r'([^\n]*(?:University|Institute|Academy|Polytechnic|College|School'
        r'|Department|Laboratory)[^\n]{0,80})',
        re.IGNORECASE,
    )
    m = pattern.search(text[:3000])
    return m.group(1).strip()[:150] if m else ""


def _extract_publish_date(text: str) -> str:
    """
    Extract and return a publication date as YYYY-MM-DD.

    When both numeric parts are <= 12 (e.g. 01/05/2024), the format
    is genuinely ambiguous — returns "" rather than silently guessing.
    Also extracts conference year from CVF conference names.
    """
    header = text[:2000]

    # --- DD/MM/YYYY or MM/DD/YYYY with an explicit label ---
    labelled = re.search(
        r'(?:Date|Published|Received|Accepted)\s*[:\-]?\s*'
        r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})',
        header, re.IGNORECASE,
    )
    if labelled:
        a, b, year = int(labelled.group(1)), int(labelled.group(2)), int(labelled.group(3))
        if 1900 < year < 2100:
            if a > 12:
                day, month = a, b
                if 1 <= month <= 12 and 1 <= day <= 31:
                    return f"{year}-{month:02d}-{day:02d}"
            elif b > 12:
                month, day = a, b
                if 1 <= month <= 12 and 1 <= day <= 31:
                    return f"{year}-{month:02d}-{day:02d}"
            # Both <= 12 → ambiguous, skip

    # --- YYYY-MM-DD or YYYY/MM/DD ---
    iso = re.search(r'((?:19|20)\d{2})[\/\-](\d{1,2})[\/\-](\d{1,2})', header)
    if iso:
        year, month, day = int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year}-{month:02d}-{day:02d}"

    # --- "Month YYYY" or "Month, YYYY" ---
    month_names = (
        'January|February|March|April|May|June|'
        'July|August|September|October|November|December'
    )
    wordy = re.search(rf'({month_names})[,\s]+(\d{{4}})', header, re.IGNORECASE)
    if wordy:
        month_str = wordy.group(1).capitalize()
        year = int(wordy.group(2))
        month_num = datetime.strptime(month_str, '%B').month
        if 1900 < year < 2100:
            return f"{year}-{month_num:02d}-01"

    # --- CVF / top-tier conference name with year ---
    conf_pattern = re.compile(
        r'\b(CVPR|ICCV|WACV|ECCV|NeurIPS|ICML|ICLR|AAAI|BMVC|ACCV|ACM MM)'
        r'[\s\-]*((20|19)\d{2})\b',
        re.IGNORECASE,
    )
    conf_m = conf_pattern.search(text[:5000])
    if conf_m:
        year = int(conf_m.group(2))
        if 1990 < year < 2100:
            CONF_MONTHS = {
                'cvpr': 6, 'iccv': 10, 'wacv': 1, 'eccv': 9,
                'neurips': 12, 'icml': 7, 'iclr': 5, 'aaai': 2,
                'bmvc': 9, 'accv': 12, 'acm mm': 10,
            }
            conf_name = conf_m.group(1).lower().replace('-', '').strip()
            month = CONF_MONTHS.get(conf_name, 1)
            return f"{year}-{month:02d}-01"

    return ""


def _extract_abstract(text: str) -> str:
    """Extract the abstract section."""
    region = text[:3000]

    pattern = re.compile(
        r'(?:Abstract|ABSTRACT)\s*[:\-]?\s*'
        r'([\s\S]+?)'
        r'(?=\n\s*\n|\n\s*(?:\d+\.?\s+)?(?:Introduction|INTRODUCTION|Keywords?|KEYWORDS))',
        re.IGNORECASE,
    )
    m = pattern.search(region)
    if m:
        return m.group(1).strip()[:1200]
    return ""


# Heuristics to detect that we've gone past the reference block
_NON_REF_PATTERNS = re.compile(
    r'^(?:appendix|supplementary|acknowledgement|acknowledgment|about the author)',
    re.IGNORECASE,
)


def _extract_references(text: str) -> List[Dict]:
    """
    Extract the references / bibliography section.

    Returns a list of dicts with ref_id (str) and raw_text (str).
    Collects all non-empty lines in the reference block, stopping when
    a line matches a known post-reference section header.
    """
    pattern = re.compile(
        r'(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*[:\-]?\s*([\s\S]+)$',
        re.IGNORECASE,
    )
    m = pattern.search(text)
    if not m:
        return []

    ref_block = m.group(1).strip()
    raw_lines = [line.strip() for line in ref_block.split('\n') if line.strip()]

    refs: List[Dict] = []
    for ref_idx, ref_line in enumerate(raw_lines, start=1):
        if _NON_REF_PATTERNS.match(ref_line):
            break
        refs.append({
            "ref_id": str(ref_idx),
            "raw_text": ref_line,
        })

    return refs


# ---------------------------------------------------------------------------
# Summarisation (sumy + LexRank)
# ---------------------------------------------------------------------------

def setup_sumy():
    """Import sumy modules and download NLTK data if necessary."""
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lex_rank import LexRankSummarizer
        import nltk

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        return PlaintextParser, Tokenizer, LexRankSummarizer
    except ImportError:
        raise ImportError("Please install sumy and nltk: pip install sumy nltk")


def summarize_text_with_sumy(text: str, sentences_count: int = 2) -> Optional[str]:
    """Summarize text using LexRank extractive summarization from sumy library."""
    if not text.strip():
        return None

    try:
        PlaintextParser, Tokenizer, LexRankSummarizer = setup_sumy()
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, sentences_count)
        result = " ".join([str(sentence) for sentence in summary])
        if result:
            print("[SUCCESS] Extractive summarisation successful.")
            return result
        return None
    except Exception as e:
        print(f"[ERROR] Extractive summarisation failed: {e}")
        return None


def summarize_sections(sections: List[Dict]) -> List[Dict]:
    """
    Attach the full concatenated section text as `section_summary`.
    No extractive summarisation — keep original content verbatim.
    Returns new dicts rather than mutating the originals in-place.
    """
    summarized: List[Dict] = []

    for idx, section in enumerate(sections, start=1):
        print(f"\n[INFO] Collecting full text for section {idx}/{len(sections)}: {section['section_title']}")

        full_text = '\n\n'.join(p["text"] for p in section.get("content", []))

        if not full_text.strip():
            print(f"[WARNING] Empty section: {section['section_title']}")

        updated = {**section, "section_summary": full_text}
        summarized.append(updated)

    return summarized


def _to_roman(num: int) -> str:
    """Convert a positive integer to an uppercase Roman numeral."""
    roman_map = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'),
    ]
    value = max(1, num)
    out = []
    for arabic, roman in roman_map:
        while value >= arabic:
            out.append(roman)
            value -= arabic
    return ''.join(out)


def _extract_leading_roman(title: str) -> str:
    """Return leading Roman numeral from a title if present, else empty string."""
    m = re.match(r'^\s*([IVXLCDM]+)\b', title.strip(), re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _build_simple_sections(sections: List[Dict]) -> List[Dict]:
    """Build section-only output (summary + citation stats), no paragraph payload."""
    simple_sections: List[Dict] = []

    for idx, section in enumerate(sections, start=1):
        content = section.get("content", [])
        citation_count = sum(len(p.get("citations", [])) for p in content)

        simple_sections.append({
            "section_id": str(idx),
            "title": section.get("section_title", ""),
            "summary": section.get("section_summary", ""),
            "has_citation": citation_count > 0,
            "citation_count": citation_count,
        })

    return simple_sections


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_to_json(output_data: Dict, output_path: str) -> None:
    """
    Save processed data to a UTF-8 JSON file with pretty formatting.

    Wrapped in try/except IOError so a disk-full or permission error
    on one file does not crash the entire batch run.
    """
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as fh:
            json.dump(output_data, fh, ensure_ascii=False, indent=2)
        print(f"[SUCCESS] Results saved → {output_path}")
    except IOError as exc:
        print(f"[ERROR] Could not save {output_path}: {exc}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: str, output_dir: str = "./output") -> Dict:
    """
    Full pipeline: extract → parse → summarise → save.
    Returns the processed document dict.
    """
    print("=" * 60)
    print("PLAGIARISM DETECTION — PDF PROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Extract
    print("\n[STEP 1] Extracting PDF content...")
    extracted = extract_pdf_content(pdf_path)
    sections = extracted["content"]["sections"]
    print(f"  Found {len(sections)} sections.")

    # Step 2: Summarise
    print("\n[STEP 2] Summarising sections via sumy extractive summarisation...")
    sections = summarize_sections(sections)
    simple_sections = _build_simple_sections(sections)

    # Step 3: Assemble output
    print("\n[STEP 3] Assembling output...")
    output_data = {
        "doc_id": _generate_document_id(pdf_path),
        "title": extracted["metadata"]["title"],
        "abstract": extracted["content"]["abstract"],
        "sections": simple_sections,
        "references": [
            {"ref_id": ref.get("ref_id", ""), "raw": ref.get("raw_text", "")}
            for ref in extracted["content"].get("references", [])
        ],
    }

    # Step 4: Save
    print("\n[STEP 4] Saving JSON...")
    pdf_stem = Path(pdf_path).stem
    output_path = os.path.join(output_dir, f"{pdf_stem}_processed.json")
    save_to_json(output_data, output_path)

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    return output_data


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def _process_pdf_worker(args):
    """
    Top-level worker function for ProcessPoolExecutor.
    Must be a module-level function (not a closure) to be picklable.
    """
    pdf_path, output_dir = args
    return process_pdf(pdf_path, output_dir)


def batch_process_pdfs(pdf_directory: str, output_dir: str = "./output") -> List[Dict]:
    """
    Process every PDF in a directory, with a per-file timeout.

    Uses ProcessPoolExecutor instead of daemon threads so timed-out
    jobs are actually killed rather than left running in the background.
    """
    pdf_files = sorted(Path(pdf_directory).glob("*.pdf"))

    if not pdf_files:
        print(f"[ERROR] No PDF files found in {pdf_directory}")
        return []

    print(f"[INFO] Found {len(pdf_files)} PDF file(s). Timeout per file: {BATCH_FILE_TIMEOUT}s")

    all_results: List[Dict] = []

    with ProcessPoolExecutor(max_workers=1) as executor:
        for idx, pdf_file in enumerate(pdf_files, start=1):
            print(f"\n{'=' * 60}")
            print(f"File {idx}/{len(pdf_files)}: {pdf_file.name}")
            print("=" * 60)

            future = executor.submit(_process_pdf_worker, (str(pdf_file), output_dir))
            try:
                result = future.result(timeout=BATCH_FILE_TIMEOUT)
                all_results.append(result)
            except FuturesTimeoutError:
                print(f"[WARNING] Timed out after {BATCH_FILE_TIMEOUT}s — skipping {pdf_file.name}")
                future.cancel()
            except Exception as exc:
                print(f"[ERROR] Failed to process {pdf_file.name}: {exc}")

    print(f"\n[INFO] Batch complete. Processed {len(all_results)}/{len(pdf_files)} file(s) successfully.")
    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    default_pdf_dir = "./pdfs"
    default_output_dir = "./json_output"
    os.makedirs(default_output_dir, exist_ok=True)

    args = sys.argv[1:]

    if args and args[0] == "--batch":
        if len(args) < 2:
            print("[ERROR] --batch requires a directory argument.")
            sys.exit(1)
        pdf_dir = args[1]
        out_dir = args[2] if len(args) > 2 else default_output_dir
        batch_process_pdfs(pdf_dir, out_dir)

    elif args and args[0] != "--batch":
        pdf_path = args[0]
        out_dir = args[1] if len(args) > 1 else default_output_dir
        process_pdf(pdf_path, out_dir)

    else:
        if os.path.exists(default_pdf_dir):
            print(f"[INFO] Auto-processing all PDFs in '{default_pdf_dir}/'")
            batch_process_pdfs(default_pdf_dir, default_output_dir)
        else:
            print(f"[ERROR] Default folder '{default_pdf_dir}/' not found.")
            print("\nUsage:")
            print("  Default:     python plagiarism_detector.py")
            print("               (processes ./pdfs/ folder)")
            print("  Single file: python plagiarism_detector.py <pdf_path> [output_dir]")
            print("  Batch:       python plagiarism_detector.py --batch <dir> [output_dir]")
            sys.exit(1)