"""
extract_v2.py — Scientific paper PDF extractor.

Input:  PDF file(s) of scientific papers (WACV/CVPR/ICCV/NeurIPS/arXiv-style).
Output: JSON per paper, schema identical to json_output/:
    {
      "doc_id": str,
      "title": str,
      "abstract": str,
      "sections": [
        {
          "section_id": str,
          "title": str,
          "summary": str,          # full section text (NOT summarised)
          "has_citation": bool,
          "citation_count": int,
        }, ...
      ],
      "references": [
        {"ref_id": str, "raw": str}, ...
      ]
    }

Pipeline:
    PDF  ──(Docling)──▶  Markdown  ──(regex)──▶  sections + metadata  ──▶  JSON
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BATCH_FILE_TIMEOUT_SEC = 600          # per-file timeout in batch mode
DEVICE = os.environ.get("DOCLING_DEVICE", "cpu")   # "cpu" | "cuda" | "mps"
DO_OCR = os.environ.get("DOCLING_OCR", "1") == "1"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("extract_v2")


# ---------------------------------------------------------------------------
# Citation patterns (3 common academic styles)
# ---------------------------------------------------------------------------

_CITATION_PATTERNS = [
    r"\[\d+(?:,\s*\d+)*\]",                          # [1], [1, 2, 3]
    r"\([\w\s]+,\s*\d{4}[a-z]?\)",                   # (Author, 2020a)
    r"\([\w\s]+\s+et\s+al\.,\s*\d{4}[a-z]?\)",       # (Author et al., 2020)
]
CITATION_REGEX = re.compile("|".join(_CITATION_PATTERNS))


# ---------------------------------------------------------------------------
# CVF / publisher watermark (so we don't mistake it for the paper title)
# ---------------------------------------------------------------------------

_WATERMARK_REGEX = re.compile(
    r"open access|computer vision foundation|ieee xplore|watermark"
    r"|proceedings|conference on computer vision|arxiv|preprint",
    re.IGNORECASE,
)

_SKIP_TITLES = frozenset({
    "abstract", "introduction", "references", "conclusion",
    "related work", "acknowledgements", "acknowledgments",
    "preamble", "appendix", "supplementary material",
})

_NON_REF_PATTERNS = re.compile(
    r"^(?:appendix|supplementary|acknowledgement|acknowledgment|about the author)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Paragraph:
    text: str
    citations: tuple[str, ...]


@dataclass
class Section:
    title: str
    paragraphs: List[Paragraph]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)

    @property
    def citation_count(self) -> int:
        return sum(len(p.citations) for p in self.paragraphs)


# ---------------------------------------------------------------------------
# Docling wrapper
# ---------------------------------------------------------------------------


def _load_docling():
    """Import Docling lazily so the module can be inspected without it installed."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise ImportError(
            "Docling is required: pip install docling"
        ) from exc
    return DocumentConverter, InputFormat, PdfFormatOption, PdfPipelineOptions


def pdf_to_markdown(pdf_path: str) -> str:
    """Convert a PDF into Markdown via Docling."""
    logger.info("Converting PDF → markdown: %s", pdf_path)

    DocumentConverter, InputFormat, PdfFormatOption, PdfPipelineOptions = _load_docling()

    opts = PdfPipelineOptions()
    opts.accelerator_options.device = DEVICE
    opts.do_ocr = DO_OCR

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    result = converter.convert(pdf_path)
    document = result.document

    if hasattr(document, "export_to_markdown"):
        return document.export_to_markdown()
    return str(document)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _match_paragraph_citations(text: str) -> tuple[str, ...]:
    """Return all citation strings found in a paragraph."""
    found = CITATION_REGEX.findall(text)
    # findall returns str for single-group patterns
    return tuple(m if isinstance(m, str) else next(s for s in m if s) for m in found)


def _split_paragraphs(block: str) -> List[Paragraph]:
    """Split a text block into paragraphs on blank lines."""
    paragraphs: List[Paragraph] = []
    for chunk in re.split(r"\n\s*\n", block.strip()):
        chunk = chunk.strip()
        if not chunk:
            continue
        paragraphs.append(
            Paragraph(text=chunk, citations=_match_paragraph_citations(chunk))
        )
    return paragraphs


_MD_HEADER_REGEX = re.compile(r"^#+\s+(.+?)\s*#*\s*$")
_ROMAN_REGEX = re.compile(
    r"^\s*(?P<roman>M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))"
    r"(?:[\.)]|\s+)\s*(?P<title>.+)$",
    re.IGNORECASE,
)


def _match_md_header(line: str) -> Optional[str]:
    m = _MD_HEADER_REGEX.match(line)
    return m.group(1).strip() if m else None


def _match_roman_header(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None
    m = _ROMAN_REGEX.fullmatch(stripped)
    if not m:
        return None
    roman = m.group("roman").upper()
    title = m.group("title").strip(" .:-")
    if not roman or len(title) < 2:
        return None
    return f"{roman}. {title}"


def _match_special_header(line: str) -> Optional[str]:
    stripped = line.strip().lower().rstrip(":.-")
    if stripped == "abstract":
        return "Abstract"
    return None


def parse_sections(markdown: str) -> List[Section]:
    """
    Split Docling markdown into Section objects.

    Content before the first header is preserved as a 'Preamble' section.
    """
    sections: List[Section] = []
    current_title = "Preamble"
    current_lines: List[str] = []

    def flush() -> None:
        paragraphs = _split_paragraphs("\n".join(current_lines))
        sections.append(Section(title=current_title, paragraphs=paragraphs))

    for line in markdown.split("\n"):
        new_title = (
            _match_md_header(line)
            or _match_roman_header(line)
            or _match_special_header(line)
        )
        if new_title:
            flush()
            current_title = new_title
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    # Drop empty leading Preamble
    if sections and sections[0].title == "Preamble" and not sections[0].full_text.strip():
        sections.pop(0)

    return sections


# ---------------------------------------------------------------------------
# Metadata extractors
# ---------------------------------------------------------------------------


def extract_title(markdown: str) -> str:
    """Extract paper title from the first meaningful H1/H2 heading."""
    region = markdown[:4000]

    for m in re.finditer(r"^#{1,2}\s+(.+)$", region, re.MULTILINE):
        title = m.group(1).strip().rstrip("#").strip()
        if _WATERMARK_REGEX.search(title):
            continue
        if title.lower() in _SKIP_TITLES:
            continue
        if len(title) < 10:
            continue
        return title[:300]

    for line in region.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if len(stripped) > 30 and not _WATERMARK_REGEX.search(stripped):
            if not re.search(r"[@{]|^\d+\.", stripped):
                return stripped[:300]

    return "Untitled"


def extract_abstract(markdown: str) -> str:
    """Extract abstract text between 'Abstract' and the next major section."""
    region = markdown[:5000]
    pattern = re.compile(
        r"(?:Abstract|ABSTRACT)\s*[:\-]?\s*"
        r"([\s\S]+?)"
        r"(?=\n\s*\n|\n\s*(?:\d+\.?\s+)?(?:Introduction|INTRODUCTION|Keywords?|KEYWORDS))",
        re.IGNORECASE,
    )
    m = pattern.search(region)
    return m.group(1).strip()[:2000] if m else ""


def extract_references(markdown: str) -> List[Dict[str, str]]:
    """
    Extract references, joining multi-line entries.

    Each entry starts with '[N]' (or a numbered line). Lines without a leading
    marker are treated as continuations of the previous entry.
    """
    pattern = re.compile(
        r"(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*[:\-]?\s*([\s\S]+)$",
        re.IGNORECASE,
    )
    m = pattern.search(markdown)
    if not m:
        return []

    block = m.group(1).strip()
    lines = [ln.rstrip() for ln in block.split("\n") if ln.strip()]

    entries: List[str] = []
    current: List[str] = []
    # Accept optional markdown bullet (`-` / `*`) before [N] or N.
    entry_start = re.compile(r"^\s*(?:[-*]\s+)?(?:\[(\d+)\]|(\d+)\.)\s+")
    bullet_strip = re.compile(r"^\s*[-*]\s+")

    for line in lines:
        clean = bullet_strip.sub("", line).strip()
        if _NON_REF_PATTERNS.match(clean):
            break
        if entry_start.match(line):
            if current:
                entries.append(" ".join(current))
            current = [clean]
        else:
            if current:
                current.append(clean)
    if current:
        entries.append(" ".join(current))

    return [{"ref_id": str(i), "raw": raw} for i, raw in enumerate(entries, start=1)]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _generate_doc_id(pdf_path: str) -> str:
    stem = Path(pdf_path).stem
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", stem)
    return f"paper_{safe}_{datetime.now().strftime('%Y%m%d')}"


_EXCLUDE_SECTION_TITLES = frozenset({
    "references", "bibliography",
})


def _sections_to_output(sections: List[Section]) -> List[Dict]:
    filtered = [s for s in sections if s.title.strip().lower() not in _EXCLUDE_SECTION_TITLES]
    return [
        {
            "section_id": str(idx),
            "title": sec.title,
            "summary": sec.full_text,
            "has_citation": sec.citation_count > 0,
            "citation_count": sec.citation_count,
        }
        for idx, sec in enumerate(filtered, start=1)
    ]


def process_pdf(pdf_path: str, output_dir: str) -> Dict:
    """Full pipeline for one PDF; returns the output dict and writes JSON."""
    markdown = pdf_to_markdown(pdf_path)

    sections = parse_sections(markdown)
    output = {
        "doc_id": _generate_doc_id(pdf_path),
        "title": extract_title(markdown),
        "abstract": extract_abstract(markdown),
        "sections": _sections_to_output(sections),
        "references": extract_references(markdown),
    }

    out_path = Path(output_dir) / f"{Path(pdf_path).stem}_processed.json"
    save_json(output, out_path)
    return output


def save_json(data: Dict, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        logger.info("Saved → %s", path)
    except OSError as exc:
        logger.error("Could not save %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def _worker(args: tuple[str, str]) -> Dict:
    pdf_path, out_dir = args
    return process_pdf(pdf_path, out_dir)


def batch_process(pdf_dir: str, output_dir: str) -> List[Dict]:
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDFs found in %s", pdf_dir)
        return []

    logger.info("Found %d PDF(s). Timeout/file: %ds", len(pdf_files), BATCH_FILE_TIMEOUT_SEC)
    results: List[Dict] = []

    with ProcessPoolExecutor(max_workers=1) as pool:
        for idx, pdf in enumerate(pdf_files, start=1):
            logger.info("[%d/%d] %s", idx, len(pdf_files), pdf.name)
            future = pool.submit(_worker, (str(pdf), output_dir))
            try:
                results.append(future.result(timeout=BATCH_FILE_TIMEOUT_SEC))
            except FuturesTimeoutError:
                logger.warning("Timeout — skipping %s", pdf.name)
                future.cancel()
            except Exception as exc:
                logger.error("Failed %s: %s", pdf.name, exc)

    logger.info("Batch done: %d/%d succeeded", len(results), len(pdf_files))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_usage() -> None:
    print(
        "Usage:\n"
        "  Single:  python extract_v2.py <pdf_path> [output_dir]\n"
        "  Batch:   python extract_v2.py --batch <pdf_dir> [output_dir]\n"
        "  Default: python extract_v2.py            (uses ./pdf/ → ./json_output/)"
    )


def main(argv: List[str]) -> int:
    here = Path(__file__).parent
    default_pdf_dir = str(here / "pdf")
    default_out_dir = str(here / "json_output")

    if argv and argv[0] == "--batch":
        if len(argv) < 2:
            _print_usage()
            return 1
        pdf_dir = argv[1]
        out_dir = argv[2] if len(argv) > 2 else default_out_dir
        batch_process(pdf_dir, out_dir)
        return 0

    if argv:
        pdf_path = argv[0]
        out_dir = argv[1] if len(argv) > 1 else default_out_dir
        process_pdf(pdf_path, out_dir)
        return 0

    if Path(default_pdf_dir).exists():
        logger.info("Auto-processing %s", default_pdf_dir)
        batch_process(default_pdf_dir, default_out_dir)
        return 0

    _print_usage()
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
