from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import TOKEN_SIZE, TOKEN_OVERLAP


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _trunc(text: str, max_bytes: int = 4096) -> str:
    """Truncate string so its UTF-8 byte length stays within Chroma Cloud's limit."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def chunk_text_by_tokens(
    text: str,
    token_size: int,
    overlap: int,
    tokenizer,
) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    sent_lengths = [
        len(ids) for ids in tokenizer(sentences, add_special_tokens=False)["input_ids"]
    ]

    chunks: List[str] = []
    start = 0

    while start < len(sentences):
        total = 0
        end = start

        while end < len(sentences):
            if total + sent_lengths[end] > token_size and end > start:
                break
            total += sent_lengths[end]
            end += 1

        chunks.append(" ".join(sentences[start:end]))

        if end == len(sentences):
            break

        overlap_acc = 0
        next_start = end
        while next_start > start + 1:
            overlap_acc += sent_lengths[next_start - 1]
            if overlap_acc >= overlap:
                break
            next_start -= 1

        start = next_start

    return chunks


def parse_pipeline_json(json_path: Path) -> Optional[Dict[str, Any]]:
    from loguru import logger
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Cannot read {json_path.name}: {exc}")
        return None

    doc_id = data.get("doc_id", "")
    title = data.get("title", "")
    abstract = data.get("abstract", "")

    sections: List[Dict[str, str]] = []
    for sec in data.get("sections", []):
        summary = normalize_text(sec.get("summary", ""))
        if summary:
            sections.append({
                "section_id": sec.get("section_id", ""),
                "section_title": sec.get("title", ""),
                "summary": summary,
            })

    if not sections:
        return None

    return {"doc_id": doc_id, "title": title, "abstract": abstract, "sections": sections}


def build_records(record: Dict[str, Any], tokenizer) -> List[tuple]:
    doc_id = record["doc_id"]
    title = record["title"]
    abstract = record["abstract"]

    records: List[tuple] = []
    for sec in record["sections"]:
        section_title = sec.get("section_title", "")
        summary = sec["summary"]

        chunks = chunk_text_by_tokens(summary, TOKEN_SIZE, TOKEN_OVERLAP, tokenizer)

        for idx, chunk_body in enumerate(chunks):
            text_to_embed = f"{section_title} | {chunk_body}" if section_title else chunk_body
            sec_id = sec.get("section_id", str(uuid.uuid4())[:8])
            rec_id = f"{doc_id}_sec_{sec_id}_chunk_{idx}"
            meta = {
                "doc_id": doc_id,
                "title": _trunc(title),
                "abstract": _trunc(abstract),
                "section_id": sec.get("section_id", ""),
                "section_title": _trunc(section_title),
                "chunk_index": idx,
                "chunk_text": _trunc(chunk_body),
            }
            records.append((rec_id, text_to_embed, meta))

    return records
