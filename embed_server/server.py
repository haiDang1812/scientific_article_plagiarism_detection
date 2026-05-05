"""
Embedding Server module.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from loguru import logger
from sentence_transformers import SentenceTransformer

load_dotenv()


# ============================================================================
# CONFIG
# ============================================================================

_model = None
_client = None

COLLECTION_NAME = "pipeline-summaries"
CHROMA_MODE = os.getenv("CHROMA_MODE", "local").strip().lower()
CHROMA_LOCAL_PATH = os.getenv("CHROMA_LOCAL_PATH", "./.chroma_db")
CHROMA_HTTP_HOST = os.getenv("CHROMA_HTTP_HOST", "localhost")
CHROMA_HTTP_PORT = int(os.getenv("CHROMA_HTTP_PORT", "8001"))
CHROMA_HTTP_SSL = os.getenv("CHROMA_HTTP_SSL", "false").strip().lower() == "true"
CHROMA_CLOUD_TENANT = os.getenv("CHROMA_CLOUD_TENANT", "")
CHROMA_CLOUD_DATABASE = os.getenv("CHROMA_CLOUD_DATABASE", "")
CHROMA_CLOUD_API_KEY = os.getenv("CHROMA_CLOUD_API_KEY", "")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MIN_CHUNK_WORDS = 20
TXT_TOKEN_SIZE = 1024
TXT_TOKEN_OVERLAP = 256
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence_transformers").strip().lower()
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "")
LITELLM_TOKENIZER = os.getenv("LITELLM_TOKENIZER", "bert-base-uncased")


# ============================================================================
# OpenAI-compatible embedder via LiteLLM client
# ============================================================================


class LiteLLMEmbedder:
    """Wraps litellm.embedding() to call any self-hosted OpenAI-compatible model."""

    def __init__(self, model: str, api_key: str, base_url: str, tokenizer_name: str) -> None:
        if not model:
            raise ValueError("LITELLM_MODEL must be set when EMBEDDING_BACKEND='litellm'.")
        try:
            import litellm as _litellm
        except ImportError:
            raise ImportError("Install litellm: pip install litellm")
        from transformers import AutoTokenizer

        self._litellm = _litellm
        self._model = model
        self._api_key = api_key or None
        self._base_url = base_url or None

        # Try to load tokenizer matching the actual model name first.
        # model string may be "openai/BAAI/bge-m3" or "huggingface/intfloat/e5-large" etc.
        # Strip the provider prefix to get the HF repo id, then fall back to tokenizer_name.
        hf_model_id = re.sub(r"^[^/]+/", "", model, count=1) if "/" in model else model
        candidates = [hf_model_id, tokenizer_name]
        self.tokenizer = None
        for candidate in candidates:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(candidate)
                if candidate != tokenizer_name:
                    logger.info(f"LiteLLM tokenizer auto-detected from model id: {candidate}")
                break
            except Exception:
                continue
        if self.tokenizer is None:
            raise ValueError(
                f"Could not load tokenizer from '{hf_model_id}' or fallback '{tokenizer_name}'. "
                "Set LITELLM_TOKENIZER to a valid HuggingFace tokenizer name."
            )

    def encode(
        self,
        texts: List[str],
        normalize_embeddings: bool = True,
        **_,
    ) -> np.ndarray:
        response = self._litellm.embedding(
            model=self._model,
            input=texts,
            api_key=self._api_key,
            api_base=self._base_url,
            encoding_format="float",
        )
        embeddings = np.array([
            item["embedding"] if isinstance(item, dict) else item.embedding
            for item in response.data
        ])
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.where(norms == 0, 1, norms)
        return embeddings


# ============================================================================
# ChromaDB helpers
# ============================================================================


def build_chroma_client(mode: str) -> chromadb.ClientAPI:
    if mode == "local":
        return chromadb.PersistentClient(
            path=CHROMA_LOCAL_PATH,
            settings=Settings(anonymized_telemetry=False),
        )

    if mode == "http":
        return chromadb.HttpClient(
            host=CHROMA_HTTP_HOST,
            port=CHROMA_HTTP_PORT,
            ssl=CHROMA_HTTP_SSL,
            settings=Settings(anonymized_telemetry=False),
        )

    if mode == "cloud":
        if not CHROMA_CLOUD_TENANT:
            raise ValueError("CHROMA_CLOUD_TENANT is required when CHROMA_MODE='cloud'.")
        if not CHROMA_CLOUD_DATABASE:
            raise ValueError("CHROMA_CLOUD_DATABASE is required when CHROMA_MODE='cloud'.")
        if not CHROMA_CLOUD_API_KEY:
            raise ValueError("CHROMA_CLOUD_API_KEY is required when CHROMA_MODE='cloud'.")

        cloud_client_ctor = getattr(chromadb, "CloudClient", None)
        if cloud_client_ctor is None:
            raise ValueError(
                "Installed chromadb package does not expose CloudClient. "
                "Please update chromadb to a version that supports Chroma Cloud."
            )
        return cloud_client_ctor(
            tenant=CHROMA_CLOUD_TENANT,
            database=CHROMA_CLOUD_DATABASE,
            api_key=CHROMA_CLOUD_API_KEY,
        )

    raise ValueError("CHROMA_MODE must be 'local', 'http', or 'cloud'.")


def _get_model():
    global _model
    if _model is None:
        if EMBEDDING_BACKEND == "litellm":
            try:
                _model = LiteLLMEmbedder(
                    LITELLM_MODEL, LITELLM_API_KEY, LITELLM_BASE_URL, LITELLM_TOKENIZER
                )
                logger.info("Using LiteLLM embedding backend.")
            except Exception as exc:
                logger.warning(f"LiteLLM init failed ({exc}). Falling back to local SentenceTransformer.")
                _model = SentenceTransformer(EMBEDDING_MODEL)
        else:
            _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _client
    if _client is None:
        _client = build_chroma_client(CHROMA_MODE)
    return _client.get_or_create_collection(name=COLLECTION_NAME)


# ============================================================================
# Text helpers
# ============================================================================


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


def chunk_text(text: str, chunk_size: int, overlap: int, min_words: int) -> List[str]:
    """Chunk text into windows of <= chunk_size words with overlap."""
    words = normalize_text(text).split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [" ".join(words)] if len(words) >= min_words else []

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(words) - chunk_size + 1, step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if len(words[start : start + chunk_size]) >= min_words:
            chunks.append(chunk)

    # Final tail
    tail = words[start + step :]
    if tail:
        t = " ".join(tail).strip()
        if len(tail) >= min_words and t != chunks[-1] if chunks else True:
            chunks.append(t)

    return chunks


def chunk_text_by_tokens(
    text: str,
    token_size: int,
    overlap: int,
    tokenizer,
) -> List[str]:
    """Sentence-aware sliding-window chunk.

    Chunks always start at a sentence boundary. Sentences are accumulated
    until adding the next one would exceed token_size. Overlap is achieved
    by backtracking from the end of each chunk to include sentences whose
    combined token count approximates `overlap`.
    """
    text = normalize_text(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    # Batch-encode all sentences at once instead of one-by-one
    sent_lengths = [
        len(ids) for ids in tokenizer(
            sentences, add_special_tokens=False
        )["input_ids"]
    ]

    chunks: List[str] = []
    start = 0

    while start < len(sentences):
        total = 0
        end = start

        # Accumulate sentences until token budget is exhausted
        while end < len(sentences):
            if total + sent_lengths[end] > token_size and end > start:
                break
            total += sent_lengths[end]
            end += 1

        chunks.append(" ".join(sentences[start:end]))

        if end == len(sentences):
            break

        # Backtrack from end to find next start with ~overlap tokens
        overlap_acc = 0
        next_start = end
        while next_start > start + 1:
            overlap_acc += sent_lengths[next_start - 1]
            if overlap_acc >= overlap:
                break
            next_start -= 1

        start = next_start

    return chunks


# ============================================================================
# JSON -> records
# ============================================================================


def parse_pipeline_json(json_path: Path) -> Optional[Dict[str, Any]]:
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
            sections.append(
                {
                    "section_id": sec.get("section_id", ""),
                    "section_title": sec.get("title", ""),
                    "summary": summary,
                }
            )

    if not sections:
        return None

    return {"doc_id": doc_id, "title": title, "abstract": abstract, "sections": sections}


def build_records(record: Dict[str, Any]) -> List[tuple]:
    """
    Chunk summary + prefix with section_title -> (id, text_to_embed, metadata)
    """
    doc_id = record["doc_id"]
    title = record["title"]
    abstract = record["abstract"]

    records: List[tuple] = []
    for sec in record["sections"]:
        section_title = sec.get("section_title", "")
        summary = sec["summary"]

        chunks = chunk_text(summary, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_WORDS)

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


def _upsert_batch(collection, records: List[tuple], model, batch_size: int = 64) -> int:
    ids_all = [r[0] for r in records]
    docs_all = [r[1] for r in records]
    metas_all = [r[2] for r in records]

    # Embed in batches (HTTP round-trips), collect all embeddings first
    all_embeddings: List[np.ndarray] = []
    for idx in range(0, len(records), batch_size):
        docs = docs_all[idx : idx + batch_size]
        emb = model.encode(
            docs,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        all_embeddings.append(emb)
        logger.debug(f"Embedded batch {idx // batch_size + 1}/{-(-len(records) // batch_size)}")

    embeddings = np.concatenate(all_embeddings, axis=0)

    chroma_max = collection._client.get_max_batch_size()
    for idx in range(0, len(records), chroma_max):
        collection.upsert(
            ids=ids_all[idx : idx + chroma_max],
            documents=docs_all[idx : idx + chroma_max],
            metadatas=metas_all[idx : idx + chroma_max],
            embeddings=embeddings[idx : idx + chroma_max].tolist(),
        )

    return len(records)


# ============================================================================
# FastAPI app
# ============================================================================

app = FastAPI(
    title="Embedding Server",
    description="Embed section summaries from pipeline JSON into ChromaDB.",
    version="0.1.0",
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    where_filter: Optional[Dict] = None


class SearchResultItem(BaseModel):
    doc_id: str
    section_id: str
    section_title: str
    title: str
    abstract: str
    chunk_index: int
    chunk_text: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    total_in_db: int
    results: List[SearchResultItem]


class UpsertResponse(BaseModel):
    status: str
    upserted: int
    skipped: int
    doc_ids: List[str]


@app.get("/")
def root():
    info: Dict[str, Any] = {
        "service": "Embedding Server",
        "version": "0.1.0",
        "embedding_backend": EMBEDDING_BACKEND,
        "collection": COLLECTION_NAME,
        "chroma_mode": CHROMA_MODE,
    }
    if EMBEDDING_BACKEND == "litellm":
        info["litellm_model"] = LITELLM_MODEL
        info["litellm_base_url"] = LITELLM_BASE_URL
    else:
        info["model"] = EMBEDDING_MODEL
    return info


@app.post("/upsert/single", response_model=UpsertResponse)
async def upsert_single(json_path: str = Form(...)):
    path = Path(json_path).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {json_path}")

    record = parse_pipeline_json(path)
    if not record:
        raise HTTPException(status_code=422, detail="No valid sections found.")

    records = build_records(record)
    model = _get_model()
    collection = _get_collection()
    total = _upsert_batch(collection, records, model, BATCH_SIZE)

    return UpsertResponse(
        status="ok",
        upserted=total,
        skipped=0,
        doc_ids=[r[0] for r in records],
    )


@app.post("/upsert/batch", response_model=UpsertResponse)
async def upsert_batch(directory: str = Form(...), glob_pattern: str = Form("*.json")):
    dir_path = Path(directory).resolve()
    if not dir_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    json_files = sorted(dir_path.glob(glob_pattern))
    if not json_files:
        raise HTTPException(status_code=404, detail=f"No files matching '{glob_pattern}'")

    model = _get_model()
    collection = _get_collection()
    all_records: List[tuple] = []
    skipped = 0

    for f in json_files:
        record = parse_pipeline_json(f)
        if not record:
            skipped += 1
            continue
        all_records.extend(build_records(record))

    if not all_records:
        raise HTTPException(status_code=422, detail="No valid sections found in any file.")

    total = _upsert_batch(collection, all_records, model, BATCH_SIZE)

    return UpsertResponse(
        status="ok",
        upserted=total,
        skipped=skipped,
        doc_ids=[r[0] for r in all_records],
    )


@app.post("/upsert/upload", response_model=UpsertResponse)
async def upsert_upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files accepted.")

    try:
        data = json.loads(await file.read())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {exc}")

    sections: List[Dict] = []
    for sec in data.get("sections", []):
        summary = normalize_text(sec.get("summary", ""))
        if summary:
            sections.append(
                {
                    "section_id": sec.get("section_id", ""),
                    "section_title": sec.get("title", ""),
                    "summary": summary,
                }
            )

    if not sections:
        raise HTTPException(status_code=422, detail="No valid sections found.")

    record = {
        "doc_id": data.get("doc_id", ""),
        "title": data.get("title", ""),
        "abstract": data.get("abstract", ""),
        "sections": sections,
    }

    model = _get_model()
    collection = _get_collection()
    recs = build_records(record)
    total = _upsert_batch(collection, recs, model, BATCH_SIZE)

    return UpsertResponse(
        status="ok",
        upserted=total,
        skipped=0,
        doc_ids=[r[0] for r in recs],
    )


@app.post("/upsert/txt", response_model=UpsertResponse)
async def upsert_txt(
    file: UploadFile = File(...),
    doc_id: str = Form(""),
    token_size: int = Form(TXT_TOKEN_SIZE),
    overlap: int = Form(TXT_TOKEN_OVERLAP),
):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files accepted.")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    if not normalize_text(text):
        raise HTTPException(status_code=422, detail="File is empty or unreadable.")

    effective_doc_id = doc_id or Path(file.filename).stem

    model = _get_model()
    chunks = chunk_text_by_tokens(text, token_size, overlap, model.tokenizer)

    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks generated from file.")

    collection = _get_collection()
    records: List[tuple] = []
    for idx, chunk in enumerate(chunks):
        rec_id = f"{effective_doc_id}_chunk_{idx}"
        meta = {
            "doc_id": effective_doc_id,
            "title": effective_doc_id,
            "abstract": "",
            "section_id": "",
            "section_title": "",
            "chunk_index": idx,
            "chunk_text": _trunc(chunk),
        }
        records.append((rec_id, chunk, meta))

    total = _upsert_batch(collection, records, model, BATCH_SIZE)

    return UpsertResponse(
        status="ok",
        upserted=total,
        skipped=0,
        doc_ids=[r[0] for r in records],
    )


@app.post("/upsert/txt/batch", response_model=UpsertResponse)
async def upsert_txt_batch(
    directory: str = Form(...),
    token_size: int = Form(TXT_TOKEN_SIZE),
    overlap: int = Form(TXT_TOKEN_OVERLAP),
):
    dir_path = Path(directory).resolve()
    if not dir_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    txt_files = sorted(dir_path.glob("*.txt"))
    if not txt_files:
        raise HTTPException(status_code=404, detail="No .txt files found in directory.")

    model = _get_model()
    collection = _get_collection()
    all_records: List[tuple] = []
    skipped = 0

    for txt_path in txt_files:
        try:
            raw = txt_path.read_bytes()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
        except Exception as exc:
            logger.warning(f"Cannot read {txt_path.name}: {exc}")
            skipped += 1
            continue

        if not normalize_text(text):
            skipped += 1
            continue

        chunks = chunk_text_by_tokens(text, token_size, overlap, model.tokenizer)
        if not chunks:
            skipped += 1
            continue

        doc_id = txt_path.stem
        for idx, chunk in enumerate(chunks):
            rec_id = f"{doc_id}_chunk_{idx}"
            meta = {
                "doc_id": doc_id,
                "title": doc_id,
                "abstract": "",
                "section_id": "",
                "section_title": "",
                "chunk_index": idx,
                "chunk_text": chunk,
            }
            all_records.append((rec_id, chunk, meta))

    if not all_records:
        raise HTTPException(status_code=422, detail="No chunks generated from any file.")

    total = _upsert_batch(collection, all_records, model, BATCH_SIZE)

    return UpsertResponse(
        status="ok",
        upserted=total,
        skipped=skipped,
        doc_ids=[r[0] for r in all_records],
    )


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    collection = _get_collection()

    where = req.where_filter if req.where_filter else None
    raw = collection.query(
        query_texts=[req.query],
        n_results=req.top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]

    results = []
    for i in range(len(ids)):
        m = metas[i] if metas else {}
        dist = dists[i] if dists and dists[i] is not None else None
        results.append(
            SearchResultItem(
                doc_id=m.get("doc_id", ""),
                section_id=m.get("section_id", ""),
                section_title=m.get("section_title", ""),
                title=m.get("title", ""),
                abstract=m.get("abstract", ""),
                chunk_index=m.get("chunk_index", 0),
                chunk_text=m.get("chunk_text", ""),
                text=docs[i] if docs else "",
                score=round(float(dist), 6) if dist is not None else 0.0,
            )
        )

    return SearchResponse(
        query=req.query,
        total_in_db=collection.count(),
        results=results,
    )


@app.get("/stats")
def stats():
    try:
        collection = _get_collection()
        return {"collection": COLLECTION_NAME, "total_records": collection.count()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/reset")
def reset():
    global _client
    try:
        _get_collection()  # ensure _client is initialized
        _client.delete_collection(COLLECTION_NAME)
        _client.get_or_create_collection(name=COLLECTION_NAME)
        return {"status": "ok", "message": "Collection cleared."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
