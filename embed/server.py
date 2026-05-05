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
import uvicorn
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
CHROMA_CLOUD_TENANT = os.getenv("CHROMA_CLOUD_TENANT", "b2272094-71db-4499-8493-f2f113d76080")
CHROMA_CLOUD_DATABASE = os.getenv("CHROMA_CLOUD_DATABASE", "testing")
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
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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
        )
        embeddings = np.array([item.embedding for item in response.data])
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.where(norms == 0, 1, norms)
        return embeddings


# ============================================================================
# ChromaDB helpers
# ============================================================================


def build_chroma_client(
    mode: str,
    local_path: str,
    cloud_tenant: str,
    cloud_database: str,
    cloud_api_key: str,
) -> chromadb.ClientAPI:
    if mode == "local":
        return chromadb.PersistentClient(
            path=local_path,
            settings=Settings(anonymized_telemetry=False),
        )

    if mode == "cloud":
        if not cloud_tenant:
            raise ValueError("CHROMA_CLOUD_TENANT is required when CHROMA_MODE='cloud'.")
        if not cloud_database:
            raise ValueError("CHROMA_CLOUD_DATABASE is required when CHROMA_MODE='cloud'.")
        if not cloud_api_key:
            raise ValueError("CHROMA_CLOUD_API_KEY is required when CHROMA_MODE='cloud'.")

        cloud_client_ctor = getattr(chromadb, "CloudClient", None)
        if cloud_client_ctor is None:
            raise ValueError(
                "Installed chromadb package does not expose CloudClient. "
                "Please update chromadb to a version that supports Chroma Cloud."
            )

        try:
            return cloud_client_ctor(
                tenant=cloud_tenant,
                database=cloud_database,
                api_key=cloud_api_key,
            )
        except TypeError:
            return cloud_client_ctor(cloud_tenant, cloud_database, cloud_api_key)

    raise ValueError("CHROMA_MODE must be either 'local' or 'cloud'.")


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
        _client = build_chroma_client(
            CHROMA_MODE,
            CHROMA_LOCAL_PATH,
            CHROMA_CLOUD_TENANT,
            CHROMA_CLOUD_DATABASE,
            CHROMA_CLOUD_API_KEY,
        )
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

    # Precompute token count per sentence to avoid repeated encoding
    sent_lengths = [
        len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences
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
                "title": title,
                "abstract": abstract,
                "section_id": sec.get("section_id", ""),
                "section_title": section_title,
                "chunk_index": idx,
                "chunk_text": chunk_body,
            }
            records.append((rec_id, text_to_embed, meta))

    return records


def _upsert_batch(collection, records: List[tuple], model, batch_size: int = 64) -> int:
    total = 0
    for idx in range(0, len(records), batch_size):
        batch = records[idx : idx + batch_size]
        ids = [r[0] for r in batch]
        docs = [r[1] for r in batch]
        metas = [r[2] for r in batch]

        embeddings = model.encode(
            docs,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings.tolist(),
        )
        total += len(batch)
    return total


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
            "chunk_text": chunk,
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


def run_from_cli() -> None:
    import argparse

    global COLLECTION_NAME, CHROMA_MODE, CHROMA_LOCAL_PATH
    global CHROMA_CLOUD_TENANT, CHROMA_CLOUD_DATABASE, CHROMA_CLOUD_API_KEY
    global EMBEDDING_MODEL, BATCH_SIZE, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_WORDS
    global EMBEDDING_BACKEND, LITELLM_API_KEY, LITELLM_BASE_URL, LITELLM_MODEL, LITELLM_TOKENIZER

    parser = argparse.ArgumentParser(description="Embedding FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--chroma-mode", choices=["local", "cloud"], default=CHROMA_MODE)
    parser.add_argument("--chroma-path", default=CHROMA_LOCAL_PATH)
    parser.add_argument("--chroma-cloud-tenant", default=CHROMA_CLOUD_TENANT)
    parser.add_argument("--chroma-cloud-database", default=CHROMA_CLOUD_DATABASE)
    parser.add_argument("--chroma-cloud-api-key", default=CHROMA_CLOUD_API_KEY)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--min-chunk-words", type=int, default=MIN_CHUNK_WORDS)
    parser.add_argument("--embedding-backend", choices=["sentence_transformers", "litellm"], default=EMBEDDING_BACKEND)
    parser.add_argument("--litellm-api-key", default=LITELLM_API_KEY)
    parser.add_argument("--litellm-base-url", default=LITELLM_BASE_URL)
    parser.add_argument("--litellm-model", default=LITELLM_MODEL)
    parser.add_argument("--litellm-tokenizer", default=LITELLM_TOKENIZER)
    args = parser.parse_args()

    COLLECTION_NAME = args.collection
    CHROMA_MODE = args.chroma_mode
    CHROMA_LOCAL_PATH = args.chroma_path
    CHROMA_CLOUD_TENANT = args.chroma_cloud_tenant
    CHROMA_CLOUD_DATABASE = args.chroma_cloud_database
    CHROMA_CLOUD_API_KEY = args.chroma_cloud_api_key
    EMBEDDING_MODEL = args.model
    BATCH_SIZE = args.batch_size
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    MIN_CHUNK_WORDS = args.min_chunk_words
    EMBEDDING_BACKEND = args.embedding_backend
    LITELLM_API_KEY = args.litellm_api_key
    LITELLM_BASE_URL = args.litellm_base_url
    LITELLM_MODEL = args.litellm_model
    LITELLM_TOKENIZER = args.litellm_tokenizer

    logger.info("=" * 60)
    logger.info("EMBEDDING SERVER")
    logger.info("=" * 60)
    logger.info(f"  Backend      : {EMBEDDING_BACKEND}")
    if EMBEDDING_BACKEND == "litellm":
        logger.info(f"  LiteLLM model: {LITELLM_MODEL}")
        logger.info(f"  LiteLLM URL  : {LITELLM_BASE_URL}")
    else:
        logger.info(f"  Model        : {EMBEDDING_MODEL}")
    logger.info(f"  Collection   : {COLLECTION_NAME}")
    logger.info(f"  Chroma mode  : {CHROMA_MODE}")
    if CHROMA_MODE == "local":
        logger.info(f"  ChromaDB     : {CHROMA_LOCAL_PATH}")
    else:
        logger.info(f"  Tenant/DB    : {CHROMA_CLOUD_TENANT}/{CHROMA_CLOUD_DATABASE}")
    logger.info(f"  Chunking     : size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, min_words={MIN_CHUNK_WORDS}")
    logger.info(f"  Server       : http://{args.host}:{args.port}")
    logger.info("=" * 60)

    uvicorn.run(
        "embed.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    run_from_cli()
