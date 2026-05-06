from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .chunking import _trunc, build_records, chunk_text_by_tokens, normalize_text, parse_pipeline_json
from .config import BATCH_SIZE, CHROMA_MODE, COLLECTION_NAME, EMBEDDING_BACKEND, EMBEDDING_MODEL, LITELLM_BASE_URL, LITELLM_MODEL, TOKEN_OVERLAP, TOKEN_SIZE
from .embedding import _get_collection, _get_model, _upsert_batch

app = FastAPI(
    title="Embedding Server",
    description="Embed section summaries from pipeline JSON into ChromaDB.",
    version="0.1.0",
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


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
        all_records.extend(build_records(record, model.tokenizer))

    if not all_records:
        raise HTTPException(status_code=422, detail="No valid sections found in any file.")

    total = _upsert_batch(collection, all_records, model, BATCH_SIZE)

    return UpsertResponse(status="ok", upserted=total, skipped=skipped, doc_ids=[r[0] for r in all_records])


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
            sections.append({
                "section_id": sec.get("section_id", ""),
                "section_title": sec.get("title", ""),
                "summary": summary,
            })

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
    recs = build_records(record, model.tokenizer)
    total = _upsert_batch(collection, recs, model, BATCH_SIZE)

    return UpsertResponse(status="ok", upserted=total, skipped=0, doc_ids=[r[0] for r in recs])


@app.post("/upsert/txt", response_model=UpsertResponse)
async def upsert_txt(
    file: UploadFile = File(...),
    doc_id: str = Form(""),
    token_size: int = Form(TOKEN_SIZE),
    overlap: int = Form(TOKEN_OVERLAP),
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

    return UpsertResponse(status="ok", upserted=total, skipped=0, doc_ids=[r[0] for r in records])


@app.post("/upsert/txt/batch", response_model=UpsertResponse)
async def upsert_txt_batch(
    directory: str = Form(...),
    token_size: int = Form(TOKEN_SIZE),
    overlap: int = Form(TOKEN_OVERLAP),
    max_files: Optional[int] = Form(None),
):
    dir_path = Path(directory).resolve()
    if not dir_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    txt_files = sorted(dir_path.glob("*.txt"))
    if not txt_files:
        raise HTTPException(status_code=404, detail="No .txt files found in directory.")

    if max_files is not None and max_files > 0:
        txt_files = txt_files[:max_files]

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
            from loguru import logger
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

    return UpsertResponse(status="ok", upserted=total, skipped=skipped, doc_ids=[r[0] for r in all_records])


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([req.query], normalize_embeddings=True)

    raw = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=req.top_k,
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
        results.append(SearchResultItem(
            doc_id=m.get("doc_id", ""),
            section_id=m.get("section_id", ""),
            section_title=m.get("section_title", ""),
            title=m.get("title", ""),
            abstract=m.get("abstract", ""),
            chunk_index=m.get("chunk_index", 0),
            chunk_text=m.get("chunk_text", ""),
            text=docs[i] if docs else "",
            score=round(float(dist), 6) if dist is not None else 0.0,
        ))

    return SearchResponse(query=req.query, total_in_db=collection.count(), results=results)


@app.get("/stats")
def stats():
    try:
        collection = _get_collection()
        return {"collection": COLLECTION_NAME, "total_records": collection.count()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/reset")
def reset():
    from .embedding import _client
    try:
        _get_collection()
        _client.delete_collection(COLLECTION_NAME)
        _client.get_or_create_collection(name=COLLECTION_NAME)
        return {"status": "ok", "message": "Collection cleared."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
