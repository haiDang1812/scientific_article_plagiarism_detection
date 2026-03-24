import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

# Edit these values for your day-to-day runs, then execute: python embed_abstract_to_chroma.py
DEFAULT_CONFIG: Dict[str, Any] = {
    "workspace_root": ".",
    "json_dir": "CVF_MAIN_FIXED/jsons",
    "collection_name": "cvf-paper-abstracts-local",
    "chroma_mode": "local",  # local | cloud
    "chroma_local_path": "./.chroma_test",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 64,
    "max_pages": 2,
    "min_abstract_chars": 120,
    "limit": 0,  # 0 = all
}

ABSTRACT_PATTERN = re.compile(
    r"(?is)\babstract\b[:\s-]*(.*?)(?=\b(?:1\.?\s*introduction|introduction|keywords|index terms)\b)"
)


def get_pymupdf_module():
    """Load PyMuPDF safely and give actionable errors for common package conflicts."""
    try:
        import pymupdf as fitz  # type: ignore

        return fitz
    except Exception:
        try:
            import fitz  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyMuPDF is not installed. Install it with: pip install pymupdf") from exc

        if not hasattr(fitz, "open") or not hasattr(fitz, "Matrix"):
            raise RuntimeError(
                "Detected incompatible 'fitz' package (not PyMuPDF). "
                "Run: pip uninstall -y fitz && pip install -U pymupdf"
            )

        return fitz


class PdfTextExtractor:
    @staticmethod
    def _extract_with_pymupdf(pdf_path: Path, max_pages: int) -> str:
        fitz = get_pymupdf_module()
        chunks: List[str] = []
        with fitz.open(pdf_path) as doc:
            for page_idx, page in enumerate(doc):
                if page_idx >= max_pages:
                    break
                chunks.append(page.get_text("text"))
        return "\n".join(chunks)

    def extract_text(self, pdf_path: Path, max_pages: int) -> str:
        return self._extract_with_pymupdf(pdf_path, max_pages)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_abstract(full_text: str, min_len: int = 120) -> Optional[str]:
    text = normalize_text(full_text)
    if not text:
        return None

    match = ABSTRACT_PATTERN.search(text)
    if match:
        abstract = normalize_text(match.group(1))
        if len(abstract) >= min_len:
            return abstract

    # Fallback: take initial content window if explicit abstract header is missing.
    head = text[:2500]
    if len(head) >= min_len:
        return head
    return None


def iter_metadata_files(json_dir: Path) -> Iterable[Path]:
    return sorted(json_dir.glob("*.json"))


def load_metadata(meta_path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Cannot read JSON {meta_path.name}: {exc}")
        return None


def resolve_pdf_path(meta: Dict[str, Any], workspace_root: Path) -> Optional[Path]:
    raw = meta.get("pdf_path")
    if not raw:
        return None

    pdf_path = Path(raw)
    if pdf_path.is_absolute() and pdf_path.exists():
        return pdf_path

    candidate = (workspace_root / pdf_path).resolve()
    if candidate.exists():
        return candidate

    candidate = (workspace_root / str(raw).replace("/", os.sep)).resolve()
    if candidate.exists():
        return candidate

    return None


def build_cloud_chroma_client() -> chromadb.ClientAPI:
    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")

    if not api_key or not tenant or not database:
        raise ValueError("Missing CHROMA_API_KEY, CHROMA_TENANT, or CHROMA_DATABASE environment variables.")

    if hasattr(chromadb, "CloudClient"):
        return chromadb.CloudClient(api_key=api_key, tenant=tenant, database=database)

    host = os.getenv("CHROMA_HOST")
    if not host:
        raise ValueError("Your chromadb package has no CloudClient. Set CHROMA_HOST for HttpClient cloud fallback.")

    port = int(os.getenv("CHROMA_PORT", "443"))
    ssl = os.getenv("CHROMA_SSL", "true").lower() in {"1", "true", "yes"}

    headers = {
        "x-chroma-token": api_key,
        "Authorization": f"Bearer {api_key}",
    }

    return chromadb.HttpClient(
        host=host,
        port=port,
        ssl=ssl,
        headers=headers,
        tenant=tenant,
        database=database,
    )


def build_local_chroma_client(local_path: str) -> chromadb.ClientAPI:
    # Persistent local storage is useful for quick offline testing.
    return chromadb.PersistentClient(path=local_path)


def build_chroma_client(mode: str, local_path: str) -> chromadb.ClientAPI:
    mode = mode.lower().strip()
    if mode == "local":
        return build_local_chroma_client(local_path)
    if mode == "cloud":
        return build_cloud_chroma_client()
    raise ValueError("Invalid --chroma-mode. Use 'local' or 'cloud'.")


def upsert_batches(
    collection: Any,
    records: List[Tuple[str, str, Dict[str, Any]]],
    model: SentenceTransformer,
    batch_size: int,
) -> int:
    total = 0
    for idx in range(0, len(records), batch_size):
        batch = records[idx : idx + batch_size]
        ids = [item[0] for item in batch]
        docs = [item[1] for item in batch]
        metas = [item[2] for item in batch]

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
        print(f"[INFO] Upserted {total}/{len(records)} records")

    return total


def run(args: argparse.Namespace) -> None:
    workspace_root = Path(args.workspace_root).resolve()
    json_dir = Path(args.json_dir).resolve()

    extractor = PdfTextExtractor()
    model = SentenceTransformer(args.embedding_model)

    client = build_chroma_client(mode=args.chroma_mode, local_path=args.chroma_local_path)
    collection = client.get_or_create_collection(name=args.collection_name)

    prepared: List[Tuple[str, str, Dict[str, Any]]] = []

    for meta_path in iter_metadata_files(json_dir):
        meta = load_metadata(meta_path)
        if meta is None:
            continue

        pdf_path = resolve_pdf_path(meta, workspace_root)
        if not pdf_path or not pdf_path.exists():
            print(f"[WARN] Missing PDF for {meta_path.name}")
            continue

        full_text = extractor.extract_text(pdf_path, max_pages=args.max_pages)
        abstract = extract_abstract(full_text, min_len=args.min_abstract_chars)
        if not abstract:
            print(f"[WARN] Cannot detect abstract from {pdf_path.name}")
            continue

        doc_id = meta_path.stem
        metadata = {
            "year": str(meta.get("year") or ""),
            "conference": meta.get("conference") or "",
            "track": meta.get("track") or "",
            "pdf_path": str(pdf_path).replace("\\", "/"),
            "json_file": meta_path.name,
            "source": "cvf",
        }

        prepared.append((doc_id, abstract, metadata))

        if args.limit and len(prepared) >= args.limit:
            break

    if not prepared:
        print("[INFO] No abstracts prepared. Nothing to upsert.")
        return

    inserted = upsert_batches(
        collection=collection,
        records=prepared,
        model=model,
        batch_size=args.batch_size,
    )

    print(f"[DONE] Uploaded {inserted} abstract embeddings to collection '{args.collection_name}'.")


def build_args_from_config() -> argparse.Namespace:
    return argparse.Namespace(**DEFAULT_CONFIG)


if __name__ == "__main__":
    run(build_args_from_config())
