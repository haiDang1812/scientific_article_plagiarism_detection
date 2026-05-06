from __future__ import annotations

import concurrent.futures
from typing import List

import chromadb
import numpy as np
from chromadb.config import Settings
from loguru import logger
from sentence_transformers import SentenceTransformer

from .config import (
    BATCH_SIZE,
    CHROMA_CLOUD_API_KEY,
    CHROMA_CLOUD_DATABASE,
    CHROMA_CLOUD_TENANT,
    CHROMA_HTTP_HOST,
    CHROMA_HTTP_PORT,
    CHROMA_HTTP_SSL,
    CHROMA_LOCAL_PATH,
    CHROMA_MODE,
    COLLECTION_NAME,
    EMBED_MAX_WORKERS,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    LITELLM_MODEL,
    LITELLM_TOKENIZER,
)
from .embedder import LiteLLMEmbedder

_model = None
_client = None


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


def _upsert_batch(collection, records: List[tuple], model, batch_size: int = BATCH_SIZE) -> int:
    ids_all = [r[0] for r in records]
    docs_all = [r[1] for r in records]
    metas_all = [r[2] for r in records]

    total_batches = -(-len(records) // batch_size)
    batch_slices = [docs_all[i : i + batch_size] for i in range(0, len(records), batch_size)]
    all_embeddings: List[np.ndarray] = [None] * total_batches

    def _embed_one(batch_idx: int) -> tuple:
        emb = model.encode(
            batch_slices[batch_idx],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        logger.debug(f"Embedded batch {batch_idx + 1}/{total_batches}")
        return batch_idx, emb

    with concurrent.futures.ThreadPoolExecutor(max_workers=EMBED_MAX_WORKERS) as executor:
        futures = [executor.submit(_embed_one, i) for i in range(total_batches)]
        for future in concurrent.futures.as_completed(futures):
            batch_idx, emb = future.result()
            all_embeddings[batch_idx] = emb

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
