from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "pipeline-summaries"
CHROMA_MODE = os.getenv("CHROMA_MODE", "local").strip().lower()
CHROMA_LOCAL_PATH = os.getenv("CHROMA_LOCAL_PATH", "./.chroma_db")
CHROMA_HTTP_HOST = os.getenv("CHROMA_HTTP_HOST", "localhost")
CHROMA_HTTP_PORT = int(os.getenv("CHROMA_HTTP_PORT", "8001"))
CHROMA_HTTP_SSL = os.getenv("CHROMA_HTTP_SSL", "false").strip().lower() == "true"
CHROMA_CLOUD_TENANT = os.getenv("CHROMA_CLOUD_TENANT", "")
CHROMA_CLOUD_DATABASE = os.getenv("CHROMA_CLOUD_DATABASE", "")
CHROMA_CLOUD_API_KEY = os.getenv("CHROMA_CLOUD_API_KEY", "")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence_transformers").strip().lower()
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "")
LITELLM_TOKENIZER = os.getenv("LITELLM_TOKENIZER", "bert-base-uncased")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
EMBED_MAX_WORKERS = int(os.getenv("EMBED_MAX_WORKERS", "8"))
TOKEN_SIZE = int(os.getenv("TOKEN_SIZE", "1024"))
TOKEN_OVERLAP = int(os.getenv("TOKEN_OVERLAP", "256"))
