from __future__ import annotations

import re
from typing import List

import numpy as np


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
        from loguru import logger

        self._litellm = _litellm
        self._model = model
        self._api_key = api_key or None
        self._base_url = base_url or None

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
