"""One shared sentence-transformers instance per model configuration.

Memory search, document query, and the optional startup preload each used to
construct their own ``SentenceTransformer``. The extra copies stay in RSS
after they are discarded because the weight buffers are not returned to the
OS. Callers in this process share the loaded module instead.

This module is imported only when a local embedding model is actually built.
Importing it loads torch, so it stays off the UI startup path.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from python.helpers.ram import configure_torch_threads, release_heap

_models: dict[str, SentenceTransformer] = {}
_lock = threading.Lock()


def _cache_key(model_name: str, kwargs: dict[str, Any]) -> str:
    return json.dumps(
        {"model": model_name, "kwargs": kwargs},
        sort_keys=True,
        default=str,
    )


def _loader_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    st_kwargs = dict(kwargs)
    raw_model_kwargs = st_kwargs.get("model_kwargs") or {}
    model_kwargs = dict(raw_model_kwargs) if isinstance(raw_model_kwargs, dict) else {}
    # Stream weights into the model instead of materializing a full extra copy
    # during from_pretrained. Callers can still override it.
    model_kwargs.setdefault("low_cpu_mem_usage", True)
    st_kwargs["model_kwargs"] = model_kwargs
    return st_kwargs


def load_model(model_name: str, kwargs: dict[str, Any]) -> SentenceTransformer:
    st_kwargs = _loader_kwargs(kwargs)
    key = _cache_key(model_name, st_kwargs)
    with _lock:
        cached = _models.get(key)
        if cached is not None:
            return cached
        configure_torch_threads()
        model = SentenceTransformer(model_name, **st_kwargs)
        model.eval()
        _models[key] = model
    # Drop the loader's temporary copies now, while nothing else needs them.
    release_heap()
    return model


def encode(model: SentenceTransformer, texts: list[str]) -> Any:
    # inference_mode skips autograd bookkeeping that would retain the batch.
    with torch.inference_mode():
        return model.encode(texts, convert_to_tensor=False)


def clear_cache() -> None:
    with _lock:
        _models.clear()
