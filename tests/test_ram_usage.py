"""RAM-tuning behavior that does not download or load real models."""

import ast
import asyncio
import os
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.helpers.ram import (
    RAM_DEFAULTS,
    IdleGuard,
    apply_ram_defaults,
    local_model_idle_seconds,
    preload_local_embeddings,
    reexec_if_needed,
)


def _clear_ram_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in RAM_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("A0_RAM_REEXEC", raising=False)
    monkeypatch.delenv("A0_DISABLE_RAM_TUNE", raising=False)
    monkeypatch.delenv("A0_PRELOAD_EMBEDDINGS", raising=False)
    monkeypatch.delenv("A0_LOCAL_MODEL_IDLE_SECONDS", raising=False)


def test_apply_ram_defaults_fills_only_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ram_env(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")

    assert apply_ram_defaults() is True
    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert os.environ["MALLOC_ARENA_MAX"] == "2"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"

    assert apply_ram_defaults() is False


def test_apply_ram_defaults_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ram_env(monkeypatch)
    monkeypatch.setenv("A0_DISABLE_RAM_TUNE", "1")

    assert apply_ram_defaults() is False
    assert "MALLOC_ARENA_MAX" not in os.environ


def test_reexec_once_when_defaults_are_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ram_env(monkeypatch)
    seen: dict[str, str] = {}

    def fake_execv(executable: str, argv: list[str]) -> None:
        seen["executable"] = executable
        seen["flag"] = os.environ["A0_RAM_REEXEC"]
        seen["arena"] = os.environ["MALLOC_ARENA_MAX"]
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", fake_execv)

    with pytest.raises(SystemExit):
        reexec_if_needed()

    assert seen["executable"] == sys.executable
    assert seen["flag"] == "1"
    assert seen["arena"] == "2"


def test_reexec_skips_when_already_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ram_env(monkeypatch)
    for key, value in RAM_DEFAULTS.items():
        monkeypatch.setenv(key, value)

    def fake_execv(executable: str, argv: list[str]) -> None:
        raise AssertionError(f"unexpected re-exec of {executable} {argv}")

    monkeypatch.setattr(os, "execv", fake_execv)
    reexec_if_needed()


def test_preload_embeddings_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ram_env(monkeypatch)
    assert preload_local_embeddings() is False

    monkeypatch.setenv("A0_PRELOAD_EMBEDDINGS", "yes")
    assert preload_local_embeddings() is True


def test_local_model_idle_seconds_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("A0_LOCAL_MODEL_IDLE_SECONDS", "0")
    assert local_model_idle_seconds() == 0

    monkeypatch.setenv("A0_LOCAL_MODEL_IDLE_SECONDS", "nope")
    assert local_model_idle_seconds() == 300


def test_idle_guard_drops_after_quiet_period() -> None:
    fired: list[str] = []

    async def scenario() -> None:
        guard = IdleGuard(0.05, lambda: fired.append("drop"))
        guard.acquire()
        guard.release()
        await asyncio.sleep(0.02)
        assert fired == []
        await asyncio.sleep(0.08)

    asyncio.run(scenario())
    assert fired == ["drop"]


def test_idle_guard_nested_use_and_zero_timeout() -> None:
    fired: list[str] = []

    async def scenario() -> None:
        guard = IdleGuard(0.05, lambda: fired.append("drop"))
        guard.acquire()
        guard.acquire()
        guard.release()
        await asyncio.sleep(0.08)
        assert fired == []
        guard.release()
        await asyncio.sleep(0.08)

        held = IdleGuard(0, lambda: fired.append("never"))
        held.acquire()
        held.release()
        await asyncio.sleep(0.02)

    asyncio.run(scenario())
    assert fired == ["drop"]


def test_local_embedding_weights_are_loaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    class FakeModel:
        def __init__(self, name: str, **kwargs: object) -> None:
            created.append(name)
            self.name = name
            self.kwargs = kwargs

        def eval(self) -> "FakeModel":
            return self

        def encode(self, texts: list[str], convert_to_tensor: bool = False) -> list[list[float]]:
            assert convert_to_tensor is False
            return [[float(len(text))] for text in texts]

    class _Inference:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> bool:
            return False

    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = FakeModel  # type: ignore[attr-defined]
    torch_mod = types.ModuleType("torch")
    torch_mod.set_num_threads = lambda _count: None  # type: ignore[attr-defined]
    torch_mod.set_num_interop_threads = lambda _count: None  # type: ignore[attr-defined]
    torch_mod.inference_mode = lambda: _Inference()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    import importlib

    import python.helpers.local_embeddings as local_embeddings
    import python.helpers.ram as ram

    ram._torch_configured = False
    local_embeddings = importlib.reload(local_embeddings)
    try:
        local_embeddings.clear_cache()

        first = local_embeddings.load_model("mini", {})
        second = local_embeddings.load_model("mini", {})
        other = local_embeddings.load_model("other", {"device": "cpu"})

        assert first is second
        assert first is not other
        assert created == ["mini", "other"]
        assert local_embeddings.encode(first, ["ab"]) == [[2.0]]
    finally:
        local_embeddings.clear_cache()
        ram._torch_configured = False
        sys.modules.pop("python.helpers.local_embeddings", None)


def test_whisper_package_is_not_imported_at_module_scope() -> None:
    source = (PROJECT_ROOT / "python" / "helpers" / "whisper.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name != "whisper" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "whisper"
    loader = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_whisper_lib"
    )
    assert any(
        isinstance(child, ast.Import) and any(alias.name == "whisper" for alias in child.names)
        for child in loader.body
    )
