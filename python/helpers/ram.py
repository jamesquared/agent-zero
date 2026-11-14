"""Process-wide RAM defaults for the Agent Zero runtime.

glibc reads ``MALLOC_ARENA_MAX`` when the process starts, and OpenMP / BLAS
read their thread counts when their native libraries load. Set the defaults
before that happens (``reexec_if_needed`` from the main entrypoint, or the
container start script). Local models call ``configure_torch_threads`` and
``release_heap`` once torch is already in the process.
"""

from __future__ import annotations

import asyncio
import ctypes
import gc
import os
import sys
import threading
from typing import Callable

# One glibc heap arena per core, and one BLAS workspace per thread, dominate
# RSS on a small machine long before model weights do. Users can export a
# larger value before start; these only fill in what is unset.
RAM_DEFAULTS: dict[str, str] = {
    "MALLOC_ARENA_MAX": "2",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

_TRUTHY = {"1", "true", "yes", "on"}
_torch_configured = False


def ram_tune_disabled() -> bool:
    return os.environ.get("A0_DISABLE_RAM_TUNE", "").strip().lower() in _TRUTHY


def apply_ram_defaults() -> bool:
    """Set unset allocator and thread env vars. Return True if any were added."""
    if ram_tune_disabled():
        return False
    changed = False
    for key, value in RAM_DEFAULTS.items():
        if not os.environ.get(key):
            os.environ[key] = value
            changed = True
    return changed


def reexec_if_needed() -> None:
    """Re-exec the current interpreter so glibc sees allocator limits.

    ``MALLOC_ARENA_MAX`` is read at process start. Setting it after Python has
    already allocated does nothing, so the main entrypoint fills defaults and
    replaces itself once when any default was missing.
    """
    if os.environ.get("A0_RAM_REEXEC") == "1" or ram_tune_disabled():
        return
    if not apply_ram_defaults():
        return
    os.environ["A0_RAM_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def preload_local_embeddings() -> bool:
    """Warm the in-process embedding model at startup.

    The image build still runs ``preload.py`` to fill the on-disk weight
    cache. The running server leaves the weights unloaded unless this is set,
    so an idle UI does not pay for a second copy that Python never returns.
    """
    return os.environ.get("A0_PRELOAD_EMBEDDINGS", "").strip().lower() in _TRUTHY


def local_model_idle_seconds() -> float:
    """How long an unused Whisper or Kokoro model stays resident.

    ``0`` keeps the model loaded. The default lets a voice model leave RAM
    after a few quiet minutes; the next request loads it again.
    """
    raw = os.environ.get("A0_LOCAL_MODEL_IDLE_SECONDS", "300").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 300.0


def configure_torch_threads() -> None:
    """Limit torch's own thread pools to the same cap as OpenMP.

    Imported here, not at module import, so bringing in this helper does not
    load torch. Safe to call more than once.
    """
    global _torch_configured
    if _torch_configured:
        return
    import torch

    raw = os.environ.get("OMP_NUM_THREADS", "1")
    try:
        threads = max(1, int(raw))
    except ValueError:
        threads = 1
    try:
        torch.set_num_threads(threads)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _torch_configured = True


def release_heap() -> None:
    """Collect cyclic garbage and ask glibc to return free arenas to the OS.

    PyTorch allocations are often large enough to be mmap'd, but the loader's
    temporary buffers sit in the main heap. ``malloc_trim`` is a no-op where
    glibc is not the allocator.
    """
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        cuda = getattr(torch, "cuda", None)
        try:
            if cuda is not None and cuda.is_available():
                cuda.empty_cache()
        except Exception:
            pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        return


class IdleGuard:
    """Call ``on_idle`` after the resource has been unused for ``idle_seconds``.

    ``acquire`` / ``release`` nest. A release that leaves the resource in use,
    or an ``idle_seconds`` of 0, does not schedule a drop. ``on_idle`` runs
    under the guard lock and must only drop its own reference; it must not
    call ``acquire`` or ``release``.
    """

    def __init__(self, idle_seconds: float, on_idle: Callable[[], None]) -> None:
        self.idle_seconds = idle_seconds
        self._on_idle = on_idle
        self._in_use = 0
        self._generation = 0
        self._task: asyncio.Task[None] | None = None
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            self._in_use += 1
            self._generation += 1
            task = self._task
            self._task = None
        if task is not None and not task.done():
            task.cancel()

    def release(self) -> None:
        with self._lock:
            self._in_use = max(0, self._in_use - 1)
            if self._in_use > 0 or self.idle_seconds <= 0:
                return
            self._generation += 1
            generation = self._generation
        self._schedule(generation)

    def _schedule(self, generation: int) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._wait(generation))
        with self._lock:
            if generation != self._generation:
                task.cancel()
                return
            self._task = task

    async def _wait(self, generation: int) -> None:
        try:
            await asyncio.sleep(self.idle_seconds)
        except asyncio.CancelledError:
            return
        with self._lock:
            if generation != self._generation or self._in_use > 0:
                return
            # Null the model in this critical section. acquire() takes the
            # same lock, so it cannot observe a half-dropped model or have
            # its freshly loaded instance cleared.
            self._on_idle()
        release_heap()
