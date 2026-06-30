"""Inline task mode (HINDSIGHT_API_INLINE_TASKS).

Inline mode makes the API process background tasks synchronously inside the
request (via SyncTaskBackend) with no worker poller and no maintenance loop, so
the database is touched only while a request is in flight. These tests cover the
deterministic mechanics: config parsing and task-backend selection.

MemoryEngine.__init__ wires the task backend before any DB connection or model
is loaded, and never calls into the embeddings / cross-encoder / query-analyzer
objects — it only stores them — so lightweight mocks let these tests run without
a database, an LLM, or the local embedding models.
"""

import os
from unittest.mock import MagicMock

import pytest

from hindsight_api import MemoryEngine
from hindsight_api.config import clear_config_cache, get_config
from hindsight_api.engine.task_backend import BrokerTaskBackend, SyncTaskBackend


@pytest.fixture
def restore_inline_env():
    """Save/restore HINDSIGHT_API_INLINE_TASKS and the global config cache."""
    saved = os.environ.get("HINDSIGHT_API_INLINE_TASKS")
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("HINDSIGHT_API_INLINE_TASKS", None)
        else:
            os.environ["HINDSIGHT_API_INLINE_TASKS"] = saved
        clear_config_cache()


def _build_engine(*, task_backend=None) -> MemoryEngine:
    """Construct a MemoryEngine without a DB connection or real models.

    initialize() is never called, so no pool or migrations run — __init__ alone
    selects the task backend, which is what these tests assert.
    """
    return MemoryEngine(
        db_url="postgresql://placeholder/placeholder",
        memory_llm_provider="none",
        embeddings=MagicMock(),
        cross_encoder=MagicMock(),
        query_analyzer=MagicMock(),
        run_migrations=False,
        task_backend=task_backend,
    )


@pytest.mark.parametrize(
    ("env_value", "expected_inline"),
    [("true", True), ("True", True), ("false", False), (None, False)],
)
def test_config_parses_inline_tasks(env_value, expected_inline, restore_inline_env):
    if env_value is None:
        os.environ.pop("HINDSIGHT_API_INLINE_TASKS", None)
    else:
        os.environ["HINDSIGHT_API_INLINE_TASKS"] = env_value
    clear_config_cache()

    assert get_config().inline_tasks is expected_inline


def test_inline_mode_uses_sync_task_backend(restore_inline_env):
    """With inline mode on, the engine runs tasks inline instead of via a poller."""
    os.environ["HINDSIGHT_API_INLINE_TASKS"] = "true"
    clear_config_cache()

    engine = _build_engine()

    assert isinstance(engine._task_backend, SyncTaskBackend)


def test_default_mode_uses_broker_task_backend(restore_inline_env):
    """Without inline mode, the engine keeps the broker backend (poller-driven)."""
    os.environ.pop("HINDSIGHT_API_INLINE_TASKS", None)
    clear_config_cache()

    engine = _build_engine()

    assert isinstance(engine._task_backend, BrokerTaskBackend)


def test_explicit_task_backend_overrides_inline(restore_inline_env):
    """An explicitly supplied task_backend always wins over the inline default."""
    os.environ["HINDSIGHT_API_INLINE_TASKS"] = "true"
    clear_config_cache()

    explicit = SyncTaskBackend()
    engine = _build_engine(task_backend=explicit)

    assert engine._task_backend is explicit
