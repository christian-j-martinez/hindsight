"""On-demand worker mode (HINDSIGHT_API_WORKER_ON_DEMAND).

On-demand mode keeps the async background worker embedded in the API but wakes it
via an in-process signal on each task submission instead of polling the database.
This gives async retain (returns immediately) with zero idle DB queries, so a
scale-to-zero database can auto-suspend between requests.

These tests cover the deterministic mechanics with no DB or LLM:
  * config parsing;
  * the task backend waking the poller on submit;
  * the poller blocking until notified and re-arming on a timer.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.config import clear_config_cache, get_config
from hindsight_api.engine.task_backend import BrokerTaskBackend
from hindsight_api.worker import WorkerPoller


@pytest.fixture
def config_env(monkeypatch):
    """Yield monkeypatch; drop the cached config afterwards so env changes here
    don't leak into other tests via the global config cache."""
    yield monkeypatch
    clear_config_cache()


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("true", True), ("True", True), ("false", False), (None, False)],
)
def test_config_parses_worker_on_demand(env_value, expected, config_env):
    if env_value is None:
        config_env.delenv("HINDSIGHT_API_WORKER_ON_DEMAND", raising=False)
    else:
        config_env.setenv("HINDSIGHT_API_WORKER_ON_DEMAND", env_value)
    clear_config_cache()

    assert get_config().worker_on_demand is expected


@pytest.mark.asyncio
async def test_broker_backend_invokes_wake_callback_on_submit(monkeypatch):
    """A wired wake callback fires after the row is written — this is how an
    in-process submit wakes the on-demand poller instead of it polling."""
    from hindsight_api.engine import db_utils

    class _FakeConn:
        async def execute(self, *args, **kwargs):
            return None

    class _FakeAcquire:
        async def __aenter__(self):
            return _FakeConn()

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(db_utils, "acquire_with_retry", lambda pool: _FakeAcquire())

    backend = BrokerTaskBackend(pool_getter=lambda: MagicMock())
    await backend.initialize()

    wakes = []
    backend.set_wake_callback(lambda: wakes.append(True))

    await backend.submit_task({"type": "retain", "operation_id": str(uuid.uuid4()), "bank_id": "b"})

    assert wakes == [True]


@pytest.mark.asyncio
async def test_broker_backend_no_wake_callback_is_safe(monkeypatch):
    """Without a wake callback wired (polling / dedicated-worker deployments),
    submit_task must still succeed."""
    from hindsight_api.engine import db_utils

    class _FakeConn:
        async def execute(self, *args, **kwargs):
            return None

    class _FakeAcquire:
        async def __aenter__(self):
            return _FakeConn()

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(db_utils, "acquire_with_retry", lambda pool: _FakeAcquire())

    backend = BrokerTaskBackend(pool_getter=lambda: MagicMock())
    await backend.initialize()

    # No set_wake_callback call — must not raise.
    await backend.submit_task({"type": "retain", "operation_id": str(uuid.uuid4()), "bank_id": "b"})


def _make_on_demand_poller() -> WorkerPoller:
    return WorkerPoller(backend=MagicMock(), worker_id="on-demand-test", executor=AsyncMock(), on_demand=True)


@pytest.mark.asyncio
async def test_wait_for_work_blocks_until_notify():
    """The idle loop parks on the event and issues no queries until notified."""
    poller = _make_on_demand_poller()

    waiter = asyncio.ensure_future(poller._wait_for_work())
    await asyncio.sleep(0.05)
    assert not waiter.done()  # still blocked — no work signalled

    poller.notify()
    await asyncio.wait_for(waiter, timeout=1.0)
    assert waiter.done()


@pytest.mark.asyncio
async def test_wait_for_work_unblocks_on_shutdown():
    """A graceful shutdown must release the idle wait even with no work."""
    poller = _make_on_demand_poller()

    waiter = asyncio.ensure_future(poller._wait_for_work())
    await asyncio.sleep(0.05)
    assert not waiter.done()

    poller._shutdown.set()
    await asyncio.wait_for(waiter, timeout=1.0)
    assert waiter.done()


@pytest.mark.asyncio
async def test_arm_wake_timer_sets_event_when_due():
    """A deferred/retried task re-arms the wake so it is picked up without polling."""
    poller = _make_on_demand_poller()
    assert not poller._work_available.is_set()

    poller._arm_wake_timer(datetime.now(timezone.utc) + timedelta(seconds=0.05))

    await asyncio.wait_for(poller._work_available.wait(), timeout=1.0)
    assert poller._work_available.is_set()
