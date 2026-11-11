"""On-demand worker mode (HINDSIGHT_API_WORKER_ON_DEMAND).

On-demand mode keeps the async background worker embedded in the API but wakes it
via an in-process signal on each task submission instead of polling the database.
This gives async retain (returns immediately) with zero idle DB queries, so a
scale-to-zero database can auto-suspend between requests.

These tests cover the deterministic mechanics with no DB or LLM:
  * config parsing;
  * the task backend waking the poller on submit;
  * the poll loop going silent when idle, and polling again while tasks run;
  * the wake sources that replace polling: submit, retry timers, startup recovery;
  * recovery leaving a process this one is replacing alone until it has exited;
  * maintenance running when the worker goes idle, in place of the timer;
  * operation retry re-submitting through the task backend (wake source);
  * cancel and a folded submit waking the worker, since neither submits a task.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api import MemoryEngine
from hindsight_api.config import clear_config_cache, get_config
from hindsight_api.engine import maintenance as maintenance_module
from hindsight_api.engine.maintenance import MaintenanceLoop
from hindsight_api.engine.task_backend import BrokerTaskBackend, TaskBackend
from hindsight_api.extensions.operation_validator import OperationValidationError
from hindsight_api.models import RequestContext
from hindsight_api.worker import WorkerPoller
from hindsight_api.worker import poller as poller_module

POLL_INTERVAL_MS = 10
# Long enough for a polling loop to have gone round many times.
SETTLE_SECONDS = 0.2


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


class _FakeConn:
    async def execute(self, *args, **kwargs):
        return None


class _FakeAcquire:
    async def __aenter__(self):
        return _FakeConn()

    async def __aexit__(self, *args):
        return False


async def _broker_backend(monkeypatch) -> BrokerTaskBackend:
    from hindsight_api.engine import db_utils

    monkeypatch.setattr(db_utils, "acquire_with_retry", lambda pool: _FakeAcquire())
    backend = BrokerTaskBackend(pool_getter=lambda: MagicMock())
    await backend.initialize()
    return backend


@pytest.mark.asyncio
async def test_broker_backend_invokes_wake_callback_on_submit(monkeypatch):
    """A wired wake callback fires after the row is written — this is how an
    in-process submit wakes the on-demand poller instead of it polling."""
    backend = await _broker_backend(monkeypatch)
    wakes = []
    backend.set_wake_callback(lambda: wakes.append(True))

    await backend.submit_task({"type": "retain", "operation_id": str(uuid.uuid4()), "bank_id": "b"})

    assert wakes == [True]


@pytest.mark.asyncio
async def test_broker_backend_no_wake_callback_is_safe(monkeypatch):
    """Without a wake callback wired (polling / dedicated-worker deployments),
    submit_task must still succeed."""
    backend = await _broker_backend(monkeypatch)

    await backend.submit_task({"type": "retain", "operation_id": str(uuid.uuid4()), "bank_id": "b"})


@pytest.mark.asyncio
async def test_failing_wake_callback_does_not_break_submit(monkeypatch):
    backend = await _broker_backend(monkeypatch)

    def _boom() -> None:
        raise RuntimeError("poller gone")

    backend.set_wake_callback(_boom)

    await backend.submit_task({"type": "retain", "operation_id": str(uuid.uuid4()), "bank_id": "b"})


def _make_poller(*, on_demand: bool = True, on_idle=None, recovery_settle_seconds: float = 0.0) -> WorkerPoller:
    poller = WorkerPoller(
        backend=MagicMock(),
        worker_id="on-demand-test",
        executor=AsyncMock(),
        poll_interval_ms=POLL_INTERVAL_MS,
        on_demand=on_demand,
        on_idle=on_idle,
        recovery_settle_seconds=recovery_settle_seconds,
    )
    # The loop's only database work when nothing is claimable.
    poller.claim_batch = AsyncMock(return_value=[])
    poller._log_progress_if_due = AsyncMock()
    return poller


async def _stop(poller: WorkerPoller, loop_task: asyncio.Task) -> None:
    poller.begin_shutdown()
    await asyncio.wait_for(loop_task, timeout=1.0)


@pytest.mark.asyncio
async def test_idle_on_demand_loop_stops_querying():
    """With nothing running and nothing claimable, the loop claims once and then
    issues no further queries — the property that lets the database suspend."""
    poller = _make_poller()
    loop_task = asyncio.ensure_future(poller._poll_loop())

    await asyncio.sleep(SETTLE_SECONDS)

    assert poller.claim_batch.await_count == 1
    poller._log_progress_if_due.assert_not_awaited()
    await _stop(poller, loop_task)


@pytest.mark.asyncio
async def test_polling_loop_keeps_querying():
    """The default mode is unchanged: it re-polls on the interval."""
    poller = _make_poller(on_demand=False)
    loop_task = asyncio.ensure_future(poller._poll_loop())

    await asyncio.sleep(SETTLE_SECONDS)

    assert poller.claim_batch.await_count > 2
    await _stop(poller, loop_task)


@pytest.mark.asyncio
async def test_notify_wakes_idle_loop_to_claim():
    poller = _make_poller()
    loop_task = asyncio.ensure_future(poller._poll_loop())
    await asyncio.sleep(SETTLE_SECONDS)
    assert poller.claim_batch.await_count == 1

    poller.notify()
    await asyncio.sleep(SETTLE_SECONDS)

    assert poller.claim_batch.await_count == 2
    await _stop(poller, loop_task)


@pytest.mark.asyncio
async def test_on_demand_loop_polls_while_tasks_are_in_flight():
    """A running task can make rows claimable without a submit (a freed
    serialization key, a webhook delivery), so the loop polls until it drains."""
    poller = _make_poller()
    poller._in_flight_count = 1
    loop_task = asyncio.ensure_future(poller._poll_loop())

    await asyncio.sleep(SETTLE_SECONDS)
    assert poller.claim_batch.await_count > 2

    poller._in_flight_count = 0
    await asyncio.sleep(SETTLE_SECONDS)
    settled = poller.claim_batch.await_count
    await asyncio.sleep(SETTLE_SECONDS)

    assert poller.claim_batch.await_count == settled
    await _stop(poller, loop_task)


@pytest.mark.asyncio
async def test_shutdown_releases_idle_loop():
    poller = _make_poller()
    loop_task = asyncio.ensure_future(poller._poll_loop())
    await asyncio.sleep(SETTLE_SECONDS)
    assert not loop_task.done()

    await _stop(poller, loop_task)

    assert loop_task.done()


@pytest.mark.asyncio
async def test_wake_timer_fires_when_task_is_due(monkeypatch):
    """A deferred/retried task re-arms the wake so it is picked up without polling."""
    monkeypatch.setattr(poller_module, "_WAKE_TIMER_SLACK_SECONDS", 0.0)
    poller = _make_poller()

    poller._arm_wake_timer(datetime.now(timezone.utc) + timedelta(seconds=0.05))

    assert not poller._work_available.is_set()
    await asyncio.wait_for(poller._work_available.wait(), timeout=1.0)
    assert not poller._wake_timers


@pytest.mark.asyncio
async def test_wake_timer_waits_out_the_clock_slack(monkeypatch):
    """The claim gates on the database clock, so a timer must not fire the instant
    this process thinks the row is due."""
    monkeypatch.setattr(poller_module, "_WAKE_TIMER_SLACK_SECONDS", 30.0)
    poller = _make_poller()

    poller._arm_wake_timer(datetime.now(timezone.utc))
    await asyncio.sleep(0.05)

    assert not poller._work_available.is_set()
    poller.begin_shutdown()


@pytest.mark.asyncio
async def test_wake_timer_is_not_armed_in_polling_mode():
    poller = _make_poller(on_demand=False)

    poller._arm_wake_timer(datetime.now(timezone.utc))

    assert not poller._wake_timers


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_wake_timers():
    poller = _make_poller()
    poller._arm_wake_timer(datetime.now(timezone.utc) + timedelta(hours=1))
    (handle,) = poller._wake_timers

    poller.begin_shutdown()

    assert handle.cancelled()
    assert not poller._wake_timers


@pytest.mark.asyncio
async def test_startup_recovery_wakes_the_loop():
    """Recovery hands stale rows back to 'pending' after the loop's first claim;
    without a wake an idle on-demand loop would never claim them."""
    poller = _make_poller()
    poller.recover_own_tasks = AsyncMock(return_value=1)

    await poller._run_recovery(datetime.now(timezone.utc))

    assert poller._work_available.is_set()


class TestReplacedProcess:
    """A deploy starts the new container before stopping the old one, so for a few
    seconds two processes share a worker id. Taking the old one's rows while it still
    runs left them claimed by a process that was then killed."""

    @pytest.mark.asyncio
    async def test_recovery_waits_for_the_old_process_to_exit(self):
        poller = _make_poller(recovery_settle_seconds=SETTLE_SECONDS)
        poller.recover_own_tasks = AsyncMock(return_value=1)
        recovery = asyncio.ensure_future(poller._run_recovery(datetime.now(timezone.utc)))

        await asyncio.sleep(SETTLE_SECONDS / 2)
        poller.recover_own_tasks.assert_not_awaited()
        assert not poller._work_available.is_set()

        await asyncio.wait_for(recovery, timeout=1.0)
        poller.recover_own_tasks.assert_awaited_once()
        assert poller._work_available.is_set()

    @pytest.mark.asyncio
    async def test_polling_mode_recovers_at_once(self):
        poller = _make_poller(on_demand=False, recovery_settle_seconds=60.0)
        poller.recover_own_tasks = AsyncMock(return_value=0)

        await asyncio.wait_for(poller._run_recovery(datetime.now(timezone.utc)), timeout=1.0)

        poller.recover_own_tasks.assert_awaited_once()
        assert not poller._orphan_sweep_due

    @pytest.mark.asyncio
    async def test_shutdown_during_the_wait_skips_recovery(self):
        poller = _make_poller(recovery_settle_seconds=60.0)
        poller.recover_own_tasks = AsyncMock(return_value=0)
        recovery = asyncio.ensure_future(poller._run_recovery(datetime.now(timezone.utc)))
        await asyncio.sleep(0)

        poller.begin_shutdown()
        await asyncio.wait_for(recovery, timeout=1.0)

        poller.recover_own_tasks.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_idle_loop_reclaims_what_the_old_process_claimed_late(self):
        """Recovery only reaches rows claimed before this process started. One the old
        process claimed after that is handed back the first time nothing runs here,
        and claimed in the same pass rather than after the next wake."""
        poller = _make_poller()
        poller.recover_own_tasks = AsyncMock(return_value=0)
        poller._reclaim_orphaned_tasks = AsyncMock(return_value=1)
        await poller._run_recovery(datetime.now(timezone.utc))

        loop_task = asyncio.ensure_future(poller._poll_loop())
        await asyncio.sleep(SETTLE_SECONDS)

        poller._reclaim_orphaned_tasks.assert_awaited_once()
        assert poller.claim_batch.await_count == 2
        await _stop(poller, loop_task)

    @pytest.mark.asyncio
    async def test_nothing_is_reclaimed_while_a_task_runs_here(self):
        """A row 'processing' under this worker id may be the one this process is
        running, so the sweep waits until nothing is."""
        poller = _make_poller()
        poller.recover_own_tasks = AsyncMock(return_value=0)
        poller._reclaim_orphaned_tasks = AsyncMock(return_value=0)
        await poller._run_recovery(datetime.now(timezone.utc))
        poller._in_flight_count = 1

        loop_task = asyncio.ensure_future(poller._poll_loop())
        await asyncio.sleep(SETTLE_SECONDS)
        poller._reclaim_orphaned_tasks.assert_not_awaited()

        poller._in_flight_count = 0
        await asyncio.sleep(SETTLE_SECONDS)

        poller._reclaim_orphaned_tasks.assert_awaited_once()
        await _stop(poller, loop_task)

    @pytest.mark.asyncio
    async def test_reclaim_covers_every_schema_without_a_claim_time_cutoff(self):
        poller = _make_poller()
        poller._get_schemas = AsyncMock(return_value=[None, "tenant_a"])
        poller._reclaim_own_processing_tasks = AsyncMock(side_effect=[1, 2])

        assert await poller._reclaim_orphaned_tasks() == 3

        assert [call.args for call in poller._reclaim_own_processing_tasks.await_args_list] == [
            (None,),
            ("tenant_a",),
        ]
        assert all(not call.kwargs for call in poller._reclaim_own_processing_tasks.await_args_list)


class TestIdleAnnouncement:
    """The loop tells ``on_idle`` each time it goes silent: the moment the database is
    awake with nothing left to do, which is when maintenance runs in this mode."""

    @pytest.mark.asyncio
    async def test_announced_each_time_the_loop_goes_silent(self):
        idles = []
        poller = _make_poller(on_idle=lambda: idles.append(True))
        loop_task = asyncio.ensure_future(poller._poll_loop())

        await asyncio.sleep(SETTLE_SECONDS)
        assert idles == [True]

        poller.notify()
        await asyncio.sleep(SETTLE_SECONDS)

        assert idles == [True, True]
        await _stop(poller, loop_task)

    @pytest.mark.asyncio
    async def test_not_announced_until_running_tasks_drain(self):
        idles = []
        poller = _make_poller(on_idle=lambda: idles.append(True))
        poller._in_flight_count = 1
        loop_task = asyncio.ensure_future(poller._poll_loop())

        await asyncio.sleep(SETTLE_SECONDS)
        assert idles == []

        poller._in_flight_count = 0
        await asyncio.sleep(SETTLE_SECONDS)

        assert idles == [True]
        await _stop(poller, loop_task)

    @pytest.mark.asyncio
    async def test_never_announced_in_polling_mode(self):
        idles = []
        poller = _make_poller(on_demand=False, on_idle=lambda: idles.append(True))
        loop_task = asyncio.ensure_future(poller._poll_loop())

        await asyncio.sleep(SETTLE_SECONDS)

        assert idles == []
        await _stop(poller, loop_task)

    @pytest.mark.asyncio
    async def test_a_failing_callback_does_not_stop_the_loop(self):
        def _boom() -> None:
            raise RuntimeError("maintenance gone")

        poller = _make_poller(on_idle=_boom)
        loop_task = asyncio.ensure_future(poller._poll_loop())
        await asyncio.sleep(SETTLE_SECONDS)

        poller.notify()
        await asyncio.sleep(SETTLE_SECONDS)

        assert poller.claim_batch.await_count == 2
        await _stop(poller, loop_task)


def _maintenance_loop(monkeypatch, **cfg_overrides) -> MaintenanceLoop:
    """A MaintenanceLoop whose only enabled job is the hourly LLM-trace retention sweep."""
    cfg = SimpleNamespace(
        consolidation_reconcile_interval_seconds=0,
        audit_log_retention_days=0,
        llm_trace_enabled=True,
        llm_trace_retention_days=1,
        mental_model_refresh_tick_seconds=0,
        retention_sweep_interval_seconds=3600,
        operation_retention_days=0,
        operation_cleanup_interval_seconds=0,
        attachment_sweep_interval_seconds=0,
    )
    for name, value in cfg_overrides.items():
        setattr(cfg, name, value)
    monkeypatch.setattr(maintenance_module, "get_config", lambda: cfg)
    monkeypatch.setattr(maintenance_module, "_is_oracle", lambda: False)
    loop = MaintenanceLoop(engine=None)
    loop._run_retention = AsyncMock()
    return loop


def _job_blocked_on(release: asyncio.Event) -> AsyncMock:
    async def _job(cfg) -> None:
        await release.wait()

    return AsyncMock(side_effect=_job)


class TestMaintenanceOnIdle:
    """On-demand mode has no maintenance timer (it would keep the database awake), so
    the jobs run when the worker goes idle instead."""

    @pytest.mark.asyncio
    async def test_runs_in_the_background_without_holding_up_the_caller(self, monkeypatch):
        loop = _maintenance_loop(monkeypatch)
        started = asyncio.Event()
        release = asyncio.Event()

        async def _slow_retention(cfg) -> None:
            started.set()
            await release.wait()

        loop._run_retention = _slow_retention

        loop.run_once_soon()

        assert not started.is_set(), "the job must not run inside the call"
        await asyncio.wait_for(started.wait(), timeout=1.0)
        release.set()
        await loop._task

    @pytest.mark.asyncio
    async def test_a_second_request_while_one_is_running_starts_nothing(self, monkeypatch):
        loop = _maintenance_loop(monkeypatch)
        release = asyncio.Event()
        loop._run_retention = _job_blocked_on(release)

        loop.run_once_soon()
        running = loop._task
        await asyncio.sleep(0)
        loop.run_once_soon()

        assert loop._task is running
        release.set()
        await running
        assert loop._run_retention.await_count == 1

    @pytest.mark.asyncio
    async def test_a_job_waits_out_its_interval_however_often_the_worker_goes_idle(self, monkeypatch):
        loop = _maintenance_loop(monkeypatch)

        for _ in range(3):
            loop.run_once_soon()
            await loop._task

        assert loop._run_retention.await_count == 1

    @pytest.mark.asyncio
    async def test_nothing_starts_when_no_job_is_enabled(self, monkeypatch):
        loop = _maintenance_loop(monkeypatch, llm_trace_enabled=False)

        loop.run_once_soon()

        assert loop._task is None

    @pytest.mark.asyncio
    async def test_a_failing_job_is_contained(self, monkeypatch):
        loop = _maintenance_loop(monkeypatch)
        loop._run_retention = AsyncMock(side_effect=RuntimeError("database gone"))

        loop.run_once_soon()

        await loop._task

    @pytest.mark.asyncio
    async def test_shutdown_stops_a_job_in_progress(self, monkeypatch):
        monkeypatch.setattr(maintenance_module, "_STOP_GRACE_SECONDS", 0.05)
        loop = _maintenance_loop(monkeypatch)
        loop._run_retention = _job_blocked_on(asyncio.Event())
        loop.run_once_soon()
        running = loop._task
        await asyncio.sleep(0)

        await asyncio.wait_for(loop.stop(), timeout=2.0)

        assert running.cancelled()

    @pytest.mark.asyncio
    async def test_an_idle_worker_triggers_the_due_jobs_once(self, monkeypatch):
        """The two halves together: every idle period asks, the interval decides."""
        loop = _maintenance_loop(monkeypatch)
        poller = _make_poller(on_idle=loop.run_once_soon)
        loop_task = asyncio.ensure_future(poller._poll_loop())
        await asyncio.sleep(SETTLE_SECONDS)
        assert loop._run_retention.await_count == 1

        poller.notify()
        await asyncio.sleep(SETTLE_SECONDS)

        assert loop._run_retention.await_count == 1
        await _stop(poller, loop_task)

    def test_the_engine_passes_the_request_to_its_loop(self, monkeypatch):
        engine, _ = _engine_with_recorder(monkeypatch, _FakeRetryConn([]))
        engine.run_maintenance_soon()

        engine._maintenance_loop = MagicMock()
        engine.run_maintenance_soon()

        engine._maintenance_loop.run_once_soon.assert_called_once_with()


class _RecordingTaskBackend(TaskBackend):
    """Records submitted payloads instead of executing or queueing them."""

    def __init__(self):
        super().__init__()
        self.submitted: list[dict] = []

    async def initialize(self):
        self._initialized = True

    async def submit_task(self, task_dict):
        self.submitted.append(task_dict)

    async def shutdown(self):
        self._initialized = False


class _FakeRetryConn:
    """Stands in for a database connection during retry_operation: hands back the
    queued fetchrow results in order."""

    def __init__(self, fetchrow_results: list[Any], children: list[dict] | None = None):
        self._fetchrow_results = list(fetchrow_results)
        self._children = children or []
        self.executed: list[str] = []

    async def fetchrow(self, *args, **kwargs):
        return self._fetchrow_results.pop(0)

    async def fetch(self, *args, **kwargs):
        return self._children

    async def execute(self, query, *args):
        self.executed.append(query)


class _FakeAcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        return False


def _engine_with_recorder(monkeypatch, conn: _FakeRetryConn) -> tuple[MemoryEngine, _RecordingTaskBackend]:
    """MemoryEngine with mock models and a recording task backend (no DB/LLM)."""
    from hindsight_api.engine import memory_engine as me

    monkeypatch.setattr(me, "acquire_with_retry", lambda pool: _FakeAcquireCtx(conn))
    recorder = _RecordingTaskBackend()
    engine = MemoryEngine(
        db_url="postgresql://placeholder/placeholder",
        memory_llm_provider="none",
        embeddings=MagicMock(),
        cross_encoder=MagicMock(),
        query_analyzer=MagicMock(),
        run_migrations=False,
        task_backend=recorder,
    )
    engine._authenticate_tenant = AsyncMock()
    engine._refuse_bank_wide_if_scoped = AsyncMock()
    engine._get_backend = AsyncMock(return_value=MagicMock())
    return engine, recorder


def _payload(bank_id: str = "b1") -> dict:
    return {"type": "batch_retain", "operation_id": str(uuid.uuid4()), "bank_id": bank_id}


@pytest.mark.asyncio
async def test_retry_operation_resubmits_task(monkeypatch):
    """Retrying a failed op must re-submit through the task backend: that wake is
    how the on-demand worker learns about the row (and how inline mode executes
    it) — a raw status reset would strand it until unrelated activity."""
    payload = _payload()
    conn = _FakeRetryConn([{"operation_id": payload["operation_id"], "task_payload": json.dumps(payload)}])
    engine, recorder = _engine_with_recorder(monkeypatch, conn)

    result = await engine.retry_operation("b1", payload["operation_id"], request_context=RequestContext())

    assert result["success"] is True
    assert recorder.submitted == [payload]


@pytest.mark.asyncio
async def test_retry_of_batch_parent_resubmits_its_requeued_children(monkeypatch):
    """A batch_retain parent has no payload of its own: retrying it re-queues its
    failed children, and those are what the worker must be told about."""
    first, second = _payload(), _payload()
    parent_id = str(uuid.uuid4())
    conn = _FakeRetryConn(
        [
            None,  # the single-operation reset skips a payload-less parent
            {"status": "failed", "operation_type": "batch_retain", "task_payload": None},
            {"operation_id": parent_id},  # the parent was revived
        ],
        children=[{"task_payload": json.dumps(first)}, {"task_payload": second}],
    )
    engine, recorder = _engine_with_recorder(monkeypatch, conn)

    result = await engine.retry_operation("b1", parent_id, request_context=RequestContext())

    assert result["success"] is True
    assert recorder.submitted == [first, second]
    assert len(conn.executed) == 1  # the children were reset before being resubmitted


@pytest.mark.asyncio
async def test_retry_of_a_non_retryable_operation_submits_nothing(monkeypatch):
    conn = _FakeRetryConn([None, {"status": "completed", "operation_type": "retain", "task_payload": "{}"}])
    engine, recorder = _engine_with_recorder(monkeypatch, conn)

    with pytest.raises(OperationValidationError) as exc:
        await engine.retry_operation("b1", str(uuid.uuid4()), request_context=RequestContext())

    assert exc.value.status_code == 409
    assert recorder.submitted == []


class _FakeTransactionConn(_FakeRetryConn):
    """Adds what the submit and cancel paths need on top of the retry stand-in."""

    def __init__(self, fetchrow_results: list[Any] | None = None, pending: list[dict] | None = None):
        super().__init__(fetchrow_results or [], children=pending)

    def transaction(self):
        return _FakeAcquireCtx(self)

    async def fetchval(self, *args, **kwargs):
        return 1


def _wakes(recorder: _RecordingTaskBackend) -> list[bool]:
    wakes: list[bool] = []
    recorder.set_wake_callback(lambda: wakes.append(True))
    return wakes


@pytest.mark.asyncio
async def test_cancel_operation_wakes_the_worker(monkeypatch):
    """Cancelling a 'processing' row frees its bank's slot: the operation queued behind
    it is claimable, and no submit happened to tell an idle on-demand worker."""
    operation_id = str(uuid.uuid4())
    conn = _FakeTransactionConn([{"bank_id": "b1", "status": "processing"}, {"status": "cancelled"}])
    engine, recorder = _engine_with_recorder(monkeypatch, conn)
    engine._maybe_update_parent_operation = AsyncMock()
    wakes = _wakes(recorder)

    result = await engine.cancel_operation("b1", operation_id, request_context=RequestContext())

    assert result["success"] is True
    assert wakes == [True]


@pytest.mark.asyncio
async def test_a_folded_submit_wakes_the_worker(monkeypatch):
    """A consolidation request that folds into the one already queued submits no task,
    but it is how a caller nudges a queue that looks stuck, so it must reach the worker."""
    existing_id = uuid.uuid4()
    conn = _FakeTransactionConn(pending=[{"operation_id": existing_id, "task_payload": "{}", "status": "pending"}])
    engine, recorder = _engine_with_recorder(monkeypatch, conn)
    wakes = _wakes(recorder)

    result = await engine._submit_async_operation("b1", "consolidation", "consolidation", {}, dedupe_by_bank=True)

    assert result == {"operation_id": str(existing_id), "deduplicated": True}
    assert recorder.submitted == []
    assert wakes == [True]
