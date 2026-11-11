"""A deploy that runs the old and new container side by side does not strand work.

Railway, like any rolling deploy, starts the new container, waits for it to answer
its health check, and only then stops the old one. For a few seconds two processes
share one database and one worker id — the fixed id such a deployment sets so the
new container can find what the old one left unfinished.

In on-demand mode that overlap used to leave an operation `processing` for good.
The new container reclaimed the old one's rows the moment it started; the old one,
still alive and still polling, claimed its operation straight back and was killed
holding it. An on-demand worker sleeps until something is submitted, so nothing
ever looked at that row again, and the bank's consolidation queued behind it.

Composition, which is why it lives here: recovery, claiming and shutdown are each
correct for one process. What broke was two of them alive at once, and only two
real processes and a real kill show it.

Both stories kill the old container without letting it drain, the way a platform
does when its grace period is shorter than the work.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from pathlib import Path

import pytest
from hindsight_client import Hindsight

from hindsight_system_tests import start_hindsight_server
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

#: Its own schema, so no other story's worker claims this one's operation.
SCHEMA = "systest_overlap"
#: Long enough that the old container is dead before the new one takes over: the
#: stories kill it within a couple of seconds of the new one starting.
RECOVERY_DELAY_SECONDS = 10
#: Shorter than the old container's shutdown drain, so it dies holding its work.
KILL_GRACE_SECONDS = 1.0
COMPLETION_TIMEOUT_SECONDS = 60.0


def _container(stub_server, log_path: Path):
    return start_hindsight_server(
        stub_url=stub_server.url,
        log_path=log_path,
        extra_env={
            "HINDSIGHT_API_DATABASE_SCHEMA": SCHEMA,
            "HINDSIGHT_API_WORKER_ON_DEMAND": "true",
            "HINDSIGHT_API_WORKER_ID": "the-one-container",
            "HINDSIGHT_API_WORKER_ON_DEMAND_RECOVERY_DELAY_SECONDS": str(RECOVERY_DELAY_SECONDS),
            # The interrupted run counts against the retry budget; the suite's 0 would
            # fail the operation instead of letting the new container run it.
            "HINDSIGHT_API_WORKER_MAX_RETRIES": "1",
        },
    )


def _declare_retain(llm) -> None:
    llm.on_step("extract_facts").returns(extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice"])))
    llm.on_step("consolidate").returns(consolidation())


async def _final_status(client: Hindsight, bank_id: str, operation_id: str) -> str:
    deadline = time.monotonic() + COMPLETION_TIMEOUT_SECONDS
    status = await client.operations.get_operation_status(bank_id, operation_id)
    while status.status in ("pending", "processing") and time.monotonic() < deadline:
        await asyncio.sleep(0.5)
        status = await client.operations.get_operation_status(bank_id, operation_id)
    return status.status


async def test_work_in_flight_at_the_deploy_is_finished_by_the_new_container(stub_server, llm, tmp_path: Path):
    _declare_retain(llm)
    bank_id = f"systest-{uuid.uuid4().hex[:12]}"
    old = _container(stub_server, tmp_path / "old.log")
    old_client = Hindsight(base_url=old.url)
    new = None
    try:
        async with llm.hold("extract_facts") as held:
            receipt = await old_client.aretain(bank_id=bank_id, content="Alice moved to Berlin.", retain_async=True)
            await held.reached()  # the old container is mid-operation
            new = _container(stub_server, tmp_path / "new.log")  # healthy while the old one still runs
            await asyncio.to_thread(old.terminate_like_a_platform, KILL_GRACE_SECONDS)

        client = Hindsight(base_url=new.url)
        try:
            status = await _final_status(client, bank_id, receipt.operation_id)
            assert status == "completed", (
                f"the old container's operation is {status!r} on the new one:\n{new.logs()[-3000:]}"
            )
        finally:
            with contextlib.suppress(Exception):
                await client.banks.delete_bank(bank_id)
            await client.aclose()
    finally:
        await old_client.aclose()
        old.stop()
        if new is not None:
            new.stop()


async def test_work_the_old_container_took_during_the_overlap_is_finished_too(stub_server, llm, tmp_path: Path):
    """The operation is claimed *after* the new container started, so nothing about
    its claim time marks it as left over from an earlier run."""
    _declare_retain(llm)
    bank_id = f"systest-{uuid.uuid4().hex[:12]}"
    old = _container(stub_server, tmp_path / "old.log")
    old_client = Hindsight(base_url=old.url)
    new = _container(stub_server, tmp_path / "new.log")
    client = Hindsight(base_url=new.url)
    try:
        async with llm.hold("extract_facts") as held:
            receipt = await old_client.aretain(bank_id=bank_id, content="Alice moved to Berlin.", retain_async=True)
            await held.reached()
            await asyncio.to_thread(old.terminate_like_a_platform, KILL_GRACE_SECONDS)

        status = await _final_status(client, bank_id, receipt.operation_id)
        assert status == "completed", (
            f"the old container's operation is {status!r} on the new one:\n{new.logs()[-3000:]}"
        )
    finally:
        with contextlib.suppress(Exception):
            await client.banks.delete_bank(bank_id)
        await client.aclose()
        await old_client.aclose()
        old.stop()
        new.stop()
