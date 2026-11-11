"""A bank can set the relevance floor for callers that never pass one.

`min_scores.final` is a per-request quality bar, so it only helps a caller that
knows to send it. An MCP client does not: it asks its question and is handed the
top matches up to the token budget, relevant or not — and every one of them is
injected into a conversation that did not need it.

The bank's `recall_min_score` is the default for that floor. Set once, it applies
to every recall that names none, including the one that should come back empty.
A caller that does state a floor still gets exactly what it asked for.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"

BERLIN = "Alice moved to Berlin in 2021 | Involving: Alice"
CELLO = "Alice plays the cello professionally | Involving: Alice"

# Same fixture and scores as test_09_min_scores: Berlin's final score is 0.4870,
# the cello fact's 0.3667.
BETWEEN_THE_TWO = 0.40
ABOVE_BOTH = 0.60


@pytest.fixture
async def scored_bank(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin in 2021", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays the cello professionally", who="Alice", entities=["Alice", "cello"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin in 2021 and works as a cellist.")
    await settled(bank_id)
    return bank_id


async def _set_floor(client, bank_id: str, floor: float) -> None:
    await client.banks.update_bank_config(bank_id, {"updates": {"recall_min_score": floor}})


async def test_without_a_floor_everything_that_ranks_comes_back(client, scored_bank):
    """The default the bank floor changes, pinned so the next tests mean something."""
    response = await client.arecall(bank_id=scored_bank, query=QUERY)

    assert sorted(r.text for r in response.results) == sorted([BERLIN, CELLO])


async def test_the_bank_floor_filters_a_recall_that_names_none(client, scored_bank):
    await _set_floor(client, scored_bank, BETWEEN_THE_TWO)

    response = await client.arecall(bank_id=scored_bank, query=QUERY)

    assert [r.text for r in response.results] == [BERLIN]


async def test_nothing_relevant_means_nothing_returned(client, scored_bank):
    """The point of the floor: an honest empty answer instead of the least-bad
    memories filling the token budget."""
    await _set_floor(client, scored_bank, ABOVE_BOTH)

    response = await client.arecall(bank_id=scored_bank, query=QUERY)

    assert response.results == []


async def test_a_caller_that_states_a_floor_overrides_the_bank(client, scored_bank):
    await _set_floor(client, scored_bank, ABOVE_BOTH)

    response = await client.arecall(bank_id=scored_bank, query=QUERY, min_scores={"final": 0.0})

    assert sorted(r.text for r in response.results) == sorted([BERLIN, CELLO])
