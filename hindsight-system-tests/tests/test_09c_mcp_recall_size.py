"""A bank can bound how much one recall puts into an MCP client's conversation.

The score floor (test_09b) decides *which* memories are relevant. This is about
how much of them arrives. An MCP caller is a model: it picks `max_tokens` itself,
and every result comes back with scores, tags and document ids that it has no use
for and that outweigh the memory text several times over.

Two bank settings cover that, and both apply to the MCP `recall` tool only:
`mcp_recall_max_tokens` is a ceiling the caller cannot raise, and
`mcp_recall_compact` trims each result to what a chat reads. The HTTP API is the
control throughout: the same bank, asked there, answers in full.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from hindsight_system_tests.mcp import call_mcp_tool
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"

# The fixture and budget of test_03_token_budget: at 12 tokens only the short,
# second-ranked fact fits.
LONG_TOP_FACT = "Alice moved to Berlin in 2021 | When: 2021 | Involving: Alice"
SHORT_SECOND_FACT = "Alice plays the cello professionally | Involving: Alice"
FITS_ONLY_THE_SHORT_FACT = 12

McpRecall = Callable[..., Awaitable[dict[str, Any]]]


@pytest.fixture
async def bank_with_two_facts(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin in 2021", when="2021", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays the cello professionally", who="Alice", entities=["Alice", "cello"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin in 2021 and works as a cellist.")
    await settled(bank_id)
    return bank_id


@pytest.fixture
def mcp_recall(hindsight_server, bank_with_two_facts) -> McpRecall:
    async def _recall(**arguments: Any) -> dict[str, Any]:
        return await call_mcp_tool(hindsight_server.url, bank_with_two_facts, "recall", {"query": QUERY, **arguments})

    return _recall


async def _configure(client, bank_id: str, **updates: Any) -> None:
    await client.banks.update_bank_config(bank_id, {"updates": updates})


async def test_by_default_an_mcp_client_gets_every_field_of_every_result(mcp_recall):
    """The default the two settings change, pinned so the next tests mean something."""
    response = await mcp_recall()

    assert [r["text"] for r in response["results"]] == [LONG_TOP_FACT, SHORT_SECOND_FACT]
    assert all({"scores", "tags", "document_id", "chunk_id"} <= set(r) for r in response["results"])


async def test_the_bank_ceiling_holds_whatever_the_caller_asks_for(client, bank_with_two_facts, mcp_recall):
    await _configure(client, bank_with_two_facts, mcp_recall_max_tokens=FITS_ONLY_THE_SHORT_FACT)

    by_default = await mcp_recall()
    asking_for_more = await mcp_recall(max_tokens=20000)

    assert [r["text"] for r in by_default["results"]] == [SHORT_SECOND_FACT]
    assert [r["text"] for r in asking_for_more["results"]] == [SHORT_SECOND_FACT]


async def test_a_caller_may_still_ask_for_less_than_the_ceiling(client, bank_with_two_facts, mcp_recall):
    await _configure(client, bank_with_two_facts, mcp_recall_max_tokens=4096)

    response = await mcp_recall(max_tokens=FITS_ONLY_THE_SHORT_FACT)

    assert [r["text"] for r in response["results"]] == [SHORT_SECOND_FACT]


async def test_compact_results_carry_only_what_a_chat_reads(client, bank_with_two_facts, mcp_recall):
    await _configure(client, bank_with_two_facts, mcp_recall_compact=True)

    response = await mcp_recall()

    assert [r["text"] for r in response["results"]] == [LONG_TOP_FACT, SHORT_SECOND_FACT]
    assert [set(r) for r in response["results"]] == [{"id", "text", "fact_type", "date"}] * 2
    assert set(response) == {"results"}


async def test_the_id_left_on_a_compact_result_fetches_the_rest(
    client, hindsight_server, bank_with_two_facts, mcp_recall
):
    """Nothing is lost by trimming: the id is the handle `get_memory` takes."""
    await _configure(client, bank_with_two_facts, mcp_recall_compact=True)
    first = (await mcp_recall())["results"][0]

    memory = await call_mcp_tool(hindsight_server.url, bank_with_two_facts, "get_memory", {"memory_id": first["id"]})

    assert memory["text"] == first["text"]


async def test_the_http_api_is_not_limited(client, bank_with_two_facts):
    """Both settings are for chat callers. A program on the HTTP API sized its own
    request and parses the fields, so it keeps its budget and the full result."""
    await _configure(
        client, bank_with_two_facts, mcp_recall_max_tokens=FITS_ONLY_THE_SHORT_FACT, mcp_recall_compact=True
    )

    response = await client.arecall(bank_id=bank_with_two_facts, query=QUERY, max_tokens=4096)

    assert [r.text for r in response.results] == [LONG_TOP_FACT, SHORT_SECOND_FACT]
    assert all(r.scores is not None for r in response.results)
