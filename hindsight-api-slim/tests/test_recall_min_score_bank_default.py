"""The bank-level recall relevance floor (``recall_min_score``).

``min_scores.final`` is per request; a caller that cannot set it (an MCP client)
gets the top matches up to its token budget whether or not they are relevant. The
bank's ``recall_min_score`` is the default for it, so such a recall returns nothing
when nothing clears the floor.

Seeds three facts with real embeddings and ranks them with a stub reranker whose
scores are known, so every assertion here is deterministic (no LLM, no judge).
"""

import uuid

import pytest_asyncio

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.config import HindsightConfig
from hindsight_api.engine.cross_encoder import CrossEncoderModel
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.response_models import MinScores
from hindsight_api.engine.retain import embedding_utils
from tests.store_seed import store_write_facts

RC = RequestContext(tenant_id="default")

# Above any final score the stub reranker can produce.
UNREACHABLE_FLOOR = 0.99


class _StubReranker(CrossEncoderModel):
    """A reranker under a chosen provider name, scoring evenly down from 1.0."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    async def initialize(self) -> None:
        pass

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        n = len(pairs)
        return [(n - i) / n for i in range(n)]


@pytest_asyncio.fixture
async def seeded_memory(memory_no_llm_verify: MemoryEngine, monkeypatch):
    """Three seeded facts ranked by a pointwise stub reranker; yields (engine, bank_id)."""
    engine = memory_no_llm_verify
    bank_id = f"test-bank-floor-{uuid.uuid4().hex[:8]}"
    await engine.ensure_bank_profile(bank_id, request_context=RC)

    ids = [str(uuid.uuid4()) for _ in range(3)]
    texts = ["the cat sat on the mat", "dogs are loyal animals", "birds can fly in the sky"]
    embeddings = await embedding_utils.generate_embeddings_batch(engine.embeddings, texts)

    if get_memories().store_owned_for(bank_id):
        await store_write_facts(engine, bank_id, ids, texts, embeddings=embeddings, text_search=False)
    else:
        pool = await engine._get_pool()
        async with pool.acquire() as conn:
            for fact_id, text, emb in zip(ids, texts, embeddings):
                await conn.execute(
                    "INSERT INTO memory_units (id, bank_id, text, fact_type, embedding) "
                    "VALUES ($1, $2, $3, 'world', $4::vector)",
                    fact_id,
                    bank_id,
                    text,
                    "[" + ",".join(str(v) for v in emb) + "]",
                )

    monkeypatch.setattr(engine._cross_encoder_reranker, "cross_encoder", _StubReranker("cohere"))

    yield engine, bank_id

    await engine.delete_bank(bank_id, request_context=RC)


def _ids(result) -> set[str]:
    return {str(r.id) for r in result.results}


async def _recall(engine, bank_id, **kwargs):
    return await engine.recall_async(
        bank_id=bank_id, query="animals and nature", request_context=RC, max_tokens=10000, **kwargs
    )


async def _set_floor(engine, bank_id: str, floor: float) -> None:
    await engine._config_resolver.update_bank_config(bank_id, {"recall_min_score": floor})


class TestBankFloor:
    async def test_floor_drops_results_below_it(self, seeded_memory):
        engine, bank_id = seeded_memory
        baseline = await _recall(engine, bank_id)
        finals = sorted((r.scores.final for r in baseline.results), reverse=True)
        assert len(finals) >= 2, "need at least two results to place a floor between them"
        floor = finals[-1] + (finals[-2] - finals[-1]) / 2

        await _set_floor(engine, bank_id, floor)
        floored = await _recall(engine, bank_id)

        assert _ids(floored), "the floor sits below the top result"
        assert _ids(floored) < _ids(baseline)
        assert all(r.scores.final >= floor for r in floored.results)

    async def test_nothing_relevant_returns_nothing(self, seeded_memory):
        engine, bank_id = seeded_memory
        assert (await _recall(engine, bank_id)).results

        await _set_floor(engine, bank_id, UNREACHABLE_FLOOR + 10)

        assert (await _recall(engine, bank_id)).results == []

    async def test_zero_floor_filters_nothing(self, seeded_memory):
        engine, bank_id = seeded_memory
        baseline = await _recall(engine, bank_id)

        await _set_floor(engine, bank_id, 0.0)

        assert _ids(await _recall(engine, bank_id)) == _ids(baseline)

    async def test_request_floor_wins_over_bank_floor(self, seeded_memory):
        engine, bank_id = seeded_memory
        baseline = await _recall(engine, bank_id)
        await _set_floor(engine, bank_id, UNREACHABLE_FLOOR + 10)

        unfloored = await _recall(engine, bank_id, min_scores=MinScores(final=0.0))

        assert _ids(unfloored) == _ids(baseline)

    async def test_other_request_floors_leave_the_bank_floor_in_place(self, seeded_memory):
        """Setting only min_scores.reranker is not a statement about the final floor."""
        engine, bank_id = seeded_memory
        await _set_floor(engine, bank_id, UNREACHABLE_FLOOR + 10)

        assert (await _recall(engine, bank_id, min_scores=MinScores(reranker=0.0))).results == []


class TestBankFloorNeedsRelevanceScores:
    """The floor is a cutoff on relevance; rankings that publish positions are left alone."""

    async def test_skipped_under_a_decision_model(self, seeded_memory, monkeypatch):
        engine, bank_id = seeded_memory
        monkeypatch.setattr(engine._cross_encoder_reranker, "cross_encoder", _StubReranker("decision_model"))
        baseline = await _recall(engine, bank_id)
        assert len(baseline.results) >= 2

        await _set_floor(engine, bank_id, UNREACHABLE_FLOOR)

        assert _ids(await _recall(engine, bank_id)) == _ids(baseline)

    async def test_skipped_under_fusion_passthrough(self, seeded_memory):
        engine, bank_id = seeded_memory
        baseline = await _recall(engine, bank_id, reranking="rrf")
        assert baseline.results

        await _set_floor(engine, bank_id, UNREACHABLE_FLOOR + 10)

        assert _ids(await _recall(engine, bank_id, reranking="rrf")) == _ids(baseline)


class TestConfig:
    def test_is_off_by_default_and_overridable_per_bank(self, monkeypatch):
        monkeypatch.delenv("HINDSIGHT_API_RECALL_MIN_SCORE", raising=False)

        assert HindsightConfig.from_env().recall_min_score == 0.0
        assert "recall_min_score" in HindsightConfig.get_configurable_fields()

    def test_env_sets_the_server_default(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_RECALL_MIN_SCORE", "0.25")

        assert HindsightConfig.from_env().recall_min_score == 0.25
