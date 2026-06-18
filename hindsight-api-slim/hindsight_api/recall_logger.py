"""RecallLogger — an OperationValidatorExtension that logs every recall's
returned facts to stdout as one structured JSON line.

WHY THIS EXISTS
---------------
Your hard requirement #1 is "see the exact text snippets injected into a
conversation." For Claude Code, the plugin writes a local `last_recall.json`
you can tail. But when *claude.ai* (the browser) calls recall over MCP, there
is no local file — the only place that truth exists is the server. This
extension makes the server print it.

HOW HINDSIGHT LOADS IT
----------------------
Hindsight loads ONE operation-validator extension, named by an env var
(set in the Dockerfile / Railway):

    HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION=recall_logger:RecallLogger

Because `PYTHONPATH=/app/extensions`, `recall_logger` is importable and the
`:RecallLogger` part selects the class. No upstream source is edited.

Railway (and `docker logs`) capture stdout, so:

    railway logs | grep RECALL_INJECTION | jq '.facts[]'

gives a live window of exactly which facts entered the context window each time
any client calls recall. That's what `watch_server_injections.sh` does.

API NOTES (verified against hindsight_api ~0.8.x)
-------------------------------------------------
* Public import path is the package `hindsight_api.extensions` (it re-exports
  these names), NOT the `...operation_validator` submodule.
* Three methods are abstract and MUST be implemented: validate_retain,
  validate_recall, validate_reflect. Each takes a single `ctx` dataclass and
  returns a ValidationResult. We accept everything (this is audit-only).
* on_recall_complete(result) fires after each recall. `result.result.results`
  is the list of MemoryFact objects; each fact has `.text`, `.fact_type`,
  `.entities`, `.context` (and more) — but NO relevance/`activation` score.
  Field access stays defensive (getattr) so upgrades can't break logging.
"""

from __future__ import annotations

import json
import sys

from hindsight_api.extensions import (
    OperationValidatorExtension,
    RecallResult,
    ValidationResult,
)


def _emit(payload: dict) -> None:
    """Write one JSON line to stdout, flushed so logs capture it immediately."""
    sys.stdout.write("RECALL_INJECTION " + json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


class RecallLogger(OperationValidatorExtension):
    """Audit-only validator: passes every operation through, logs recall results."""

    # --- Required abstract hooks: accept everything, modify nothing ---------
    # Each receives a single context dataclass (RetainContext / RecallContext /
    # ReflectContext). We never block, so we ignore ctx and accept.

    async def validate_retain(self, ctx) -> ValidationResult:  # noqa: ANN001
        return ValidationResult.accept()

    async def validate_recall(self, ctx) -> ValidationResult:  # noqa: ANN001
        return ValidationResult.accept()

    async def validate_reflect(self, ctx) -> ValidationResult:  # noqa: ANN001
        return ValidationResult.accept()

    # --- The hook we actually care about ------------------------------------

    async def on_recall_complete(self, result: RecallResult) -> None:
        """Fires after every recall. Logs the exact facts that were returned."""
        try:
            if (
                not getattr(result, "success", False)
                or getattr(result, "result", None) is None
            ):
                _emit(
                    {
                        "ok": False,
                        "bank": getattr(result, "bank_id", None),
                        "query": getattr(result, "query", None),
                        "error": getattr(result, "error", None),
                    }
                )
                return

            facts = []
            for f in getattr(result.result, "results", None) or []:
                facts.append(
                    {
                        "text": getattr(f, "text", None),
                        "type": getattr(f, "fact_type", None),
                        # No per-fact relevance score exists in this API; entities
                        # and context are the useful "why was this pulled" signal.
                        "entities": getattr(f, "entities", None),
                        "context": getattr(f, "context", None),
                    }
                )

            _emit(
                {
                    "ok": True,
                    "bank": getattr(result, "bank_id", None),
                    "query": getattr(result, "query", None),
                    "budget": getattr(result, "budget", None),
                    "max_tokens": getattr(result, "max_tokens", None),
                    "fact_count": len(facts),
                    "facts": facts,
                }
            )
        except Exception as e:  # never let logging break a recall
            _emit({"ok": False, "logger_error": repr(e)})
