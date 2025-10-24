"""
Conversation service mediating between API layer, knowledge service, and agents.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from knowledge import KnowledgeService

try:
    from agents.runtime import AgentRuntime  # pragma: no cover - optional during bootstrap
except ImportError:  # pragma: no cover - agents module may not be ready yet
    AgentRuntime = None  # type: ignore


class ConversationService:
    """Facade for handling conversational queries.

    The service prefers routing through an AgentRuntime when available;
    otherwise it falls back to pure knowledge service (vector / keyword search).
    """

    def __init__(
        self,
        knowledge_service: KnowledgeService,
        agent_runtime: Optional["AgentRuntime"] = None,
    ):
        self._knowledge_service = knowledge_service
        self._agent_runtime = agent_runtime

    @property
    def knowledge(self) -> KnowledgeService:
        """Expose the underlying knowledge service."""
        return self._knowledge_service

    def set_agent_runtime(self, runtime: Optional["AgentRuntime"]):
        """Attach or swap the active agent runtime."""
        self._agent_runtime = runtime

    async def rag_query(
        self,
        message: str,
        *,
        search_params: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle semantic retrieval augmented generation."""
        params = search_params or {}
        top_k = params.get("top_k") or params.get("similarity_top_k")
        response_mode = params.get("response_mode", "compact")

        # Future: route through agent runtime when configured
        return await self._knowledge_service.rag_search_async(
            message,
            similarity_top_k=top_k,
            response_mode=response_mode,
            tenant_id=tenant_id,
        )

    async def keyword_query(
        self,
        message: str,
        *,
        search_params: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle keyword search via Elasticsearch."""
        params = search_params or {}
        top_k = params.get("top_k") or params.get("keyword_top_k") or 5
        min_score = params.get("min_score")

        return await self._knowledge_service.keyword_search_async(
            message,
            top_k=top_k,
            min_score=min_score,
            tenant_id=tenant_id,
        )

    async def measure_latency(self, coro) -> Dict[str, Any]:
        """Utility to measure latency for downstream consumers."""
        start = time.time()
        result = await coro
        duration = time.time() - start
        result.setdefault("metadata", {})
        result["metadata"]["response_time_secs"] = duration
        return result

