"""
Minimal Agno runtime placeholder.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class AgentRuntime:
    """Stub runtime that will be extended with Agno-based orchestration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    async def run(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Placeholder implementation."""
        raise NotImplementedError("AgentRuntime integration coming soon.")
