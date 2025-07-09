from __future__ import annotations

from typing import Optional

from .base_agent_config import BaseAgentConfig


class LoopAgentConfig(BaseAgentConfig):
  """Configuration for a loop agent."""

  max_iterations: Optional[int] = None
  """LoopAgent.max_iterations. Optional"""
