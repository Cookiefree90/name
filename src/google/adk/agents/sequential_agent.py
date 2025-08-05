# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sequential agent implementation."""


from __future__ import annotations

from typing import AsyncGenerator
from typing import Type

from typing_extensions import override

from ..events.event import Event
from ..utils.feature_decorator import working_in_progress
from .base_agent import BaseAgent
from .base_agent import BaseAgentConfig
from .invocation_context import InvocationContext
from .llm_agent import LlmAgent
from .sequential_agent_config import SequentialAgentConfig




class SequentialAgent(BaseAgent):
  """A shell agent that runs its sub-agents in sequence."""

  config_type: Type[BaseAgentConfig] = SequentialAgentConfig
  """The config type for this agent."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    for sub_agent in self.sub_agents:
      async for event in sub_agent.run_async(ctx):
        yield event

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Implementation for live SequentialAgent.

    In a live run, this agent executes its sub-agents one by one. It relies
    on the `generation_complete` event from the underlying model to determine
    when a sub-agent has finished its turn. Once a sub-agent's `run_live`
    stream concludes (triggered by the `generation_complete` event), the
    `SequentialAgent` will proceed to execute the next sub-agent in the
    sequence.

    Args:
      ctx: The invocation context of the agent.
    """
    for sub_agent in self.sub_agents:
      async for event in sub_agent.run_live(ctx):
        yield event
        if event.generation_complete:
          break

  @classmethod
  @override
  @working_in_progress('SequentialAgent.from_config is not ready for use.')
  def from_config(
      cls: Type[SequentialAgent],
      config: SequentialAgentConfig,
      config_abs_path: str,
  ) -> SequentialAgent:
    return super().from_config(config, config_abs_path)
