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
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ..events.event import Event


class BaseCompactionStrategy(BaseModel):
  """Base interface for compacting events."""

  def compact_events(
      self,
      current_branch: Optional[str],
      events: list[Event],
      agent_name: str = '',
  ) -> Event:
    """Compacts the events.

      This method will summarize the events and return a new summray event
      indicating the range of events it summarized.

    When sending events to the LLM, if a summary event is present, the events it
    replaces (those identified in itssummary_range) should not be included.

      Args:
        current_branch: The current branch of the agent.
        events: Events to compact.
        agent_name: The name of the agent.

      Returns:
        The new compaction event.
    """
    raise NotImplementedError()
