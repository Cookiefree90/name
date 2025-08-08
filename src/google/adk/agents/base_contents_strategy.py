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

from typing import List, Optional
from pydantic import BaseModel
from ..events.event import Event


class BaseContentsStrategy(BaseModel):
  """Base interface for handling event filtering or select logics."""

  def get_events(
      self,
      current_branch: Optional[str],
      events: list[Event],
      agent_name: str = '',
  ) -> List[Event]:
    """Get the contents for the LLM request.

    Applies filtering, rearrangement, and content processing to events.

    Args:
      current_branch: The current branch of the agent.
      events: Events to process.
      agent_name: The name of the agent.

    Returns:
      A list of returned contents.
    """
    raise NotImplementedError()
