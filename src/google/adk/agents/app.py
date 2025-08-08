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

from typing import Union
from pydantic import BaseModel, Field
from ..plugins.base_plugin import BasePlugin
from .base_agent import BaseAgent
from .base_compaction_strategy import BaseCompactionStrategy
from .base_contents_strategy import BaseContentsStrategy
from ..events.event import Event

class App(BaseModel):
  """Agentic application."""

  name: str
  """The name of the application."""

  root_agent: BaseAgent = None
  """The root agent in the application."""

  plugins: list[BasePlugin] = Field(default_factory=list)
  """The plugins in the application."""

  include_contents: Union[str, BaseContentsStrategy] = None
  """The strategy to include contents in the application."""

  compaction_strategy: Union[str, BaseCompactionStrategy] = None
  """The strategy to compact the contents in the application."""
