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

from typing import List
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .custom_agent_config import CustomAgentConfig
from .llm_agent_config import LlmAgentConfig
from .loop_agent_config import LoopAgentConfig

AgentConfig = Union[LlmAgentConfig, CustomAgentConfig, LoopAgentConfig]


class AgentsConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")

  agents: List[AgentConfig]
