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
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict

from .common_config import CodeConfig
from .common_config import SubAgentConfig


# Should not be used directly, only as a sub-class
class BaseAgentConfig(BaseModel):
  """The config for a BaseAgent."""

  model_config = ConfigDict(extra="forbid")

  name: str
  """BaseAgent.name. Required."""

  description: str = ""
  """BaseAgent.description. Optional."""

  before_agent_callbacks: Optional[List[CodeConfig]] = None
  """BaseAgent.before_agent_callbacks. Optional.

  Examples:

    ```
    before_agent_callbacks:
      - name: my_library.security_callbacks.before_agent_callback
    ```
  """

  after_agent_callbacks: Optional[List[CodeConfig]] = None
  """BaseAgent.after_agent_callbacks. Optional."""

  sub_agents: Optional[List[SubAgentConfig]] = None
  """The sub_agents of this agent.

  Examples:

    Below is a sample with two sub-agents in yaml.
    - The first agent is defined via config.
    - The second agent is implemented python file.

    ```
    sub_agents:
      - config: search_agent.yaml
      - code:
          name: my_library.my_custom_agent
    ```
  """
