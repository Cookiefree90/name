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

from typing import Any
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict


class SubAgentConfig(BaseModel):
  """The config for a sub-agent."""

  model_config = ConfigDict(extra="forbid")

  config: Optional[str] = None
  """The config file path of the sub-agent defined via agent config. It could
   also be the name of the agent defined in the same file.

  Example:

    ```
    sub_agents:
      - config: search_agent.yaml
      - config: my_library/my_custom_agent.yaml
      - config: another_agent_defined_in_the_same_file
    ```
  """

  code: Optional[CodeConfig] = None
  """The agent defined by code.

  When this exists, it override the config.

  Example:

    For below agent defined in Python code:

    ```
    # my_library/custom_agents.py
    from google.adk.agents import LlmAgent

    my_custom_agent = LlmAgent(
        name="my_custom_agent",
        instruction="You are a helpful custom agent.",
        model="gemini-2.0-flash",
    )
    ```

    The yaml config should be:

    ```
    sub_agents:
      - code:
          name: my_library.custom_agents.my_custom_agent
    ```

    For below agent defined in Java code:

    ```
    package com.acme.agents

    class CustomAgent {
      public static final LlmAgent customAgent = LlmAgent.builder()
        .name("custom_agent")
        .model("gemini-2.0-flash")
        .instruction("You are a helpful assistant.")
        .build();
    }
    ```

    ```
    sub_agents:
      - code:
          name: com.acme.agents.customAgent.customAgent
    ```
  """


class ArgumentConfig(BaseModel):
  """An argument passed to a method or constructor."""

  model_config = ConfigDict(extra="forbid")

  name: Optional[str] = None
  """The argument name. Optional

  When the argument is for a positional argument, this can be omitted.
  """

  value: Any
  """The argument value."""


class CodeConfig(BaseModel):
  """Reference the code"""

  model_config = ConfigDict(extra="forbid")

  name: str
  """Required. The name of the variable, method, class, etc. in Python or Java.

  Examples:

    When used for agent_class, it could be ADK builtin agents, such as LlmAgent,
    SequetialAgent, etc.

    When used for tools,

      - It can be ADK builtin tools, such as google_search, etc.
      - It can be users' custom tools, e.g. my_library.my_tools.my_tool.

    When used for clalbacks, it refers to a function, e.g. my_library.my_callbacks.my_callback
  """

  args: Optional[List[ArgumentConfig]] = None
  """The arguments for the code when name references a method or a contructor.

  Examples:

    Below is a sample usage:

    ```
    tools
      - name: AgentTool
        args:
          - name: agent
            value: search_agent.yaml
          - name: skip_summarization
            value: True
    ```
  """
