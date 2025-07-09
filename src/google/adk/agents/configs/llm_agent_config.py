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
from typing import Literal
from typing import Optional

from .base_agent_config import BaseAgentConfig
from .common_config import CodeConfig


class LlmAgentConfig(BaseAgentConfig):
  """Config for LlmAgent."""

  model: Optional[str] = None
  """LlmAgent.model. Optional.

  When not set, using the same model as the parent model.
  """

  instruction: str
  """LlmAgent.instruction. Required."""

  tools: Optional[List[CodeConfig]] = None
  """LlmAgent.tools. Optional.

  Examples:

    For ADK builtin tools in google.adk.tools package, they can be referenced
    directly with the name.

    ```
    - tools:
      - name: google_search
      - name: load_memory
    ```

    For user-defined tools, they can be referenced with fully qualified name.

    ```
    - tools:
      - name: my_library.my_tools.my_tool
    ```

    For tools that needs to be created via functions.

    ```
    - tools:
      - name: my_library.my_tools.create_tool
        - args:
          - name: param1
            value: value1
          - name: param2
            value: value2
    ```

    For more advanced tools, instead of specifying arguments in config, it's
    recommended defining them in python files and reference them.

    ```
    # tools.py
    my_mcp_toolset = MCPToolset(
        connection_params=StdioServerParameters(
            command="npx",
            args=["-y", "@notionhq/notion-mcp-server"],
            env={"OPENAPI_MCP_HEADERS": NOTION_HEADERS},
        )
    )
    ```

    Then, reference the toolset in config:

    ```
    - tools:
      - name: tools.my_mcp_toolset
    ```
  """

  input_schema: Optional[CodeConfig] = None
  """LlmAgent.input_schema. Optional."""
  output_schema: Optional[CodeConfig] = None
  """LlmAgent.output_schema. Optional."""

  before_model_callbacks: Optional[List[CodeConfig]] = None
  after_model_callbacks: Optional[List[CodeConfig]] = None
  before_tool_callbacks: Optional[List[CodeConfig]] = None
  after_tool_callbacks: Optional[List[CodeConfig]] = None

  disallow_transfer_to_parent: Optional[bool] = None
  """LlmAgent.disallow_transfer_to_parent. Optional."""

  disallow_transfer_to_peers: Optional[bool] = None
  """LlmAgent.disallow_transfer_to_peers. Optional."""

  include_contents: Literal['default', 'none'] = 'default'
  """LlmAgent.include_contents. Optional."""

  output_key: Optional[str] = None
  """LlmAgent.output_key. Optional."""
