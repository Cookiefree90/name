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
from typing import Union

from google.adk.agents.readonly_context import ReadonlyContext
from typing_extensions import override

from . import metadata_tool
from . import query_tool
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.base_toolset import ToolPredicate
from ...utils.feature_decorator import experimental
from .bigtable_credentials import BigtableCredentialsConfig
from .bigtable_tool import BigtableTool
from .config import BigtableToolConfig


@experimental
class BigtableToolset(BaseToolset):
  """Bigtable Toolset contains tools for interacting with Bigtable data and metadata."""

  def __init__(
      self,
      *,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
      credentials_config: Optional[BigtableCredentialsConfig] = None,
      bigtable_tool_config: Optional[BigtableToolConfig] = None,
  ):
    self.tool_filter = tool_filter
    self._credentials_config = credentials_config
    self._tool_config = bigtable_tool_config

  def _is_tool_selected(
      self, tool: BaseTool, readonly_context: ReadonlyContext
  ) -> bool:
    if self.tool_filter is None:
      return True

    if isinstance(self.tool_filter, ToolPredicate):
      return self.tool_filter(tool, readonly_context)

    if isinstance(self.tool_filter, list):
      return tool.name in self.tool_filter

    return False

  @override
  async def get_tools(
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> List[BaseTool]:
    """Get tools from the toolset."""
    all_tools = [
        # Example of how tools would be added:
        BigtableTool(
            func=func,
            credentials_config=self._credentials_config,
            bigtable_tool_config=self._tool_config,
        )
        for func in [
            metadata_tool.list_instances,
            metadata_tool.get_instance_info,
            metadata_tool.list_tables,
            metadata_tool.get_table_info,
            query_tool.get_execute_sql(self._tool_config),
        ]
    ]
    #  No tools are added yet, this is a placeholder
    #  Uncomment and modify the above section when you add tool functions.

    return [
        tool
        for tool in all_tools
        if self._is_tool_selected(tool, readonly_context)
    ]

  @override
  async def close(self):
    pass
