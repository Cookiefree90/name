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

from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from ...utils.feature_decorator import experimental
from ..base_toolset import BaseToolset
from .computer import Computer
from .computer_use_tool import ComputerUseTool


@experimental
class ComputerUseToolset(BaseToolset):

  def __init__(
      self,
      *,
      computer: Computer,
  ):
    super().__init__()
    self._computer = computer
    self._initialized = False

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self._computer.initialize()
      self._initialized = True

  @override
  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> list[ComputerUseTool]:
    await self._ensure_initialized()
    # Get screen size and environment for tool configuration
    screen_size = await self._computer.screen_size()
    environment = await self._computer.environment()

    # Get all methods defined in Computer abstract base class, excluding specified methods
    excluded_methods = {'screen_size', 'environment', 'close'}
    computer_methods = []

    # Get all methods defined in the Computer ABC interface
    for method_name in dir(Computer):
      # Skip private methods (starting with underscore)
      if method_name.startswith('_'):
        continue

      # Skip excluded methods
      if method_name in excluded_methods:
        continue

      # Check if it's a method defined in Computer class
      attr = getattr(Computer, method_name, None)
      if attr is not None and callable(attr):
        # Get the corresponding method from the concrete instance
        instance_method = getattr(self._computer, method_name)
        computer_methods.append(instance_method)

    # Create ComputerUseTool instances for each method

    return [
        ComputerUseTool(
            func=method,
            screen_size=screen_size,
            environment=environment,
        )
        for method in computer_methods
    ]

  @override
  async def close(self) -> None:
    await self._computer.close()
