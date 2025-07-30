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

import json
from typing import Any
from typing import Literal
from typing import Optional

from pydantic import BaseModel

from ..tools.base_tool import BaseTool
from ..tools.tool_context import ToolContext
from .base_plugin import BasePlugin


class ReflectAndRetryPluginResponse(BaseModel):
  """Response from ReflectAndRetryPlugin."""

  response_type: Literal[str] = "ERROR_HANDLED_BY_REFLEX_AND_RETRY_PLUGIN"
  error_type: str = ""
  error_details: str = ""
  retry_count: int = 0
  reflection_guidance: str = ""


class ReflectAndRetryPlugin(BasePlugin):
  """A plugin that provides error recovery through reflection and retry logic.

  When tool calls fail with exception, this plugin generates instructional
  responses that encourage the model to reflect on the error and try a
  different approach, rather than simply propagating the error.

  This plugin is particularly useful for handling transient errors, API
  limitations, or cases where the model might need to adjust its strategy
  based on encountered obstacles.

  Example:
      >>> reflect_retry_plugin = ReflectAndRetryPlugin()
      >>> runner = Runner(
      ...     agents=[my_agent],
      ...     plugins=[reflect_retry_plugin],
      ... )
  """

  def __init__(self, name: str = "reflect_retry_plugin", max_retries: int = 3):
    """Initialize the reflect and retry plugin.

    Args:
      name: The name of the plugin instance.
      max_retries: Maximum number of retries to attempt before giving up.
    """
    super().__init__(name)
    self.max_retries = max_retries
    self._retry_counts: dict[str, int] = {}

  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> Optional[dict]:
    """Handle tool execution errors with reflection and retry logic."""
    retry_key = self._get_retry_key(
        tool_context.invocation_id, f"tool:{tool.name}"
    )

    if not self._should_retry(retry_key):
      return self._get_tool_retry_exceed_msg(tool, error)

    retry_count = self._increment_retry_count(retry_key)

    # Create a reflective response instead of propagating the error
    return self._create_tool_reflection_response(
        tool, tool_args, error, retry_count
    )

  def _get_retry_key(self, context_id: str, operation: str) -> str:
    """Generate a unique key for tracking retries."""
    return f"{context_id}:{operation}"

  def _should_retry(self, retry_key: str) -> bool:
    """Check if we should attempt a retry for this operation."""
    current_count = self._retry_counts.get(retry_key, 0)
    return current_count < self.max_retries

  def _increment_retry_count(self, retry_key: str) -> int:
    """Increment and return the retry count for an operation."""
    self._retry_counts[retry_key] = self._retry_counts.get(retry_key, 0) + 1
    return self._retry_counts[retry_key]

  def _format_error_details(self, error: Exception) -> str:
    """Format error details for inclusion in reflection message."""
    error_type = type(error).__name__
    error_message = str(error)
    return f"{error_type}: {error_message}"

  def _create_tool_reflection_response(
      self,
      tool: BaseTool,
      tool_args: dict[str, Any],
      error: Exception,
      retry_count: int,
  ) -> dict[str, Any]:
    """Create a reflection response for tool errors."""
    args_summary = json.dumps(tool_args, indent=2, default=str)
    error_details = self._format_error_details(error)

    reflection_message = f"""
The tool call to '{tool.name}' failed with the following error:

Error: {error_details}

Tool Arguments Used:
{args_summary}

**Reflection Instructions:**
When realizing the current approach won't work, think about the potential issues and explicitly try a different approach. Consider:

1. **Parameter Issues**: Are the arguments correctly formatted or within expected ranges?
2. **Alternative Methods**: Is there a different tool or approach that might work better?
3. **Error Context**: What does this specific error tell you about what went wrong?
4. **Incremental Steps**: Can you break down the task into smaller, more manageable steps?

This is retry attempt {retry_count} of {self.max_retries}. Please analyze the error and adjust your strategy accordingly.

Instead of repeating the same approach, explicitly state what you learned from this error and how you plan to modify your approach.
"""

    return ReflectAndRetryPluginResponse(
        error_type=type(error).__name__,
        error_details=str(error),
        retry_count=retry_count,
        reflection_guidance=reflection_message.strip(),
    ).model_dump(mode="json")

  def _get_tool_retry_exceed_msg(
      self,
      tool: BaseTool,
      error: Exception,
  ) -> dict[str, Any]:
    """Create a reflection response for tool errors."""
    reflection_message = f"""
The tool call to '{tool.name}' has failed {self.max_retries} times and has exceeded the maximum retry limit.

Last Error: {self._format_error_details(error)}

**Instructions:**
Do not attempt to use this tool ('{tool.name}') again for this task.
You must try a different approach, using a different tool or strategy to accomplish the goal.
"""
    return ReflectAndRetryPluginResponse(
        error_type=type(error).__name__,
        error_details=str(error),
        retry_count=self.max_retries,
        reflection_guidance=reflection_message.strip(),
    ).model_dump(mode="json")
