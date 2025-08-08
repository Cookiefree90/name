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

import abc
import logging
from typing import Optional

from ..agents.callback_context import CallbackContext
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from .base_plugin import BasePlugin

logger = logging.getLogger('google_adk.' + __name__)

class BaseContextReductionStrategy(abc.ABC):
  """Abstract base class for context reduction strategies."""

  @abc.abstractmethod
  def reduce(self, llm_request: LlmRequest):
    """Reduces the context of the LLM request in place."""
    raise NotImplementedError


class TotalMessagesStrategy(BaseContextReductionStrategy):
  """Reduces context by keeping the last N total messages."""

  def __init__(self, limit: int):
    """Initializes the strategy.

    Args:
        limit: The maximum number of total messages to keep.
    """
    self.limit = limit

  def reduce(self, llm_request: LlmRequest):
    """Reduces context by keeping the last N total messages, always preserving the first message."""
    contents = llm_request.contents
    if len(contents) > self.limit:
      original_count = len(contents)
      system_prompt = contents[0]
      history = contents[1:]

      # The number of history messages to keep is the limit minus the system prompt.
      history_to_keep_count = self.limit - 1

      reduced_history = (
          history[-history_to_keep_count:] if history_to_keep_count > 0 else []
      )

      llm_request.contents = [system_prompt] + reduced_history
      logger.info(
          f"Reduced context from {original_count} to"
          f" {len(llm_request.contents)} messages using TotalMessagesStrategy."
      )


class UserMessagesStrategy(BaseContextReductionStrategy):
  """Reduces context by keeping the last N user messages, preserving the system prompt."""

  def __init__(self, limit: int):
    """Initializes the strategy.

    Args:
        limit: The maximum number of user messages to keep.
    """
    self.limit = limit

  def reduce(self, llm_request: LlmRequest):
    """Reduces context by keeping the last N user messages and any model message in-between, always preserving the first message."""
    contents = llm_request.contents
    # The first message is typically a system prompt and should always be preserved.
    # The history starts from the second message.
    if len(contents) <= 1:
      return

    history = contents[1:]
    user_messages_in_history = sum(1 for msg in history if msg.role == "user")

    if user_messages_in_history <= self.limit:
      return  # No reduction needed.

    original_count = len(contents)
    system_prompt = contents[0]

    user_message_count = 0
    # The index from which to slice the history to keep the desired messages.
    cutoff_index = len(history)

    # Iterate backwards through the history to find the cutoff point.
    for i in range(len(history) - 1, -1, -1):
      if history[i].role == "user":
        user_message_count += 1
      if user_message_count >= self.limit:
        cutoff_index = i
        break

    reduced_history = history[cutoff_index:]
    llm_request.contents = [system_prompt] + reduced_history
    logger.info(
        f"Reduced context from {original_count} to"
        f" {len(llm_request.contents)} messages using UserMessagesStrategy."
    )


class ContextManagementPlugin(BasePlugin):
  """A plugin that manages the LLM context by reducing its size using a given strategy.

  This plugin delegates the context reduction logic to a strategy object.

  Example:
      # To reduce context based on the total number of messages.
      >>> plugin1 = ContextManagementPlugin(
      ...     strategy=TotalMessagesStrategy(limit=15)
      ... )

      # To reduce context based on the number of user messages.
      >>> plugin2 = ContextManagementPlugin(
      ...     strategy=UserMessagesStrategy(limit=5)
      ... )

      # Then, add the plugin to your runner.
      >>> runner = Runner(
      ...     agents=[my_agent],
      ...     # ...
      ...     plugins=[plugin1],
      ... )
  """

  def __init__(
      self,
      strategy: BaseContextReductionStrategy,
      name: str = "context_management_plugin",
  ):
    """Initializes the context management plugin.

    Args:
      strategy: An instance of a context reduction strategy class.
      name: The name of the plugin instance.
    """
    super().__init__(name)
    self.strategy = strategy

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Compresses the context by applying the configured strategy."""
    try:
      self.strategy.reduce(llm_request)
    except Exception as e:
      logger.error(f"Failed to reduce context for request: {e}")

    return None
