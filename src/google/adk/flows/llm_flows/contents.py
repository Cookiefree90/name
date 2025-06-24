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

import copy
from collections import defaultdict
from operator import attrgetter, itemgetter
from typing import AsyncGenerator
from typing import Optional

from google.genai import types
from pydantic import BaseModel
from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ._base_llm_processor import BaseLlmRequestProcessor
from .functions import remove_client_function_call_id
from .functions import REQUEST_EUC_FUNCTION_CALL_NAME


class _ContentLlmRequestProcessor(BaseLlmRequestProcessor):
  """Builds the contents for the LLM request."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    if agent.include_contents != 'none':
      llm_request.contents = _get_contents(
          invocation_context.branch,
          invocation_context.session.events,
          agent.name,
      )

    # Maintain async generator behavior
    if False:  # Ensures it behaves as a generator
      yield  # This is a no-op but maintains generator structure


request_processor = _ContentLlmRequestProcessor()


class OrderedPart(BaseModel):
  event_index: int
  part_index: int
  event_timestamp: float
  role: str
  part: types.Part

  @property
  def sort_key(self):
    return (self.event_index, self.event_timestamp, self.part_index)


def _extract_latest_function_parts(
    events: list[Event],
) -> list[types.Content]:
  """Rearrange the async function_response events in the history."""

  function_call_id_to_parts: dict[str, list[OrderedPart]] = defaultdict(list)
  all_parts: list[OrderedPart] = []

  for i, event in enumerate(events):
    for j, part in enumerate(event.content.parts):
      ordered_part = OrderedPart(
          event_index=i,
          part_index=j,
          event_timestamp=event.timestamp,
          role=event.content.role,
          part=part,
      )
      if part.function_response:
        function_call_id_to_parts[part.function_response.id].append(
            ordered_part
        )
      elif part.function_call:
        function_call_id_to_parts[part.function_call.id].append(ordered_part)
      else:
        all_parts.append(ordered_part)

  for function_call_id, parts in function_call_id_to_parts.items():
    fc_ordered_part = max(
        [
            ordered_part
            for ordered_part in parts
            if ordered_part.part.function_call
        ],
        key=attrgetter('sort_key'),
        default=None,
    )
    fr_ordered_part = max(
        [
            ordered_part
            for ordered_part in parts
            if ordered_part.part.function_response
        ],
        key=attrgetter('sort_key'),
        default=None,
    )
    if fr_ordered_part:
      if fc_ordered_part:
        fc_ordered_part = fc_ordered_part.model_copy()
        fc_ordered_part.event_index = fr_ordered_part.event_index - 1
        all_parts.append(fc_ordered_part)
      all_parts.append(fr_ordered_part)
    else:
      all_parts.append(fc_ordered_part)

  sorted_parts = sorted(all_parts, key=attrgetter('sort_key'))

  all_content = []
  current_parts = []
  current_role = None
  for ordered_part in sorted_parts:
    part = ordered_part.part
    if ordered_part.role is None:
      continue
    if ordered_part.role == current_role:
      current_parts.append((ordered_part.event_timestamp, part))
    else:
      if current_parts:
        all_content.append(
            types.Content(
                role=current_role,
                parts=[
                    _part
                    for _ts, _part in sorted(current_parts, key=itemgetter(0))
                ],
            )
        )
      current_role = ordered_part.role
      current_parts = [(ordered_part.event_timestamp, part)]
  if current_parts:
    all_content.append(
        types.Content(
            role=current_role,
            parts=[
                _part for _ts, _part in sorted(current_parts, key=itemgetter(0))
            ],
        )
    )

  return all_content


def _get_contents(
    current_branch: Optional[str], events: list[Event], agent_name: str = ''
) -> list[types.Content]:
  """Get the contents for the LLM request.

  Args:
    current_branch: The current branch of the agent.
    events: A list of events.
    agent_name: The name of the agent.

  Returns:
    A list of contents.
  """
  filtered_events = []
  # Parse the events, leaving the contents and the function calls and
  # responses from the current agent.
  for event in events:
    if (
        not event.content
        or not event.content.role
        or not event.content.parts
        or event.content.parts[0].text == ''
    ):
      # Skip events without content, or generated neither by user nor by model
      # or has empty text.
      # E.g. events purely for mutating session states.
      continue
    if not _is_event_belongs_to_branch(current_branch, event):
      # Skip events not belong to current branch.
      continue
    if _is_auth_event(event):
      # skip auth event
      continue
    filtered_events.append(
        _convert_foreign_event(event)
        if _is_other_agent_reply(agent_name, event)
        else event
    )
  contents = []
  for content in _extract_latest_function_parts(filtered_events):
    content = copy.deepcopy(content)
    remove_client_function_call_id(content)
    contents.append(content)
  return contents


def _is_other_agent_reply(current_agent_name: str, event: Event) -> bool:
  """Whether the event is a reply from another agent."""
  return bool(
      current_agent_name
      and event.author != current_agent_name
      and event.author != 'user'
  )


def _convert_foreign_event(event: Event) -> Event:
  """Converts an event authored by another agent as a user-content event.

  This is to provide another agent's output as context to the current agent, so
  that current agent can continue to respond, such as summarizing previous
  agent's reply, etc.

  Args:
    event: The event to convert.

  Returns:
    The converted event.

  """
  if not event.content or not event.content.parts:
    return event

  content = types.Content()
  content.role = 'user'
  content.parts = [types.Part(text='For context:')]
  for part in event.content.parts:
    if part.text:
      content.parts.append(
          types.Part(text=f'[{event.author}] said: {part.text}')
      )
    elif part.function_call:
      content.parts.append(
          types.Part(
              text=(
                  f'[{event.author}] called tool `{part.function_call.name}`'
                  f' with parameters: {part.function_call.args}'
              )
          )
      )
    elif part.function_response:
      # Otherwise, create a new text part.
      content.parts.append(
          types.Part(
              text=(
                  f'[{event.author}] `{part.function_response.name}` tool'
                  f' returned result: {part.function_response.response}'
              )
          )
      )
    # Fallback to the original part for non-text and non-functionCall parts.
    else:
      content.parts.append(part)

  return Event(
      timestamp=event.timestamp,
      author='user',
      content=content,
      branch=event.branch,
  )


def _is_event_belongs_to_branch(
    invocation_branch: Optional[str], event: Event
) -> bool:
  """Event belongs to a branch, when event.branch is prefix of the invocation branch."""
  if not invocation_branch or not event.branch:
    return True
  return invocation_branch.startswith(event.branch)


def _is_auth_event(event: Event) -> bool:
  if not event.content.parts:
    return False
  for part in event.content.parts:
    if (
        part.function_call
        and part.function_call.name == REQUEST_EUC_FUNCTION_CALL_NAME
    ):
      return True
    if (
        part.function_response
        and part.function_response.name == REQUEST_EUC_FUNCTION_CALL_NAME
    ):
      return True
  return False
