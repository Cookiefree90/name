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

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.genai.types import Part
import pytest

from ... import testing_utils


def transfer_call_part(agent_name: str) -> Part:
  return Part.from_function_call(
      name='transfer_to_agent', args={'agent_name': agent_name}
  )


TRANSFER_RESPONSE_PART = Part.from_function_response(
    name='transfer_to_agent', response={'result': None}
)


def test_fallback_to_parent_basic():
  """Tests basic fallback_to_parent functionality."""
  response = [
      'child_response',
      'parent_response',
  ]
  mock_model = testing_utils.MockModel.create(responses=response)

  child_agent = LlmAgent(
      name='child_agent',
      model=mock_model,
      fallback_to_parent=True,
  )
  parent_agent = LlmAgent(
      name='parent_agent',
      model=mock_model,
      sub_agents=[child_agent],
  )

  # Run child agent directly
  runner = testing_utils.InMemoryRunner(child_agent)

  result = testing_utils.simplify_events(runner.run('test1'))

  # Should have child response followed by forced transfer to parent and parent response
  assert len(result) >= 2
  assert result[0] == ('child_agent', 'child_response')
  # The fallback should cause a transfer to parent, so parent should respond
  assert result[-1] == ('parent_agent', 'parent_response')


def test_fallback_to_parent_with_transfer_call():
  """Tests that fallback doesn't trigger when model already has transfer call."""
  response = [
      transfer_call_part('parent_agent'),  # Model explicitly transfers
      'parent_response',
  ]
  mock_model = testing_utils.MockModel.create(responses=response)

  child_agent = LlmAgent(
      name='child_agent',
      model=mock_model,
      fallback_to_parent=True,
  )
  parent_agent = LlmAgent(
      name='parent_agent',
      model=mock_model,
      sub_agents=[child_agent],
  )

  runner = testing_utils.InMemoryRunner(child_agent)

  result = testing_utils.simplify_events(runner.run('test1'))

  # Should have explicit transfer (not fallback), so no duplicate transfers
  expected_events = [
      ('child_agent', transfer_call_part('parent_agent')),
      ('child_agent', TRANSFER_RESPONSE_PART),
      ('parent_agent', 'parent_response'),
  ]
  assert result == expected_events


@pytest.mark.asyncio
async def test_fallback_to_parent_without_parent_agent():
  """Tests behavior when fallback_to_parent is True but no parent_agent exists."""
  mock_model = testing_utils.MockModel.create(responses=['response_from_child'])

  child_agent = LlmAgent(
      name='child',
      model=mock_model,
      fallback_to_parent=True,
  )
  # parent_agent is not set (None by default)

  runner = testing_utils.InMemoryRunner(child_agent)

  # Should work normally without fallback since there's no parent
  result = testing_utils.simplify_events(runner.run('test1'))
  assert result == [('child', 'response_from_child')]


def test_fallback_to_parent_disabled():
  """Tests that fallback doesn't happen when fallback_to_parent is False."""
  response = [
      'child_response',
      'should_not_be_called',
  ]
  mock_model = testing_utils.MockModel.create(responses=response)

  child_agent = LlmAgent(
      name='child_agent',
      model=mock_model,
      fallback_to_parent=False,  # Explicitly disabled
  )
  parent_agent = LlmAgent(
      name='parent_agent',
      model=mock_model,
      sub_agents=[child_agent],
  )

  runner = testing_utils.InMemoryRunner(child_agent)

  result = testing_utils.simplify_events(runner.run('test1'))

  # Should only have child response, no fallback
  assert result == [('child_agent', 'child_response')]


def test_fallback_to_parent_with_non_llm_agent():
  """Tests that fallback doesn't happen with non-LlmAgent instances."""

  class MockNonLlmAgent(BaseAgent):

    def __init__(self, name: str):
      super().__init__(name=name)
      # This is not an LlmAgent, so fallback should not apply

    async def _run_async_impl(self, ctx):
      # Simple implementation that yields a single event
      from google.adk.events.event import Event

      event = Event(
          id=Event.new_id(),
          invocation_id=ctx.invocation_id,
          author=self.name,
      )
      event.content = testing_utils.UserContent('non_llm_response')
      yield event

  mock_model = testing_utils.MockModel.create(responses=['parent_response'])

  non_llm_agent = MockNonLlmAgent(name='non_llm_agent')
  parent_agent = LlmAgent(
      name='parent_agent',
      model=mock_model,
      sub_agents=[non_llm_agent],
  )

  runner = testing_utils.InMemoryRunner(non_llm_agent)

  result = testing_utils.simplify_events(runner.run('test1'))

  # Should only have non-LLM agent response, no fallback
  assert result == [('non_llm_agent', 'non_llm_response')]


def test_fallback_to_parent_hierarchy():
  """Tests fallback behavior in a multi-level hierarchy."""
  response = [
      'grandchild_response',
      'child_response',
      'parent_response',
  ]
  mock_model = testing_utils.MockModel.create(responses=response)

  grandchild_agent = LlmAgent(
      name='grandchild_agent',
      model=mock_model,
      fallback_to_parent=True,
  )
  child_agent = LlmAgent(
      name='child_agent',
      model=mock_model,
      fallback_to_parent=True,
      sub_agents=[grandchild_agent],
  )
  parent_agent = LlmAgent(
      name='parent_agent',
      model=mock_model,
      sub_agents=[child_agent],
  )

  runner = testing_utils.InMemoryRunner(grandchild_agent)

  result = testing_utils.simplify_events(runner.run('test1'))

  # Should cascade fallbacks: grandchild -> child -> parent
  assert len(result) >= 3
  assert result[0] == ('grandchild_agent', 'grandchild_response')
  # Middle events are transfer events, final should be parent
  assert result[-1] == ('parent_agent', 'parent_response')
