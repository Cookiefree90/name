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

from unittest import mock

from google.adk.agents.live_request_queue import LiveRequest
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest

from ... import testing_utils


class TestBaseLlmFlow(BaseLlmFlow):
  """Test implementation of BaseLlmFlow for testing purposes."""

  pass


@pytest.fixture
def test_blob():
  """Test blob for audio data."""
  return types.Blob(data=b'\x00\xFF\x00\xFF', mime_type='audio/pcm')


@pytest.fixture
def mock_llm_connection():
  """Mock LLM connection for testing."""
  connection = mock.AsyncMock()
  connection.send_realtime = mock.AsyncMock()
  return connection


@pytest.mark.asyncio
async def test_send_to_model_with_disabled_vad(test_blob, mock_llm_connection):
  """Test _send_to_model with automatic_activity_detection.disabled=True."""
  # Create LlmRequest with disabled VAD
  realtime_input_config = types.RealtimeInputConfig(
      automatic_activity_detection=types.AutomaticActivityDetection(
          disabled=True
      )
  )

  # Create invocation context with live request queue
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='',
      run_config=RunConfig(realtime_input_config=realtime_input_config),
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  # Create flow and start _send_to_model task
  flow = TestBaseLlmFlow()

  # Send a blob to the queue
  live_request = LiveRequest(blob=test_blob)
  invocation_context.live_request_queue.send(live_request)
  invocation_context.live_request_queue.close()

  # Run _send_to_model
  await flow._send_to_model(mock_llm_connection, invocation_context)

  mock_llm_connection.send_realtime.assert_called_once_with(test_blob)


@pytest.mark.asyncio
async def test_send_to_model_with_enabled_vad(test_blob, mock_llm_connection):
  """Test _send_to_model with automatic_activity_detection.disabled=False.

  Custom VAD activity signal is not supported so we should still disable it.
  """
  # Create LlmRequest with enabled VAD
  realtime_input_config = types.RealtimeInputConfig(
      automatic_activity_detection=types.AutomaticActivityDetection(
          disabled=False
      )
  )

  # Create invocation context with live request queue
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  # Create flow and start _send_to_model task
  flow = TestBaseLlmFlow()

  # Send a blob to the queue
  live_request = LiveRequest(blob=test_blob)
  invocation_context.live_request_queue.send(live_request)
  invocation_context.live_request_queue.close()

  # Run _send_to_model
  await flow._send_to_model(mock_llm_connection, invocation_context)

  mock_llm_connection.send_realtime.assert_called_once_with(test_blob)


@pytest.mark.asyncio
async def test_send_to_model_without_realtime_config(
    test_blob, mock_llm_connection
):
  """Test _send_to_model without realtime_input_config (default behavior)."""
  # Create invocation context with live request queue
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  # Create flow and start _send_to_model task
  flow = TestBaseLlmFlow()

  # Send a blob to the queue
  live_request = LiveRequest(blob=test_blob)
  invocation_context.live_request_queue.send(live_request)
  invocation_context.live_request_queue.close()

  # Run _send_to_model
  await flow._send_to_model(mock_llm_connection, invocation_context)

  mock_llm_connection.send_realtime.assert_called_once_with(test_blob)


@pytest.mark.asyncio
async def test_send_to_model_with_none_automatic_activity_detection(
    test_blob, mock_llm_connection
):
  """Test _send_to_model with automatic_activity_detection=None."""
  # Create LlmRequest with None automatic_activity_detection
  realtime_input_config = types.RealtimeInputConfig(
      automatic_activity_detection=None
  )

  # Create invocation context with live request queue
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='',
      run_config=RunConfig(realtime_input_config=realtime_input_config),
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  # Create flow and start _send_to_model task
  flow = TestBaseLlmFlow()

  # Send a blob to the queue
  live_request = LiveRequest(blob=test_blob)
  invocation_context.live_request_queue.send(live_request)
  invocation_context.live_request_queue.close()

  # Run _send_to_model
  await flow._send_to_model(mock_llm_connection, invocation_context)

  mock_llm_connection.send_realtime.assert_called_once_with(test_blob)


@pytest.mark.asyncio
async def test_send_to_model_with_text_content(mock_llm_connection):
  """Test _send_to_model with text content (not blob)."""
  # Create invocation context with live request queue
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  # Create flow and start _send_to_model task
  flow = TestBaseLlmFlow()

  # Send text content to the queue
  content = types.Content(
      role='user', parts=[types.Part.from_text(text='Hello')]
  )
  live_request = LiveRequest(content=content)
  invocation_context.live_request_queue.send(live_request)
  invocation_context.live_request_queue.close()

  # Run _send_to_model
  await flow._send_to_model(mock_llm_connection, invocation_context)

  # Verify send_content was called instead of send_realtime
  mock_llm_connection.send_content.assert_called_once_with(content)
  mock_llm_connection.send_realtime.assert_not_called()


@pytest.mark.asyncio
async def test_receive_from_model_calls_telemetry_trace(monkeypatch):
  """Test that _receive_from_model calls trace_call_llm for telemetry."""
  # Mock the trace_call_llm function
  mock_trace_call_llm = mock.AsyncMock()
  monkeypatch.setattr(
      'google.adk.flows.llm_flows.base_llm_flow.trace_call_llm',
      mock_trace_call_llm,
  )

  # Create mock LLM connection that yields responses
  mock_llm_connection = mock.AsyncMock()

  # Create test LLM response with usage metadata
  from google.adk.models.llm_response import LlmResponse

  test_llm_response = LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text='Test response')]
      ),
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          total_token_count=100,
          prompt_token_count=50,
          candidates_token_count=50,
      ),
  )

  # Mock the receive method to yield our test response
  async def mock_receive():
    yield test_llm_response

  mock_llm_connection.receive = mock_receive

  # Create agent and invocation context
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  # Create flow and test data
  flow = TestBaseLlmFlow()
  event_id = 'test_event_123'
  llm_request = LlmRequest()

  # Call _receive_from_model and consume the generator
  events = []
  async for event in flow._receive_from_model(
      mock_llm_connection, event_id, invocation_context, llm_request
  ):
    events.append(event)
    break  # Exit after first event to avoid infinite loop

  # Verify trace_call_llm was called
  mock_trace_call_llm.assert_called()

  # Verify the call arguments
  call_args = mock_trace_call_llm.call_args
  assert call_args[0][0] == invocation_context  # First arg: invocation_context
  assert call_args[0][2] == llm_request  # Third arg: llm_request
  assert call_args[0][3] == test_llm_response  # Fourth arg: llm_response

  # Second arg should be the event ID from the generated event
  assert len(call_args[0][1]) > 0  # Event ID should be non-empty string


@pytest.mark.asyncio
async def test_receive_from_model_telemetry_integration_with_live_queue(
    monkeypatch,
):
  """Test telemetry integration in live mode with actual live request queue."""
  # Mock the telemetry tracer to capture span creation
  mock_tracer = mock.MagicMock()
  mock_span = mock.MagicMock()
  mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
      mock_span
  )

  monkeypatch.setattr('google.adk.telemetry.tracer', mock_tracer)

  # Create mock LLM connection
  mock_llm_connection = mock.AsyncMock()

  # Create test responses - one with usage metadata, one without
  from google.adk.models.llm_response import LlmResponse

  response_with_usage = LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text='Response 1')]
      ),
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          total_token_count=100,
          prompt_token_count=50,
          candidates_token_count=50,
      ),
  )

  response_without_usage = LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text='Response 2')]
      ),
      usage_metadata=None,
  )

  # Mock receive to yield both responses
  async def mock_receive():
    yield response_with_usage
    yield response_without_usage

  mock_llm_connection.receive = mock_receive

  # Create agent and invocation context with live request queue
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  # Create flow
  flow = TestBaseLlmFlow()
  event_id = 'test_event_integration'
  llm_request = LlmRequest()

  # Process events from _receive_from_model
  events = []
  async for event in flow._receive_from_model(
      mock_llm_connection, event_id, invocation_context, llm_request
  ):
    events.append(event)
    if len(events) >= 2:  # Stop after processing both responses
      break

  # Verify new spans were created for live events with usage metadata
  assert mock_tracer.start_as_current_span.call_count >= 1

  # Check that at least one span was created with live event naming
  span_calls = mock_tracer.start_as_current_span.call_args_list
  live_event_spans = [
      call for call in span_calls if 'llm_call_live_event' in call[0][0]
  ]
  assert len(live_event_spans) >= 1, 'Should create live event spans'
