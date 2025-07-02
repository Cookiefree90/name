import pytest
from unittest.mock import MagicMock
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow, LlmResponse, Event, ConnectionClosedOK

@pytest.mark.asyncio
async def test_receive_from_model_yields_events():
    flow = BaseLlmFlow()

    fake_response_1 = LlmResponse(
        content=MagicMock(role='assistant'),
        error_code=None,
        interrupted=False
    )
    fake_response_2 = LlmResponse(
        content=MagicMock(role='user'),
        error_code=None,
        interrupted=False
    )

    async def fake_receive():
        yield fake_response_1
        yield fake_response_2
        raise ConnectionClosedOK(rcvd=None, sent=None)


    llm_connection = MagicMock()
    llm_connection.receive = fake_receive

    invocation_context = MagicMock()
    invocation_context.agent.name = "TestAgent"
    invocation_context.live_request_queue = MagicMock()
    invocation_context.transcription_cache = []
    invocation_context.invocation_id = "test_invocation_id_123"

    events = []
    async for event in flow._receive_from_model(
        llm_connection, event_id="test_event", invocation_context=invocation_context, llm_request=MagicMock()
    ):
        events.append(event)

    # Add your assertions here, e.g.:
    assert len(events) == 2
    assert events[0].author == "TestAgent"
    assert events[1].author == "user"
