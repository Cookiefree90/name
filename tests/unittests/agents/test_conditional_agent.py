"""Tests for the ConditionalAgent."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest
from typing_extensions import override

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.conditional_agent import ConditionalAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types


class _TestingAgent(BaseAgent):
    """A simple testing agent that emits a single event."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(parts=[types.Part(text=f"Hello, async {self.name}!")]),
        )

    @override
    async def _run_live_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(parts=[types.Part(text=f"Hello, live {self.name}!")]),
        )


async def _create_parent_invocation_context(
    test_name: str, agent: BaseAgent
) -> InvocationContext:
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="test_app", user_id="test_user"
    )
    return InvocationContext(
        invocation_id=f"{test_name}_invocation_id",
        agent=agent,
        session=session,
        session_service=session_service,
    )


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("predicate,expected_idx", [(lambda _ctx: True, 0), (lambda _ctx: False, 1)])
async def test_run_async_branching(request: pytest.FixtureRequest, predicate, expected_idx):
    agents = [
        _TestingAgent(name=f"{request.function.__name__}_agent_true"),
        _TestingAgent(name=f"{request.function.__name__}_agent_false"),
    ]
    conditional_agent = ConditionalAgent(
        name=f"{request.function.__name__}_conditional_agent",
        sub_agents=agents,
        condition=predicate,
    )
    parent_ctx = await _create_parent_invocation_context(request.function.__name__, conditional_agent)
    events = [e async for e in conditional_agent.run_async(parent_ctx)]

    # Only the chosen agent should produce events
    assert len(events) == 1
    assert events[0].author == agents[expected_idx].name
    assert events[0].content.parts[0].text == f"Hello, async {agents[expected_idx].name}!"


@pytest.mark.asyncio
async def test_run_async_async_predicate(request: pytest.FixtureRequest):
    async def async_predicate(_ctx):
        await asyncio.sleep(0.01)
        return True

    agent_true = _TestingAgent(name=f"{request.function.__name__}_agent_true")
    agent_false = _TestingAgent(name=f"{request.function.__name__}_agent_false")

    conditional_agent = ConditionalAgent(
        name=f"{request.function.__name__}_conditional_agent",
        sub_agents=[agent_true, agent_false],
        condition=async_predicate,
    )
    parent_ctx = await _create_parent_invocation_context(request.function.__name__, conditional_agent)
    events = [e async for e in conditional_agent.run_async(parent_ctx)]

    assert len(events) == 1 and events[0].author == agent_true.name


# ---------------------------------------------------------------------------
# Live tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_live_branching(request: pytest.FixtureRequest):
    agent_true = _TestingAgent(name=f"{request.function.__name__}_agent_true")
    agent_false = _TestingAgent(name=f"{request.function.__name__}_agent_false")
    conditional_agent = ConditionalAgent(
        name=f"{request.function.__name__}_conditional_agent",
        sub_agents=[agent_true, agent_false],
        condition=lambda _ctx: False,
    )
    parent_ctx = await _create_parent_invocation_context(request.function.__name__, conditional_agent)
    events = [e async for e in conditional_agent.run_live(parent_ctx)]

    assert len(events) == 1
    assert events[0].author == agent_false.name
    assert events[0].content.parts[0].text == f"Hello, live {agent_false.name}!"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_invalid_sub_agent_count():
    with pytest.raises(ValueError):
        ConditionalAgent(name="invalid", sub_agents=[], condition=lambda _ctx: True)
