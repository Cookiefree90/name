"""Conditional agent implementation.

This agent evaluates a provided predicate (condition) against the current
`InvocationContext` and delegates the call to one of *exactly two* sub-agents
based on the result:

  * ``condition(ctx) == True``  -> ``sub_agents[0]`` ("true" agent)
  * ``condition(ctx) == False`` -> ``sub_agents[1]`` ("false" agent)

Typical usages include simple routing / decision making, e.g. directing the
conversation to either a *search* agent or *calculation* agent depending on the
user request type.

Example:

```python
router = ConditionalAgent(
    name="router",
    description="Route to calc or search agent based on context flag",
    sub_agents=[calc_agent, search_agent],
    condition=lambda ctx: ctx.state.get("needs_calc", False),
)
```

The condition can be either synchronous or asynchronous. If the predicate is
awaitable (i.e. returns an ``Awaitable[bool]``), it will be awaited.
"""
from __future__ import annotations

import inspect
from typing import Awaitable, Callable, AsyncGenerator, Union

from typing_extensions import override

from ..events.event import Event
from .base_agent import BaseAgent
from .invocation_context import InvocationContext

# Type alias for the predicate
Condition = Callable[[InvocationContext], Union[bool, Awaitable[bool]]]


class ConditionalAgent(BaseAgent):
    """A shell agent that chooses between two sub-agents based on a condition."""

    # NOTE: The predicate function itself is **not** serialisable; exclude from
    # model dump to avoid pydantic complaining when exporting to json.
    condition: Condition  # type: ignore[assignment]

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    async def _evaluate_condition(self, ctx: InvocationContext) -> bool:
        """Evaluates *self.condition* and returns the boolean result."""
        result = self.condition(ctx)
        if inspect.isawaitable(result):
            result = await result  # type: ignore[assignment]
        return bool(result)

    def _validate_sub_agent_count(self) -> None:
        if len(self.sub_agents) != 2:
            raise ValueError(
                "ConditionalAgent requires *exactly* two sub-agents (true / false)."
            )

    # ---------------------------------------------------------------------
    # Core execution implementations
    # ---------------------------------------------------------------------
    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Delegates to either ``sub_agents[0]`` or ``sub_agents[1]``."""
        self._validate_sub_agent_count()
        chosen_agent_idx = 0 if await self._evaluate_condition(ctx) else 1
        chosen_agent = self.sub_agents[chosen_agent_idx]
        async for event in chosen_agent.run_async(ctx):
            yield event

    @override
    async def _run_live_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Live mode implementation mirroring the async version."""
        self._validate_sub_agent_count()
        chosen_agent_idx = 0 if await self._evaluate_condition(ctx) else 1
        chosen_agent = self.sub_agents[chosen_agent_idx]
        async for event in chosen_agent.run_live(ctx):
            yield event
