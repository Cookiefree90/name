# tests/integration/models/test_lite_llm_retry.py
import pytest
from unittest import mock
from litellm.exceptions import RateLimitError

from google.adk.models.lite_llm import LiteLLMClient


@pytest.mark.asyncio
async def test_acompletion_succeeds_immediately():
    with mock.patch(
        "google.adk.models.lite_llm.acompletion", return_value="ok"
    ) as m:
        client = LiteLLMClient()
        res = await client.acompletion(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
        )
    assert res == "ok"
    m.assert_called_once()


@pytest.mark.asyncio
async def test_acompletion_retries_then_succeeds():
    exc = RateLimitError(
        message="rate limited",
        llm_provider="openai",
        model="gpt-3.5-turbo",
        response=None,
    )
    with mock.patch(
        "google.adk.models.lite_llm.acompletion",
        side_effect=[exc, "ok"],
    ) as m:
        client = LiteLLMClient()
        res = await client.acompletion(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
        )
    assert res == "ok"
    assert m.call_count == 2


@pytest.mark.asyncio
async def test_acompletion_exhausts_retries():
    exc = RateLimitError(
        message="rate limited",
        llm_provider="openai",
        model="gpt-3.5-turbo",
        response=None,
    )
    with mock.patch(
        "google.adk.models.lite_llm.acompletion",
        side_effect=[exc, exc, exc],
    ) as m:
        client = LiteLLMClient()
        with pytest.raises(RateLimitError):
            await client.acompletion(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
            )
    assert m.call_count == 3