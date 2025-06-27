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

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from google.adk.models.retry_utils import calculate_delay
from google.adk.models.retry_utils import is_retryable_error
from google.adk.models.retry_utils import retry_async
from google.adk.models.retry_utils import retry_async_generator
from google.adk.models.retry_utils import RetryableError
from google.adk.models.retry_utils import RetryConfig
import pytest


class MockApiError(Exception):
  """Mock API error for testing."""

  def __init__(self, message: str, code: int = None):
    super().__init__(message)
    self.code = code


class TestRetryConfig:
  """Tests for RetryConfig class."""

  def test_default_values(self):
    """Test that RetryConfig has correct default values."""
    config = RetryConfig()
    assert config.max_retries == 3
    assert config.initial_delay == 1.0
    assert config.max_delay == 60.0
    assert config.exponential_base == 2.0
    assert config.jitter is True

  def test_custom_values(self):
    """Test that RetryConfig accepts custom values."""
    config = RetryConfig(
        max_retries=5,
        initial_delay=0.5,
        max_delay=30.0,
        exponential_base=3.0,
        jitter=False,
    )
    assert config.max_retries == 5
    assert config.initial_delay == 0.5
    assert config.max_delay == 30.0
    assert config.exponential_base == 3.0
    assert config.jitter is False


class TestIsRetryableError:
  """Tests for is_retryable_error function."""

  def test_api_error_with_retryable_codes(self):
    """Test that API errors with retryable codes are considered retryable."""
    # Mock ApiError with code 429
    error_429 = MockApiError("rate limit exceeded", code=429)
    assert is_retryable_error(error_429) is True

    # Mock ApiError with code 503
    error_503 = MockApiError("service unavailable", code=503)
    assert is_retryable_error(error_503) is True

  def test_api_error_with_non_retryable_codes(self):
    """Test that API errors with non-retryable codes are not retryable."""
    # Mock ApiError with code 400
    error_400 = MockApiError("bad request", code=400)
    assert is_retryable_error(error_400) is False

  def test_api_error_with_retryable_message_patterns(self):
    """Test that API errors with retryable message patterns are retryable."""
    retryable_messages = [
        "resource exhausted",
        "too many requests",
        "rate limit exceeded",
        "service unavailable",
        "timeout",
        "connection error",
        "network error",
        "internal server error",
    ]

    for message in retryable_messages:
      error = MockApiError(message)
      assert is_retryable_error(error) is True, f"Failed for message: {message}"

  def test_api_error_with_non_retryable_message(self):
    """Test that API errors with non-retryable messages are not retryable."""
    error = MockApiError("Invalid request format")
    assert is_retryable_error(error) is False

  def test_timeout_error_is_retryable(self):
    """Test that timeout errors are retryable."""
    error = asyncio.TimeoutError()
    assert is_retryable_error(error) is True

  def test_connection_error_is_retryable(self):
    """Test that connection errors are retryable."""
    error = ConnectionError("Connection failed")
    assert is_retryable_error(error) is True

  def test_other_errors_not_retryable(self):
    """Test that other errors are not retryable."""
    error = ValueError("Invalid value")
    assert is_retryable_error(error) is False


class TestCalculateDelay:
  """Tests for calculate_delay function."""

  def test_exponential_backoff_calculation(self):
    """Test that delay calculation follows exponential backoff."""
    config = RetryConfig(
        initial_delay=1.0, exponential_base=2.0, max_delay=60.0, jitter=False
    )

    # Test first few attempts
    assert calculate_delay(0, config) == 1.0  # 1.0 * 2^0 = 1.0
    assert calculate_delay(1, config) == 2.0  # 1.0 * 2^1 = 2.0
    assert calculate_delay(2, config) == 4.0  # 1.0 * 2^2 = 4.0
    assert calculate_delay(3, config) == 8.0  # 1.0 * 2^3 = 8.0

  def test_max_delay_cap(self):
    """Test that delay is capped at max_delay."""
    config = RetryConfig(
        initial_delay=1.0, exponential_base=2.0, max_delay=10.0, jitter=False
    )

    # Large attempt number should be capped
    delay = calculate_delay(10, config)  # Would be 1024 without cap
    assert delay == 10.0

  def test_jitter_adds_randomness(self):
    """Test that jitter adds randomness to delay."""
    config = RetryConfig(
        initial_delay=4.0, exponential_base=2.0, max_delay=60.0, jitter=True
    )

    # With jitter, delay should be between base delay and base delay + 25%
    delays = [calculate_delay(1, config) for _ in range(10)]
    base_delay = 8.0  # 4.0 * 2^1 = 8.0

    for delay in delays:
      assert base_delay <= delay <= base_delay * 1.25


class TestRetryAsync:
  """Tests for retry_async function."""

  @pytest.mark.asyncio
  async def test_success_on_first_attempt(self):
    """Test that function succeeds on first attempt without retry."""
    mock_func = AsyncMock(return_value="success")
    config = RetryConfig(max_retries=3)

    result = await retry_async(mock_func, config)

    assert result == "success"
    mock_func.assert_called_once()

  @pytest.mark.asyncio
  async def test_success_after_retries(self):
    """Test that function succeeds after some retries."""
    mock_func = AsyncMock()
    # Fail twice, then succeed
    mock_func.side_effect = [
        MockApiError("rate limit exceeded"),
        MockApiError("service unavailable"),
        "success",
    ]
    config = RetryConfig(max_retries=3, initial_delay=0.01)  # Fast for testing

    result = await retry_async(mock_func, config)

    assert result == "success"
    assert mock_func.call_count == 3

  @pytest.mark.asyncio
  async def test_failure_after_max_retries(self):
    """Test that function fails after max retries are exhausted."""
    mock_func = AsyncMock()
    error = MockApiError("rate limit exceeded")
    mock_func.side_effect = error
    config = RetryConfig(max_retries=2, initial_delay=0.01)

    with pytest.raises(MockApiError):
      await retry_async(mock_func, config)

    assert mock_func.call_count == 3  # Initial + 2 retries

  @pytest.mark.asyncio
  async def test_non_retryable_error_not_retried(self):
    """Test that non-retryable errors are not retried."""
    mock_func = AsyncMock()
    error = ValueError("Invalid value")
    mock_func.side_effect = error
    config = RetryConfig(max_retries=3)

    with pytest.raises(ValueError):
      await retry_async(mock_func, config)

    mock_func.assert_called_once()


class TestRetryAsyncGenerator:
  """Tests for retry_async_generator function."""

  @pytest.mark.asyncio
  async def test_generator_success_on_first_attempt(self):
    """Test that generator succeeds on first attempt without retry."""

    async def mock_generator():
      yield "item1"
      yield "item2"
      yield "item3"

    config = RetryConfig(max_retries=3)
    items = []
    async for item in retry_async_generator(mock_generator, config):
      items.append(item)

    assert items == ["item1", "item2", "item3"]

  @pytest.mark.asyncio
  async def test_generator_success_after_retries(self):
    """Test that generator succeeds after some retries."""
    attempt_count = 0

    async def mock_generator():
      nonlocal attempt_count
      attempt_count += 1
      if attempt_count < 3:
        raise MockApiError("service unavailable")
      yield "item1"
      yield "item2"

    config = RetryConfig(max_retries=3, initial_delay=0.01)
    items = []
    async for item in retry_async_generator(mock_generator, config):
      items.append(item)

    assert items == ["item1", "item2"]
    assert attempt_count == 3

  @pytest.mark.asyncio
  async def test_generator_failure_after_max_retries(self):
    """Test that generator fails after max retries are exhausted."""

    async def mock_generator():
      raise MockApiError("rate limit exceeded")
      yield  # This line is never reached but makes it an async generator

    config = RetryConfig(max_retries=2, initial_delay=0.01)

    with pytest.raises(MockApiError):
      async for _ in retry_async_generator(mock_generator, config):
        pass

  @pytest.mark.asyncio
  async def test_generator_partial_yield_then_error(self):
    """Test generator that yields some items then fails."""
    attempt_count = 0

    async def mock_generator():
      nonlocal attempt_count
      attempt_count += 1
      yield "item1"
      if attempt_count == 1:
        raise MockApiError("timeout")
      yield "item2"

    config = RetryConfig(max_retries=3, initial_delay=0.01)
    items = []
    async for item in retry_async_generator(mock_generator, config):
      items.append(item)

    # Should get items from both attempts - first yields "item1", second yields "item1", "item2"
    assert items == ["item1", "item1", "item2"]
    assert attempt_count == 2

  @pytest.mark.asyncio
  async def test_generator_non_retryable_error_not_retried(self):
    """Test that non-retryable errors in generator are not retried."""
    attempt_count = 0

    async def mock_generator():
      nonlocal attempt_count
      attempt_count += 1
      raise ValueError("Invalid value")
      yield  # This line is never reached but makes it an async generator

    config = RetryConfig(max_retries=3)

    with pytest.raises(ValueError):
      async for _ in retry_async_generator(mock_generator, config):
        pass

    assert attempt_count == 1
