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

import asyncio
import logging
import random
from typing import Any
from typing import AsyncGenerator
from typing import Callable
from typing import TypeVar

# Note: We don't import ApiError from google.genai.types as it may not exist
# Instead, we check for exceptions with hasattr pattern matching
from pydantic import BaseModel

logger = logging.getLogger("google_adk." + __name__)

T = TypeVar("T")


class RetryConfig(BaseModel):
  """Configuration for exponential backoff retry."""

  max_retries: int = 3
  """Maximum number of retry attempts."""

  initial_delay: float = 1.0
  """Initial delay in seconds before first retry."""

  max_delay: float = 60.0
  """Maximum delay in seconds between retries."""

  exponential_base: float = 2.0
  """Base for exponential backoff calculation."""

  jitter: bool = True
  """Whether to add random jitter to delay calculation."""


class RetryableError(Exception):
  """Exception that indicates an operation should be retried."""

  pass


def is_retryable_error(error: Exception) -> bool:
  """Determines if an error is retryable.

  Args:
      error: The exception to check.

  Returns:
      True if the error should be retried, False otherwise.
  """
  # Check for specific error codes that should be retried
  if hasattr(error, "code"):
    # Resource exhausted (429) and service unavailable (503)
    if error.code in [429, 503]:
      return True

  # Check error message for specific patterns
  error_message = str(error).lower()
  retryable_patterns = [
      "resource exhausted",
      "too many requests",
      "rate limit exceeded",
      "service unavailable",
      "timeout",
      "connection error",
      "network error",
      "internal server error",
  ]

  for pattern in retryable_patterns:
    if pattern in error_message:
      return True

  # Check for common network/timeout errors
  if isinstance(error, (asyncio.TimeoutError, ConnectionError)):
    return True

  return False


def calculate_delay(attempt: int, config: RetryConfig) -> float:
  """Calculate delay for exponential backoff.

  Args:
      attempt: Current attempt number (0-based).
      config: Retry configuration.

  Returns:
      Delay in seconds.
  """
  delay = config.initial_delay * (config.exponential_base**attempt)
  delay = min(delay, config.max_delay)

  if config.jitter:
    # Add random jitter of up to 25% of the delay
    jitter_amount = delay * 0.25 * random.random()
    delay += jitter_amount

  return delay


async def retry_async(
    func: Callable[[], Any],
    config: RetryConfig,
    operation_name: str = "LLM API call",
) -> Any:
  """Retry an async function with exponential backoff.

  Args:
      func: The async function to retry.
      config: Retry configuration.
      operation_name: Name of the operation for logging.

  Returns:
      The result of the function call.

  Raises:
      The last exception encountered if all retries fail.
  """
  last_error = None

  for attempt in range(config.max_retries + 1):
    try:
      result = await func()
      if attempt > 0:
        logger.info(f"{operation_name} succeeded after {attempt} retries")
      return result
    except Exception as error:
      last_error = error

      if attempt == config.max_retries:
        logger.error(
            f"{operation_name} failed after {config.max_retries} retries. "
            f"Final error: {error}"
        )
        break

      if not is_retryable_error(error):
        logger.warning(
            f"{operation_name} failed with non-retryable error: {error}"
        )
        break

      delay = calculate_delay(attempt, config)
      logger.warning(
          f"{operation_name} failed (attempt"
          f" {attempt + 1}/{config.max_retries + 1}): {error}. Retrying in"
          f" {delay:.2f} seconds."
      )

      await asyncio.sleep(delay)

  raise last_error


async def retry_async_generator(
    func: Callable[[], AsyncGenerator[T, None]],
    config: RetryConfig,
    operation_name: str = "LLM streaming API call",
) -> AsyncGenerator[T, None]:
  """Retry an async generator function with exponential backoff.

  Args:
      func: The async generator function to retry.
      config: Retry configuration.
      operation_name: Name of the operation for logging.

  Yields:
      Items from the async generator.

  Raises:
      The last exception encountered if all retries fail.
  """
  last_error = None

  for attempt in range(config.max_retries + 1):
    try:
      async_gen = func()
      async for item in async_gen:
        yield item

      if attempt > 0:
        logger.info(f"{operation_name} succeeded after {attempt} retries")
      return

    except Exception as error:
      last_error = error

      if attempt == config.max_retries:
        logger.error(
            f"{operation_name} failed after {config.max_retries} retries. "
            f"Final error: {error}"
        )
        break

      if not is_retryable_error(error):
        logger.warning(
            f"{operation_name} failed with non-retryable error: {error}"
        )
        break

      delay = calculate_delay(attempt, config)
      logger.warning(
          f"{operation_name} failed (attempt"
          f" {attempt + 1}/{config.max_retries + 1}): {error}. Retrying in"
          f" {delay:.2f} seconds."
      )

      await asyncio.sleep(delay)

  raise last_error
