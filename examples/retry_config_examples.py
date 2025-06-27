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

"""Examples showing how to configure retry behavior for LLM API calls."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.models import Claude
from google.adk.models import Gemini
from google.adk.models import LiteLlm
from google.adk.models.retry_utils import RetryConfig


def example_default_retry():
  """Example using default retry configuration."""
  # Default: max_retries=3, initial_delay=1.0, max_delay=60.0
  agent = LlmAgent(
      model=Gemini(model="gemini-1.5-flash"),
      instruction="You are a helpful assistant",
  )
  return agent


def example_custom_retry_gemini():
  """Example with custom retry configuration for Gemini."""
  retry_config = RetryConfig(
      max_retries=5,  # Retry up to 5 times
      initial_delay=0.5,  # Start with 0.5 second delay
      max_delay=30.0,  # Cap at 30 seconds
      exponential_base=2.0,  # Double delay each retry
      jitter=True,  # Add random variation
  )

  model = Gemini(model="gemini-1.5-flash", retry_config=retry_config)

  agent = LlmAgent(
      model=model,
      instruction="You are a helpful assistant with enhanced retry capability",
  )
  return agent


def example_custom_retry_claude():
  """Example with custom retry configuration for Claude."""
  retry_config = RetryConfig(
      max_retries=3,
      initial_delay=2.0,  # Start with longer delay for Claude
      max_delay=120.0,  # Allow longer max delay
      exponential_base=3.0,  # More aggressive backoff
      jitter=False,  # No jitter for predictable timing
  )

  model = Claude(
      model="claude-3-5-sonnet-v2@20241022", retry_config=retry_config
  )

  agent = LlmAgent(
      model=model, instruction="You are Claude with custom retry behavior"
  )
  return agent


def example_custom_retry_litellm():
  """Example with custom retry configuration for LiteLLM."""
  # Conservative retry for third-party APIs
  retry_config = RetryConfig(
      max_retries=2,  # Fewer retries for external APIs
      initial_delay=1.0,
      max_delay=60.0,
      exponential_base=2.0,
      jitter=True,
  )

  model = LiteLlm(model="gpt-4", retry_config=retry_config)

  agent = LlmAgent(
      model=model, instruction="You are GPT-4 with conservative retry behavior"
  )
  return agent


def example_no_retry():
  """Example with retry disabled."""
  no_retry_config = RetryConfig(
      max_retries=0,  # Disable retry
      initial_delay=0.0,
      max_delay=0.0,
      exponential_base=1.0,
      jitter=False,
  )

  model = Gemini(model="gemini-1.5-flash", retry_config=no_retry_config)

  agent = LlmAgent(
      model=model, instruction="You are an assistant with no retry behavior"
  )
  return agent


def example_high_throughput_retry():
  """Example optimized for high-throughput scenarios."""
  fast_retry_config = RetryConfig(
      max_retries=2,  # Quick retries only
      initial_delay=0.1,  # Very short initial delay
      max_delay=5.0,  # Short max delay
      exponential_base=2.0,
      jitter=True,  # Spread out concurrent requests
  )

  model = Gemini(model="gemini-1.5-flash", retry_config=fast_retry_config)

  agent = LlmAgent(
      model=model,
      instruction="You are optimized for high-throughput processing",
  )
  return agent


async def example_usage():
  """Example showing how to use agents with different retry configurations."""

  # Create agents with different retry behaviors
  default_agent = example_default_retry()
  custom_agent = example_custom_retry_gemini()
  conservative_agent = example_custom_retry_litellm()

  # Use the agents
  try:
    # This will automatically retry on rate limits/resource exhaustion
    async for event in default_agent.run_async("What is AI?"):
      print(f"Default agent: {event.content}")

    async for event in custom_agent.run_async("Explain machine learning"):
      print(f"Custom agent: {event.content}")

  except Exception as e:
    print(f"Request failed after all retries: {e}")


if __name__ == "__main__":
  import asyncio

  # Run the example
  asyncio.run(example_usage())
