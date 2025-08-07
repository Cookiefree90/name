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
from typing import Any
from typing import AsyncGenerator
from typing import Dict
from typing import Optional

from google.genai import types
from typing_extensions import override

from .base_llm import BaseLlm
from .llm_request import LlmRequest
from .llm_response import LlmResponse

logger = logging.getLogger("google_adk." + __name__)


class KerasLlm(BaseLlm):
  """Wrapper around KerasHub for local model inference.

  This wrapper can be used with any of the models supported by KerasHub. The
  models run locally without requiring external API calls.

  Example usage:
  ```
  agent = Agent(
      model=KerasLlm(model="gpt2_base_en"),
      ...
  )
  ```

  Attributes:
      model: The name of the KerasHub model preset.
      _keras_model: The loaded KerasHub model instance.
      _preprocessor: The preprocessor/tokenizer for the model.
      _additional_args: Additional generation parameters.
  """

  _keras_model: Optional[Any] = None
  """The loaded KerasHub model instance."""

  _preprocessor: Optional[Any] = None
  """The preprocessor/tokenizer for the model."""

  _additional_args: Dict[str, Any] = None
  """Additional generation parameters."""

  def __init__(self, model: str, **kwargs):
    """Initializes the KerasLlm class.

    Args:
        model: The name of the KerasHub model preset.
        **kwargs: Additional arguments to pass to the model generation.
    """
    super().__init__(model=model, **kwargs)
    self._additional_args = kwargs
    # Remove internal fields from kwargs
    self._additional_args.pop("_keras_model", None)
    self._additional_args.pop("_preprocessor", None)
    self._additional_args.pop("_additional_args", None)

  def _load_model(self):
    """Loads the KerasHub model and preprocessor."""
    try:
      import keras_hub
    except ImportError:
      raise ImportError(
          "KerasHub is not installed. Please install it with: pip install"
          " keras-hub"
      )

    # Strip 'keras/' prefix if present
    preset = self.model
    if preset.startswith("keras/"):
      preset = preset[6:]

    try:
      # Load the model using the appropriate model class
      # For causal language models, use CausalLM.from_preset
      from keras_hub.models import CausalLM

      self._keras_model = CausalLM.from_preset(preset)
    except Exception as e:
      raise ValueError(
          f"Failed to load model '{self.model}'. "
          "This might be due to an unsupported preset or network issues. "
          f"Error: {e}"
      )

    # Configure the sampler using KerasHub's native samplers
    sampler = self._additional_args.get("sampler", "greedy")
    temperature = self._additional_args.get("temperature", 1.0)

    if sampler == "greedy":
      self._keras_model.compile(sampler="greedy")
    elif sampler == "top_k":
      top_k = self._additional_args.get("top_k", 50)
      self._keras_model.compile(sampler=f"top_k(top_k={top_k})")
    elif sampler == "top_p":
      top_p = self._additional_args.get("top_p", 0.9)
      self._keras_model.compile(sampler=f"top_p(top_p={top_p})")
    elif sampler == "temperature":
      self._keras_model.compile(
          sampler=f"temperature(temperature={temperature})"
      )
    elif sampler == "beam_search":
      beam_size = self._additional_args.get("beam_size", 5)
      self._keras_model.compile(sampler=f"beam_search(beam_size={beam_size})")
    elif sampler == "nucleus":
      nucleus_p = self._additional_args.get("nucleus_p", 0.9)
      self._keras_model.compile(sampler=f"nucleus(nucleus_p={nucleus_p})")
    else:
      # Default to greedy sampling
      self._keras_model.compile(sampler="greedy")

  def _flatten_conversation_to_prompt(self, llm_request: LlmRequest) -> str:
    """Flattens the conversation into a single text prompt.

    Args:
        llm_request: The LLM request containing conversation history.

    Returns:
        A single text prompt string.
    """
    prompt_parts = []

    for content in llm_request.contents:
      if content.role == "system":
        # Add system instruction at the beginning
        for part in content.parts:
          if hasattr(part, "text") and part.text:
            prompt_parts.append(f"System: {part.text}")
      elif content.role == "user":
        # Add user message
        for part in content.parts:
          if hasattr(part, "text") and part.text:
            prompt_parts.append(f"User: {part.text}")
      elif content.role == "assistant":
        # Add assistant response
        for part in content.parts:
          if hasattr(part, "text") and part.text:
            prompt_parts.append(f"Assistant: {part.text}")

    # Add the final "Assistant:" to prompt for completion
    prompt_parts.append("Assistant:")

    return "\n".join(prompt_parts)

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generate content using the KerasHub model."""
    self._maybe_append_user_content(llm_request)

    if self._keras_model is None:
      self._load_model()

    prompt = self._flatten_conversation_to_prompt(llm_request)
    max_length = self._additional_args.get("max_length", 100)

    def generate_text():
      return self._keras_model.generate(prompt, max_length=max_length)

    try:
      generated_text = await asyncio.to_thread(generate_text)
    except Exception as e:
      logger.error(f"Error generating text with KerasHub: {e}")
      raise RuntimeError(f"Failed to generate text with KerasHub model: {e}")

    # Create response content
    response_content = types.Content(
        role="assistant", parts=[types.Part.from_text(text=generated_text)]
    )

    response = LlmResponse(content=response_content)
    yield response

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    """Returns a list of supported models in regex for LlmRegistry.
    
    This is not exhaustive - KerasHub supports many more models dynamically.
    These patterns cover the most common model families that are actually available.
    """
    return [
        # KerasHub prefix for explicit local model selection
        "keras/.*",
        
        # GPT Family (actual presets: gpt2_base_en, gpt2_large_en, etc.)
        "gpt2_.*_en",
        "gpt_neo_.*_en",
        "gpt_neox_.*_en",
        
        # OPT Family
        "opt_.*_en",
        
        # BLOOM Family
        "bloom_.*_en",
        
        # LLaMA Family (actual presets: llama2_7b_en, llama3_8b_en, etc.)
        "llama2_.*_en",
        "llama3_.*_en",
        
        # Gemma Family (actual presets: gemma2_27b_en, gemma2_2b_en, etc.)
        "gemma_.*_en",
        "gemma2_.*_en",
        "gemma3_.*_en",
        "shieldgemma_.*_en",
        "code_gemma_.*_en",
        
        # BERT Family
        "bert_.*_en",
        "roberta_.*_en",
        "distilbert_.*_en",
        "albert_.*_en",
        "deberta_.*_en",
        
        # T5 Family
        "t5_.*_en",
        "flan_t5_.*_en",
        "ul2_.*_en",
        
        # Modern Models (actually available)
        "falcon_.*_en",
        "mistral_.*_en",
        "mixtral_.*_en",
        "qwen_.*_en",
        "qwen2_.*_en",
        "qwen_moe_.*_en",
        "phi3_.*_en",
        
        # Vision-Language Models (actually available)
        "pali_gemma_.*_en",
        "pali_gemma2_.*_en",
        
        # Audio Models (actually available)
        "whisper_.*_en",
        "moonshine_.*_en",
        
        # Text-to-Image Models (actually available)
        "stable_diffusion_.*_en",
        "flux_.*_en",
        
        # Generic patterns for any KerasHub model
        ".*_en",  # Any English model
        ".*_multi",  # Multilingual models
    ]
