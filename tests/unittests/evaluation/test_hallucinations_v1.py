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

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_case import ToolUseWithResponse
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import JudgeModelOptions
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import PerInvocationResult
from google.adk.evaluation.hallucinations_v1 import _format_function_call
from google.adk.evaluation.hallucinations_v1 import _format_function_response
from google.adk.evaluation.hallucinations_v1 import _format_tool_use_with_response
from google.adk.evaluation.hallucinations_v1 import HallucinationsV1Evaluator
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
import pytest


def test_format_function_call():
  function_call = genai_types.FunctionCall(
      id="test_function_id",
      name="test_function",
      args={"arg1": "arg1_value", "arg2": "arg2_value"},
  )
  formatted_function_call = _format_function_call(function_call)
  assert formatted_function_call == """Function call
Name: test_function
Args: {"arg1": "arg1_value", "arg2": "arg2_value"}"""


def test_format_function_response():
  function_response = genai_types.FunctionResponse(
      id="test_function_id",
      name="test_function",
      response={"result": "return_value"},
  )
  formatted_function_response = _format_function_response(function_response)
  assert formatted_function_response == """Function response
Name: test_function
Response: {"result": "return_value"}"""


def test_format_tool_use_with_response():
  tool_use_with_response = ToolUseWithResponse(
      function_call=genai_types.FunctionCall(
          id="test_function_id",
          name="test_function",
          args={"arg1": "arg1_value", "arg2": "arg2_value"},
      ),
      function_response=genai_types.FunctionResponse(
          id="test_function_id",
          name="test_function",
          response={"result": "return_value"},
      ),
  )
  formatted_tool_use_with_response = _format_tool_use_with_response(
      tool_use_with_response
  )
  assert formatted_tool_use_with_response == (
      """{"name": "test_function", "args": {"arg1": "arg1_value", "arg2": "arg2_value"}, "response": {"result": "return_value"}}"""
  )


def _create_test_evaluator_gemini(
    threshold: float,
) -> HallucinationsV1Evaluator:
  evaluator = HallucinationsV1Evaluator(
      EvalMetric(
          metric_name="hallucinations_v1",
          threshold=threshold,
          judge_model_options=JudgeModelOptions(
              judge_model="gemini-2.5-flash",
              num_samples=5,
          ),
      ),
  )
  return evaluator


def test_convert_auto_rater_response_to_score():
  llm_response = LlmResponse(
      content=genai_types.Content(
          parts=[
              genai_types.Part(
                  text=(
                      "label: supported\n"
                      "rationale: The sentence is supported.\n\n"
                      "label: unsupported\n"
                      "rationale: The sentence is unsupported.\n\n"
                      "label: contradictory\n"
                      "rationale: The sentence is contradictory.\n\n"
                      "label: disputed\n"
                      "rationale: The sentence is disputed.\n\n"
                      "label: no_rad\n"
                      "rationale: The sentence is no_rad.\n\n"
                  )
              )
          ],
          role="model",
      )
  )
  evaluator = _create_test_evaluator_gemini(threshold=0.8)
  score = evaluator.convert_auto_rater_response_to_score(llm_response)
  # The score is the average of the labels, with 1.0 for supported and -1.0 for
  # unsupported and contradictory.
  assert score == -1 / 5


def test_aggregate_per_invocation_samples():
  per_invocation_samples = [
      PerInvocationResult(
          actual_invocation=Invocation(
              user_content=genai_types.Content(
                  parts=[genai_types.Part(text="This is a test query.")],
                  role="user",
              ),
              final_response=genai_types.Content(
                  parts=[genai_types.Part(text="This is a test response.")],
                  role="model",
              ),
          ),
          expected_invocation=Invocation(
              user_content=genai_types.Content(
                  parts=[genai_types.Part(text="This is a test query.")],
                  role="user",
              ),
              final_response=genai_types.Content(
                  parts=[genai_types.Part(text="This is a test response.")],
                  role="model",
              ),
          ),
          score=0.5,
          eval_status=EvalStatus.FAILED,
      ),
      PerInvocationResult(
          actual_invocation=Invocation(
              user_content=genai_types.Content(
                  parts=[genai_types.Part(text="This is a test query.")],
                  role="user",
              ),
              final_response=genai_types.Content(
                  parts=[genai_types.Part(text="This is a test response.")],
                  role="model",
              ),
          ),
          expected_invocation=Invocation(
              user_content=genai_types.Content(
                  parts=[genai_types.Part(text="This is a test query.")],
                  role="user",
              ),
              final_response=genai_types.Content(
                  parts=[genai_types.Part(text="This is a test response.")],
                  role="model",
              ),
          ),
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
  ]
  evaluator = _create_test_evaluator_gemini(threshold=0.8)
  result = evaluator.aggregate_per_invocation_samples(per_invocation_samples)
  assert result.score == 0.75
  assert result.eval_status == EvalStatus.FAILED


def test_aggregate_invocation_results():
  per_invocation_samples = [
      PerInvocationResult(
          actual_invocation=Invocation(
              user_content=genai_types.Content(
                  parts=[genai_types.Part(text="This is a test query.")],
                  role="user",
              ),
              final_response=genai_types.Content(
                  parts=[genai_types.Part(text="This is a test response.")],
                  role="model",
              ),
          ),
          expected_invocation=Invocation(
              user_content=genai_types.Content(
                  parts=[genai_types.Part(text="This is a test query.")],
                  role="user",
              ),
              final_response=genai_types.Content(
                  parts=[genai_types.Part(text="This is a test response.")],
                  role="model",
              ),
          ),
          score=0.5,
          eval_status=EvalStatus.FAILED,
      ),
      PerInvocationResult(
          actual_invocation=Invocation(
              user_content=genai_types.Content(
                  parts=[genai_types.Part(text="This is a test query.")],
                  role="user",
              ),
              final_response=genai_types.Content(
                  parts=[genai_types.Part(text="This is a test response.")],
                  role="model",
              ),
          ),
          expected_invocation=Invocation(
              user_content=genai_types.Content(
                  parts=[genai_types.Part(text="This is a test query.")],
                  role="user",
              ),
              final_response=genai_types.Content(
                  parts=[genai_types.Part(text="This is a test response.")],
                  role="model",
              ),
          ),
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
  ]
  evaluator = _create_test_evaluator_gemini(threshold=0.7)
  result = evaluator.aggregate_per_invocation_samples(per_invocation_samples)
  assert result.score == 0.75
  assert result.eval_status == EvalStatus.PASSED
