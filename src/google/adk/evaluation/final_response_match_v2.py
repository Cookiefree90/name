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

import json
import logging
import re
from typing import Optional

from typing_extensions import override

from ..models.llm_response import LlmResponse
from .eval_case import FunctionSpec
from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import PerInvocationResult
from .llm_as_judge import get_eval_status
from .llm_as_judge import get_text_from_content
from .llm_as_judge import LlmAsJudge

logger = logging.getLogger("google_adk." + __name__)

FINAL_RESPONSE_MATCH_V2_PROMPT = """
You are an expert rater for an AI agent. The AI agent is going to call an API to answer the user query and generate API tool use code based for the choice of the API and API arguments. The ideal model response should be a function call that fulfills user query, or a natural language response hedges or asks users for further clarification if a function call does not apply.
The primary focus of this rating task is to check correctness of the model responses.

The data consists of:
- A set of python function definitions available to the agent.
- A user query.
- A model generated response for the prompt. The responses can consist of:
  - Natural language, when the model is asking for clarification, or tells the user it does not possess the requested functionality / option.
  - Code, in the form of one or multiple python function calls, and additional code as needed, for when the model is fulfilling the user request.
You can use the help from a reference response annotated by a human rater. This reference response is of high quality. You can compare the agent's response with the reference response and decide if the agent's response is valid.
Note sometimes the reference response only contains the key entities of the correct answer and you need to be flexible to allow the agent response to contain more information than the reference response, or to present the key entities in a different format or structure or in shorter or longer format.
When the agent response is provided in the form of tables/dataframes or should be best provided in the form of tables/dataframes: focus on the key entities and main components requested in the user query and check whether you can retrieve those from the agent response. Likewise, if you have the reference response, then find out the key entities and main components in them and check whether you can retrieve those from the agent response. If the prompt does not specify any format instructions and the main items/components are included in the response then tolerate the differences in the formatting of those tables/dataframes.

You should follow the constitutions below very carefully to rate the model response:
- Allow flexibility of format even when reference code only uses one of the possible format, unless API spec or user prompt has explicit format requirement
  - e.g. For state name, allow both abbreviation and full name unless API spec has explicit requirement. e.g. both 'tx' and 'Texas' should be allowed in the agent response even when reference code only uses one of them.
  - e.g. If a reference response list outputs in a list format, the agent response is allowed to use sentence format and vice versa unless user prompt explicitly asks for a specific format.
  - e.g. For numbers, allow flexibility of formatting, e.g. 1000000 vs 1,000,000.
- The model shouldn't assume that it doesn't have access to according data or incapable of answering the question if reference response is able to find a legit answer.
- If the model response contains the correct final answer, rate it as valid even when the model response contains more information than the reference response.
- If the user prompt has csv or other table format data, don't read it yourself. Trust the reference response final answer instead.
- When the validation needs maths, date calculations, do not use your own calculator. Trust the reference response final answer instead.
- Be mindful about unit of numbers. For example, if the reference response says 100 miles, but the model response says 100 km, it is invalid.
- When the agent response or the reference response is provided in the form of tables/dataframes: focus on the key entities and main components requested in the user query and check whether you can retrieve those from the agent response and whether those match the reference response. If the user query does not specify any format instructions and the main items/components are included in the response then tolerate the differences in the formatting of those tables/dataframes.
- When the answer is in numeric format, check whether there are any format requirements in the numeric format, rounding, precision, number of decimals, etc. specified in the user query and the prompt. If there are no such instructions, then tolerate different numerical formats.
- When the answer is in numeric format and there are rounding or precision differences between the agent response and the reference response, if no further instructions are provided evaluate if the rounding strategy or precision in the agent response follows the standards for that entity. For instance, model accuracy scores must be reported with at least two decimal places (e.g., 0.798 → 0.80 is acceptable,  but 0.7 is not).

Below are the inputs:
{{
  "Function API spec": {function_api_spec},
  "User prompt": {prompt},
  "Agent response": {response},
  "Reference response": {golden_response},
}}

The answer should be a json alone which follows the json structure below:
{{
  "is_the_agent_response_valid": [valid or invalid],
  "reasoning":
}}
Answer with assertiveness:
"""


def _extract_validity_label(response_text: str) -> Optional[str]:
  label_match_is_response_valid = re.search(
      r'"is_the_agent_response_valid":\s*\[*[\n\s]*"*([^"^\]^\s]*)"*[\n\s]*\]*\s*[,\n\}]',
      response_text,
  )
  label_match_is_response_invalid = re.search(
      r'"is_the_agent_response_invalid":\s*\[*[\n\s]*"*([^"^\]^\s]*)"*[\n\s]*\]*\s*[,\n\}]',
      response_text,
  )
  if label_match_is_response_valid:
    return label_match_is_response_valid.group(1)
  elif label_match_is_response_invalid:
    return label_match_is_response_invalid.group(1)
  else:
    return None


def _format_function_api_spec(functions: list[FunctionSpec]) -> str:
  function_api_spec = []
  for function in functions:
    function_spec = {
        "Function name": function.name,
        "Function description": function.description,
    }
    function_api_spec.append(function_spec)
  return json.dumps(function_api_spec)


class FinalResponseMatchV2Evaluator(LlmAsJudge):
  """V2 final response match evaluator which uses an LLM to judge responses."""

  def __init__(
      self,
      eval_metric: EvalMetric,
  ):
    super().__init__(eval_metric)
    self._auto_rater_prompt_template = FINAL_RESPONSE_MATCH_V2_PROMPT

  @override
  def format_auto_rater_prompt(
      self, actual_invocation: Invocation, expected_invocation: Invocation
  ) -> str:
    reference = get_text_from_content(expected_invocation.final_response)
    response = get_text_from_content(actual_invocation.final_response)
    user_prompt = get_text_from_content(expected_invocation.user_content)
    function_api_spec = _format_function_api_spec(
        actual_invocation.function_api_spec
    )
    return self._auto_rater_prompt_template.format(
        function_api_spec=function_api_spec,
        prompt=user_prompt,
        response=response,
        golden_response=reference,
    )

  @override
  def convert_auto_rater_response_to_score(
      self, llm_response: LlmResponse
  ) -> Optional[float]:
    try:
      response_text = get_text_from_content(llm_response.content).strip()
      label = _extract_validity_label(response_text)
    except json.JSONDecodeError:
      logger.error("Failed to parse auto rater response: %s", llm_response)
      return None
    if label == "valid":
      return 1.0
    elif label == "invalid":
      return 0.0
    else:
      return None

  @override
  def aggregate_invocation_results(
      self, per_invocation_results: list[PerInvocationResult]
  ) -> EvaluationResult:
    """Computes the fraction of invocation results that are valid."""
    num_valid = 0
    num_evaluated = 0
    for result in per_invocation_results:
      if result.score is None or result.eval_status == EvalStatus.NOT_EVALUATED:
        continue
      num_evaluated += 1
      num_valid += result.score
    overall_score = num_valid / num_evaluated
    return EvaluationResult(
        overall_score=overall_score,
        overall_eval_status=get_eval_status(
            overall_score, self._eval_metric.threshold
        ),
        per_invocation_results=per_invocation_results,
    )
