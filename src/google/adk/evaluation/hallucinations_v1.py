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

from google.genai import types as genai_types
from typing_extensions import override

from ..models.llm_response import LlmResponse
from ..utils.feature_decorator import working_in_progress
from .eval_case import Invocation
from .eval_case import ToolUseWithResponse
from .eval_metrics import EvalMetric
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import PerInvocationResult
from .llm_as_judge import LlmAsJudge
from .llm_as_judge_utils import get_eval_status
from .llm_as_judge_utils import get_text_from_content

logger = logging.getLogger("google_adk." + __name__)


HALLUCINATIONS_V1_PROMPT = """
You are a helpful and harmless AI assistant. You will be provided with a textual context and a model-generated response.
Your task is to analyze the response sentence by sentence and classify each sentence according to its relationship with the provided context.

**Instructions:**

1. **Decompose the response into individual sentences.**
2. **For each sentence, assign one of the following labels:**
    * **`supported`**: The sentence is entailed by the given context. Provide a supporting excerpt from the context. The supporting except must *fully* entail the sentence.
    * **`unsupported`**: The sentence is not entailed by the given context. No excerpt is needed for this label.
    * **`contradictory`**: The sentence is falsified by the given context. Provide a contradicting excerpt from the context.
    * **`disputed`**: The given context contains both supporting and contradicting information. Provide both supporting and contradicting excerpt from the context.
    * **`no_rad`**: The sentence does not require factual attribution (e.g., opinions, planning steps, greetings, questions, disclaimers, mathematical calculation).
3. **For each label, provide a short rationale explaining your decision.** The rationale should be separate from the excerpt.
4. **Be very strict with your `supported`, `contradictory` and `disputed` decisions.** Unless you can find straightforward, indisputable evidence excepts *in the context* that a sentence is `supported`, `contradictory` or `disputed`, consider it `unsupported`.  You should not employ world knowledge unless it is truly trivial.
5. "tool_outputs" blocks contain code execution results of the "tool_code" blocks immediately above them. If any sentence is based on "tool_outputs" results, first analyze if the corresponding "tool_code" is supported and if the results are error-free. Only if the "tool_code" block is supported, you can treat code execution results as correct.
6. If you need to cite multiple supporting excerpts, simply concatenate them. Excerpt could be summary from the context if it is too long.

**Input Format:**

The input will consist of two parts, clearly separated:

* **Context:**  The textual context used to generate the response.
* **Response:** The model-generated response to be analyzed.

**Output Format:**

For each sentence in the response, output a block of text with the following fields:

* sentence: The sentence being analyzed.
* label: One of `supported`, `unsupported`, `contradictory`, `disputed` or `no_rad`.
* rationale: A brief explanation for the assessment
* supporting_excerpt: A relevant excerpt from the context that supports the sentence. Only required for `supported` and `disputed` labels.
* contradicting_excerpt: A relevant excerpt from the context that contradicts with the sentence. Only required for `contradictory` and `disputed` labels.

**Example:**

**Input:**

**Context Begin**
Apples are red fruits. Bananas are yellow fruits. Pears are purple fruits. Pears are blue fruits.
**Context End**

**Response Begin**
Apples are red. Bananas are green. Pears are purple. Bananas are cheaper than apples. Enjoy your fruit!
**Response End**

**Output:**
sentence: Apples are red.
label: supported
rationale: The context explicitly states that apples are red.
supporting_excerpt: Apples are red fruits.
contradicting_excerpt: null

sentence: Bananas are green.
label: contradictory
rationale: The context states that bananas are yellow, not green.
supporting_excerpt: null
contradicting_excerpt: Bananas are yellow fruits.

sentence: Pears are purple.
label: disputed
rationale: The context states that pears are purple but it also states that pears are blue.
supporting_excerpt: Pears are purple fruits
contradicting_excerpt: Pears are blue fruits

sentence: Bananas are cheaper than apples.
label: unsupported
rationale: The context does not mention the price of bananas or apples.
supporting_excerpt: null
contradicting_excerpt: null

sentence: Enjoy your fruit!
label: no_rad
rationale: This is a general expression and does not require factual attribution.
supporting_excerpt: null
contradicting_excerpt: null

**Now, please analyze the following context and response:**

**Input:**

**Context Begin**
{context}
**Context End**

**Response Begin**
{response}
**Response End**

**Output:**
""".strip()


def _format_function_call(function_call: genai_types.FunctionCall) -> str:
  """Formats a function call as a string."""
  formatted_function_call = f"Function call\nName: {function_call.name}\n"
  formatted_function_call += f"Args: {json.dumps(function_call.args)}"
  return formatted_function_call


def _format_function_response(
    function_response: genai_types.FunctionResponse,
) -> str:
  """Formats a function response as a string."""
  formatted_function_response = (
      f"Function response\nName: {function_response.name}\n"
  )
  formatted_function_response += (
      f"Response: {json.dumps(function_response.response)}"
  )
  return formatted_function_response


def _format_tool_use_with_response(
    tool_use_with_response: ToolUseWithResponse,
) -> str:
  """Formats a tool use with response as a string."""
  formatted_tool_use = {
      "name": tool_use_with_response.function_call.name,
      "args": tool_use_with_response.function_call.args,
  }
  if tool_use_with_response.function_response:
    formatted_tool_use["response"] = (
        tool_use_with_response.function_response.response
    )
  return json.dumps(formatted_tool_use)


def _extract_labels_from_critique(response: str) -> list[str]:
  """Extracts the label from the LLM critique."""
  label_matches = re.findall(
      r"label: (supported|unsupported|contradictory|disputed|no_rad)", response
  )
  return label_matches


@working_in_progress
class HallucinationsV1Evaluator(LlmAsJudge):
  """LLM-based evaluator to judge factuality and whether the response contains hallucinations."""

  def __init__(
      self,
      eval_metric: EvalMetric,
  ):
    super().__init__(eval_metric)
    self._auto_rater_prompt_template = HALLUCINATIONS_V1_PROMPT

  @override
  def format_auto_rater_prompt(
      self, actual_invocation: Invocation, expected_invocation: Invocation
  ) -> str:
    response = get_text_from_content(actual_invocation.final_response)
    user_content = get_text_from_content(expected_invocation.user_content)
    context = [f"User prompt: {user_content}"]
    if not actual_invocation.intermediate_data:
      return self._auto_rater_prompt_template.format(
          context=context[0], response=response
      )
    # Only support text and function calls/tool uses for now.
    for (
        tool_use
    ) in actual_invocation.intermediate_data.tool_uses_with_responses:
      context.append(f"Tool use:\n{_format_tool_use_with_response(tool_use)}")
    for (
        author,
        parts,
    ) in actual_invocation.intermediate_data.intermediate_responses:
      parts_formatted = []
      for part in parts:
        if part.text:
          parts_formatted.append(part.text)
        elif part.function_call:
          parts_formatted.append(_format_function_call(part.function_call))
        elif part.function_response:
          parts_formatted.append(
              _format_function_response(part.function_response)
          )
      parts_formatted = "\n".join(parts_formatted)
      context.append(f"Sub-agent {author} response:\n{parts_formatted}")
    return self._auto_rater_prompt_template.format(
        context="\n\n".join(context), response=response
    )

  @override
  def convert_auto_rater_response_to_score(
      self, llm_response: LlmResponse
  ) -> Optional[float]:
    text = get_text_from_content(llm_response.content)
    labels = _extract_labels_from_critique(text)
    if not labels:
      return None
    final_score = 0.0
    # If label is 'disputed' or 'no_rad', the score is not affected.
    for label in labels:
      if label == "supported":
        final_score += 1.0
      elif label in ["unsupported", "contradictory"]:
        final_score -= 1.0
    return final_score / len(labels)

  @override
  def aggregate_per_invocation_samples(
      self, per_invocation_samples: list[PerInvocationResult]
  ) -> PerInvocationResult:
    """Computes the fraction of invocation samples that are valid."""
    final_score = 0.0
    num_evaluated = 0
    for sample in per_invocation_samples:
      if sample.score is None or sample.eval_status == EvalStatus.NOT_EVALUATED:
        continue
      num_evaluated += 1
      final_score += sample.score
    final_score /= num_evaluated
    return PerInvocationResult(
        actual_invocation=per_invocation_samples[0].actual_invocation,
        expected_invocation=per_invocation_samples[0].expected_invocation,
        score=final_score,
        eval_status=get_eval_status(final_score, self._eval_metric.threshold),
    )

  @override
  def aggregate_invocation_results(
      self, per_invocation_results: list[PerInvocationResult]
  ) -> EvaluationResult:
    """Computes the fraction of invocation results that are valid."""
    final_score = 0.0
    num_evaluated = 0
    for result in per_invocation_results:
      if result.score is None or result.eval_status == EvalStatus.NOT_EVALUATED:
        continue
      num_evaluated += 1
      final_score += result.score
    final_score /= num_evaluated
    return EvaluationResult(
        overall_score=final_score,
        overall_eval_status=get_eval_status(
            final_score, self._eval_metric.threshold
        ),
        per_invocation_results=per_invocation_results,
    )
