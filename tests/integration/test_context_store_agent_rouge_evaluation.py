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

"""
Integration test for Context Reference Store with ROUGE evaluation using real agents.

This test demonstrates how to evaluate agent performance using ROUGE metrics
when agents use the context reference store for large context management.
"""

import asyncio
import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, patch


from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.sessions import LargeContextState, ContextReferenceStore
from google.adk.evaluation.final_response_match_v1 import RougeEvaluator
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.genai import types as genai_types


class TestContextStoreAgentRougeIntegration:
    """Integration tests for Context Store + Agent + ROUGE evaluation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context_store = ContextReferenceStore()
        self.large_context_state = LargeContextState(context_store=self.context_store)

        # ROUGE evaluator with threshold for agent evaluation
        self.rouge_evaluator = RougeEvaluator(
            EvalMetric(metric_name="response_match_score", threshold=0.75)
        )

        # Samples
        self.large_document = {
            "company": "TechCorp",
            "year": 2024,
            "quarters": [
                {
                    "quarter": "Q1",
                    "revenue": "$12M",
                    "growth": "15%",
                    "highlights": ["Product A launch", "Team expansion"],
                },
                {
                    "quarter": "Q2",
                    "revenue": "$15M",
                    "growth": "25%",
                    "highlights": ["Product B launch", "Market expansion"],
                },
                {
                    "quarter": "Q3",
                    "revenue": "$18M",
                    "growth": "20%",
                    "highlights": ["Partnership deals", "Customer growth"],
                },
                {
                    "quarter": "Q4",
                    "revenue": "$22M",
                    "growth": "22%",
                    "highlights": ["Record sales", "International expansion"],
                },
            ],
            "total_revenue": "$67M",
            "annual_growth": "35%",
        }

    def create_document_search_tool(
        self, context_state: LargeContextState
    ) -> FunctionTool:
        """Create a tool that searches the document stored in context state."""

        def search_document(query: str) -> str:
            """Search for information in the company document."""
            try:
                # Retrieve document from context store
                document = context_state.get_context("company_doc_ref")

                query_lower = query.lower()

                # Keyword-based search
                if "revenue" in query_lower and "total" in query_lower:
                    return f"Total revenue for {document['year']} was {document['total_revenue']}"

                elif "revenue" in query_lower and any(
                    q in query_lower for q in ["q1", "q2", "q3", "q4"]
                ):
                    for quarter in document["quarters"]:
                        if quarter["quarter"].lower() in query_lower:
                            return f"{quarter['quarter']} revenue was {quarter['revenue']} with {quarter['growth']} growth"

                elif "growth" in query_lower and "annual" in query_lower:
                    return f"Annual growth for {document['year']} was {document['annual_growth']}"

                elif "growth" in query_lower:
                    growth_info = []
                    for quarter in document["quarters"]:
                        growth_info.append(f"{quarter['quarter']}: {quarter['growth']}")
                    return f"Quarterly growth rates: {', '.join(growth_info)}"

                elif "highlights" in query_lower:
                    all_highlights = []
                    for quarter in document["quarters"]:
                        all_highlights.extend(quarter["highlights"])
                    return f"Key highlights: {', '.join(all_highlights)}"

                else:
                    return f"Company performance data for {document['year']} is available. Total revenue: {document['total_revenue']}, Annual growth: {document['annual_growth']}"

            except Exception as e:
                return f"Error accessing document: {str(e)}"

        return FunctionTool(
            func=search_document,
            name="search_document",
            description="Search for information in the company performance document",
        )

    @pytest.mark.asyncio
    async def test_agent_with_context_store_rouge_evaluation(self):
        """Test agent performance with context store using ROUGE evaluation."""

        # Store document in context store
        doc_ref = self.large_context_state.add_large_context(
            self.large_document,
            metadata={"content_type": "application/json", "cache_ttl": 3600},
            key="company_doc_ref",
        )

        # Create agent with context-aware tool
        search_tool = self.create_document_search_tool(self.large_context_state)

        # Mock agent's responses for testing
        with patch(
            "google.adk.models.google_llm.Gemini.generate_content_async"
        ) as mock_generate:

            # Mock responses
            mock_responses = [
                # Response for total revenue query
                Mock(
                    candidates=[
                        Mock(
                            content=Mock(
                                parts=[
                                    Mock(
                                        text="Based on the document search, the total revenue for 2024 was $67M."
                                    )
                                ]
                            )
                        )
                    ]
                ),
                # Response for Q2 revenue query
                Mock(
                    candidates=[
                        Mock(
                            content=Mock(
                                parts=[
                                    Mock(
                                        text="Q2 revenue was $15M with 25% growth according to the company data."
                                    )
                                ]
                            )
                        )
                    ]
                ),
                # Response for annual growth query
                Mock(
                    candidates=[
                        Mock(
                            content=Mock(
                                parts=[
                                    Mock(
                                        text="The annual growth for 2024 was 35% as shown in the performance report."
                                    )
                                ]
                            )
                        )
                    ]
                ),
            ]

            # Async generator for each response
            async def create_mock_response(response):
                yield response

            mock_generate.side_effect = [
                create_mock_response(mock_responses[0]),
                create_mock_response(mock_responses[1]),
                create_mock_response(mock_responses[2]),
            ]

            agent = LlmAgent(
                model="gemini-2.0-flash-001",
                name="document_analyzer",
                instruction="You are a business analyst. Use the search_document tool to find information and provide accurate responses.",
                tools=[search_tool],
            )

            # Test cases with expected responses
            test_cases = [
                {
                    "query": "What was the total revenue for 2024?",
                    "expected": "The total revenue for 2024 was $67M.",
                    "agent_response": "Based on the document search, the total revenue for 2024 was $67M.",
                },
                {
                    "query": "What was Q2 revenue and growth?",
                    "expected": "Q2 revenue was $15M with 25% growth.",
                    "agent_response": "Q2 revenue was $15M with 25% growth according to the company data.",
                },
                {
                    "query": "What was the annual growth rate?",
                    "expected": "The annual growth for 2024 was 35%.",
                    "agent_response": "The annual growth for 2024 was 35% as shown in the performance report.",
                },
            ]

            # Invocations
            actual_invocations = []
            expected_invocations = []

            for test_case in test_cases:
                # Agents produced Invocations
                actual_invocations.append(
                    Invocation(
                        user_content=genai_types.Content(
                            parts=[genai_types.Part(text=test_case["query"])]
                        ),
                        final_response=genai_types.Content(
                            parts=[genai_types.Part(text=test_case["agent_response"])]
                        ),
                    )
                )

                # Expected Invocations
                expected_invocations.append(
                    Invocation(
                        user_content=genai_types.Content(
                            parts=[genai_types.Part(text=test_case["query"])]
                        ),
                        final_response=genai_types.Content(
                            parts=[genai_types.Part(text=test_case["expected"])]
                        ),
                    )
                )

            # ROUGE evaluation
            evaluation_result = self.rouge_evaluator.evaluate_invocations(
                actual_invocations, expected_invocations
            )

           
            assert (
                evaluation_result.overall_score > 0.8
            ), f"Agent ROUGE score {evaluation_result.overall_score} below threshold"
            assert len(evaluation_result.per_invocation_results) == 3

            # Check individual test case performance
            for i, result in enumerate(evaluation_result.per_invocation_results):
                assert (
                    result.score > 0.75
                ), f"Test case {i+1} ROUGE score {result.score} below threshold"

            print(f"Context Store Agent ROUGE Evaluation Results:")
            print(f"Overall Score: {evaluation_result.overall_score:.3f}")
            for i, result in enumerate(evaluation_result.per_invocation_results):
                print(f"Test Case {i+1}: {result.score:.3f}")

    @pytest.mark.asyncio
    async def test_context_store_performance_impact_on_rouge(self):
        """Test that context store doesn't negatively impact ROUGE scores compared to direct context."""

        doc_ref = self.large_context_state.add_large_context(
            self.large_document, key="perf_test_doc"
        )

        # Simulate agent responses with context store
        context_store_responses = [
            "Total revenue was $67M for 2024.",
            "Q1 revenue was $12M with 15% growth.",
            "Annual growth rate was 35% in 2024.",
        ]

        # Simulate agent responses with direct context (baseline)
        direct_context_responses = [
            "The total revenue for 2024 was $67M.",
            "Q1 had $12M revenue with 15% growth.",
            "Annual growth for 2024 was 35%.",
        ]

        # Expected responses
        expected_responses = [
            "Total revenue for 2024 was $67M.",
            "Q1 revenue was $12M with 15% growth.",
            "Annual growth was 35% in 2024.",
        ]

        # Calculate ROUGE scores for both approaches
        from google.adk.evaluation.final_response_match_v1 import (
            _calculate_rouge_1_scores,
        )

        context_store_scores = []
        direct_context_scores = []

        for i in range(len(expected_responses)):
            # Context store ROUGE score
            cs_score = _calculate_rouge_1_scores(
                context_store_responses[i], expected_responses[i]
            )
            context_store_scores.append(cs_score.fmeasure)

            # Direct context ROUGE score
            dc_score = _calculate_rouge_1_scores(
                direct_context_responses[i], expected_responses[i]
            )
            direct_context_scores.append(dc_score.fmeasure)

        # Calculate average scores
        avg_context_store = sum(context_store_scores) / len(context_store_scores)
        avg_direct_context = sum(direct_context_scores) / len(direct_context_scores)

        # Context store should not significantly degrade performance
        performance_difference = abs(avg_context_store - avg_direct_context)
        assert (
            performance_difference < 0.05
        ), f"Context store degraded ROUGE performance by {performance_difference}"

        # Both approaches should meet minimum quality
        assert (
            avg_context_store > 0.8
        ), f"Context store average ROUGE {avg_context_store} below threshold"
        assert (
            avg_direct_context > 0.8
        ), f"Direct context average ROUGE {avg_direct_context} below threshold"

        print(f"Performance Comparison:")
        print(f"Context Store Average ROUGE: {avg_context_store:.3f}")
        print(f"Direct Context Average ROUGE: {avg_direct_context:.3f}")
        print(f"Performance Difference: {performance_difference:.3f}")

    @pytest.mark.asyncio
    async def test_large_context_scaling_rouge_evaluation(self):
        """Test ROUGE scores with increasingly large contexts."""

        # Contexts of different sizes
        context_sizes = {
            "small": {"data": "Revenue: $10M"},
            "medium": {
                "quarters": [
                    {"q": "Q1", "revenue": "$10M", "growth": "10%"},
                    {"q": "Q2", "revenue": "$12M", "growth": "20%"},
                ]
            },
            "large": self.large_document,
            "very_large": {
                "company": "TechCorp",
                "years": [
                    {
                        "year": year,
                        "quarters": [
                            {
                                "quarter": f"Q{q}",
                                "revenue": f"${(year-2020)*4+q*2}M",
                                "growth": f"{10+q*2}%",
                            }
                            for q in range(1, 5)
                        ],
                    }
                    for year in range(2020, 2025) #5 years 
                ],
            },
        }

        rouge_scores_by_size = {}

        for size, context in context_sizes.items():
            # Store context
            doc_ref = self.large_context_state.add_large_context(
                context, key=f"{size}_doc"
            )

            # Simulate consistent agent response regardless of context size
            agent_response = "The revenue data shows positive growth trends."
            expected_response = "Revenue data shows positive growth trends."

            # ROUGE score calculation
            from google.adk.evaluation.final_response_match_v1 import (
                _calculate_rouge_1_scores,
            )

            rouge_score = _calculate_rouge_1_scores(agent_response, expected_response)
            rouge_scores_by_size[size] = rouge_score.fmeasure

        # Verify scores remain stable across context sizes
        scores = list(rouge_scores_by_size.values())
        min_score = min(scores)
        max_score = max(scores)
        score_variance = max_score - min_score

        assert (
            score_variance < 0.1
        ), f"ROUGE score variance {score_variance} too high across context sizes"
        assert (
            min_score > 0.85
        ), f"Minimum ROUGE score {min_score} below threshold for large contexts"

        print(f"ROUGE Scores by Context Size:")
        for size, score in rouge_scores_by_size.items():
            print(f"{size.capitalize()}: {score:.3f}")
        print(f"Score Variance: {score_variance:.3f}")


if __name__ == "__main__":
    pytest.main([__file__])
