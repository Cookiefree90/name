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
ROUGE-based evaluation tests for Context Reference Store.

This module tests whether the context reference store maintains response quality
by using ROUGE metrics to compare agent responses with and without the context store.
"""

import pytest
import json
from typing import Dict, Any, List
from unittest.mock import Mock, patch


from google.adk.sessions.context_reference_store import (
    ContextReferenceStore,
    ContextMetadata,
)
from google.adk.sessions.large_context_state import LargeContextState


from google.adk.evaluation.final_response_match_v1 import (
    RougeEvaluator,
    _calculate_rouge_1_scores,
)
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluator import EvalStatus
from google.genai import types as genai_types


class TestContextStoreRougeEvaluation:
    """Tests for validating Context Reference Store using ROUGE metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context_store = ContextReferenceStore()
        self.large_context_state = LargeContextState(context_store=self.context_store)
        self.rouge_evaluator = RougeEvaluator(
            EvalMetric(metric_name="response_match_score", threshold=0.8)
        )

    def test_rouge_score_with_small_context(self):
        """Test that ROUGE scores are maintained with small contexts."""
        # Test document
        small_document = {
            "title": "Company Report",
            "content": "Revenue increased by 25% last quarter due to strong sales performance.",
        }

        # Store in context store
        doc_ref = self.large_context_state.add_large_context(
            small_document, key="doc_ref"
        )

        # Simulate agent responses
        expected_response = "The revenue increased by 25% last quarter."
        actual_response_with_context = "Revenue increased by 25% in the last quarter."


        rouge_score = _calculate_rouge_1_scores(
            actual_response_with_context, expected_response
        )


        assert (
            rouge_score.fmeasure > 0.8
        ), f"ROUGE F1 score {rouge_score.fmeasure} is below threshold"
        assert (
            rouge_score.precision > 0.7
        ), f"ROUGE precision {rouge_score.precision} is low"
        assert rouge_score.recall > 0.7, f"ROUGE recall {rouge_score.recall} is low"

    def test_rouge_evaluation_with_context_references(self):
        """Test full ROUGE evaluation using context references."""
        # Test documents with varying sizes
        test_documents = [
            {
                "id": "doc1",
                "content": "The quarterly financial report shows revenue of $10M with 15% growth.",
            },
            {
                "id": "doc2",
                "content": "Customer satisfaction scores improved to 4.2/5 with reduced response times.",
            },
            {
                "id": "doc3",
                "content": "Product launch was successful with 50K pre-orders in the first week.",
            },
        ]


        doc_refs = []
        for i, doc in enumerate(test_documents):
            ref = self.large_context_state.add_large_context(doc, key=f"doc_ref_{i}")
            doc_refs.append(ref)

        # Test invocations
        actual_invocations = []
        expected_invocations = []

        # Test case 1: Revenue question
        actual_invocations.append(
            Invocation(
                user_content=genai_types.Content(
                    parts=[genai_types.Part(text="What was the revenue last quarter?")]
                ),
                final_response=genai_types.Content(
                    parts=[
                        genai_types.Part(
                            text="The revenue was $10M with 15% growth last quarter."
                        )
                    ]
                ),
            )
        )
        expected_invocations.append(
            Invocation(
                user_content=genai_types.Content(
                    parts=[genai_types.Part(text="What was the revenue last quarter?")]
                ),
                final_response=genai_types.Content(
                    parts=[
                        genai_types.Part(
                            text="Revenue was $10M with 15% growth in the quarter."
                        )
                    ]
                ),
            )
        )

        # Test case 2: Customer satisfaction question
        actual_invocations.append(
            Invocation(
                user_content=genai_types.Content(
                    parts=[
                        genai_types.Part(text="How did customer satisfaction change?")
                    ]
                ),
                final_response=genai_types.Content(
                    parts=[
                        genai_types.Part(
                            text="Customer satisfaction improved to 4.2 out of 5 scores."
                        )
                    ]
                ),
            )
        )
        expected_invocations.append(
            Invocation(
                user_content=genai_types.Content(
                    parts=[
                        genai_types.Part(text="How did customer satisfaction change?")
                    ]
                ),
                final_response=genai_types.Content(
                    parts=[
                        genai_types.Part(
                            text="Customer satisfaction scores improved to 4.2/5."
                        )
                    ]
                ),
            )
        )

        # Run ROUGE evaluation
        evaluation_result = self.rouge_evaluator.evaluate_invocations(
            actual_invocations, expected_invocations
        )


        assert (
            evaluation_result.overall_score > 0.7
        ), f"Overall ROUGE score {evaluation_result.overall_score} is too low"
        assert len(evaluation_result.per_invocation_results) == 2


        for result in evaluation_result.per_invocation_results:
            assert (
                result.score > 0.6
            ), f"Individual ROUGE score {result.score} is too low"

    def test_context_store_vs_direct_context_rouge_comparison(self):
        """Compare ROUGE scores between context store and direct context approaches."""

        large_document = {
            "sections": [
                {
                    "title": "Executive Summary",
                    "content": "The company achieved record revenue of $50M this year, representing 30% growth. Key achievements include product launch success, improved customer satisfaction, and operational efficiency gains.",
                },
                {
                    "title": "Financial Performance",
                    "content": "Revenue breakdown shows 40% from existing products, 35% from new product launches, and 25% from service offerings. Profit margins improved by 5% due to cost optimization initiatives.",
                },
                {
                    "title": "Customer Metrics",
                    "content": "Customer satisfaction increased to 4.4/5, with Net Promoter Score reaching 75. Customer retention rate improved to 92%, the highest in company history.",
                },
            ]
        }


        doc_ref = self.large_context_state.add_large_context(
            large_document, key="large_doc_ref"
        )

        query = "What was the company's revenue performance this year?"
        expected_response = (
            "The company achieved record revenue of $50M this year with 30% growth."
        )

        # Simulate two agent responses:
        # 1. Using context store (should maintain quality)
        response_with_context_store = "The company achieved record revenue of $50M representing 30% growth this year."

        # 2. Using direct context (baseline)
        response_with_direct_context = (
            "The company achieved $50M revenue this year, a 30% growth record."
        )

        # ROUGE scores for both approaches
        rouge_context_store = _calculate_rouge_1_scores(
            response_with_context_store, expected_response
        )
        rouge_direct_context = _calculate_rouge_1_scores(
            response_with_direct_context, expected_response
        )

        # Context store should maintain similar or better quality
        score_difference = abs(
            rouge_context_store.fmeasure - rouge_direct_context.fmeasure
        )
        assert (
            score_difference < 0.1
        ), f"Context store degraded quality by {score_difference}"

        # Both should meet minimum quality threshold
        assert (
            rouge_context_store.fmeasure > 0.75
        ), f"Context store ROUGE score {rouge_context_store.fmeasure} below threshold"
        assert (
            rouge_direct_context.fmeasure > 0.75
        ), f"Direct context ROUGE score {rouge_direct_context.fmeasure} below threshold"

    def test_rouge_scores_across_context_sizes(self):
        """Test that ROUGE scores remain stable across different context sizes."""
        context_sizes = ["small", "medium", "large", "very_large"]
        rouge_scores = []

        # Contexts of different sizes
        contexts = {
            "small": "Revenue was $10M.",
            "medium": "The quarterly financial report shows total revenue of $10M with strong growth in all segments.",
            "large": " ".join(
                [
                    "The comprehensive quarterly financial report demonstrates exceptional performance with total revenue reaching $10M."
                ]
                * 10
            ),
            "very_large": " ".join(
                [
                    "The detailed quarterly financial report shows comprehensive analysis of revenue performance reaching $10M with growth across all business segments."
                ]
                * 50
            ),
        }

        expected_response = "Revenue was $10M in the quarter."

        for size in context_sizes:
            # Store context
            context_ref = self.large_context_state.add_large_context(
                contexts[size], key=f"{size}_context"
            )

            # Simulate agent response using context
            agent_response = "The revenue reached $10M in the quarterly report."

            # Calculate ROUGE score
            rouge_score = _calculate_rouge_1_scores(agent_response, expected_response)
            rouge_scores.append(rouge_score.fmeasure)

        # Verify scores remain relatively stable
        min_score = min(rouge_scores)
        max_score = max(rouge_scores)
        score_variance = max_score - min_score

        assert (
            score_variance < 0.2
        ), f"ROUGE score variance {score_variance} is too high across context sizes"
        assert (
            min_score > 0.5
        ), f"Minimum ROUGE score {min_score} is below acceptable threshold"

    def test_context_reference_retrieval_accuracy(self):
        """Test that context retrieval maintains content accuracy for ROUGE evaluation."""
        # Original structured document
        original_document = {
            "company": "TechCorp",
            "year": 2024,
            "metrics": {"revenue": "$50M", "growth": "30%", "customers": 10000},
            "highlights": [
                "Record revenue achievement",
                "Customer satisfaction improved",
                "Successful product launches",
            ],
        }

        # Store and retrieve
        doc_ref = self.large_context_state.add_large_context(
            original_document, key="structured_doc"
        )
        retrieved_document = self.large_context_state.get_context("structured_doc")

        # Verify exact match 
        assert (
            original_document == retrieved_document
        ), "Context store altered document content"

        # ROUGE evaluation test with retrieved context
        expected_response = "TechCorp achieved $50M revenue with 30% growth in 2024."
        actual_response = "TechCorp achieved $50M revenue and 30% growth in 2024."

        rouge_score = _calculate_rouge_1_scores(actual_response, expected_response)

        # Should achieve high ROUGE score with accurate context
        assert (
            rouge_score.fmeasure > 0.85
        ), f"ROUGE score {rouge_score.fmeasure} indicates context accuracy issues"

    def test_cache_hint_impact_on_rouge_scores(self):
        """Test that cache hints don't affect ROUGE evaluation quality."""

        document_with_cache = {
            "content": "Performance metrics show significant improvement across all key indicators this quarter.",
            "source": "Q4 Performance Report",
        }

        # Store with cache hint
        doc_ref = self.large_context_state.add_large_context(
            document_with_cache,
            metadata={"cache_ttl": 3600, "priority": "high"},
            key="cached_doc",
        )

        # Get cache hint
        cache_hint = self.large_context_state.with_cache_hint("cached_doc")

        # Verify cache hint exists
        assert "cache_id" in cache_hint
        assert cache_hint["cache_level"] == "HIGH"

        # Test ROUGE score with cached context
        expected_response = "Performance metrics improved significantly this quarter."
        actual_response = (
            "Performance metrics show significant improvement this quarter."
        )

        rouge_score = _calculate_rouge_1_scores(actual_response, expected_response)

        # Cache shouldn't affect response quality
        assert (
            rouge_score.fmeasure > 0.7
        ), f"Cache implementation affected ROUGE quality: {rouge_score.fmeasure}"

    def test_multiple_context_references_rouge_evaluation(self):
        """Test ROUGE evaluation when using multiple context references."""
        # Multiple related documents
        documents = [
            {"id": "sales", "content": "Sales increased by 40% compared to last year."},
            {
                "id": "marketing",
                "content": "Marketing campaigns reached 2M people with 15% conversion.",
            },
            {
                "id": "operations",
                "content": "Operational efficiency improved by 25% through automation.",
            },
        ]


        refs = []
        for doc in documents:
            ref = self.large_context_state.add_large_context(
                doc, key=f"{doc['id']}_ref"
            )
            refs.append(ref)

        # Test query requiring multiple contexts
        expected_response = "Sales increased 40%, marketing reached 2M people with 15% conversion, and operations improved 25%."
        actual_response = "Sales increased by 40%, marketing campaigns reached 2M people with 15% conversion rate, and operational efficiency improved 25%."

        rouge_score = _calculate_rouge_1_scores(actual_response, expected_response)

        # Multiple context handling should maintain high ROUGE scores
        assert (
            rouge_score.fmeasure > 0.75
        ), f"Multiple context ROUGE score {rouge_score.fmeasure} below threshold"
        assert (
            rouge_score.precision > 0.7
        ), f"Multiple context precision {rouge_score.precision} is low"
        assert (
            rouge_score.recall > 0.8
        ), f"Multiple context recall {rouge_score.recall} is low"

    def test_rouge_evaluation_failure_cases(self):
        """Test ROUGE evaluation for cases where context store might fail."""
        # Edge case: Empty context
        empty_doc_ref = self.large_context_state.add_large_context("", key="empty_doc")

        # Edge case: Very short context
        short_doc_ref = self.large_context_state.add_large_context(
            "OK", key="short_doc"
        )

        # Edge case: Context with special characters
        special_doc = {"data": "Revenue: $50M (30% ↑) - Q4'24"}
        special_doc_ref = self.large_context_state.add_large_context(
            special_doc, key="special_doc"
        )

        empty_retrieved = self.large_context_state.get_context("empty_doc")
        short_retrieved = self.large_context_state.get_context("short_doc")
        special_retrieved = self.large_context_state.get_context("special_doc")

        # Verify edge cases are handled correctly
        assert empty_retrieved == ""
        assert short_retrieved == "OK"
        assert special_retrieved == special_doc

        # Test ROUGE with edge case (should handle gracefully)
        rouge_score = _calculate_rouge_1_scores("No data available", "")
        assert (
            rouge_score.fmeasure == 0.0
        ), "Empty reference should result in 0 ROUGE score"


if __name__ == "__main__":
    pytest.main([__file__])
