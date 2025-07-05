# Large Context Reference Store Implementation Summary

## Overview

We've implemented a solution for efficiently handling large context windows (1M-2M tokens) in ADK and LangGraph by using a reference-based approach rather than direct context serialization.

## Files Created

1. **`src/google/adk/sessions/context_reference_store.py`**

   - Implements `ContextReferenceStore` class that stores contexts and provides reference-based access
   - Handles both text and structured data
   - Implements caching with LRU eviction

2. **`src/google/adk/sessions/large_context_state.py`**

   - Extends ADK's `State` class for efficient handling of large contexts
   - Provides methods for adding, retrieving, and managing context references
   - Supports Gemini's context caching feature for cost optimization

3. **`src/google/adk/utils/langgraph_utils.py`**

   - Provides utilities for integrating with LangGraph
   - Implements `LangGraphContextManager` for adding context to LangGraph state
   - Provides reference-aware merge functions for LangGraph

4. **`src/google/adk/examples/large_context_example.py`**

   - Demonstrates how to use the context reference store and large context state
   - Shows integration with LlmAgent and FunctionTool

5. **`tests/sessions/test_large_context_state.py`**

   - Unit tests for `ContextReferenceStore` and `LargeContextState`

6. **`tests/utils/test_langgraph_utils.py`**
   - Unit tests for LangGraph integration utilities

## Updated Files

1. **`src/google/adk/sessions/__init__.py`**

   - Added exports for new classes

2. **`src/google/adk/utils/__init__.py`**

   - Added exports for new LangGraph utilities

3. **`README.md`**
   - Added section on large context management

## Key Features Implemented

1. **Reference-based Context Storage**

   - Store large contexts once and reference them by ID
   - Prevent unnecessary serialization of large data

2. **Efficient Caching**

   - LRU eviction strategy for managing memory usage
   - Integration with Gemini's context caching for API cost optimization

3. **Structured Data Support**

   - Handle both text and JSON/dict data types
   - Automatic conversion between formats

4. **LangGraph Integration**
   - Special handling for context references in LangGraph state
   - Reference-aware merge functions for state updates

## Benefits

1. **Performance Improvements**

   - Reduces memory usage when handling large contexts
   - Prevents repeated serialization/deserialization
   - Enables efficient passing of large contexts between nodes in a graph

2. **Cost Optimization**

   - Integrates with Gemini's context caching to reduce API costs
   - Allows reuse of large contexts without re-tokenization

3. **Developer Experience**
   - Simple API for managing large contexts
   - Seamless integration with existing ADK and LangGraph code

## Usage Example

```python
from google.adk.sessions import LargeContextState, ContextReferenceStore
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

# Create a context store and state
context_store = ContextReferenceStore()
state = LargeContextState(context_store=context_store)

# Store large context by reference
state.add_large_context(large_document, key="document_ref")

# Define a function that efficiently uses the context
def search_document(context_state: LargeContextState, query: str):
    # Retrieve the document from the store without copying
    document = context_state.get_context("document_ref")
    # Process and return results...

# Create an agent that can work with the large context
agent = LlmAgent(
    model="gemini-1.5-pro",
    tools=[FunctionTool(func=search_document, name="search_document", description="...")],
    instruction="You have access to a large document through reference-based context..."
)

# Run the agent with the state containing the reference
result = agent.run({"user_input": "Find information about X", "state": state})
```
