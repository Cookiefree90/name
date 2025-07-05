# Add Large Context Reference Store for Efficient Context Management

## Summary

This PR adds a new `ContextReferenceStore` and `LargeContextState` implementation to efficiently handle large context windows (1M-2M tokens) when working with Gemini models. The key features include:

- Reference-based context storage to prevent unnecessary serialization of large data
- Efficient caching with LRU eviction strategy
- Support for both text and structured data formats
- Integration with Gemini's context caching feature for cost optimization
- LangGraph integration for reference-aware state management

## Problem Solved

When working with large context windows (1M-2M tokens), several inefficiencies can arise:

1. **Inefficient State Serialization**: Repeatedly serializing and deserializing large contexts causes performance bottlenecks
2. **LangGraph State Passing**: Passing the full context between nodes in a LangGraph creates performance issues
3. **No Context Caching**: Lacking a mechanism to cache large contexts that are reused multiple times
4. **Multi-agent Inefficiency**: Duplicating large contexts across multiple agents wastes memory

This implementation solves these issues by storing context once and passing references instead of the full content.

## Implementation Details

The implementation includes:

1. **`ContextReferenceStore`**: A store for large contexts that provides reference-based access
2. **`LargeContextState`**: An extension of ADK's State class that integrates with the context store
3. **`LangGraphContextManager`**: Utilities for LangGraph integration
4. **Example and Unit Tests**: Comprehensive example and tests for the new functionality

## Testing

Added comprehensive unit tests for:

- `ContextReferenceStore`
- `LargeContextState`
- LangGraph utilities

## Documentation

Added:

- Detailed docstrings for all classes and methods
- Example usage in `large_context_example.py`
- README section explaining the feature

## Related Issues

This PR addresses the need for more efficient handling of large contexts with Gemini 1.5's massive context window capabilities.
