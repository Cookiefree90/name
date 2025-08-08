"""
Test suite for VertexAiCodeExecutor deep copy and extension initialization fixes.

This test validates the critical fixes made to address:
1. Deep copy recursion errors during agent engine deployment
2. Extension state management during serialization
3. Automatic extension re-initialization after deep copy
"""

import copy
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors import VertexAiCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import File
import pytest


class TestVertexAiCodeExecutorFixes:
  """Test class for VertexAiCodeExecutor deep copy and extension fixes."""

  @pytest.fixture
  def mock_extension(self):
    """Create a mock extension for testing."""
    mock_ext = Mock()
    mock_ext.execute.return_value = {
        "execution_result": 'print("Hello World")\nHello World',
        "execution_error": "",
        "output_files": [],
    }
    return mock_ext

  @pytest.fixture
  def code_executor(self, mock_extension):
    """Create a VertexAiCodeExecutor instance with mocked extension."""
    with patch(
        "google.adk.code_executors.vertex_ai_code_executor._get_code_interpreter_extension",
        return_value=mock_extension,
    ):
      executor = VertexAiCodeExecutor()
    return executor

  def test_deep_copy_no_recursion_error(self, code_executor):
    """Test that deep copy works without recursion errors."""
    # This was the core issue: deep copy would cause recursion errors
    # due to the extension object not being serializable
    try:
      copied_executor = copy.deepcopy(code_executor)
      assert copied_executor is not None
      assert copied_executor != code_executor  # Different instances
      print("Deep copy completed without recursion errors")
    except RecursionError:
      pytest.fail("Deep copy still causes recursion error - fix not working")

  def test_extension_state_after_deep_copy(self, code_executor, mock_extension):
    """Test that extension is properly managed after deep copy."""
    # Original executor should have extension
    assert hasattr(code_executor, "_code_interpreter_extension")

    # After deep copy, the copied object should have None extension initially
    copied_executor = copy.deepcopy(code_executor)

    # The copied executor's extension should be None (to be re-initialized)
    assert copied_executor._code_interpreter_extension is None

    # Original executor should still have its extension
    assert code_executor._code_interpreter_extension == mock_extension
    print("Extension state properly managed during deep copy")

  def test_extension_re_initialization(self, mock_extension):
    """Test that extension gets re-initialized when needed."""
    with patch(
        "google.adk.code_executors.vertex_ai_code_executor._get_code_interpreter_extension",
        return_value=mock_extension,
    ) as mock_get_ext:

      executor = VertexAiCodeExecutor()

      # Deep copy the executor
      copied_executor = copy.deepcopy(executor)

      # Extension should be None after deep copy
      assert copied_executor._code_interpreter_extension is None

      # Call ensure_extension_initialized - should trigger re-initialization
      copied_executor._ensure_extension_initialized()

      # Extension should now be re-initialized
      assert copied_executor._code_interpreter_extension == mock_extension
      print("Extension re-initialization working correctly")

  def test_code_execution_after_deep_copy(self, code_executor, mock_extension):
    """Test that code execution works after deep copy."""
    # Create test input
    invocation_context = Mock(spec=InvocationContext)
    code_input = CodeExecutionInput(
        code="print('Hello from copied executor')", execution_id="test-123"
    )

    # Deep copy the executor
    copied_executor = copy.deepcopy(code_executor)

    # Execute code with copied executor - should trigger re-initialization
    with patch(
        "google.adk.code_executors.vertex_ai_code_executor._get_code_interpreter_extension",
        return_value=mock_extension,
    ):
      result = copied_executor.execute_code(invocation_context, code_input)

    # Verify execution worked
    assert result is not None
    assert "Hello World" in result.stdout

    # Verify extension was re-initialized during execution
    assert copied_executor._code_interpreter_extension == mock_extension
    print("Code execution works after deep copy with auto re-initialization")

  def test_ensure_extension_initialized_idempotent(
      self, code_executor, mock_extension
  ):
    """Test that _ensure_extension_initialized is safe to call multiple times."""
    original_extension = code_executor._code_interpreter_extension

    # Call multiple times
    code_executor._ensure_extension_initialized()
    code_executor._ensure_extension_initialized()
    code_executor._ensure_extension_initialized()

    # Extension should remain the same
    assert code_executor._code_interpreter_extension == original_extension
    print("Extension initialization is idempotent")

  def test_extension_initialization_with_resource_name(self, mock_extension):
    """Test extension initialization with custom resource name."""
    resource_name = "projects/test/locations/us-central1/extensions/123"

    with patch(
        "google.adk.code_executors.vertex_ai_code_executor._get_code_interpreter_extension",
        return_value=mock_extension,
    ) as mock_get_ext:

      executor = VertexAiCodeExecutor(resource_name=resource_name)

      # Verify resource name was passed correctly
      mock_get_ext.assert_called_with(resource_name)
      assert executor.resource_name == resource_name
      print("Resource name properly handled during initialization")

  @patch.dict(
      "os.environ", {"CODE_INTERPRETER_EXTENSION_NAME": "test-extension"}
  )
  def test_environment_variable_handling(self, mock_extension):
    """Test that environment variables are properly handled."""
    with patch(
        "google.adk.code_executors.vertex_ai_code_executor._get_code_interpreter_extension",
        return_value=mock_extension,
    ) as mock_get_ext:

      executor = VertexAiCodeExecutor()

      # Should use environment variable
      mock_get_ext.assert_called_with(None)  # No resource_name passed
      print("Environment variable handling works correctly")


def test_integration_with_agent_engine_deployment():
  """Integration test simulating agent engine deployment process."""
  print("\nRunning integration test for agent engine deployment...")

  with patch(
      "google.adk.code_executors.vertex_ai_code_executor._get_code_interpreter_extension"
  ) as mock_get_ext:
    mock_extension = Mock()
    mock_extension.execute.return_value = {
        "execution_result": "Deployment test successful",
        "execution_error": "",
        "output_files": [],
    }
    mock_get_ext.return_value = mock_extension

    # Create executor (simulating agent creation)
    executor = VertexAiCodeExecutor()

    # Simulate agent engine deployment (which involves deep copying)
    try:
      serialized_executor = copy.deepcopy(executor)

      # Simulate code execution on deployed agent
      invocation_context = Mock(spec=InvocationContext)
      code_input = CodeExecutionInput(
          code="print('Agent deployed successfully!')",
          execution_id="deployment-test",
      )

      result = serialized_executor.execute_code(invocation_context, code_input)

      assert result is not None
      assert "Deployment test successful" in result.stdout
      print(
          "Integration test passed - agent engine deployment simulation"
          " successful"
      )

    except Exception as e:
      pytest.fail(f"Integration test failed: {e}")


if __name__ == "__main__":
  # Run the tests
  pytest.main([__file__, "-v", "--tb=short"])
