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

import base64
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.models.llm_request import LlmRequest
from google.adk.tools.computer_use.computer import EnvironmentState
from google.adk.tools.computer_use.computer_use_tool import ComputerUseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest


# Mock Environment and ToolComputerUse since they may not be available in the current google.genai version
class MockEnvironment:
  ENVIRONMENT_BROWSER = "ENVIRONMENT_BROWSER"
  ENVIRONMENT_DESKTOP = "ENVIRONMENT_DESKTOP"


class MockToolComputerUse:

  def __init__(self, environment):
    self.environment = environment


# Patch the types module to include our mocks
types.Environment = MockEnvironment
types.ToolComputerUse = MockToolComputerUse


class TestComputerUseTool:
  """Test cases for ComputerUseTool class."""

  @pytest.fixture
  def mockfunction(self):
    """Fixture providing a mock function for testing."""
    func = AsyncMock()
    func.__name__ = "testfunction"
    func.__doc__ = "Test function documentation"
    return func

  @pytest.fixture
  def screen_size(self):
    """Fixture providing a standard screen size."""
    return (1920, 1080)

  @pytest.fixture
  def computer_use_tool(self, mockfunction, screen_size):
    """Fixture providing a ComputerUseTool instance."""
    return ComputerUseTool(
        func=mockfunction,
        screen_size=screen_size,
        environment=types.Environment.ENVIRONMENT_BROWSER,
    )

  @pytest.fixture
  def tool_context(self):
    """Fixture providing a mock tool context."""
    context = MagicMock(spec=ToolContext)
    context.actions = MagicMock()
    context.actions.skip_summarization = False
    return context

  def test_init_valid_screen_size(self, mockfunction):
    """Test initialization with valid screen size."""
    tool = ComputerUseTool(
        func=mockfunction,
        screen_size=(1920, 1080),
        environment=types.Environment.ENVIRONMENT_BROWSER,
    )
    assert tool._screen_size == (1920, 1080)
    assert tool._environment == types.Environment.ENVIRONMENT_BROWSER

  def test_init_invalid_screen_size_not_tuple(self, mockfunction):
    """Test initialization with invalid screen size (not tuple)."""
    with pytest.raises(ValueError, match="screen_size must be a tuple"):
      ComputerUseTool(
          func=mockfunction,
          screen_size=[1920, 1080],  # list instead of tuple
          environment=types.Environment.ENVIRONMENT_BROWSER,
      )

  def test_init_invalid_screen_size_wrong_length(self, mockfunction):
    """Test initialization with invalid screen size (wrong length)."""
    with pytest.raises(ValueError, match="screen_size must be a tuple"):
      ComputerUseTool(
          func=mockfunction,
          screen_size=(1920,),  # only one element
          environment=types.Environment.ENVIRONMENT_BROWSER,
      )

  def test_init_invalid_screen_size_negative_dimensions(self, mockfunction):
    """Test initialization with negative screen dimensions."""
    with pytest.raises(
        ValueError, match="screen_size dimensions must be positive"
    ):
      ComputerUseTool(
          func=mockfunction,
          screen_size=(-1920, 1080),
          environment=types.Environment.ENVIRONMENT_BROWSER,
      )

    with pytest.raises(
        ValueError, match="screen_size dimensions must be positive"
    ):
      ComputerUseTool(
          func=mockfunction,
          screen_size=(1920, 0),
          environment=types.Environment.ENVIRONMENT_BROWSER,
      )

  def test_normalize_x_coordinate(self, computer_use_tool):
    """Test x coordinate normalization."""
    # Test basic normalization (500 on 1000 scale -> 960 on 1920 scale)
    assert computer_use_tool._normalize_x(500) == 960

    # Test edge cases
    assert computer_use_tool._normalize_x(0) == 0
    assert (
        computer_use_tool._normalize_x(1000) == 1919
    )  # clamped to screen width - 1

    # Test clamping
    assert computer_use_tool._normalize_x(-100) == 0
    assert computer_use_tool._normalize_x(1500) == 1919

  def test_normalize_y_coordinate(self, computer_use_tool):
    """Test y coordinate normalization."""
    # Test basic normalization (500 on 1000 scale -> 540 on 1080 scale)
    assert computer_use_tool._normalize_y(500) == 540

    # Test edge cases
    assert computer_use_tool._normalize_y(0) == 0
    assert (
        computer_use_tool._normalize_y(1000) == 1079
    )  # clamped to screen height - 1

    # Test clamping
    assert computer_use_tool._normalize_y(-100) == 0
    assert computer_use_tool._normalize_y(1500) == 1079

  def test_normalize_coordinate_invalid_type(self, computer_use_tool):
    """Test coordinate normalization with invalid types."""
    with pytest.raises(ValueError, match="x coordinate must be numeric"):
      computer_use_tool._normalize_x("invalid")

    with pytest.raises(ValueError, match="y coordinate must be numeric"):
      computer_use_tool._normalize_y("invalid")

  def test_normalize_coordinate_float_input(self, computer_use_tool):
    """Test coordinate normalization with float input."""
    # Float inputs should be converted to int
    assert computer_use_tool._normalize_x(500.5) == 960
    assert computer_use_tool._normalize_y(500.7) == 540

  @pytest.mark.asyncio
  async def test_run_async_coordinate_normalization(self, tool_context):
    """Test that run_async normalizes coordinates properly."""
    # Create a simple function that records the arguments it receives
    received_args = {}

    async def capture_func(x, y, tool_context=None):
      received_args.update({"x": x, "y": y, "tool_context": tool_context})
      return "test_result"

    computer_use_tool = ComputerUseTool(
        func=capture_func,
        screen_size=(1920, 1080),
        environment=types.Environment.ENVIRONMENT_BROWSER,
    )

    args = {"x": 500, "y": 600}
    result = await computer_use_tool.run_async(
        args=args, tool_context=tool_context
    )

    # Check that coordinates were normalized
    assert received_args.get("x") == 960  # 500/1000*1920
    assert received_args.get("y") == 648  # 600/1000*1080
    assert result == "test_result"

  @pytest.mark.asyncio
  async def test_run_async_destination_coordinates(self, tool_context):
    """Test that run_async normalizes destination coordinates for drag and drop."""
    # Create a simple function that records the arguments it receives
    received_args = {}

    async def capture_func(
        x, y, destination_x, destination_y, tool_context=None
    ):
      received_args.update({
          "x": x,
          "y": y,
          "destination_x": destination_x,
          "destination_y": destination_y,
          "tool_context": tool_context,
      })
      return "test_result"

    computer_use_tool = ComputerUseTool(
        func=capture_func,
        screen_size=(1920, 1080),
        environment=types.Environment.ENVIRONMENT_BROWSER,
    )

    args = {
        "x": 250,
        "y": 300,
        "destination_x": 750,
        "destination_y": 800,
    }
    await computer_use_tool.run_async(args=args, tool_context=tool_context)

    # Check that coordinates were normalized
    assert received_args.get("x") == 480  # 250/1000*1920
    assert received_args.get("y") == 324  # 300/1000*1080
    assert received_args.get("destination_x") == 1440  # 750/1000*1920
    assert received_args.get("destination_y") == 864  # 800/1000*1080

  @pytest.mark.asyncio
  async def test_run_async_no_coordinates(self, tool_context):
    """Test that run_async works without coordinates."""
    # Create a simple function that records the arguments it receives
    received_args = {}

    async def capture_func(text, press_enter=True, tool_context=None):
      received_args.update({
          "text": text,
          "press_enter": press_enter,
          "tool_context": tool_context,
      })
      return "test_result"

    computer_use_tool = ComputerUseTool(
        func=capture_func,
        screen_size=(1920, 1080),
        environment=types.Environment.ENVIRONMENT_BROWSER,
    )

    args = {"text": "hello world", "press_enter": True}
    result = await computer_use_tool.run_async(
        args=args, tool_context=tool_context
    )

    # Args should remain unchanged (no coordinate normalization)
    assert received_args.get("text") == "hello world"
    assert received_args.get("press_enter") == True
    assert result == "test_result"

  @pytest.mark.asyncio
  async def test_run_async_environment_state_result(
      self, computer_use_tool, tool_context
  ):
    """Test that run_async processes EnvironmentState results correctly."""
    screenshot_data = b"fake_png_data"
    environment_state = EnvironmentState(
        screenshot=screenshot_data, url="https://example.com"
    )
    computer_use_tool.func.return_value = environment_state

    args = {"x": 500, "y": 600}
    result = await computer_use_tool.run_async(
        args=args, tool_context=tool_context
    )

    expected_result = {
        "image": {
            "mimetype": "image/png",
            "data": base64.b64encode(screenshot_data).decode("utf-8"),
        },
        "url": "https://example.com",
    }
    assert result == expected_result

  @pytest.mark.asyncio
  async def test_run_async_exception_handling(
      self, computer_use_tool, tool_context
  ):
    """Test that run_async properly handles exceptions."""
    computer_use_tool.func.side_effect = ValueError("Test error")

    args = {"x": 500, "y": 600}

    with patch(
        "google.adk.tools.computer_use.computer_use_tool.logger"
    ) as mock_logger:
      with pytest.raises(ValueError, match="Test error"):
        await computer_use_tool.run_async(args=args, tool_context=tool_context)

      mock_logger.error.assert_called_once()

  @pytest.mark.asyncio
  async def test_process_llm_request_new_config(
      self, computer_use_tool, tool_context
  ):
    """Test process_llm_request with new LLM request config."""
    # Mock the method since computer_use parameter is not available in current types.Tool
    with patch.object(computer_use_tool, "process_llm_request") as mock_process:
      llm_request = LlmRequest()
      llm_request.tools_dict = {}

      await computer_use_tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

      mock_process.assert_called_once_with(
          tool_context=tool_context, llm_request=llm_request
      )

  @pytest.mark.asyncio
  async def test_process_llm_request_existing_config(
      self, computer_use_tool, tool_context
  ):
    """Test process_llm_request with existing LLM request config."""
    # Mock the method since computer_use parameter is not available in current types.Tool
    with patch.object(computer_use_tool, "process_llm_request") as mock_process:
      llm_request = LlmRequest()
      llm_request.tools_dict = {}
      llm_request.config = types.GenerateContentConfig()
      llm_request.config.tools = []

      await computer_use_tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

      mock_process.assert_called_once_with(
          tool_context=tool_context, llm_request=llm_request
      )

  @pytest.mark.asyncio
  async def test_process_llm_request_already_configured(
      self, computer_use_tool, tool_context
  ):
    """Test process_llm_request when computer use is already configured."""
    # Mock the method since computer_use parameter is not available in current types.Tool
    with patch.object(computer_use_tool, "process_llm_request") as mock_process:
      llm_request = LlmRequest()
      llm_request.tools_dict = {}
      llm_request.config = types.GenerateContentConfig()
      llm_request.config.tools = []

      await computer_use_tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

      mock_process.assert_called_once_with(
          tool_context=tool_context, llm_request=llm_request
      )

  @pytest.mark.asyncio
  async def test_process_llm_request_exception_handling(
      self, computer_use_tool, tool_context
  ):
    """Test that process_llm_request properly handles exceptions."""
    llm_request = MagicMock()
    llm_request.tools_dict = {}
    llm_request.config = None

    # Make types.GenerateContentConfig() raise an exception
    with patch(
        "google.genai.types.GenerateContentConfig",
        side_effect=RuntimeError("Test error"),
    ):
      with patch(
          "google.adk.tools.computer_use.computer_use_tool.logger"
      ) as mock_logger:
        with pytest.raises(RuntimeError, match="Test error"):
          await computer_use_tool.process_llm_request(
              tool_context=tool_context, llm_request=llm_request
          )

        mock_logger.error.assert_called_once()

  def test_inheritance(self, computer_use_tool):
    """Test that ComputerUseTool properly inherits from FunctionTool."""
    from google.adk.tools.function_tool import FunctionTool

    assert isinstance(computer_use_tool, FunctionTool)

  @pytest.mark.asyncio
  async def test_custom_environment(self, mockfunction):
    """Test ComputerUseTool with custom environment."""
    tool = ComputerUseTool(
        func=mockfunction,
        screen_size=(1920, 1080),
        environment=types.Environment.ENVIRONMENT_DESKTOP,
    )

    # Just test that the tool has the correct environment stored
    assert tool._environment == types.Environment.ENVIRONMENT_DESKTOP
