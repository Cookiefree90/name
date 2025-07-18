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

from unittest.mock import MagicMock

from google.adk.tools.computer_use.computer import Computer
from google.adk.tools.computer_use.computer import EnvironmentState
from google.adk.tools.computer_use.computer_use_tool import ComputerUseTool
from google.adk.tools.computer_use.computer_use_toolset import ComputerUseToolset
import pytest


# Mock Environment enum since it may not be available in the current google.genai version
class MockEnvironment:
  ENVIRONMENT_BROWSER = "ENVIRONMENT_BROWSER"
  ENVIRONMENT_DESKTOP = "ENVIRONMENT_DESKTOP"


Environment = MockEnvironment


class MockComputer(Computer):
  """Mock Computer implementation for testing."""

  def __init__(self):
    self.initialize_called = False
    self.close_called = False
    self._screen_size = (1920, 1080)
    self._environment = Environment.ENVIRONMENT_BROWSER

  async def initialize(self):
    self.initialize_called = True

  async def close(self):
    self.close_called = True

  async def screen_size(self) -> tuple[int, int]:
    return self._screen_size

  async def environment(self) -> Environment:
    return self._environment

  # Implement all abstract methods to make this a concrete class
  async def open_web_browser(self) -> EnvironmentState:
    return EnvironmentState(screenshot=b"test", url="https://example.com")

  async def click_at(self, x: int, y: int) -> EnvironmentState:
    return EnvironmentState(screenshot=b"test", url="https://example.com")

  async def hover_at(self, x: int, y: int) -> EnvironmentState:
    return EnvironmentState(screenshot=b"test", url="https://example.com")

  async def type_text_at(
      self,
      x: int,
      y: int,
      text: str,
      press_enter: bool = True,
      clear_before_typing: bool = True,
  ) -> EnvironmentState:
    return EnvironmentState(screenshot=b"test", url="https://example.com")

  async def scroll_document(self, direction: str) -> EnvironmentState:
    return EnvironmentState(screenshot=b"test", url="https://example.com")

  async def scroll_at(
      self, x: int, y: int, direction: str, magnitude: int
  ) -> EnvironmentState:
    return EnvironmentState(screenshot=b"test", url="https://example.com")

  async def wait_5_seconds(self) -> EnvironmentState:
    return EnvironmentState(screenshot=b"test", url="https://example.com")

  async def go_back(self) -> EnvironmentState:
    return EnvironmentState(screenshot=b"test", url="https://example.com")

  async def go_forward(self) -> EnvironmentState:
    return EnvironmentState(screenshot=b"test", url="https://example.com")

  async def search(self) -> EnvironmentState:
    return EnvironmentState(screenshot=b"test", url="https://example.com")

  async def navigate(self, url: str) -> EnvironmentState:
    return EnvironmentState(screenshot=b"test", url=url)

  async def key_combination(self, keys: list[str]) -> EnvironmentState:
    return EnvironmentState(screenshot=b"test", url="https://example.com")

  async def drag_and_drop(
      self, x: int, y: int, destination_x: int, destination_y: int
  ) -> EnvironmentState:
    return EnvironmentState(screenshot=b"test", url="https://example.com")

  async def current_state(self) -> EnvironmentState:
    return EnvironmentState(screenshot=b"test", url="https://example.com")


class TestComputerUseToolset:
  """Test cases for ComputerUseToolset class."""

  @pytest.fixture
  def mock_computer(self):
    """Fixture providing a mock computer."""
    return MockComputer()

  @pytest.fixture
  def toolset(self, mock_computer):
    """Fixture providing a ComputerUseToolset instance."""
    return ComputerUseToolset(computer=mock_computer)

  def test_init(self, mock_computer):
    """Test ComputerUseToolset initialization."""
    toolset = ComputerUseToolset(computer=mock_computer)

    assert toolset._computer == mock_computer
    assert toolset._initialized is False

  @pytest.mark.asyncio
  async def test_ensure_initialized(self, toolset, mock_computer):
    """Test that _ensure_initialized calls computer.initialize()."""
    assert not mock_computer.initialize_called
    assert not toolset._initialized

    await toolset._ensure_initialized()

    assert mock_computer.initialize_called
    assert toolset._initialized

  @pytest.mark.asyncio
  async def test_ensure_initialized_only_once(self, toolset, mock_computer):
    """Test that _ensure_initialized only calls initialize once."""
    await toolset._ensure_initialized()

    # Reset the flag to test it's not called again
    mock_computer.initialize_called = False

    await toolset._ensure_initialized()

    # Should not be called again
    assert not mock_computer.initialize_called
    assert toolset._initialized

  @pytest.mark.asyncio
  async def test_get_tools(self, toolset, mock_computer):
    """Test that get_tools returns ComputerUseTool instances."""
    tools = await toolset.get_tools()

    # Should initialize the computer
    assert mock_computer.initialize_called

    # Should return a list of ComputerUseTool instances
    assert isinstance(tools, list)
    assert len(tools) > 0
    assert all(isinstance(tool, ComputerUseTool) for tool in tools)

    # Each tool should have the correct configuration
    for tool in tools:
      assert tool._screen_size == (1920, 1080)
      assert tool._environment == Environment.ENVIRONMENT_BROWSER

  @pytest.mark.asyncio
  async def test_get_tools_excludes_utility_methods(self, toolset):
    """Test that get_tools excludes utility methods like screen_size, environment, close."""
    tools = await toolset.get_tools()

    # Get tool function names
    tool_names = [tool.func.__name__ for tool in tools]

    # Should exclude utility methods
    excluded_methods = {"screen_size", "environment", "close"}
    for method in excluded_methods:
      assert method not in tool_names

    # initialize might be included since it's a concrete method, not just abstract
    # This is acceptable behavior

    # Should include action methods
    expected_methods = {
        "open_web_browser",
        "click_at",
        "hover_at",
        "type_text_at",
        "scroll_document",
        "scroll_at",
        "wait_5_seconds",
        "go_back",
        "go_forward",
        "search",
        "navigate",
        "key_combination",
        "drag_and_drop",
        "current_state",
    }
    for method in expected_methods:
      assert method in tool_names

  @pytest.mark.asyncio
  async def test_get_tools_with_readonly_context(self, toolset):
    """Test get_tools with readonly_context parameter."""
    from google.adk.agents.readonly_context import ReadonlyContext

    readonly_context = MagicMock(spec=ReadonlyContext)

    tools = await toolset.get_tools(readonly_context=readonly_context)

    # Should still return tools (readonly_context doesn't affect behavior currently)
    assert isinstance(tools, list)
    assert len(tools) > 0

  @pytest.mark.asyncio
  async def test_close(self, toolset, mock_computer):
    """Test that close calls computer.close()."""
    await toolset.close()

    assert mock_computer.close_called

  @pytest.mark.asyncio
  async def test_get_tools_creates_tools_with_correct_methods(
      self, toolset, mock_computer
  ):
    """Test that get_tools creates tools with the correct underlying methods."""
    tools = await toolset.get_tools()

    # Find the click_at tool
    click_tool = None
    for tool in tools:
      if tool.func.__name__ == "click_at":
        click_tool = tool
        break

    assert click_tool is not None

    # The tool's function should be bound to the mock computer instance
    assert click_tool.func.__self__ == mock_computer

  @pytest.mark.asyncio
  async def test_get_tools_handles_custom_screen_size(self, mock_computer):
    """Test get_tools with custom screen size."""
    mock_computer._screen_size = (2560, 1440)

    toolset = ComputerUseToolset(computer=mock_computer)
    tools = await toolset.get_tools()

    # All tools should have the custom screen size
    for tool in tools:
      assert tool._screen_size == (2560, 1440)

  @pytest.mark.asyncio
  async def test_get_tools_handles_custom_environment(self, mock_computer):
    """Test get_tools with custom environment."""
    mock_computer._environment = Environment.ENVIRONMENT_DESKTOP

    toolset = ComputerUseToolset(computer=mock_computer)
    tools = await toolset.get_tools()

    # All tools should have the custom environment
    for tool in tools:
      assert tool._environment == Environment.ENVIRONMENT_DESKTOP

  @pytest.mark.asyncio
  async def test_multiple_get_tools_calls_return_different_instances(
      self, toolset
  ):
    """Test that multiple get_tools calls return different tool instances."""
    tools1 = await toolset.get_tools()
    tools2 = await toolset.get_tools()

    # Should return different instances
    for tool1, tool2 in zip(tools1, tools2):
      assert tool1 is not tool2
      # But should have the same configuration
      assert tool1._screen_size == tool2._screen_size
      assert tool1._environment == tool2._environment

  def test_inheritance(self, toolset):
    """Test that ComputerUseToolset inherits from BaseToolset."""
    from google.adk.tools.base_toolset import BaseToolset

    assert isinstance(toolset, BaseToolset)

  @pytest.mark.asyncio
  async def test_get_tools_method_filtering(self, toolset):
    """Test that get_tools properly filters methods from Computer ABC."""
    tools = await toolset.get_tools()
    tool_names = [tool.func.__name__ for tool in tools]

    # Should not include private methods
    assert not any(name.startswith("_") for name in tool_names)

    # Should not include class methods, static methods, or properties
    # that aren't actual computer action methods
    forbidden_names = {
        "__init__",
        "__new__",
        "__class__",
        "__dict__",
        "__module__",
        "__doc__",
        "__annotations__",
        "__abstractmethods__",
    }
    for forbidden in forbidden_names:
      assert forbidden not in tool_names

  @pytest.mark.asyncio
  async def test_computer_method_binding(self, toolset, mock_computer):
    """Test that computer methods are properly bound to the computer instance."""
    tools = await toolset.get_tools()

    for tool in tools:
      # Each tool's function should be a bound method of the computer
      assert hasattr(tool.func, "__self__")
      assert tool.func.__self__ == mock_computer

  @pytest.mark.asyncio
  async def test_toolset_handles_computer_initialization_failure(
      self, mock_computer
  ):
    """Test toolset behavior when computer initialization fails."""

    async def failing_initialize():
      raise RuntimeError("Initialization failed")

    mock_computer.initialize = failing_initialize
    toolset = ComputerUseToolset(computer=mock_computer)

    with pytest.raises(RuntimeError, match="Initialization failed"):
      await toolset.get_tools()

    # Should not be marked as initialized
    assert not toolset._initialized
