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
import pydantic
import pytest

# Mock Environment enum since it may not be available in the current google.genai version
Environment = MagicMock()
Environment.ENVIRONMENT_BROWSER = "ENVIRONMENT_BROWSER"
Environment.ENVIRONMENT_DESKTOP = "ENVIRONMENT_DESKTOP"


class TestEnvironmentState:
  """Test cases for EnvironmentState model."""

  def test_valid_environment_state(self):
    """Test creating a valid EnvironmentState."""
    screenshot_data = b"fake_png_data"
    url = "https://example.com"

    state = EnvironmentState(screenshot=screenshot_data, url=url)

    assert state.screenshot == screenshot_data
    assert state.url == url

  def test_empty_url_raises_validation_error(self):
    """Test that empty URL raises validation error."""
    screenshot_data = b"fake_png_data"

    with pytest.raises(pydantic.ValidationError, match="URL cannot be empty"):
      EnvironmentState(screenshot=screenshot_data, url="")

  def test_whitespace_only_url_raises_validation_error(self):
    """Test that whitespace-only URL raises validation error."""
    screenshot_data = b"fake_png_data"

    with pytest.raises(pydantic.ValidationError, match="URL cannot be empty"):
      EnvironmentState(screenshot=screenshot_data, url="   ")

  def test_valid_url_with_spaces_is_accepted(self):
    """Test that URL with trailing/leading spaces is trimmed and accepted."""
    screenshot_data = b"fake_png_data"
    url = "  https://example.com  "

    state = EnvironmentState(screenshot=screenshot_data, url=url)
    assert state.url == url  # pydantic validation doesn't auto-strip

  def test_missing_required_fields_raise_validation_error(self):
    """Test that missing required fields raise validation errors."""
    with pytest.raises(pydantic.ValidationError):
      EnvironmentState()

    with pytest.raises(pydantic.ValidationError):
      EnvironmentState(screenshot=b"data")

    with pytest.raises(pydantic.ValidationError):
      EnvironmentState(url="https://example.com")

  def test_environment_state_serialization(self):
    """Test that EnvironmentState can be serialized and deserialized."""
    screenshot_data = b"fake_png_data"
    url = "https://example.com"

    original_state = EnvironmentState(screenshot=screenshot_data, url=url)

    # Test dict conversion
    state_dict = original_state.model_dump()
    assert state_dict["screenshot"] == screenshot_data
    assert state_dict["url"] == url

    # Test reconstruction from dict
    reconstructed_state = EnvironmentState(**state_dict)
    assert reconstructed_state.screenshot == original_state.screenshot
    assert reconstructed_state.url == original_state.url


class MockComputer(Computer):
  """Mock implementation of Computer for testing."""

  def __init__(self):
    self.screen_width = 1920
    self.screen_height = 1080
    self.current_url = "https://example.com"
    self.screenshot_data = b"mock_screenshot_data"

  async def screen_size(self) -> tuple[int, int]:
    return (self.screen_width, self.screen_height)

  async def open_web_browser(self) -> EnvironmentState:
    return EnvironmentState(
        screenshot=self.screenshot_data, url=self.current_url
    )

  async def click_at(self, x: int, y: int) -> EnvironmentState:
    return EnvironmentState(
        screenshot=self.screenshot_data, url=self.current_url
    )

  async def hover_at(self, x: int, y: int) -> EnvironmentState:
    return EnvironmentState(
        screenshot=self.screenshot_data, url=self.current_url
    )

  async def type_text_at(
      self,
      x: int,
      y: int,
      text: str,
      press_enter: bool = True,
      clear_before_typing: bool = True,
  ) -> EnvironmentState:
    return EnvironmentState(
        screenshot=self.screenshot_data, url=self.current_url
    )

  async def scroll_document(self, direction: str) -> EnvironmentState:
    return EnvironmentState(
        screenshot=self.screenshot_data, url=self.current_url
    )

  async def scroll_at(
      self, x: int, y: int, direction: str, magnitude: int
  ) -> EnvironmentState:
    return EnvironmentState(
        screenshot=self.screenshot_data, url=self.current_url
    )

  async def wait_5_seconds(self) -> EnvironmentState:
    return EnvironmentState(
        screenshot=self.screenshot_data, url=self.current_url
    )

  async def go_back(self) -> EnvironmentState:
    return EnvironmentState(
        screenshot=self.screenshot_data, url=self.current_url
    )

  async def go_forward(self) -> EnvironmentState:
    return EnvironmentState(
        screenshot=self.screenshot_data, url=self.current_url
    )

  async def search(self) -> EnvironmentState:
    return EnvironmentState(
        screenshot=self.screenshot_data, url=self.current_url
    )

  async def navigate(self, url: str) -> EnvironmentState:
    self.current_url = url
    return EnvironmentState(screenshot=self.screenshot_data, url=url)

  async def key_combination(self, keys: list[str]) -> EnvironmentState:
    return EnvironmentState(
        screenshot=self.screenshot_data, url=self.current_url
    )

  async def drag_and_drop(
      self, x: int, y: int, destination_x: int, destination_y: int
  ) -> EnvironmentState:
    return EnvironmentState(
        screenshot=self.screenshot_data, url=self.current_url
    )

  async def current_state(self) -> EnvironmentState:
    return EnvironmentState(
        screenshot=self.screenshot_data, url=self.current_url
    )


class TestComputer:
  """Test cases for Computer abstract class."""

  @pytest.fixture
  def mock_computer(self):
    """Fixture providing a mock computer instance."""
    return MockComputer()

  @pytest.mark.asyncio
  async def test_initialize_default_implementation(self, mock_computer):
    """Test that default initialize method works."""
    # Should not raise any exception
    await mock_computer.initialize()

  @pytest.mark.asyncio
  async def test_close_default_implementation(self, mock_computer):
    """Test that default close method works."""
    # Should not raise any exception
    await mock_computer.close()

  @pytest.mark.asyncio
  async def test_environment_default_implementation(self, mock_computer):
    """Test that default environment method returns ENVIRONMENT_BROWSER."""
    environment = await mock_computer.environment()
    assert environment == Environment.ENVIRONMENT_BROWSER

  @pytest.mark.asyncio
  async def test_screen_size(self, mock_computer):
    """Test screen_size method."""
    size = await mock_computer.screen_size()
    assert size == (1920, 1080)
    assert isinstance(size, tuple)
    assert len(size) == 2
    assert isinstance(size[0], int)
    assert isinstance(size[1], int)

  @pytest.mark.asyncio
  async def test_open_web_browser(self, mock_computer):
    """Test open_web_browser method."""
    state = await mock_computer.open_web_browser()
    assert isinstance(state, EnvironmentState)
    assert state.screenshot == b"mock_screenshot_data"
    assert state.url == "https://example.com"

  @pytest.mark.asyncio
  async def test_click_at(self, mock_computer):
    """Test click_at method."""
    state = await mock_computer.click_at(100, 200)
    assert isinstance(state, EnvironmentState)

  @pytest.mark.asyncio
  async def test_hover_at(self, mock_computer):
    """Test hover_at method."""
    state = await mock_computer.hover_at(150, 250)
    assert isinstance(state, EnvironmentState)

  @pytest.mark.asyncio
  async def test_type_text_at(self, mock_computer):
    """Test type_text_at method."""
    state = await mock_computer.type_text_at(100, 200, "test text")
    assert isinstance(state, EnvironmentState)

  @pytest.mark.asyncio
  async def test_type_text_at_with_options(self, mock_computer):
    """Test type_text_at method with optional parameters."""
    state = await mock_computer.type_text_at(
        100, 200, "test text", press_enter=False, clear_before_typing=False
    )
    assert isinstance(state, EnvironmentState)

  @pytest.mark.asyncio
  async def test_scroll_document(self, mock_computer):
    """Test scroll_document method."""
    for direction in ["up", "down", "left", "right"]:
      state = await mock_computer.scroll_document(direction)
      assert isinstance(state, EnvironmentState)

  @pytest.mark.asyncio
  async def test_scroll_at(self, mock_computer):
    """Test scroll_at method."""
    state = await mock_computer.scroll_at(100, 200, "down", 5)
    assert isinstance(state, EnvironmentState)

  @pytest.mark.asyncio
  async def test_wait_5_seconds(self, mock_computer):
    """Test wait_5_seconds method."""
    state = await mock_computer.wait_5_seconds()
    assert isinstance(state, EnvironmentState)

  @pytest.mark.asyncio
  async def test_go_back(self, mock_computer):
    """Test go_back method."""
    state = await mock_computer.go_back()
    assert isinstance(state, EnvironmentState)

  @pytest.mark.asyncio
  async def test_go_forward(self, mock_computer):
    """Test go_forward method."""
    state = await mock_computer.go_forward()
    assert isinstance(state, EnvironmentState)

  @pytest.mark.asyncio
  async def test_search(self, mock_computer):
    """Test search method."""
    state = await mock_computer.search()
    assert isinstance(state, EnvironmentState)

  @pytest.mark.asyncio
  async def test_navigate(self, mock_computer):
    """Test navigate method."""
    new_url = "https://google.com"
    state = await mock_computer.navigate(new_url)
    assert isinstance(state, EnvironmentState)
    assert state.url == new_url

  @pytest.mark.asyncio
  async def test_key_combination(self, mock_computer):
    """Test key_combination method."""
    state = await mock_computer.key_combination(["ctrl", "c"])
    assert isinstance(state, EnvironmentState)

  @pytest.mark.asyncio
  async def test_drag_and_drop(self, mock_computer):
    """Test drag_and_drop method."""
    state = await mock_computer.drag_and_drop(100, 200, 300, 400)
    assert isinstance(state, EnvironmentState)

  @pytest.mark.asyncio
  async def test_current_state(self, mock_computer):
    """Test current_state method."""
    state = await mock_computer.current_state()
    assert isinstance(state, EnvironmentState)

  def test_computer_is_abstract(self):
    """Test that Computer cannot be instantiated directly."""
    with pytest.raises(TypeError):
      Computer()
