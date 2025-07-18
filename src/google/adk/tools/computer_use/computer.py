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

import abc
from typing import Literal

from google.genai.types import Environment
import pydantic

from ...utils.feature_decorator import experimental


@experimental
class EnvironmentState(pydantic.BaseModel):
  """Represents the current state of the computer environment.

  Attributes:
    screenshot: The screenshot in PNG format as bytes.
    url: The current URL of the webpage being displayed.
  """

  screenshot: bytes = pydantic.Field(
      ..., description="Screenshot in PNG format"
  )
  url: str = pydantic.Field(..., description="Current webpage URL")

  @pydantic.field_validator("url")
  @classmethod
  def validate_url(cls, v: str) -> str:
    """Validate that URL is not empty."""
    if not v.strip():
      raise ValueError("URL cannot be empty")
    return v


@experimental
class Computer(abc.ABC):
  """async defines an interface for computer environments.

  This abstract base class async defines the standard interface for controlling
  computer environments, including web browsers and other interactive systems.
  """

  @abc.abstractmethod
  async def screen_size(self) -> tuple[int, int]:
    """Returns the screen size of the environment.

    Returns:
      A tuple of (width, height) in pixels.
    """

  @abc.abstractmethod
  async def open_web_browser(self) -> EnvironmentState:
    """Opens the web browser.

    Returns:
      The current state after opening the browser.
    """

  @abc.abstractmethod
  async def click_at(self, x: int, y: int) -> EnvironmentState:
    """Clicks at a specific x, y coordinate on the webpage.

    The 'x' and 'y' values are absolute values, scaled to the height and width of the screen.

    Args:
      x: The x-coordinate to click at.
      y: The y-coordinate to click at.

    Returns:
      The current state after clicking.
    """

  @abc.abstractmethod
  async def hover_at(self, x: int, y: int) -> EnvironmentState:
    """Hovers at a specific x, y coordinate on the webpage.

    May be used to explore sub-menus that appear on hover.
    The 'x' and 'y' values are absolute values, scaled to the height and width of the screen.

    Args:
      x: The x-coordinate to hover at.
      y: The y-coordinate to hover at.

    Returns:
      The current state after hovering.
    """

  @abc.abstractmethod
  async def type_text_at(
      self,
      x: int,
      y: int,
      text: str,
      press_enter: bool = True,
      clear_before_typing: bool = True,
  ) -> EnvironmentState:
    """Types text at a specific x, y coordinate.

    The system automatically presses ENTER after typing. To disable this, set `press_enter` to False.
    The system automatically clears any existing content before typing the specified `text`. To disable this, set `clear_before_typing` to False.
    The 'x' and 'y' values are absolute values, scaled to the height and width of the screen.

    Args:
      x: The x-coordinate to type at.
      y: The y-coordinate to type at.
      text: The text to type.
      press_enter: Whether to press ENTER after typing.
      clear_before_typing: Whether to clear existing content before typing.

    Returns:
      The current state after typing.
    """

  @abc.abstractmethod
  async def scroll_document(
      self, direction: Literal["up", "down", "left", "right"]
  ) -> EnvironmentState:
    """Scrolls the entire webpage "up", "down", "left" or "right" based on direction.

    Args:
      direction: The direction to scroll.

    Returns:
      The current state after scrolling.
    """

  @abc.abstractmethod
  async def scroll_at(
      self,
      x: int,
      y: int,
      direction: Literal["up", "down", "left", "right"],
      magnitude: int,
  ) -> EnvironmentState:
    """Scrolls up, down, right, or left at a x, y coordinate by magnitude.

    The 'x' and 'y' values are absolute values, scaled to the height and width of the screen.

    Args:
      x: The x-coordinate to scroll at.
      y: The y-coordinate to scroll at.
      direction: The direction to scroll.
      magnitude: The amount to scroll.

    Returns:
      The current state after scrolling.
    """

  @abc.abstractmethod
  async def wait_5_seconds(self) -> EnvironmentState:
    """Waits for 5 seconds to allow unfinished webpage processes to complete.

    Returns:
      The current state after waiting.
    """

  @abc.abstractmethod
  async def go_back(self) -> EnvironmentState:
    """Navigates back to the previous webpage in the browser history.

    Returns:
      The current state after navigating back.
    """

  @abc.abstractmethod
  async def go_forward(self) -> EnvironmentState:
    """Navigates forward to the next webpage in the browser history.

    Returns:
      The current state after navigating forward.
    """

  @abc.abstractmethod
  async def search(self) -> EnvironmentState:
    """Directly jumps to a search engine home page.

    Used when you need to start with a search. For example, this is used when
    the current website doesn't have the information needed or because a new
    task is being started.

    Returns:
      The current state after navigating to search.
    """

  @abc.abstractmethod
  async def navigate(self, url: str) -> EnvironmentState:
    """Navigates directly to a specified URL.

    Args:
      url: The URL to navigate to.

    Returns:
      The current state after navigation.
    """

  @abc.abstractmethod
  async def key_combination(self, keys: list[str]) -> EnvironmentState:
    """Presses keyboard keys and combinations, such as "control+c" or "enter".

    Args:
      keys: List of keys to press in combination.

    Returns:
      The current state after key press.
    """

  @abc.abstractmethod
  async def drag_and_drop(
      self, x: int, y: int, destination_x: int, destination_y: int
  ) -> EnvironmentState:
    """Drag and drop an element from a x, y coordinate to a destination destination_y, destination_x coordinate.

    The 'x', 'y', 'destination_y' and 'destination_x' values are absolute values, scaled to the height and width of the screen.

    Args:
      x: The x-coordinate to start dragging from.
      y: The y-coordinate to start dragging from.
      destination_x: The x-coordinate to drop at.
      destination_y: The y-coordinate to drop at.

    Returns:
      The current state after drag and drop.
    """

  @abc.abstractmethod
  async def current_state(self) -> EnvironmentState:
    """Returns the current state of the current webpage.

    Returns:
      The current environment state.
    """

  async def initialize(self) -> None:
    """Initialize the computer."""
    pass

  async def close(self) -> None:
    """Cleanup resource of the computer."""
    pass

  async def environment(self) -> Environment:
    """Returns the environment of the computer.

    Returns:
      The environment type, async defaults to ENVIRONMENT_BROWSER.
    """
    return Environment.ENVIRONMENT_BROWSER
