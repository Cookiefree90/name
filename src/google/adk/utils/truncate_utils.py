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

"""Utility functions for truncating data."""

import json
from typing import Any, Dict, List, Union


def truncate_data(
    data: Any,
    max_chars: int = 1000,
    max_items: int = 10,
    truncation_marker: str = "...",
) -> Any:
    """
    Truncate data to a maximum size.

    Args:
        data: The data to truncate.
        max_chars: Maximum number of characters for strings.
        max_items: Maximum number of items for lists and dictionaries.
        truncation_marker: Marker to indicate truncation.

    Returns:
        Truncated data.
    """
    if data is None:
        return None

    if isinstance(data, str):
        if len(data) <= max_chars:
            return data
        return data[:max_chars] + truncation_marker

    if isinstance(data, (list, tuple)):
        if len(data) <= max_items:
            return [
                truncate_data(item, max_chars, max_items, truncation_marker)
                for item in data
            ]
        return [
            truncate_data(item, max_chars, max_items, truncation_marker)
            for item in data[:max_items]
        ] + [truncation_marker]

    if isinstance(data, dict):
        if len(data) <= max_items:
            return {
                k: truncate_data(v, max_chars, max_items, truncation_marker)
                for k, v in data.items()
            }
        truncated = {
            k: truncate_data(v, max_chars, max_items, truncation_marker)
            for k, v in list(data.items())[:max_items]
        }
        truncated["__truncated__"] = (
            f"{len(data) - max_items} more items {truncation_marker}"
        )
        return truncated

    if isinstance(data, (int, float, bool)):
        return data

    # Try to convert to string and truncate
    try:
        return truncate_data(str(data), max_chars, max_items, truncation_marker)
    except Exception:
        return f"<unprintable object: {type(data).__name__}>"
