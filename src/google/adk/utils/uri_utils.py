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

"""Utility functions for URI operations."""

import os
from urllib.parse import urlparse
from typing import Optional


def uri_to_file_path(uri: str) -> Optional[str]:
    """
    Convert a URI to a file path.

    Args:
        uri: The URI to convert.

    Returns:
        The file path, or None if the URI is not a file URI.
    """
    parsed = urlparse(uri)

    # Handle file:// URIs
    if parsed.scheme == "file":
        # On Windows, the path will start with a slash that needs to be removed
        if os.name == "nt" and parsed.path.startswith("/"):
            return parsed.path[1:]
        return parsed.path

    # Handle local file paths (no scheme)
    if not parsed.scheme and os.path.exists(uri):
        return uri

    return None
