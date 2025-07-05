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

"""Utility functions for image operations."""

import base64
import os
from urllib.parse import urlparse
from typing import Optional


def get_base64_image_from_uri(uri: str) -> Optional[str]:
    """
    Get base64-encoded image from a URI.

    Args:
        uri: The URI of the image file.

    Returns:
        Base64-encoded image data, or None if the file could not be read.
    """
    try:
        # Handle file:// URIs
        if uri.startswith("file://"):
            path = urlparse(uri).path
        else:
            path = uri

        # Ensure the path exists
        if not os.path.exists(path):
            return None

        # Read and encode the file
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        # Return None if any error occurs
        return None
