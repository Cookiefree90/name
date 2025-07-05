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

"""Utility functions for multipart message operations."""

import email.parser
import uuid
from typing import Dict, List, Tuple, Any, Optional


def extract_boundary_from_content_type(content_type: str) -> Optional[str]:
    """
    Extract the boundary string from a multipart content type header.

    Args:
        content_type: The Content-Type header value.

    Returns:
        The boundary string, or None if not found.
    """
    if not content_type or "boundary=" not in content_type:
        return None

    parts = content_type.split("boundary=")
    if len(parts) < 2:
        return None

    boundary = parts[1].strip()

    # Handle quoted boundaries
    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]

    return boundary


def create_multipart_message(
    parts: List[Dict[str, Any]], boundary: Optional[str] = None
) -> Tuple[str, str]:
    """
    Create a multipart message from a list of parts.

    Args:
        parts: List of parts, each with 'content_type' and 'content' keys.
        boundary: Optional boundary string to use. If not provided, a random one will be generated.

    Returns:
        A tuple of (content_type, message_body).
    """
    if boundary is None:
        boundary = f"boundary_{uuid.uuid4().hex}"

    content_type = f'multipart/mixed; boundary="{boundary}"'
    body_parts = []

    for part in parts:
        part_content_type = part.get("content_type", "text/plain")
        content = part.get("content", "")

        part_headers = [f"Content-Type: {part_content_type}"]

        # Add any additional headers
        for key, value in part.items():
            if key not in ["content_type", "content"]:
                part_headers.append(f"{key}: {value}")

        # Assemble the part
        body_parts.append("\r\n".join(part_headers) + "\r\n\r\n" + content)

    # Assemble the message
    message_body = (
        f"\r\n--{boundary}\r\n"
        + f"\r\n--{boundary}\r\n".join(body_parts)
        + f"\r\n--{boundary}--\r\n"
    )

    return content_type, message_body


def parse_multipart_message(content_type: str, body: str) -> List[Dict[str, Any]]:
    """
    Parse a multipart message into a list of parts.

    Args:
        content_type: The Content-Type header value.
        body: The message body.

    Returns:
        A list of parts, each with 'content_type' and 'content' keys.
    """
    # Create a message from the content type and body
    message = f"Content-Type: {content_type}\r\n\r\n{body}"

    # Parse the message
    parser = email.parser.Parser()
    msg = parser.parsestr(message)

    # Extract the parts
    result = []
    if not msg.is_multipart():
        return [{"content_type": content_type, "content": body}]

    for part in msg.get_payload():
        part_dict = {"content_type": part.get_content_type()}

        # Add headers
        for key in part.keys():
            part_dict[key.lower()] = part[key]

        # Add content
        part_dict["content"] = part.get_payload()
        result.append(part_dict)

    return result
