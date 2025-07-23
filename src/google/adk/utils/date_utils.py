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

"""Utility functions for date and time operations."""

import datetime
from typing import Optional


def to_rfc3339_datetime(dt: Optional[datetime.datetime] = None) -> str:
    """
    Convert a datetime to RFC 3339 format.

    Args:
        dt: The datetime to convert, or None to use the current time.

    Returns:
        The datetime in RFC 3339 format.
    """
    if dt is None:
        dt = datetime.datetime.now(datetime.timezone.utc)

    # Ensure timezone info is present
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)

    return dt.isoformat()


def from_rfc3339_datetime(datetime_str: str) -> datetime.datetime:
    """
    Parse a RFC 3339 formatted datetime string to a datetime object.

    Args:
        datetime_str: The RFC 3339 formatted datetime string.

    Returns:
        The parsed datetime object.
    """
    return datetime.datetime.fromisoformat(datetime_str)
