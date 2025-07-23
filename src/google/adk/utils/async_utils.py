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

"""Utility functions for asynchronous operations."""

import asyncio
from typing import List, Any, Awaitable, TypeVar, Callable, Dict, Optional

T = TypeVar("T")


async def gather_results(
    coroutines: List[Awaitable[T]], on_error: Optional[Callable[[Exception], T]] = None
) -> List[T]:
    """
    Gather results from multiple coroutines, handling errors.

    Args:
        coroutines: List of coroutines to execute
        on_error: Optional function to handle errors. If not provided, errors will be raised.

    Returns:
        List of results from the coroutines
    """
    if not coroutines:
        return []

    # If no error handler is provided, just use gather
    if on_error is None:
        return await asyncio.gather(*coroutines)

    # Otherwise, handle errors for each coroutine
    results = []
    for coro in coroutines:
        try:
            result = await coro
            results.append(result)
        except Exception as e:
            if on_error:
                results.append(on_error(e))
            else:
                raise

    return results
