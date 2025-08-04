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

import sys

# Check Python version first
if sys.version_info < (3, 10):
  raise ImportError(
      'A2A requires Python 3.10 or above. Please upgrade your Python version. '
      f'Current version: {sys.version_info.major}.{sys.version_info.minor}'
  )

# Import a2a packages with fallback to a2a_sdk
try:
  from a2a.server.events import Event
  from a2a.types import Message
  from a2a.types import TaskState
  from a2a.types import TaskStatusUpdateEvent
except ImportError:
  try:
    from a2a_sdk.server.events import Event
    from a2a_sdk.types import Message
    from a2a_sdk.types import TaskState
    from a2a_sdk.types import TaskStatusUpdateEvent
  except ImportError:
    raise ImportError(
        'Could not import a2a or a2a_sdk packages. Please install a2a-sdk as a'
        ' dependency.'
    )

from ...utils.feature_decorator import experimental


@experimental
class TaskResultAggregator:
  """Aggregates the task status updates and provides the final task state."""

  def __init__(self):
    self._task_state = TaskState.working
    self._task_status_message = None

  def process_event(self, event: Event):
    """Process an event from the agent run and detect signals about the task status.
    Priority of task state:
    - failed
    - auth_required
    - input_required
    - working
    """
    if isinstance(event, TaskStatusUpdateEvent):
      if event.status.state == TaskState.failed:
        self._task_state = TaskState.failed
        self._task_status_message = event.status.message
      elif (
          event.status.state == TaskState.auth_required
          and self._task_state != TaskState.failed
      ):
        self._task_state = TaskState.auth_required
        self._task_status_message = event.status.message
      elif (
          event.status.state == TaskState.input_required
          and self._task_state
          not in (TaskState.failed, TaskState.auth_required)
      ):
        self._task_state = TaskState.input_required
        self._task_status_message = event.status.message
      # final state is already recorded and make sure the intermediate state is
      # always working because other state may terminate the event aggregation
      # in a2a request handler
      elif self._task_state == TaskState.working:
        self._task_status_message = event.status.message
      event.status.state = TaskState.working

  @property
  def task_state(self) -> TaskState:
    return self._task_state

  @property
  def task_status_message(self) -> Message | None:
    return self._task_status_message
