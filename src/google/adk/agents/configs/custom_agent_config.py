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

from typing import List
from typing import Literal
from typing import Optional

from .base_agent_config import BaseAgentConfig
from .common_config import ArgumentConfig


# This is essentially a CodeConfig, maybe using CodeConfig is sufficient.
class CustomAgentConfig(BaseAgentConfig):

  agent_type: Literal['CustomAgent'] = 'CustomAgent'

  path: str

  args: Optional[List[ArgumentConfig]] = None
