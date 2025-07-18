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


class MockEnvironment:
  ENVIRONMENT_BROWSER = "ENVIRONMENT_BROWSER"
  ENVIRONMENT_DESKTOP = "ENVIRONMENT_DESKTOP"


class MockToolComputerUse:

  def __init__(self, environment=None):
    self.environment = environment


# Patch google.genai.types before imports
try:
  from google.genai import types

  # Add missing types if they don't exist
  if not hasattr(types, "Environment"):
    types.Environment = MockEnvironment
  if not hasattr(types, "ToolComputerUse"):
    types.ToolComputerUse = MockToolComputerUse
except ImportError:
  pass
