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

from unittest import mock

from google.adk.tools.bigtable import BigtableCredentialsConfig
from google.adk.tools.bigtable.bigtable_credentials import BIGTABLE_DEFAULT_SCOPE
from google.auth.credentials import Credentials
import google.oauth2.credentials
import pytest


def test_bigtable_credentials_config_client_id_secret():
  """Test BigtableCredentialsConfig with client_id and client_secret."""
  config = BigtableCredentialsConfig(client_id="abc", client_secret="def")
  assert config.client_id == "abc"
  assert config.client_secret == "def"
  assert config.scopes == BIGTABLE_DEFAULT_SCOPE
  assert config.credentials is None


def test_bigtable_credentials_config_existing_creds():
  """Test BigtableCredentialsConfig with existing credentials."""
  mock_creds = mock.create_autospec(Credentials, instance=True)
  config = BigtableCredentialsConfig(credentials=mock_creds)
  assert config.credentials == mock_creds
  assert config.client_id is None


def test_bigtable_credentials_config_oauth2_creds():
  """Test BigtableCredentialsConfig with existing OAuth2 credentials."""
  mock_creds = mock.create_autospec(
      google.oauth2.credentials.Credentials, instance=True
  )
  mock_creds.client_id = "oauth_client_id"
  mock_creds.client_secret = "oauth_client_secret"
  mock_creds.scopes = ["fake_scope"]
  config = BigtableCredentialsConfig(credentials=mock_creds)
  assert config.client_id == "oauth_client_id"
  assert config.client_secret == "oauth_client_secret"
  assert config.scopes == ["fake_scope"]


def test_bigtable_credentials_config_validation_errors():
  """Test BigtableCredentialsConfig validation errors."""
  with pytest.raises(ValueError):
    BigtableCredentialsConfig()

  with pytest.raises(ValueError):
    BigtableCredentialsConfig(client_id="abc")

  mock_creds = mock.create_autospec(Credentials, instance=True)
  with pytest.raises(ValueError):
    BigtableCredentialsConfig(
        credentials=mock_creds, client_id="abc", client_secret="def"
    )
