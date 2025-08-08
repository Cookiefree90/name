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

"""Tests for the LocalFileArtifactService."""

import json
from pathlib import Path
import tempfile

from google.adk.artifacts.local_file_artifact_service import LocalFileArtifactService
from google.genai import types
import pytest


@pytest.fixture
def temp_dir():
  """Creates a temporary directory for testing."""
  with tempfile.TemporaryDirectory() as temp_dir:
    yield temp_dir


@pytest.fixture
def artifact_service(temp_dir):
  """Creates a LocalFileArtifactService for testing."""
  return LocalFileArtifactService(base_path=temp_dir)


@pytest.mark.asyncio
async def test_load_empty(artifact_service):
  """Tests loading an artifact when none exists."""
  assert not await artifact_service.load_artifact(
      app_name="test_app",
      user_id="test_user",
      session_id="session_id",
      filename="filename",
  )


@pytest.mark.asyncio
async def test_save_load_delete(artifact_service):
  """Tests saving, loading, and deleting an artifact."""
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "file456"

  # Save artifact
  version = await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=artifact,
  )
  assert version == 0

  # Load artifact
  loaded_artifact = await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )
  assert loaded_artifact == artifact

  # Delete artifact
  await artifact_service.delete_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )
  assert not await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )


@pytest.mark.asyncio
async def test_list_keys(artifact_service):
  """Tests listing keys in the artifact service."""
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "filename"
  filenames = [filename + str(i) for i in range(5)]

  for f in filenames:
    await artifact_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=f,
        artifact=artifact,
    )

  assert (
      await artifact_service.list_artifact_keys(
          app_name=app_name, user_id=user_id, session_id=session_id
      )
      == filenames
  )


@pytest.mark.asyncio
async def test_list_versions(artifact_service):
  """Tests listing versions of an artifact."""
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "with/slash/filename"
  versions = [
      types.Part.from_bytes(
          data=i.to_bytes(2, byteorder="big"), mime_type="text/plain"
      )
      for i in range(3)
  ]

  # Save multiple versions
  for i in range(3):
    version = await artifact_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        artifact=versions[i],
    )
    assert version == i

  # List versions
  response_versions = await artifact_service.list_versions(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )

  assert response_versions == list(range(3))


@pytest.mark.asyncio
async def test_load_specific_version(artifact_service):
  """Tests loading a specific version of an artifact."""
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "versioned_file"

  # Save multiple versions
  version1 = types.Part.from_bytes(data=b"version_1", mime_type="text/plain")
  version2 = types.Part.from_bytes(data=b"version_2", mime_type="text/plain")
  version3 = types.Part.from_bytes(data=b"version_3", mime_type="text/plain")

  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=version1,
  )
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=version2,
  )
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=version3,
  )

  # Load specific versions
  loaded_v0 = await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      version=0,
  )
  loaded_v1 = await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      version=1,
  )
  loaded_v2 = await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      version=2,
  )

  # Load latest version (without specifying version)
  loaded_latest = await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )

  assert loaded_v0 == version1
  assert loaded_v1 == version2
  assert loaded_v2 == version3
  assert loaded_latest == version3


@pytest.mark.asyncio
async def test_user_namespaced_files(artifact_service):
  """Tests handling of user-namespaced files (starting with 'user:')."""
  artifact = types.Part.from_bytes(data=b"user_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  user_filename = "user:shared_file"
  regular_filename = "regular_file"

  # Save user-namespaced artifact
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=user_filename,
      artifact=artifact,
  )

  # Save regular session-scoped artifact
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=regular_filename,
      artifact=artifact,
  )

  # Load both artifacts
  loaded_user_artifact = await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=user_filename,
  )
  loaded_regular_artifact = await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=regular_filename,
  )

  assert loaded_user_artifact == artifact
  assert loaded_regular_artifact == artifact

  # List artifacts should include both
  artifact_keys = await artifact_service.list_artifact_keys(
      app_name=app_name, user_id=user_id, session_id=session_id
  )
  assert regular_filename in artifact_keys
  assert user_filename in artifact_keys


@pytest.mark.asyncio
async def test_different_mime_types(artifact_service):
  """Tests handling different MIME types."""
  app_name = "app0"
  user_id = "user0"
  session_id = "123"

  # Test different MIME types
  text_artifact = types.Part.from_bytes(
      data=b"text content", mime_type="text/plain"
  )
  json_artifact = types.Part.from_bytes(
      data=b'{"key": "value"}', mime_type="application/json"
  )
  image_artifact = types.Part.from_bytes(
      data=b"fake_image_data", mime_type="image/png"
  )

  # Save artifacts with different MIME types
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="text_file",
      artifact=text_artifact,
  )
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="json_file",
      artifact=json_artifact,
  )
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="image_file",
      artifact=image_artifact,
  )

  # Load and verify MIME types are preserved
  loaded_text = await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="text_file",
  )
  loaded_json = await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="json_file",
  )
  loaded_image = await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="image_file",
  )

  assert loaded_text == text_artifact
  assert loaded_json == json_artifact
  assert loaded_image == image_artifact


@pytest.mark.asyncio
async def test_nonexistent_version(artifact_service):
  """Tests loading a non-existent version."""
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "test_file"

  # Save one version
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=artifact,
  )

  # Try to load non-existent version
  loaded_artifact = await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      version=999,
  )

  assert loaded_artifact is None


@pytest.mark.asyncio
async def test_empty_list_versions(artifact_service):
  """Tests listing versions when no artifact exists."""
  response_versions = await artifact_service.list_versions(
      app_name="app0",
      user_id="user0",
      session_id="123",
      filename="nonexistent_file",
  )

  assert response_versions == []


@pytest.mark.asyncio
async def test_file_structure(artifact_service, temp_dir):
  """Tests that files are stored in the correct directory structure."""
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "test_app"
  user_id = "test_user"
  session_id = "test_session"
  filename = "test_file"
  user_filename = "user:shared_file"

  # Save regular file
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=artifact,
  )

  # Save user-namespaced file
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=user_filename,
      artifact=artifact,
  )

  base_path = Path(temp_dir)

  # Check regular file structure
  regular_file_path = (
      base_path / app_name / user_id / session_id / filename / "0"
  )
  regular_metadata_path = (
      base_path / app_name / user_id / session_id / filename / "0.metadata.json"
  )
  assert regular_file_path.exists()
  assert regular_metadata_path.exists()

  # Check user-namespaced file structure
  user_file_path = base_path / app_name / user_id / "user" / user_filename / "0"
  user_metadata_path = (
      base_path
      / app_name
      / user_id
      / "user"
      / user_filename
      / "0.metadata.json"
  )
  assert user_file_path.exists()
  assert user_metadata_path.exists()

  # Check metadata content
  with open(regular_metadata_path) as f:
    metadata = json.load(f)
    assert metadata["mime_type"] == "text/plain"


@pytest.mark.asyncio
async def test_corrupted_metadata_handling(artifact_service, temp_dir):
  """Tests handling of corrupted metadata files."""
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "test_app"
  user_id = "test_user"
  session_id = "test_session"
  filename = "test_file"

  # Save artifact normally
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=artifact,
  )

  # Corrupt the metadata file
  base_path = Path(temp_dir)
  metadata_path = (
      base_path / app_name / user_id / session_id / filename / "0.metadata.json"
  )
  metadata_path.write_text("invalid json{")

  # Try to load the artifact - should return None due to corrupted metadata
  loaded_artifact = await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )

  assert loaded_artifact is None


@pytest.mark.asyncio
async def test_missing_metadata_file(artifact_service, temp_dir):
  """Tests handling when metadata file is missing."""
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "test_app"
  user_id = "test_user"
  session_id = "test_session"
  filename = "test_file"

  # Save artifact normally
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=artifact,
  )

  # Delete the metadata file
  base_path = Path(temp_dir)
  metadata_path = (
      base_path / app_name / user_id / session_id / filename / "0.metadata.json"
  )
  metadata_path.unlink()

  # Try to load the artifact - should return None due to missing metadata
  loaded_artifact = await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )

  assert loaded_artifact is None
