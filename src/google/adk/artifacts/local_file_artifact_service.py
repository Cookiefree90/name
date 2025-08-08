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

"""An artifact service implementation using the local file system.

The file path format used depends on whether the filename has a user namespace:
  - For files with user namespace (starting with "user:"):
    {base_path}/{app_name}/{user_id}/user/{filename}/{version}
  - For regular session-scoped files:
    {base_path}/{app_name}/{user_id}/{session_id}/{filename}/{version}
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import shutil
from typing import Optional

from google.genai import types
from typing_extensions import override

from .base_artifact_service import BaseArtifactService

logger = logging.getLogger("google_adk." + __name__)


class LocalFileArtifactService(BaseArtifactService):
  """An artifact service implementation using the local file system."""

  def __init__(self, base_path: str = "./adk_artifacts"):
    """Initializes the LocalFileArtifactService.

    Args:
        base_path: The base directory path where artifacts will be stored.
                   Defaults to "./adk_artifacts".
    """
    self.base_path = Path(base_path).resolve()
    self.base_path.mkdir(parents=True, exist_ok=True)

  @override
  async def save_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      artifact: types.Part,
  ) -> int:
    return await asyncio.to_thread(
        self._save_artifact,
        app_name,
        user_id,
        session_id,
        filename,
        artifact,
    )

  @override
  async def load_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      version: Optional[int] = None,
  ) -> Optional[types.Part]:
    return await asyncio.to_thread(
        self._load_artifact,
        app_name,
        user_id,
        session_id,
        filename,
        version,
    )

  @override
  async def list_artifact_keys(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> list[str]:
    return await asyncio.to_thread(
        self._list_artifact_keys,
        app_name,
        user_id,
        session_id,
    )

  @override
  async def delete_artifact(
      self, *, app_name: str, user_id: str, session_id: str, filename: str
  ) -> None:
    return await asyncio.to_thread(
        self._delete_artifact,
        app_name,
        user_id,
        session_id,
        filename,
    )

  @override
  async def list_versions(
      self, *, app_name: str, user_id: str, session_id: str, filename: str
  ) -> list[int]:
    return await asyncio.to_thread(
        self._list_versions,
        app_name,
        user_id,
        session_id,
        filename,
    )

  def _file_has_user_namespace(self, filename: str) -> bool:
    """Checks if the filename has a user namespace.

    Args:
        filename: The filename to check.

    Returns:
        True if the filename has a user namespace (starts with "user:"),
        False otherwise.
    """
    return filename.startswith("user:")

  def _get_artifact_dir(
      self,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
  ) -> Path:
    """Constructs the directory path for an artifact.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        session_id: The ID of the session.
        filename: The name of the artifact file.

    Returns:
        The constructed directory path.
    """
    if self._file_has_user_namespace(filename):
      return self.base_path / app_name / user_id / "user" / filename
    return self.base_path / app_name / user_id / session_id / filename

  def _get_artifact_file_path(
      self,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      version: int,
  ) -> Path:
    """Constructs the full file path for an artifact version.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        session_id: The ID of the session.
        filename: The name of the artifact file.
        version: The version of the artifact.

    Returns:
        The constructed file path.
    """
    artifact_dir = self._get_artifact_dir(
        app_name, user_id, session_id, filename
    )
    return artifact_dir / str(version)

  def _get_metadata_file_path(
      self,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      version: int,
  ) -> Path:
    """Constructs the metadata file path for an artifact version.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        session_id: The ID of the session.
        filename: The name of the artifact file.
        version: The version of the artifact.

    Returns:
        The constructed metadata file path.
    """
    artifact_dir = self._get_artifact_dir(
        app_name, user_id, session_id, filename
    )
    return artifact_dir / f"{version}.metadata.json"

  def _save_artifact(
      self,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      artifact: types.Part,
  ) -> int:
    versions = self._list_versions(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    version = 0 if not versions else max(versions) + 1

    artifact_dir = self._get_artifact_dir(
        app_name, user_id, session_id, filename
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact_file_path = self._get_artifact_file_path(
        app_name, user_id, session_id, filename, version
    )
    metadata_file_path = self._get_metadata_file_path(
        app_name, user_id, session_id, filename, version
    )

    # Save the artifact data
    artifact_file_path.write_bytes(artifact.inline_data.data)

    # Save metadata (mime_type)
    metadata = {"mime_type": artifact.inline_data.mime_type}
    metadata_file_path.write_text(json.dumps(metadata))

    return version

  def _load_artifact(
      self,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      version: Optional[int] = None,
  ) -> Optional[types.Part]:
    if version is None:
      versions = self._list_versions(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
      )
      if not versions:
        return None
      version = max(versions)

    artifact_file_path = self._get_artifact_file_path(
        app_name, user_id, session_id, filename, version
    )
    metadata_file_path = self._get_metadata_file_path(
        app_name, user_id, session_id, filename, version
    )

    if not artifact_file_path.exists() or not metadata_file_path.exists():
      return None

    try:
      artifact_data = artifact_file_path.read_bytes()
      metadata_text = metadata_file_path.read_text()
      metadata = json.loads(metadata_text)

      artifact = types.Part.from_bytes(
          data=artifact_data, mime_type=metadata["mime_type"]
      )
      return artifact
    except (OSError, json.JSONDecodeError, KeyError):
      logger.warning(
          "Failed to load artifact %s for app %s, user %s, session %s,"
          " version %d",
          filename,
          app_name,
          user_id,
          session_id,
          version,
      )
      return None

  def _list_artifact_keys(
      self, app_name: str, user_id: str, session_id: str
  ) -> list[str]:
    filenames = set()

    # List session-scoped artifacts
    session_dir = self.base_path / app_name / user_id / session_id
    if session_dir.exists():
      for item in session_dir.iterdir():
        if item.is_dir():
          filenames.add(item.name)

    # List user-namespaced artifacts
    user_namespace_dir = self.base_path / app_name / user_id / "user"
    if user_namespace_dir.exists():
      for item in user_namespace_dir.iterdir():
        if item.is_dir():
          filenames.add(item.name)

    return sorted(list(filenames))

  def _delete_artifact(
      self, app_name: str, user_id: str, session_id: str, filename: str
  ) -> None:
    artifact_dir = self._get_artifact_dir(
        app_name, user_id, session_id, filename
    )
    if artifact_dir.exists():
      shutil.rmtree(artifact_dir)

  def _list_versions(
      self, app_name: str, user_id: str, session_id: str, filename: str
  ) -> list[int]:
    """Lists all available versions of an artifact.

    This method retrieves all versions of a specific artifact by listing
    the version directories within the artifact's directory.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user who owns the artifact.
        session_id: The ID of the session (ignored for user-namespaced files).
        filename: The name of the artifact file.

    Returns:
        A list of version numbers (integers) available for the specified
        artifact. Returns an empty list if no versions are found.
    """
    artifact_dir = self._get_artifact_dir(
        app_name, user_id, session_id, filename
    )
    if not artifact_dir.exists():
      return []

    versions = []
    for item in artifact_dir.iterdir():
      if item.is_file() and item.name.isdigit():
        versions.append(int(item.name))

    return sorted(versions)
