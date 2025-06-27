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

import json
import asyncio
import os
import datetime
from typing import Dict, Any, Optional

try:
    from google.cloud import pubsub_v1
    from google.auth.exceptions import DefaultCredentialsError
except ImportError:
    pubsub_v1 = None
    DefaultCredentialsError = None
    print("[PubSubPublisher WARNING] google-cloud-pubsub not installed. Pub/Sub features will be unavailable.")


_loop: Optional[asyncio.AbstractEventLoop] = None
_pubsub_publisher: Optional[Any] = None
_pubsub_topic_path: Optional[str] = None
_pubsub_enabled: bool = False

def _pubsub_callback(future: Any) -> None:
    """Callback for Pub/Sub publish results."""
    try:
        message_id = future.result()
        print(f"[PubSubPublisher INFO] Published Pub/Sub message with ID: {message_id}")
    except Exception as e:
        print(f"[PubSubPublisher ERROR] Failed to publish Pub/Sub message: {e}")

def setup_pubsub_publisher_async(
    gcp_project_id: Optional[str] = None,
    pubsub_topic_id: Optional[str] = None
) -> None:
    """
    Initializes the GCP Pub/Sub publisher.
    Uses provided arguments or falls back to environment variables.
    """
    global _loop, _pubsub_publisher, _pubsub_topic_path, _pubsub_enabled

    _loop = asyncio.get_running_loop()

    if pubsub_v1 is None:
        print("[PubSubPublisher INFO] GCP Pub/Sub client library not found. Publishing disabled.")
        _pubsub_enabled = False
        return

    # Use provided args or get from environment
    project_id = gcp_project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    topic_id = pubsub_topic_id or os.environ.get("PUBSUB_TOPIC_ID")

    if project_id and topic_id:
        try:
            _pubsub_publisher = pubsub_v1.PublisherClient()
            _pubsub_topic_path = _pubsub_publisher.topic_path(project_id, topic_id)
            _pubsub_enabled = True
            print(f"[PubSubPublisher] GCP Pub/Sub publishing enabled for topic: {_pubsub_topic_path}")
        except DefaultCredentialsError:
            print("[PubSubPublisher ERROR] GCP Default Credentials not found. Publishing disabled.")
            _pubsub_enabled = False
        except Exception as e:
            print(f"[PubSubPublisher ERROR] Failed to initialize GCP Pub/Sub publisher: {e}. Publishing disabled.")
            _pubsub_enabled = False
    else:
        print("[PubSubPublisher INFO] GOOGLE_CLOUD_PROJECT and/or PUBSUB_TOPIC_ID not set. Publishing disabled.")
        _pubsub_enabled = False

def publish_event(
    event_data: Dict[str, Any],
    event_type: str = "custom_event",
    gcp_project_id: Optional[str] = None,
    pubsub_topic_id: Optional[str] = None
) -> None:
    """
    Publishes a structured event to the configured GCP Pub/Sub topic.
    """
    if not _pubsub_enabled:
        setup_pubsub_publisher_async(gcp_project_id=gcp_project_id, pubsub_topic_id=pubsub_topic_id)

    if not _pubsub_enabled or _pubsub_publisher is None or _pubsub_topic_path is None or _loop is None:
        return

    timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    payload = {
        "event_type": event_type,
        "timestamp": timestamp,
        "data": event_data,
    }

    try:
        data_bytes = json.dumps(payload).encode("utf-8")

        def do_publish():
            try:
                publish_future = _pubsub_publisher.publish(_pubsub_topic_path, data_bytes)
                publish_future.add_done_callback(_pubsub_callback)
            except Exception as e:
                print(f"[PubSubPublisher ERROR] Error when trying to initiate publish: {e}")

        _loop.call_soon_threadsafe(do_publish)

    except Exception as e:
        print(f"[PubSubPublisher ERROR] Failed to prepare event for publishing: {e}")
