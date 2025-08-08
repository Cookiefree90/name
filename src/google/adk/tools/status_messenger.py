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

import asyncio
from contextvars import ContextVar
from typing import Optional, Tuple, AsyncIterator
# ContextVar to hold the current WebSocket session ID for the active async context
current_websocket_session_id_var: ContextVar[Optional[str]] = ContextVar("current_websocket_session_id_var", default=None)

# asyncio.Queue to hold (session_id, message) tuples for WebSocket status updates
AGENT_MESSAGE_QUEUE: Optional[asyncio.Queue[Tuple[Optional[str], str]]] = None
_loop: Optional[asyncio.AbstractEventLoop] = None

def setup_status_messenger_async(loop: asyncio.AbstractEventLoop) -> None:
    """
    Initializes the status messenger with the asyncio event loop and creates the WebSocket message queue.
    This should be called once from the main async application at startup.
    """
    global AGENT_MESSAGE_QUEUE, _loop

    _loop = loop
    AGENT_MESSAGE_QUEUE = asyncio.Queue()
    print("[StatusMessenger] Async WebSocket setup complete, queue created.")


def add_status_message(message: str) -> None:
    """
    Adds a status message to the queue, associating it with the WebSocket session ID
    from the current asyncio context. Prints to console.
    """
    if AGENT_MESSAGE_QUEUE is None or _loop is None:
        print("[StatusMessenger ERROR] WebSocket Messenger not initialized. Call setup_status_messenger_async first.")
        print(f"Orphaned WebSocket status message (messenger not ready): {message}")
        return

    websocket_session_id = current_websocket_session_id_var.get()

    if websocket_session_id is None:
        print(f"[StatusMessenger WARNING] No WebSocket session ID in context for WebSocket message: {message}. Message will be queued without a specific session target.")
    
    print(f"WebSocket Status for session {websocket_session_id or 'UnknownSession'}: {message}")

    try:
        _loop.call_soon_threadsafe(AGENT_MESSAGE_QUEUE.put_nowait, (websocket_session_id, message))
    except RuntimeError:
        try:
            AGENT_MESSAGE_QUEUE.put_nowait((websocket_session_id, message))
        except Exception as e:
            print(f"[StatusMessenger ERROR] Failed to queue WebSocket message directly: {e}")




async def stream_status_updates() -> AsyncIterator[Tuple[Optional[str], str]]:
    """
    Asynchronously yields (websocket_session_id, message) tuples from the WebSocket message queue.
    """
    if AGENT_MESSAGE_QUEUE is None:
        print("[StatusMessenger ERROR] WebSocket Messenger not initialized for streaming. Call setup_status_messenger_async first.")
        return

    while True:
        session_id, message = await AGENT_MESSAGE_QUEUE.get()
        yield session_id, message
        AGENT_MESSAGE_QUEUE.task_done()
