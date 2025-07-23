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

"""
Context Reference Store for Efficient Management of Large Context Windows

This module implements a solution for efficiently managing large context windows (1M-2M tokens)
by using a reference-based approach rather than direct context passing.
"""

import time
import json
import uuid
import hashlib
import psutil
import threading
import os
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from google.genai import types


class CacheEvictionPolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live based
    MEMORY_PRESSURE = "memory_pressure"  # Based on system memory usage


@dataclass
class ContextMetadata:
    """Metadata for stored context."""

    content_type: str = "text/plain"
    token_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    cache_id: Optional[str] = None
    cached_until: Optional[float] = None  # Timestamp when cache expires
    is_structured: bool = False  # Whether this is JSON or not
    priority: int = 0  # Higher priority contexts are kept longer
    frequency_score: float = 0.0  # Calculated frequency score for LFU

    def update_access_stats(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1

        # Update frequency score for LFU
        current_time = time.time()
        time_since_creation = current_time - self.created_at
        if time_since_creation > 0:
            self.frequency_score = self.access_count / time_since_creation

    def is_expired(self) -> bool:
        """Check if context has expired based on TTL."""
        if self.cached_until is None:
            return False
        return time.time() > self.cached_until


class ContextReferenceStore:
    """
    A store for large contexts that provides reference-based access with advanced caching.

    This class allows large contexts to be stored once and referenced by ID,
    preventing unnecessary duplication and serialization of large data.
    """

    def __init__(
        self,
        cache_size: int = 50,
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU,
        memory_threshold: float = 0.8,  # 80% memory usage threshold
        ttl_check_interval: int = 300,  # Check TTL every 5 minutes
        enable_cache_warming: bool = True,
        use_disk_storage: bool = True,
        binary_cache_dir: str = "./multimodal_cache",
        large_binary_threshold: int = 1024 * 1024,  # 1MB threshold
    ):
        """
        Args:
            cache_size: Maximum number of contexts to keep in memory
            eviction_policy: Cache eviction policy to use
            memory_threshold: Memory usage threshold for pressure-based eviction (0.0-1.0)
            ttl_check_interval: Interval in seconds to check for expired contexts
            enable_cache_warming: Whether to enable cache warming strategies
            use_disk_storage: Whether to use disk storage for large binaries
            binary_cache_dir: Directory to store binary cache files
            large_binary_threshold: Size threshold for using disk storage
        """
        self._contexts: Dict[str, str] = {}
        self._metadata: Dict[str, ContextMetadata] = {}
        self._cache_size = cache_size
        self._eviction_policy = eviction_policy
        self._memory_threshold = memory_threshold
        self._ttl_check_interval = ttl_check_interval
        self._enable_cache_warming = enable_cache_warming

        # Multimodal storage infrastructure
        self._use_disk_storage = use_disk_storage
        self._binary_cache_dir = binary_cache_dir
        self._large_binary_threshold = large_binary_threshold
        self._binary_store: Dict[str, Union[bytes, str]] = {}
        self._binary_metadata: Dict[str, Dict[str, Any]] = {}

        # Cache warming data
        self._warmup_contexts: List[str] = []  # Contexts to keep warm
        self._access_patterns: Dict[str, List[float]] = {}  # Track access patterns
        # Background thread for TTL cleanup
        self._cleanup_thread = None
        self._stop_cleanup = False

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_pressure_evictions": 0,
            "ttl_evictions": 0,
        }

        # Create binary cache directory if needed
        if self._use_disk_storage:
            os.makedirs(self._binary_cache_dir, exist_ok=True)

        if self._ttl_check_interval > 0:
            self._start_ttl_cleanup()

    def store(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store context and return a reference ID.

        Args:
            content: The context content to store (string or structured data)
            metadata: Optional metadata about the context

        Returns:
            A reference ID for the stored context
        """
        # Handle both string and structured data (JSON objects)
        is_structured = not isinstance(content, str)

        # Convert structured data to string for storage
        if is_structured:
            content_str = json.dumps(content)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()
        else:
            content_str = content
            content_hash = hashlib.md5(content.encode()).hexdigest()

        for context_id, existing_content in self._contexts.items():
            existing_hash = hashlib.md5(existing_content.encode()).hexdigest()
            if (
                existing_hash == content_hash
                and self._metadata[context_id].is_structured == is_structured
            ):
                # Update access stats
                self._metadata[context_id].update_access_stats()
                self._stats["hits"] += 1
                self._track_access_pattern(context_id)
                return context_id
        # Generate a new ID if not found
        context_id = str(uuid.uuid4())
        self._stats["misses"] += 1
        self._contexts[context_id] = content_str

        # Set content type based on input type
        if is_structured:
            content_type = "application/json"
        else:
            content_type = (
                metadata.get("content_type", "text/plain") if metadata else "text/plain"
            )

        meta = ContextMetadata(
            content_type=content_type,
            token_count=len(content_str) // 4,
            tags=metadata.get("tags", []) if metadata else [],
            is_structured=is_structured,
        )
        if metadata and "priority" in metadata:
            meta.priority = metadata["priority"]

        # Generate a cache ID for Gemini caching
        if metadata and "cache_id" in metadata:
            meta.cache_id = metadata["cache_id"]
        else:
            meta.cache_id = f"context_{content_hash[:16]}"

        # Set cache expiration if provided
        if metadata and "cache_ttl" in metadata:
            ttl_seconds = metadata["cache_ttl"]
            meta.cached_until = time.time() + ttl_seconds
        self._metadata[context_id] = meta
        self._track_access_pattern(context_id)

        # Check if we need to warm this context
        if self._enable_cache_warming and self._should_warm_context(context_id):
            self._warmup_contexts.append(context_id)

        self._manage_cache()

        return context_id

    def store_multimodal_content(
        self,
        content: Union[str, Dict, types.Content, types.Part],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store multimodal content including images, audio, video."""

        if isinstance(content, types.Content):
            return self._store_content_with_parts(content, metadata)
        elif isinstance(content, types.Part):
            return self._store_part(content, metadata)
        else:
            return self.store(content, metadata)

    def _store_content_with_parts(self, content: types.Content, metadata):
        """Handle Content with multiple Parts (text + images + audio)."""
        content_data = {"role": content.role, "parts": []}

        # Pre-build the content data structure for deduplication check
        for part in content.parts:
            if part.text:
                content_data["parts"].append({"type": "text", "data": part.text})
            elif part.inline_data:
                binary_hash = hashlib.sha256(part.inline_data.data).hexdigest()
                content_data["parts"].append(
                    {
                        "type": "binary_ref",
                        "mime_type": part.inline_data.mime_type,
                        "binary_hash": binary_hash,
                        "size": len(part.inline_data.data),
                    }
                )
            elif part.file_data:
                content_data["parts"].append(
                    {
                        "type": "file_ref",
                        "file_uri": part.file_data.file_uri,
                        "mime_type": part.file_data.mime_type,
                    }
                )

        # Check if this content already exists before storing binaries
        content_str = json.dumps(content_data)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()

        for context_id, existing_content in self._contexts.items():
            existing_hash = hashlib.md5(existing_content.encode()).hexdigest()
            if existing_hash == content_hash:
                self._metadata[context_id].update_access_stats()
                self._stats["hits"] += 1
                self._track_access_pattern(context_id)
                return context_id

        # If content doesn't exist, store binaries now
        for part in content.parts:
            if part.inline_data:
                binary_hash = hashlib.sha256(part.inline_data.data).hexdigest()
                self._store_binary_data(
                    binary_hash, part.inline_data.data, part.inline_data.mime_type
                )

        return self._store_structured_multimodal(
            content_data, metadata, skip_dedup_check=True
        )

    def _store_part(self, part: types.Part, metadata):
        """Handle individual Part (text, image, audio, video)."""
        if part.text:
            part_data = {"type": "text", "data": part.text}
            return self._store_structured_multimodal(part_data, metadata)
        elif part.inline_data:
            binary_hash = hashlib.sha256(part.inline_data.data).hexdigest()
            part_data = {
                "type": "binary_ref",
                "mime_type": part.inline_data.mime_type,
                "binary_hash": binary_hash,
                "size": len(part.inline_data.data),
            }

            # Check if this content already exists before storing binary
            content_str = json.dumps(part_data)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()

            for context_id, existing_content in self._contexts.items():
                existing_hash = hashlib.md5(existing_content.encode()).hexdigest()
                if existing_hash == content_hash:
                    self._metadata[context_id].update_access_stats()
                    self._stats["hits"] += 1
                    self._track_access_pattern(context_id)
                    return context_id

            # If content doesn't exist, store binary now
            self._store_binary_data(
                binary_hash, part.inline_data.data, part.inline_data.mime_type
            )
            return self._store_structured_multimodal(
                part_data, metadata, skip_dedup_check=True
            )
        elif part.file_data:
            part_data = {
                "type": "file_ref",
                "file_uri": part.file_data.file_uri,
                "mime_type": part.file_data.mime_type,
            }
            return self._store_structured_multimodal(part_data, metadata)
        else:
            raise ValueError("Part contains no recognizable content")

    def _store_binary_data(self, binary_hash: str, data: bytes, mime_type: str) -> str:
        """Store binary data SEPARATELY from JSON context."""

        if binary_hash in self._binary_store:
            self._binary_metadata[binary_hash]["ref_count"] += 1
            return binary_hash

        if self._use_disk_storage and len(data) > self._large_binary_threshold:
            file_path = os.path.join(self._binary_cache_dir, f"{binary_hash}.bin")
            with open(file_path, "wb") as f:
                f.write(data)
            self._binary_store[binary_hash] = file_path
        else:
            self._binary_store[binary_hash] = data

        self._binary_metadata[binary_hash] = {
            "ref_count": 1,
            "mime_type": mime_type,
            "size": len(data),
            "created_at": time.time(),
            "is_disk_stored": len(data) > self._large_binary_threshold
            and self._use_disk_storage,
        }

        return binary_hash

    def _store_structured_multimodal(
        self,
        content_data: Dict,
        metadata: Optional[Dict[str, Any]],
        skip_dedup_check: bool = False,
    ) -> str:
        """Store structured multimodal data using existing infrastructure."""
        content_str = json.dumps(content_data)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()

        if not skip_dedup_check:
            for context_id, existing_content in self._contexts.items():
                existing_hash = hashlib.md5(existing_content.encode()).hexdigest()
                if existing_hash == content_hash:
                    self._metadata[context_id].update_access_stats()
                    self._stats["hits"] += 1
                    self._track_access_pattern(context_id)
                    return context_id

        context_id = str(uuid.uuid4())
        self._stats["misses"] += 1
        self._contexts[context_id] = content_str

        meta = ContextMetadata(
            content_type="application/json+multimodal",
            token_count=len(content_str) // 4,
            tags=metadata.get("tags", []) if metadata else [],
            is_structured=True,
        )

        if metadata:
            if "priority" in metadata:
                meta.priority = metadata["priority"]
            if "cache_id" in metadata:
                meta.cache_id = metadata["cache_id"]
            else:
                meta.cache_id = f"multimodal_{content_hash[:16]}"
            if "cache_ttl" in metadata:
                meta.cached_until = time.time() + metadata["cache_ttl"]

        self._metadata[context_id] = meta
        self._track_access_pattern(context_id)

        if self._enable_cache_warming and self._should_warm_context(context_id):
            self._warmup_contexts.append(context_id)

        self._manage_cache()
        return context_id

    def retrieve(self, context_id: str) -> Any:
        """
        Retrieve context by its reference ID.

        Args:
            context_id: The reference ID for the context

        Returns:
            The context content (string or structured data depending on how it was stored)
        """
        if context_id not in self._contexts:
            raise KeyError(f"Context ID {context_id} not found")

        # Check if expired
        if self._metadata[context_id].is_expired():
            self._evict_context(context_id)
            raise KeyError(f"Context ID {context_id} has expired")

        self._metadata[context_id].update_access_stats()
        self._track_access_pattern(context_id)
        self._stats["hits"] += 1
        content = self._contexts[context_id]
        metadata = self._metadata[context_id]

        # If the content is JSON, parse it back
        if metadata.is_structured:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content

        return content

    def retrieve_multimodal_content(
        self, context_id: str
    ) -> Union[types.Content, types.Part]:
        """Retrieve multimodal content with lazy loading of binary data."""

        if context_id not in self._contexts:
            raise KeyError(f"Context ID {context_id} not found")

        if self._metadata[context_id].is_expired():
            self._evict_context(context_id)
            raise KeyError(f"Context ID {context_id} has expired")

        self._metadata[context_id].update_access_stats()
        self._track_access_pattern(context_id)
        self._stats["hits"] += 1

        content_str = self._contexts[context_id]
        metadata = self._metadata[context_id]

        if metadata.content_type == "application/json+multimodal":
            content_data = json.loads(content_str)

            if "role" in content_data:
                return self._reconstruct_content(content_data)
            else:
                return self._reconstruct_part(content_data)
        else:
            return self.retrieve(context_id)

    def _reconstruct_content(self, content_data: Dict) -> types.Content:
        """Reconstruct types.Content from stored data."""
        parts = []

        for part_data in content_data.get("parts", []):
            if part_data["type"] == "text":
                parts.append(types.Part.from_text(text=part_data["data"]))
            elif part_data["type"] == "binary_ref":
                parts.append(self._create_lazy_binary_part(part_data))
            elif part_data["type"] == "file_ref":
                parts.append(
                    types.Part(
                        file_data=types.FileData(
                            file_uri=part_data["file_uri"],
                            mime_type=part_data["mime_type"],
                        )
                    )
                )

        return types.Content(role=content_data["role"], parts=parts)

    def _reconstruct_part(self, part_data: Dict) -> types.Part:
        """Reconstruct types.Part from stored data."""
        if part_data["type"] == "text":
            return types.Part.from_text(text=part_data["data"])
        elif part_data["type"] == "binary_ref":
            return self._create_lazy_binary_part(part_data)
        elif part_data["type"] == "file_ref":
            return types.Part(
                file_data=types.FileData(
                    file_uri=part_data["file_uri"], mime_type=part_data["mime_type"]
                )
            )
        else:
            raise ValueError(f"Unknown part type: {part_data['type']}")

    def _create_lazy_binary_part(self, part_data: Dict) -> types.Part:
        """Create a Part that loads binary data on demand."""
        binary_hash = part_data["binary_hash"]
        mime_type = part_data["mime_type"]

        binary_data = self._load_binary_data(binary_hash)

        return types.Part(inline_data=types.Blob(data=binary_data, mime_type=mime_type))

    def _load_binary_data(self, binary_hash: str) -> bytes:
        """Load binary data from storage."""
        if binary_hash not in self._binary_store:
            raise KeyError(f"Binary hash {binary_hash} not found")

        stored_data = self._binary_store[binary_hash]

        if isinstance(stored_data, str):
            with open(stored_data, "rb") as f:
                return f.read()
        else:
            return stored_data

    def get_metadata(self, context_id: str) -> ContextMetadata:
        """Get metadata for a context."""
        if context_id not in self._metadata:
            raise KeyError(f"Context ID {context_id} not found")
        return self._metadata[context_id]

    def _manage_cache(self):
        """Manage the cache size using the selected eviction policy."""
        # remove expired contexts
        self._evict_expired_contexts()
        # For memory pressure policy, check memory first
        if self._eviction_policy == CacheEvictionPolicy.MEMORY_PRESSURE:
            self._evict_by_memory_pressure()
        elif len(self._contexts) > self._cache_size:
            # Using the eviction policy
            if self._eviction_policy == CacheEvictionPolicy.LRU:
                self._evict_by_lru()
            elif self._eviction_policy == CacheEvictionPolicy.LFU:
                self._evict_by_lfu()
            elif self._eviction_policy == CacheEvictionPolicy.TTL:
                self._evict_by_ttl()

    def _evict_expired_contexts(self):
        """Remove contexts that have expired based on TTL."""
        expired_contexts = []
        for context_id, metadata in self._metadata.items():
            if metadata.is_expired():
                expired_contexts.append(context_id)

        for context_id in expired_contexts:
            self._evict_context(context_id)
            self._stats["ttl_evictions"] += 1

    def _evict_by_lru(self):
        """Evict contexts using Least Recently Used policy."""
        # Sort by priority first (ascending - low priority first), then by last accessed time (ascending)
        sorted_contexts = sorted(
            self._metadata.items(), key=lambda x: (x[1].priority, x[1].last_accessed)
        )

        # Remove oldest contexts until we're under the limit
        contexts_to_remove = len(self._contexts) - self._cache_size
        removed_count = 0

        for context_id, metadata in sorted_contexts:
            if removed_count >= contexts_to_remove:
                break

            # Don't evict warm contexts unless necessaryx
            if context_id not in self._warmup_contexts:
                self._evict_context(context_id)
                removed_count += 1

        if removed_count < contexts_to_remove:
            for context_id, metadata in sorted_contexts:
                if removed_count >= contexts_to_remove:
                    break
                if context_id in self._contexts:
                    self._evict_context(context_id)
                    removed_count += 1

    def _evict_by_lfu(self):
        """Evict contexts using Least Frequently Used policy."""
        # Sort by priority first (ascending - low priority first), then by frequency score (ascending)
        sorted_contexts = sorted(
            self._metadata.items(), key=lambda x: (x[1].priority, x[1].frequency_score)
        )

        # Remove least frequently used contexts
        contexts_to_remove = len(self._contexts) - self._cache_size
        removed_count = 0

        for context_id, metadata in sorted_contexts:
            if removed_count >= contexts_to_remove:
                break

            # Don't evict warm contexts unless necessary
            if context_id not in self._warmup_contexts:
                self._evict_context(context_id)
                removed_count += 1

        # If we still need to remove more and only warm contexts remain
        if removed_count < contexts_to_remove:
            for context_id, metadata in sorted_contexts:
                if removed_count >= contexts_to_remove:
                    break
                if context_id in self._contexts:
                    self._evict_context(context_id)
                    removed_count += 1

    def _evict_by_ttl(self):
        """Evict contexts based on TTL, removing those expiring soonest."""
        # Sort by priority first (ascending - low priority first), then by Time To Live (ascending)
        sorted_contexts = sorted(
            self._metadata.items(),
            key=lambda x: (x[1].priority, x[1].cached_until or float("inf")),
        )

        contexts_to_remove = len(self._contexts) - self._cache_size
        removed_count = 0
        for context_id, metadata in sorted_contexts:
            if removed_count >= contexts_to_remove:
                break
            self._evict_context(context_id)
            removed_count += 1

    def _evict_by_memory_pressure(self):
        """Evict contexts based on system memory pressure."""
        try:
            memory_percent = psutil.virtual_memory().percent / 100.0
            if memory_percent > self._memory_threshold:
                target_size = int(self._cache_size * 0.7)
                contexts_to_remove = len(self._contexts) - target_size
                if contexts_to_remove > 0:
                    # Use LRU for memory pressure eviction
                    sorted_contexts = sorted(
                        self._metadata.items(),
                        key=lambda x: (x[1].priority, x[1].last_accessed),
                    )
                    removed_count = 0
                    for context_id, metadata in sorted_contexts:
                        if removed_count >= contexts_to_remove:
                            break
                        self._evict_context(context_id)
                        self._stats["memory_pressure_evictions"] += 1
                        removed_count += 1

        except Exception:
            pass

        # After memory pressure eviction, still check regular cache size
        if len(self._contexts) > self._cache_size:
            self._evict_by_lru()

    def _evict_context(self, context_id: str):
        """Remove a context from the store."""

        if context_id in self._contexts:
            self._cleanup_binary_references(context_id)
            del self._contexts[context_id]
        if context_id in self._metadata:
            del self._metadata[context_id]
        if context_id in self._warmup_contexts:
            self._warmup_contexts.remove(context_id)
        if context_id in self._access_patterns:
            del self._access_patterns[context_id]
        self._stats["evictions"] += 1

    def _cleanup_binary_references(self, context_id: str):
        """Clean up binary references when evicting multimodal contexts."""
        if context_id not in self._contexts:
            return

        content_str = self._contexts[context_id]
        metadata = self._metadata.get(context_id)

        if metadata and metadata.content_type == "application/json+multimodal":
            try:
                content_data = json.loads(content_str)
                self._decrement_binary_refs(content_data)
            except (json.JSONDecodeError, KeyError):
                pass

    def _decrement_binary_refs(self, content_data: Dict):
        """Decrement reference counts for binary data."""

        if "parts" in content_data:
            for part_data in content_data["parts"]:
                if part_data.get("type") == "binary_ref":
                    self._decrement_binary_ref(part_data["binary_hash"])
        elif content_data.get("type") == "binary_ref":
            self._decrement_binary_ref(content_data["binary_hash"])

    def _decrement_binary_ref(self, binary_hash: str):
        """Decrement reference count for a binary hash and clean up if needed."""
        if binary_hash in self._binary_metadata:
            self._binary_metadata[binary_hash]["ref_count"] -= 1

            if self._binary_metadata[binary_hash]["ref_count"] <= 0:
                self._cleanup_binary_data(binary_hash)

    def _cleanup_binary_data(self, binary_hash: str):
        """Clean up binary data when no longer referenced."""
        if binary_hash in self._binary_store:
            stored_data = self._binary_store[binary_hash]

            if isinstance(stored_data, str):
                try:
                    os.remove(stored_data)
                except OSError:
                    pass

            del self._binary_store[binary_hash]

        if binary_hash in self._binary_metadata:
            del self._binary_metadata[binary_hash]

    def _track_access_pattern(self, context_id: str):
        """Track access patterns for cache warming."""
        if not self._enable_cache_warming:
            return

        current_time = time.time()
        if context_id not in self._access_patterns:
            self._access_patterns[context_id] = []

        # Keep only recent accesses (last hour)
        self._access_patterns[context_id] = [
            t for t in self._access_patterns[context_id] if current_time - t < 3600
        ]
        self._access_patterns[context_id].append(current_time)

    def _should_warm_context(self, context_id: str) -> bool:
        """Determine if a context should be kept warm."""
        if not self._enable_cache_warming:
            return False

        # Keep contexts warm if they're accessed frequently
        if context_id in self._access_patterns:
            recent_accesses = len(self._access_patterns[context_id])
            return recent_accesses > 3  # Warm if accessed more than 3 times recently

        return False

    def _start_ttl_cleanup(self):
        """Start background thread for TTL cleanup."""

        def cleanup_worker():
            while not self._stop_cleanup:
                time.sleep(self._ttl_check_interval)
                if not self._stop_cleanup:
                    self._evict_expired_contexts()

        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def get_cache_hint(self, context_id: str) -> Dict[str, Any]:
        """
        Get a cache hint object for Gemini API calls.

        This allows the Gemini API to cache the context for reuse.
        According to Gemini API docs, context caching can significantly
        reduce costs when reusing the same context multiple times.
        """
        if context_id not in self._metadata:
            raise KeyError(f"Context ID {context_id} not found")

        metadata = self._metadata[context_id]

        # Create cache hint with recommended parameters
        cache_hint = {
            "cache_id": metadata.cache_id,
            "cache_level": "HIGH",
        }
        # If we have a cached_until timestamp, add it
        if metadata.cached_until:
            now = time.time()
            if metadata.cached_until > now:
                # Still valid, calculate remaining TTL in seconds
                cache_hint["ttl_seconds"] = int(metadata.cached_until - now)

        return cache_hint

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_accesses = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_accesses if total_accesses > 0 else 0

        multimodal_contexts = sum(
            1
            for meta in self._metadata.values()
            if meta.content_type == "application/json+multimodal"
        )

        total_binary_size = sum(meta["size"] for meta in self._binary_metadata.values())

        disk_stored_binaries = sum(
            1
            for meta in self._binary_metadata.values()
            if meta.get("is_disk_stored", False)
        )

        return {
            "total_contexts": len(self._contexts),
            "multimodal_contexts": multimodal_contexts,
            "total_binary_objects": len(self._binary_store),
            "total_binary_size_bytes": total_binary_size,
            "disk_stored_binaries": disk_stored_binaries,
            "cache_size_limit": self._cache_size,
            "hit_rate": hit_rate,
            "total_hits": self._stats["hits"],
            "total_misses": self._stats["misses"],
            "total_evictions": self._stats["evictions"],
            "memory_pressure_evictions": self._stats["memory_pressure_evictions"],
            "ttl_evictions": self._stats["ttl_evictions"],
            "warm_contexts": len(self._warmup_contexts),
            "eviction_policy": self._eviction_policy.value,
            "memory_usage_percent": psutil.virtual_memory().percent if psutil else None,
        }

    def set_context_priority(self, context_id: str, priority: int):
        """Set priority for a context (higher priority = kept longer)."""
        if context_id in self._metadata:
            self._metadata[context_id].priority = priority

    def warm_contexts(self, context_ids: List[str]):
        """Mark contexts as warm (should be kept in cache)."""
        for context_id in context_ids:
            if context_id in self._contexts and context_id not in self._warmup_contexts:
                self._warmup_contexts.append(context_id)

    def cleanup_unused_binaries(self):
        """Clean up binary data that is no longer referenced."""
        unused_hashes = []

        for binary_hash, metadata in self._binary_metadata.items():
            if metadata["ref_count"] <= 0:
                unused_hashes.append(binary_hash)

        for binary_hash in unused_hashes:
            self._cleanup_binary_data(binary_hash)

    def get_multimodal_stats(self) -> Dict[str, Any]:
        """Get detailed multimodal storage statistics."""
        memory_binaries = sum(
            1
            for meta in self._binary_metadata.values()
            if not meta.get("is_disk_stored", False)
        )

        memory_binary_size = sum(
            meta["size"]
            for meta in self._binary_metadata.values()
            if not meta.get("is_disk_stored", False)
        )

        disk_binary_size = sum(
            meta["size"]
            for meta in self._binary_metadata.values()
            if meta.get("is_disk_stored", False)
        )

        return {
            "memory_stored_binaries": memory_binaries,
            "memory_binary_size_bytes": memory_binary_size,
            "disk_stored_binaries": sum(
                1
                for meta in self._binary_metadata.values()
                if meta.get("is_disk_stored", False)
            ),
            "disk_binary_size_bytes": disk_binary_size,
            "binary_deduplication_ratio": len(self._binary_metadata)
            / max(1, sum(meta["ref_count"] for meta in self._binary_metadata.values())),
            "binary_cache_directory": self._binary_cache_dir,
            "large_binary_threshold": self._large_binary_threshold,
        }

    def clear_multimodal_cache(self):
        """Clear all multimodal binary data from cache."""
        for binary_hash in list(self._binary_store.keys()):
            self._cleanup_binary_data(binary_hash)

    def __del__(self):
        """Cleanup background threads and binary data."""
        self._stop_cleanup = True
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1)

        if hasattr(self, "_binary_store"):
            for binary_hash in list(self._binary_store.keys()):
                stored_data = self._binary_store[binary_hash]
                if isinstance(stored_data, str):
                    try:
                        os.remove(stored_data)
                    except OSError:
                        pass
