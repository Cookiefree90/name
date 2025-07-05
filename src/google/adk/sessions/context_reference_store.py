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
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


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
    ):
        """
        Args:
            cache_size: Maximum number of contexts to keep in memory
            eviction_policy: Cache eviction policy to use
            memory_threshold: Memory usage threshold for pressure-based eviction (0.0-1.0)
            ttl_check_interval: Interval in seconds to check for expired contexts
            enable_cache_warming: Whether to enable cache warming strategies
        """
        self._contexts: Dict[str, str] = {}
        self._metadata: Dict[str, ContextMetadata] = {}
        self._cache_size = cache_size
        self._eviction_policy = eviction_policy
        self._memory_threshold = memory_threshold
        self._ttl_check_interval = ttl_check_interval
        self._enable_cache_warming = enable_cache_warming
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
        # Sort by priority first (ascending - low priority first), then by TTL (ascending)
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
            # Fallback to regular eviction
            pass

        # After memory pressure eviction, still check regular cache size
        if len(self._contexts) > self._cache_size:
            self._evict_by_lru()

    def _evict_context(self, context_id: str):
        """Remove a context from the store."""
        if context_id in self._contexts:
            del self._contexts[context_id]
        if context_id in self._metadata:
            del self._metadata[context_id]
        if context_id in self._warmup_contexts:
            self._warmup_contexts.remove(context_id)
        if context_id in self._access_patterns:
            del self._access_patterns[context_id]
        self._stats["evictions"] += 1

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

        return {
            "total_contexts": len(self._contexts),
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

    def __del__(self):
        """Cleanup background threads."""
        self._stop_cleanup = True
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1)
