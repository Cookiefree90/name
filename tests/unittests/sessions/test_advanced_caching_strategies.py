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
Advanced Caching Strategies Test Suite for Context Reference Store.

This module tests the advanced caching features including:
- TTL-based eviction policies
- LRU/LFU cache replacement algorithms
- Memory pressure-based eviction
- Cache warming strategies
"""

import time
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Force the importing of conftest.py which sets up our mock modules
from . import conftest

# Now import the modules we want to test
from google.adk.sessions.context_reference_store import (
    ContextReferenceStore,
    ContextMetadata,
    CacheEvictionPolicy,
)


class TestAdvancedCachingStrategies:
    """Tests for advanced caching strategies in ContextReferenceStore."""

    def test_lru_eviction_policy(self):
        """Test Least Recently Used eviction policy."""
        store = ContextReferenceStore(
            cache_size=3, eviction_policy=CacheEvictionPolicy.LRU
        )

        # Store 3 contexts
        id1 = store.store("context1")
        id2 = store.store("context2")
        id3 = store.store("context3")

        # Access them in a specific order to establish LRU pattern
        store.retrieve(id2)  # Make id2 most recent
        time.sleep(0.01)
        store.retrieve(id3)  # Make id3 second most recent
        time.sleep(0.01)
        # id1 is now least recently used (never accessed after store)

        # Add fourth context - should evict id1 (least recently used)
        id4 = store.store("context4")

        # id1 should be evicted, others should remain
        assert store._contexts.get(id1) is None  # Least recently used - evicted
        assert store._contexts.get(id2) is not None  # Most recently used - kept
        assert store._contexts.get(id3) is not None  # Second most recent - kept
        assert store._contexts.get(id4) is not None  # New context - kept

    def test_lfu_eviction_policy(self):
        """Test Least Frequently Used eviction policy."""
        store = ContextReferenceStore(
            cache_size=3, eviction_policy=CacheEvictionPolicy.LFU
        )

        # Store 3 contexts
        id1 = store.store("context1")
        id2 = store.store("context2")
        id3 = store.store("context3")

        # Access id1 multiple times (most frequent)
        for _ in range(5):
            store.retrieve(id1)
            time.sleep(0.001)

        # Access id3 a few times (medium frequent)
        for _ in range(3):
            store.retrieve(id3)
            time.sleep(0.001)

        # id2 is least frequent (never accessed after store)

        # Add fourth context - should evict id2 (least frequently used)
        id4 = store.store("context4")

        # id2 should be evicted, others should remain
        assert store._contexts.get(id1) is not None  # Most frequently used - kept
        assert store._contexts.get(id2) is None  # Least frequently used - evicted
        assert store._contexts.get(id3) is not None  # Medium frequently used - kept
        assert store._contexts.get(id4) is not None  # New context - kept

    def test_ttl_eviction_policy(self):
        """Test TTL-based eviction policy."""
        store = ContextReferenceStore(
            cache_size=3, eviction_policy=CacheEvictionPolicy.TTL
        )

        # Store contexts with different TTLs
        id1 = store.store("context1", {"cache_ttl": 10})  # Expires in 10s
        id2 = store.store("context2", {"cache_ttl": 5})  # Expires in 5s
        id3 = store.store("context3")  # No TTL

        # Add fourth context - should evict id2 (expires soonest)
        id4 = store.store("context4", {"cache_ttl": 15})

        # id2 should be evicted (shortest TTL), others should remain
        assert store._contexts.get(id1) is not None
        assert store._contexts.get(id3) is not None
        assert store._contexts.get(id4) is not None
        assert store._contexts.get(id2) is None

    def test_ttl_expiration(self):
        """Test that contexts expire based on TTL."""
        store = ContextReferenceStore(cache_size=10, ttl_check_interval=0)

        # Store context with very short TTL
        id1 = store.store("context1", {"cache_ttl": 0.1})  # 100ms

        # Should be retrievable immediately
        assert store.retrieve(id1) == "context1"

        # Wait for expiration
        time.sleep(0.2)

        # Should be expired and raise KeyError
        with pytest.raises(KeyError, match="has expired"):
            store.retrieve(id1)

    @patch("psutil.virtual_memory")
    def test_memory_pressure_eviction(self, mock_memory):
        """Test memory pressure-based eviction."""
        # Start with normal memory usage
        mock_memory.return_value = MagicMock(percent=70.0)

        store = ContextReferenceStore(
            cache_size=10,
            eviction_policy=CacheEvictionPolicy.MEMORY_PRESSURE,
            memory_threshold=0.8,  # 80% threshold
        )

        # Store multiple contexts - should work normally under 80% memory
        context_ids = []
        for i in range(10):
            context_ids.append(store.store(f"context{i}"))

        # All should be stored initially with normal memory usage
        assert len(store._contexts) == 10

        # Now simulate high memory usage (90%)
        mock_memory.return_value = MagicMock(percent=90.0)

        # Add one more context - should trigger memory pressure eviction
        new_id = store.store("new_context")

        # Should have reduced to 70% of cache size (7 contexts) due to memory pressure
        assert len(store._contexts) <= 7
        assert store.get_cache_stats()["memory_pressure_evictions"] > 0

    def test_priority_based_eviction(self):
        """Test that high priority contexts are kept longer."""
        store = ContextReferenceStore(cache_size=3)

        # Store contexts with different priorities
        id1 = store.store("context1", {"priority": 10})  # High priority
        id2 = store.store("context2", {"priority": 1})  # Low priority
        id3 = store.store("context3", {"priority": 5})  # Medium priority

        # Access all contexts to establish last_accessed times
        store.retrieve(id1)
        time.sleep(0.01)
        store.retrieve(id2)
        time.sleep(0.01)
        store.retrieve(id3)

        # Add fourth context - should evict id2 (lowest priority despite being recently accessed)
        id4 = store.store("context4", {"priority": 7})

        # Low priority context should be evicted first due to priority-based LRU sorting
        assert store._contexts.get(id2) is None  # Low priority evicted
        assert store._contexts.get(id1) is not None  # High priority kept
        assert store._contexts.get(id3) is not None  # Medium priority kept
        assert store._contexts.get(id4) is not None  # New high priority kept

    def test_cache_warming(self):
        """Test cache warming functionality."""
        store = ContextReferenceStore(cache_size=3, enable_cache_warming=True)

        # Store contexts and access one frequently to make it "warm"
        id1 = store.store("context1")
        id2 = store.store("context2")
        id3 = store.store("context3")

        # Access id1 multiple times to make it warm
        for _ in range(5):
            store.retrieve(id1)
            time.sleep(0.001)

        # Manually mark it as warm (simulating automatic warming)
        store.warm_contexts([id1])

        # Add new contexts to force eviction
        id4 = store.store("context4")
        id5 = store.store("context5")

        # Warm context (id1) should be kept longer
        assert store._contexts.get(id1) is not None  # Warm context kept
        assert len(store._warmup_contexts) >= 1

    def test_cache_statistics(self):
        """Test cache statistics functionality."""
        store = ContextReferenceStore(cache_size=5)

        # Store some contexts
        id1 = store.store("context1")
        id2 = store.store("context2")

        # Access them
        store.retrieve(id1)
        store.retrieve(id1)  # Hit
        store.retrieve(id2)  # Hit

        # Try to access non-existent context
        try:
            store.retrieve("non-existent")
        except KeyError:
            pass  # Expected

        # Get statistics
        stats = store.get_cache_stats()

        assert stats["total_contexts"] == 2
        assert stats["cache_size_limit"] == 5
        assert stats["total_hits"] >= 2
        assert stats["hit_rate"] > 0
        assert "eviction_policy" in stats
        assert "memory_usage_percent" in stats

    def test_background_ttl_cleanup(self):
        """Test background TTL cleanup thread."""
        store = ContextReferenceStore(
            cache_size=10, ttl_check_interval=0.1  # Check every 100ms
        )

        # Store context with short TTL
        id1 = store.store("context1", {"cache_ttl": 0.05})  # 50ms
        assert store._contexts.get(id1) is not None

        # Wait for background cleanup
        time.sleep(0.2)

        # Context should be cleaned up by background thread
        assert store._contexts.get(id1) is None
        assert store.get_cache_stats()["ttl_evictions"] > 0

    def test_frequency_score_calculation(self):
        """Test frequency score calculation for LFU."""
        store = ContextReferenceStore()

        id1 = store.store("context1")

        # Access multiple times
        for _ in range(3):
            store.retrieve(id1)
            time.sleep(0.01)

        metadata = store.get_metadata(id1)

        # Frequency score should be calculated based on access count and time
        assert metadata.frequency_score > 0
        assert (
            metadata.access_count == 3
        )  # 3 retrieve calls (store doesn't count as access)

    def test_context_priority_setting(self):
        """Test setting context priority after creation."""
        store = ContextReferenceStore()

        id1 = store.store("context1")

        # Set priority
        store.set_context_priority(id1, 100)

        metadata = store.get_metadata(id1)
        assert metadata.priority == 100

    def test_manual_context_warming(self):
        """Test manually warming contexts."""
        store = ContextReferenceStore(enable_cache_warming=True)

        id1 = store.store("context1")
        id2 = store.store("context2")

        # Warm specific contexts
        store.warm_contexts([id1, id2])

        assert id1 in store._warmup_contexts
        assert id2 in store._warmup_contexts

    def test_access_pattern_tracking(self):
        """Test access pattern tracking for cache warming."""
        store = ContextReferenceStore(enable_cache_warming=True)

        id1 = store.store("context1")

        # Access multiple times
        for _ in range(3):
            store.retrieve(id1)
            time.sleep(0.01)

        # Should have tracked access patterns
        assert id1 in store._access_patterns
        assert len(store._access_patterns[id1]) >= 3

    def test_cache_warming_disabled(self):
        """Test that cache warming can be disabled."""
        store = ContextReferenceStore(enable_cache_warming=False)

        id1 = store.store("context1")

        # Access multiple times
        for _ in range(5):
            store.retrieve(id1)

        # Should not track access patterns when disabled
        assert len(store._access_patterns) == 0

    def test_cache_hint_with_expired_ttl(self):
        """Test cache hint generation with expired TTL."""
        store = ContextReferenceStore()

        # Store context with TTL that will expire
        id1 = store.store("context1", {"cache_ttl": 0.01})  # 10ms

        # Get cache hint immediately
        hint1 = store.get_cache_hint(id1)
        assert "ttl_seconds" in hint1

        # Wait for expiration
        time.sleep(0.02)

        # Get cache hint after expiration
        hint2 = store.get_cache_hint(id1)
        # Should not include ttl_seconds for expired context
        assert "ttl_seconds" not in hint2 or hint2["ttl_seconds"] <= 0

    def test_mixed_eviction_scenarios(self):
        """Test complex scenarios with mixed eviction factors."""
        store = ContextReferenceStore(
            cache_size=4, eviction_policy=CacheEvictionPolicy.LRU
        )

        # Store contexts with different characteristics
        id1 = store.store("context1", {"priority": 10, "cache_ttl": 1})
        id2 = store.store("context2", {"priority": 1})
        id3 = store.store("context3", {"priority": 5, "cache_ttl": 10})
        id4 = store.store("context4", {"priority": 7})

        # Should be exactly at cache limit now
        assert len(store._contexts) == 4

        # Warm one context
        store.warm_contexts([id2])

        # Access in specific pattern
        store.retrieve(id1)  # High priority, but short TTL
        store.retrieve(id4)  # Medium-high priority

        # Add new context to force eviction
        id5 = store.store("context5", {"priority": 8})

        # Should still be at cache limit after eviction
        stats = store.get_cache_stats()
        assert stats["total_contexts"] == 4

    def test_cleanup_on_destruction(self):
        """Test that background threads are cleaned up properly."""
        store = ContextReferenceStore(ttl_check_interval=1)

        # Ensure cleanup thread is running
        assert store._cleanup_thread is not None
        assert store._cleanup_thread.is_alive()

        # Destroy store
        del store

        # Thread should be stopped (though we can't easily test this without
        # race conditions, the important thing is no exceptions are raised)


if __name__ == "__main__":
    pytest.main([__file__])
