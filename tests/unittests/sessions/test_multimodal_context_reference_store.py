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

import os
import tempfile
import unittest
from unittest.mock import patch

from google.genai import types
from google.adk.sessions.context_reference_store import ContextReferenceStore


class TestMultimodalContextReferenceStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = ContextReferenceStore(
            cache_size=10,
            use_disk_storage=True,
            binary_cache_dir=self.temp_dir,
            large_binary_threshold=1024,
        )

        # Create test binary data
        self.small_image_data = b"fake_image_data" * 10  # 150 bytes
        self.large_image_data = b"fake_large_image_data" * 100  # 2100 bytes
        self.audio_data = b"fake_audio_data" * 50  # 800 bytes

        # Create test Parts
        self.text_part = types.Part.from_text(text="Hello, world!")
        self.image_part = types.Part(
            inline_data=types.Blob(data=self.small_image_data, mime_type="image/png")
        )
        self.large_image_part = types.Part(
            inline_data=types.Blob(data=self.large_image_data, mime_type="image/jpeg")
        )
        self.audio_part = types.Part(
            inline_data=types.Blob(data=self.audio_data, mime_type="audio/wav")
        )

        # Create test Content
        self.multimodal_content = types.Content(
            role="user", parts=[self.text_part, self.image_part, self.audio_part]
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_store_and_retrieve_text_part(self):
        """Test storing and retrieving a text part."""
        context_id = self.store.store_multimodal_content(self.text_part)

        retrieved = self.store.retrieve_multimodal_content(context_id)
        self.assertIsInstance(retrieved, types.Part)
        self.assertEqual(retrieved.text, "Hello, world!")

    def test_store_and_retrieve_image_part(self):
        """Test storing and retrieving an image part."""
        context_id = self.store.store_multimodal_content(self.image_part)

        retrieved = self.store.retrieve_multimodal_content(context_id)
        self.assertIsInstance(retrieved, types.Part)
        self.assertEqual(retrieved.inline_data.data, self.small_image_data)
        self.assertEqual(retrieved.inline_data.mime_type, "image/png")

    def test_store_and_retrieve_multimodal_content(self):
        """Test storing and retrieving multimodal content."""
        context_id = self.store.store_multimodal_content(self.multimodal_content)

        retrieved = self.store.retrieve_multimodal_content(context_id)
        self.assertIsInstance(retrieved, types.Content)
        self.assertEqual(retrieved.role, "user")
        self.assertEqual(len(retrieved.parts), 3)

        # Check text part
        self.assertEqual(retrieved.parts[0].text, "Hello, world!")
        # Check image part
        self.assertEqual(retrieved.parts[1].inline_data.data, self.small_image_data)
        self.assertEqual(retrieved.parts[1].inline_data.mime_type, "image/png")
        # Check audio part
        self.assertEqual(retrieved.parts[2].inline_data.data, self.audio_data)
        self.assertEqual(retrieved.parts[2].inline_data.mime_type, "audio/wav")

    def test_binary_deduplication(self):
        """Test that identical binary data is deduplicated."""
        # Fresh store for this test to avoid interference
        import tempfile

        temp_dir = tempfile.mkdtemp()
        fresh_store = ContextReferenceStore(
            cache_size=10,
            use_disk_storage=True,
            binary_cache_dir=temp_dir,
            large_binary_threshold=1024,
        )

        # Store the same image twice
        context_id1 = fresh_store.store_multimodal_content(self.image_part)
        context_id2 = fresh_store.store_multimodal_content(self.image_part)

        # Should be the same context ID for identical content (deduplication)
        self.assertEqual(context_id1, context_id2)

        # Check that binary data is deduplicated
        stats = fresh_store.get_multimodal_stats()
        self.assertEqual(
            stats["memory_stored_binaries"], 1
        )  # Only one copy of binary data

        # Check reference count - should be 1 since same context ID
        binary_hash = list(fresh_store._binary_metadata.keys())[0]
        self.assertEqual(fresh_store._binary_metadata[binary_hash]["ref_count"], 1)

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_large_binary_disk_storage(self):
        """Test that large binaries are stored on disk."""
        # Fresh store for this test to avoid interference
        import tempfile

        temp_dir = tempfile.mkdtemp()
        fresh_store = ContextReferenceStore(
            cache_size=10,
            use_disk_storage=True,
            binary_cache_dir=temp_dir,
            large_binary_threshold=1024,
        )

        context_id = fresh_store.store_multimodal_content(self.large_image_part)

        # Check that binary is stored on disk
        stats = fresh_store.get_multimodal_stats()
        self.assertEqual(stats["disk_stored_binaries"], 1)
        self.assertEqual(stats["memory_stored_binaries"], 0)

        # Verify file exists
        binary_hash = list(fresh_store._binary_store.keys())[0]
        file_path = fresh_store._binary_store[binary_hash]
        self.assertTrue(os.path.exists(file_path))

        # Verify retrieval works
        retrieved = fresh_store.retrieve_multimodal_content(context_id)
        self.assertEqual(retrieved.inline_data.data, self.large_image_data)

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_binary_cleanup_on_eviction(self):
        """Test that binary data is cleaned up when contexts are evicted."""
        # Store multiple multimodal contents to trigger eviction
        context_ids = []
        for i in range(15):  # More than cache size
            test_data = b"test_data" * i
            part = types.Part(
                inline_data=types.Blob(data=test_data, mime_type="image/png")
            )
            context_id = self.store.store_multimodal_content(part)
            context_ids.append(context_id)

        # Some contexts should be evicted
        self.assertLess(len(self.store._contexts), 15)

        # Check that binary data is cleaned up
        referenced_hashes = set()
        for context_str in self.store._contexts.values():
            try:
                import json

                context_data = json.loads(context_str)
                if isinstance(context_data, dict) and "binary_hash" in context_data:
                    referenced_hashes.add(context_data["binary_hash"])
            except:
                pass

        # All remaining binary hashes should be referenced
        for binary_hash in self.store._binary_metadata.keys():
            self.assertGreater(self.store._binary_metadata[binary_hash]["ref_count"], 0)

    def test_multimodal_content_type_identification(self):
        """Test that multimodal content is properly identified."""
        context_id = self.store.store_multimodal_content(self.multimodal_content)

        metadata = self.store.get_metadata(context_id)
        self.assertEqual(metadata.content_type, "application/json+multimodal")
        self.assertTrue(metadata.is_structured)

    def test_cache_stats_include_multimodal(self):
        """Test that cache statistics include multimodal information."""
        # Fresh store for this test to avoid interference
        import tempfile

        temp_dir = tempfile.mkdtemp()
        fresh_store = ContextReferenceStore(
            cache_size=10,
            use_disk_storage=True,
            binary_cache_dir=temp_dir,
            large_binary_threshold=1024,
        )

        # Store some multimodal content
        fresh_store.store_multimodal_content(self.multimodal_content)
        fresh_store.store_multimodal_content(self.image_part)

        stats = fresh_store.get_cache_stats()
        self.assertIn("multimodal_contexts", stats)
        self.assertIn("total_binary_objects", stats)
        self.assertIn("total_binary_size_bytes", stats)
        self.assertIn("disk_stored_binaries", stats)

        self.assertEqual(stats["multimodal_contexts"], 2)
        self.assertGreater(stats["total_binary_objects"], 0)
        self.assertGreater(stats["total_binary_size_bytes"], 0)

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_multimodal_specific_stats(self):
        """Test detailed multimodal statistics."""
        # Fresh store for this test to avoid interference
        import tempfile

        temp_dir = tempfile.mkdtemp()
        fresh_store = ContextReferenceStore(
            cache_size=10,
            use_disk_storage=True,
            binary_cache_dir=temp_dir,
            large_binary_threshold=1024,
        )

        # Store mixed content
        fresh_store.store_multimodal_content(self.image_part)  # Small - memory
        fresh_store.store_multimodal_content(self.large_image_part)  # Large - disk

        stats = fresh_store.get_multimodal_stats()

        self.assertEqual(stats["memory_stored_binaries"], 1)
        self.assertEqual(stats["disk_stored_binaries"], 1)
        self.assertGreater(stats["memory_binary_size_bytes"], 0)
        self.assertGreater(stats["disk_binary_size_bytes"], 0)
        self.assertEqual(stats["binary_deduplication_ratio"], 1.0)  # No duplication
        self.assertEqual(stats["binary_cache_directory"], temp_dir)
        self.assertEqual(stats["large_binary_threshold"], 1024)

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_clear_multimodal_cache(self):
        """Test clearing multimodal cache."""
        # Store some multimodal content
        self.store.store_multimodal_content(self.multimodal_content)
        self.store.store_multimodal_content(self.large_image_part)

        # Verify content exists
        self.assertGreater(len(self.store._binary_store), 0)

        # Clear cache
        self.store.clear_multimodal_cache()

        # Verify cache is cleared
        self.assertEqual(len(self.store._binary_store), 0)
        self.assertEqual(len(self.store._binary_metadata), 0)

    def test_fallback_to_regular_storage(self):
        """Test that non-multimodal content falls back to regular storage."""
        text_content = "Regular text content"

        context_id = self.store.store_multimodal_content(text_content)

        # Should use regular storage
        metadata = self.store.get_metadata(context_id)
        self.assertEqual(metadata.content_type, "text/plain")
        self.assertFalse(metadata.is_structured)

        # Should retrieve with regular method
        retrieved = self.store.retrieve(context_id)
        self.assertEqual(retrieved, text_content)

    def test_file_data_part_storage(self):
        """Test storing parts with file_data."""
        file_part = types.Part(
            file_data=types.FileData(
                file_uri="gs://bucket/file.png", mime_type="image/png"
            )
        )

        context_id = self.store.store_multimodal_content(file_part)

        retrieved = self.store.retrieve_multimodal_content(context_id)
        self.assertIsInstance(retrieved, types.Part)
        self.assertEqual(retrieved.file_data.file_uri, "gs://bucket/file.png")
        self.assertEqual(retrieved.file_data.mime_type, "image/png")


if __name__ == "__main__":
    unittest.main()
