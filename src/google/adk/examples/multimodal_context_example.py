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
Example demonstrating the multimodal Context Reference Store functionality.

This example shows how to efficiently store and retrieve multimodal content
including text, images, audio, and video using the Context Reference Store.
"""

import tempfile
from google.genai import types
from google.adk.sessions.context_reference_store import ContextReferenceStore


def main():
    print("🎯 Multimodal Context Reference Store Example")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        store = ContextReferenceStore(
            cache_size=20,
            use_disk_storage=True,
            binary_cache_dir=temp_dir,
            large_binary_threshold=1024,
        )

        # Create sample multimodal content
        print("\n📝 Creating multimodal content...")

        # Text content
        text_part = types.Part.from_text(
            "Here's an analysis of the provided image and audio:"
        )

        # Image content (simulated)
        image_data = b"PNG_IMAGE_DATA" * 100  # 1.4KB - will be stored on disk
        image_part = types.Part(
            inline_data=types.Blob(data=image_data, mime_type="image/png")
        )

        # Audio content (simulated)
        audio_data = b"AUDIO_DATA" * 50  # 500 bytes - will be stored in memory
        audio_part = types.Part(
            inline_data=types.Blob(data=audio_data, mime_type="audio/wav")
        )

        # File reference
        file_part = types.Part(
            file_data=types.FileData(
                file_uri="gs://my-bucket/video.mp4", mime_type="video/mp4"
            )
        )

        # Combine into multimodal content
        multimodal_content = types.Content(
            role="user", parts=[text_part, image_part, audio_part, file_part]
        )

        print(f"✅ Created content with {len(multimodal_content.parts)} parts")

        # Store the multimodal content
        print("\n💾 Storing multimodal content...")
        context_id = store.store_multimodal_content(multimodal_content)
        print(f"✅ Stored with context ID: {context_id}")

        # Store the same image again to demonstrate deduplication
        print("\n🔄 Testing binary deduplication...")
        duplicate_image_part = types.Part(
            inline_data=types.Blob(data=image_data, mime_type="image/png")
        )
        context_id2 = store.store_multimodal_content(duplicate_image_part)
        print(f"✅ Stored duplicate image with context ID: {context_id2}")

        # Display storage statistics
        print("\n📊 Storage Statistics:")
        stats = store.get_cache_stats()
        print(f"   Total contexts: {stats['total_contexts']}")
        print(f"   Multimodal contexts: {stats['multimodal_contexts']}")
        print(f"   Binary objects: {stats['total_binary_objects']}")
        print(f"   Binary size: {stats['total_binary_size_bytes']:,} bytes")

        multimodal_stats = store.get_multimodal_stats()
        print(f"   Memory binaries: {multimodal_stats['memory_stored_binaries']}")
        print(f"   Disk binaries: {multimodal_stats['disk_stored_binaries']}")
        print(
            f"   Deduplication ratio: {multimodal_stats['binary_deduplication_ratio']:.2f}"
        )

        # Retrieve and verify the content
        print("\n🔍 Retrieving multimodal content...")
        retrieved_content = store.retrieve_multimodal_content(context_id)

        print(f"✅ Retrieved content with {len(retrieved_content.parts)} parts:")
        for i, part in enumerate(retrieved_content.parts):
            if part.text:
                print(f"   Part {i+1}: Text ({len(part.text)} chars)")
            elif part.inline_data:
                print(
                    f"   Part {i+1}: Binary ({part.inline_data.mime_type}, {len(part.inline_data.data):,} bytes)"
                )
            elif part.file_data:
                print(
                    f"   Part {i+1}: File ({part.file_data.mime_type}, {part.file_data.file_uri})"
                )

        # Demonstrate efficient serialization
        print("\n⚡ Efficiency Demonstration:")
        print("   Traditional approach (base64 in JSON):")

        # Simulate traditional approach
        import base64
        import json

        traditional_data = {
            "text": text_part.text,
            "image": base64.b64encode(image_data).decode("utf-8"),
            "audio": base64.b64encode(audio_data).decode("utf-8"),
        }
        traditional_json = json.dumps(traditional_data)
        print(f"     JSON size: {len(traditional_json):,} bytes")

        print("   Reference-based approach:")
        context_json = store._contexts[context_id]
        print(f"     JSON size: {len(context_json):,} bytes")
        print(
            f"     Size reduction: {len(traditional_json) / len(context_json):.1f}x smaller"
        )

        # Clean up demonstration
        print("\n🧹 Cleaning up...")
        store.clear_multimodal_cache()
        final_stats = store.get_cache_stats()
        print(f"   Binary objects after cleanup: {final_stats['total_binary_objects']}")

        print("\n🎉 Example completed successfully!")


if __name__ == "__main__":
    main()
