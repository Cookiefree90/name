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
import os
import sys

from google.adk import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService

# Add the parent directory to the path so we can import the agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keras_hub_agent.agent import keras_hub_agent


async def main():
  """Main function to run the KerasHub agent example."""

  # Set Keras backend (optional - defaults to TensorFlow)
  os.environ["KERAS_BACKEND"] = "tensorflow"

  print("🚀 Starting KerasHub Agent Example")
  print("=" * 50)
  print("This example demonstrates using KerasHub for local model inference")
  print("with Google ADK. The agent will run completely offline.")
  print()

  # Create a runner with the KerasHub agent
  runner = Runner(
      agent=keras_hub_agent, session_service=InMemorySessionService()
  )

  # Example conversation
  test_queries = [
      "Hello! Can you roll a 6-sided die for me?",
      "Now can you check if the number you rolled is prime?",
      "Can you roll a 20-sided die and then check if it's prime?",
  ]

  for i, query in enumerate(test_queries, 1):
    print(f"\n📝 Query {i}: {query}")
    print("-" * 50)

    try:
      # Run the agent
      response_events = runner.run(query)

      # Process the response
      final_answer = None
      async for event in response_events:
        if event.final:
          final_answer = event.content.parts[0].text
          break

      if final_answer:
        print(f"🤖 Agent Response: {final_answer}")
      else:
        print("❌ No response received from agent")

    except Exception as e:
      print(f"❌ Error running agent: {e}")
      print("This might be due to:")
      print("- KerasHub not being installed")
      print("- Model download issues")
      print("- Memory constraints")
      print()
      print("To install KerasHub: pip install keras-hub")
      break

  print("\n" + "=" * 50)
  print("✅ KerasHub Agent Example Complete!")
  print("\nKey Benefits of KerasHub Integration:")
  print("• Runs completely offline - no API calls required")
  print("• Privacy-focused - all computation happens locally")
  print("• Cost-effective - no per-token charges")


if __name__ == "__main__":
  asyncio.run(main())
