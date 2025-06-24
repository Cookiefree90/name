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

# --- Imports ---
# Vertex Agent Modules
from google.adk.agents import Agent # Base class for creating agents


# Other Python Modules
import warnings # For suppressing warnings
import logging # For controlling logging output
from dotenv import load_dotenv # For loading environment variables from a .env file


# Modules for Publishing Events to GCP Pub/Sub
import json # For working with JSON data, such as sending events to Pub/Sub
import uuid # For generating unique identifiers used for idempotency in event publishing
from typing import Optional # For type hinting optional parameters for gcp_project_id and pubsub_topic_id
from google.adk.tools import publish_event # For publishing events to Pub/Sub (not used in this script, but included for completeness)


# --- Configuration ---
load_dotenv()


# Ignore all warnings
warnings.filterwarnings("ignore")
# Set logging level to ERROR to suppress informational messages
logging.basicConfig(level=logging.INFO)


# --- Agent Tool Definitions ---

# Tool function to publish an event to GCP Pub/Sub
def publish_to_gcp_pubsub_tool(
    event_data_json: str,
    event_type: str = "custom_agent_event",
    gcp_project_id: Optional[str] = 'YOUR_GCP_PROJECT_ID', # If gcp_project_id is not provided, it will use the environment variable GOOGLE_CLOUD_PROJECT
    pubsub_topic_id: Optional[str] = 'YOUR_PUBSUB_TOPIC_ID' # If pubsub_topic_id is not provided, it will use the environment variable PUBSUB_TOPIC_ID
) -> str:
    """
    Publishes a structured event from the agent to a GCP Pub/Sub topic.
    The agent should provide the event data as a JSON string.
    Optionally, gcp_project_id and pubsub_topic_id can be provided to override environment settings.

    Args:
        event_data_json: A JSON string representing the structured data for the event.
        event_type: A string to categorize the event (e.g., 'decision_made', 'action_taken').
        gcp_project_id: The Google Cloud project ID. If not provided, uses the environment variable.
        pubsub_topic_id: The Pub/Sub topic ID. If not provided, uses the environment variable.
    """
    try:
        event_data = json.loads(event_data_json)
        event_data["app_message_id"] = str(uuid.uuid4())
        publish_event(
            event_data=event_data,
            event_type=event_type,
            gcp_project_id=gcp_project_id,
            pubsub_topic_id=pubsub_topic_id
        )
        return f"Event (type: {event_type}) with data '{event_data_json}' has been queued for publishing."
    except json.JSONDecodeError:
        return "Error: The provided event_data_json was not valid JSON."
    except Exception as e:
        print(f"[AgentTool ERROR] Error attempting to publish event to GCP Pub/Sub: {e}")
        return f"Error attempting to publish event to GCP Pub/Sub: {str(e)}"



# --- Root Agent Definition ---
# @title Define the Root Agent

# Initialize root agent variables
root_agent = None
runner_root = None # Initialize runner variable (although runner is created later)

    # Define the root agent (coordinator)
root_agent = Agent(
    name="root_support_agent",    # Name for the root agent
    model="gemini-2.5-flash", # Model for the root agent (orchestration)
    description="The main coordinator agent. Handles user requests and delegates tasks to specialist sub-agents and tools.", 
    instruction=                  # The core instructions defining the workflow
    """
        You are the lead support coordinator agent. Your goal is to understand the customer's question or topic and provide insightful answers.

        You have access to specialized tools and sub-agents:
        1. Tool `publish_to_gcp_pubsub_tool`: Use this tool to publish information to GCP Pub/Sub.
        
      

        Your workflow:
        1. You will be provided with a request from a user.
        2. Inform the user you will begin the research (e.g., "Okay, I'll start researching that for you. Please wait a moment.").
        3. For key findings, use `publish_to_gcp_pubsub_tool` with `event_type="key_finding_identified"` and `event_data_json` containing the finding details in JSON format.
        4. Provide a summary of the research findings to the user.

       
    """,
    tools=[
        publish_to_gcp_pubsub_tool, # Tool to publish events to GCP Pub/Sub
    ],
    sub_agents=[
    ],

)
