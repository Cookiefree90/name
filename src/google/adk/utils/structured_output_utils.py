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

"""Utility functions for handling structured output."""

import inspect
import json
from typing import Dict, Any, Optional, Type, get_type_hints, get_origin, get_args


def to_function_schema(func: callable) -> Dict[str, Any]:
    """
    Convert a Python function to a function schema for LLM function calling.

    Args:
        func: The function to convert.

    Returns:
        A dictionary representing the function schema.
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    schema = {
        "name": func.__name__,
        "description": func.__doc__ or "",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    for param_name, param in sig.parameters.items():
        if param_name == "self" or param_name == "cls":
            continue

        param_type = type_hints.get(param_name, Any)
        param_schema = _type_to_schema(param_type)

        schema["parameters"]["properties"][param_name] = {
            "type": param_schema.get("type", "string"),
            "description": "",
        }

        # Add format if applicable
        if "format" in param_schema:
            schema["parameters"]["properties"][param_name]["format"] = param_schema[
                "format"
            ]

        # Add enum if applicable
        if "enum" in param_schema:
            schema["parameters"]["properties"][param_name]["enum"] = param_schema[
                "enum"
            ]

        # Add required parameters
        if param.default == inspect.Parameter.empty:
            schema["parameters"]["required"].append(param_name)

    return schema


def _type_to_schema(type_hint: Type) -> Dict[str, Any]:
    """
    Convert a Python type hint to a JSON schema type.

    Args:
        type_hint: The type hint to convert.

    Returns:
        A dictionary with the JSON schema type.
    """
    if type_hint == str:
        return {"type": "string"}
    elif type_hint == int:
        return {"type": "integer"}
    elif type_hint == float:
        return {"type": "number"}
    elif type_hint == bool:
        return {"type": "boolean"}
    elif type_hint == dict or get_origin(type_hint) == dict:
        return {"type": "object"}
    elif type_hint == list or get_origin(type_hint) == list:
        return {"type": "array"}
    else:
        return {"type": "string"}


def input_or_function_schema_to_signature(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an input schema or function schema to an LLM function signature.

    Args:
        schema: The schema to convert.

    Returns:
        A dictionary representing the function signature.
    """
    if "name" in schema and "parameters" in schema:
        # Already a function schema
        return schema

    # Convert input schema to function schema
    return {
        "name": "input",
        "description": schema.get("description", "Input schema"),
        "parameters": schema,
    }


def typescript_schema_to_pydantic(schema: str) -> str:
    """
    Convert a TypeScript schema to a Pydantic model.

    Args:
        schema: The TypeScript schema as a string.

    Returns:
        A string containing the Pydantic model code.
    """
    # Very simplified implementation
    # A real implementation would parse and convert TypeScript types
    schema = schema.replace("interface", "class")
    schema = schema.replace("string", "str")
    schema = schema.replace("number", "float")
    schema = schema.replace("boolean", "bool")
    schema = schema.replace(";", "")

    # Add Pydantic imports
    pydantic_schema = "from pydantic import BaseModel, Field\n\n"

    # Add BaseModel inheritance
    pydantic_schema += schema.replace("class ", "class ", 1).replace(
        ":", "(BaseModel):", 1
    )

    return pydantic_schema
