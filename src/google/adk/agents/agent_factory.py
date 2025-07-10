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

"""Centralized factory for building agents from configurations."""

from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from typing import Union

import yaml

from .configs.agents_config import AgentConfig
from .configs.agents_config import AgentsConfig
from .configs.base_agent_config import BaseAgentConfig
from .configs.common_config import CodeConfig
from .configs.common_config import SubAgentConfig
from .configs.custom_agent_config import CustomAgentConfig
from .configs.llm_agent_config import LlmAgentConfig
from .configs.loop_agent_config import LoopAgentConfig

# Import here to avoid circular dependencies in the typing
if TYPE_CHECKING:
  from .base_agent import BaseAgent

logger = logging.getLogger('google_adk.' + __name__)

# Built-in tool name mappings
BUILTIN_TOOLS = {
    'google_search': 'google.adk.tools.google_search',
    'load_memory': 'google.adk.tools.load_memory',
}


class AgentFactory:
  """Centralized factory for building agents from configurations."""

  _config_registry: Dict[str, Type] = {}
  """Registry mapping agent type names to their config classes."""

  _agent_registry: Dict[str, Type] = {}
  """Registry mapping agent type names to their agent classes."""

  @classmethod
  def from_config(
      cls,
      config: Union[AgentsConfig, AgentConfig, str, Path],
  ) -> BaseAgent:
    """Build agent from config object or file path.

    Args:
      config: Either a config object, or a path to a YAML config file.

    Returns:
      The created root agent instance. If multiple agents are defined, returns the
      first agent as root with others connected as sub-agents.

    Raises:
      FileNotFoundError: If config file path doesn't exist.
      ValueError: If agent type is unknown or config is invalid.
    """
    if isinstance(config, (str, Path)):
      config = cls._load_config_from_path(config)

    if isinstance(config, AgentsConfig):
      # Multiple agents defined - first agent is root, others become sub-agents
      if not config.agents:
        raise ValueError('AgentsConfig must contain at least one agent')

      # Build the root agent (by default the first in the list)
      root_agent = cls._build_agent_from_agent_config(config.agents[0])

      # If there are additional agents, they should be connected as sub-agents.
      # This would be handled through the sub_agents field in the config.
      return root_agent

    elif isinstance(config, AgentConfig):
      # Single agent with type
      return cls._build_agent_from_agent_config(config)
    else:
      raise ValueError('Invalid config type')

  @classmethod
  def _load_config_from_path(
      cls, config_path: Union[str, Path]
  ) -> Union[AgentsConfig, AgentConfig]:
    """Load configuration from YAML file.

    Args:
      config_path: Path to the YAML config file.

    Returns:
      The loaded and validated config object.

    Raises:
      FileNotFoundError: If config file doesn't exist.
      ValueError: If config is invalid or agent type unknown.
    """
    config_path = Path(config_path)

    if not config_path.exists():
      raise FileNotFoundError(f'Config file not found: {config_path}')

    with open(config_path, 'r', encoding='utf-8') as f:
      config_data = yaml.safe_load(f)

    # Check if this is the new schema with top-level agents list
    if 'agents' in config_data and isinstance(config_data['agents'], list):
      return AgentsConfig.model_validate(config_data)

    # Check if this is a single agent config with type field
    if 'type' in config_data and 'config' in config_data:
      return AgentConfig.model_validate(config_data)

    raise ValueError('Could not determine config type from data structure')

  @classmethod
  def _build_agent_from_agent_config(
      cls, agent_config: AgentConfig
  ) -> BaseAgent:
    """Build agent from AgentConfig object.

    Args:
      agent_config: The agent configuration object with type and config.

    Returns:
      The created agent instance.
    """
    # Get the actual config object
    config = agent_config.config

    # 1. Resolve agent class
    agent_class = cls._resolve_agent_class(agent_config)

    # 2. Build constructor arguments
    kwargs = cls._build_agent_kwargs(config)
    logger.info(f'WULIANG Agent kwargs: {kwargs}')

    # 3. Create agent instance
    agent = agent_class(**kwargs)
    logger.info(f'WULIANG Agent: {agent}')

    # 4. Build and attach sub-agents recursively
    if config.sub_agents:
      for sub_config in config.sub_agents:
        sub_agent = cls._build_sub_agent(sub_config)
        sub_agent.parent_agent = agent
        agent.sub_agents.append(sub_agent)
    return agent

  @classmethod
  def _resolve_agent_class(cls, agent_config: AgentConfig) -> Type[BaseAgent]:
    """Resolve agent class.

    Returns:
      The resolved agent class.
    """
    agent_type = agent_config.type
    if agent_type in cls._agent_registry:
      return cls._agent_registry[agent_type]
    elif agent_type == 'CustomAgent':
      return cls._resolve_custom_agent_class(agent_config)
    else:
      raise ValueError(f'Unknown agent type: {agent_type}')

  @classmethod
  def _resolve_custom_agent_class(
      cls, agent_config: AgentConfig
  ) -> Type[BaseAgent]:
    # TODO: Add support for custom agent types.
    """Resolve custom agent class.

    Returns:
      The resolved custom agent class.
    """
    raise NotImplementedError('Custom agent type is not supported yet')

  @classmethod
  def _build_agent_kwargs(cls, config: BaseAgentConfig) -> Dict[str, Any]:
    """Build constructor arguments for agent.

    Args:
      config: The agent configuration object.

    Returns:
      Dictionary of constructor arguments.
    """
    kwargs: Dict[str, Any] = {
        'name': config.name,
        'description': config.description,
    }

    # Handle BaseAgent callbacks
    if config.before_agent_callbacks:
      kwargs['before_agent_callback'] = cls._resolve_callbacks(
          config.before_agent_callbacks
      )
    if config.after_agent_callbacks:
      kwargs['after_agent_callback'] = cls._resolve_callbacks(
          config.after_agent_callbacks
      )

    # Handle LlmAgent specific fields
    if isinstance(config, LlmAgentConfig):
      if config.model:
        kwargs['model'] = config.model
      if config.instruction:
        kwargs['instruction'] = config.instruction
      if config.tools:
        kwargs['tools'] = cls._resolve_tools(config.tools)
      if config.input_schema:
        kwargs['input_schema'] = cls._resolve_code_reference(
            config.input_schema
        )
      if config.output_schema:
        kwargs['output_schema'] = cls._resolve_code_reference(
            config.output_schema
        )
      if config.disallow_transfer_to_parent is not None:
        kwargs['disallow_transfer_to_parent'] = (
            config.disallow_transfer_to_parent
        )
      if config.disallow_transfer_to_peers is not None:
        kwargs['disallow_transfer_to_peers'] = config.disallow_transfer_to_peers
      if config.include_contents != 'default':
        kwargs['include_contents'] = config.include_contents
      if config.output_key:
        kwargs['output_key'] = config.output_key

      # Handle LlmAgent callbacks
      if config.before_model_callbacks:
        kwargs['before_model_callback'] = cls._resolve_callbacks(
            config.before_model_callbacks
        )
      if config.after_model_callbacks:
        kwargs['after_model_callback'] = cls._resolve_callbacks(
            config.after_model_callbacks
        )
      if config.before_tool_callbacks:
        kwargs['before_tool_callback'] = cls._resolve_callbacks(
            config.before_tool_callbacks
        )
      if config.after_tool_callbacks:
        kwargs['after_tool_callback'] = cls._resolve_callbacks(
            config.after_tool_callbacks
        )

    # Handle LoopAgent specific fields
    elif isinstance(config, LoopAgentConfig):
      if config.max_iterations is not None:
        kwargs['max_iterations'] = config.max_iterations  # type: ignore

    # Handle CustomAgent specific fields
    elif isinstance(config, CustomAgentConfig):
      # For custom agents, we need to load the agent from the specified path
      if config.path:
        kwargs['agent'] = cls._resolve_code_reference(
            CodeConfig(name=config.path, args=config.args)
        )

    return kwargs

  @classmethod
  def _build_sub_agent(cls, sub_config: SubAgentConfig) -> BaseAgent:
    """Build a sub-agent from configuration.

    Args:
      sub_config: The sub-agent configuration (SubAgentConfig).

    Returns:
      The created sub-agent instance.
    """
    if sub_config.code:
      # Agent defined in code
      return cls._resolve_code_reference(sub_config.code)
    elif sub_config.config:
      # Agent defined in config file
      return cls.from_config(sub_config.config)
    else:
      raise ValueError("SubAgentConfig must have either 'code' or 'config'")

  @classmethod
  def _resolve_tools(cls, tools_config: List[CodeConfig]) -> List[Any]:
    """Resolve tools from configuration.

    Args:
      tools_config: List of tool configurations (CodeConfig objects).

    Returns:
      List of resolved tool objects.
    """
    return [
        cls._resolve_code_reference(tool_config) for tool_config in tools_config
    ]

  @classmethod
  def _resolve_callbacks(cls, callbacks_config: List[CodeConfig]) -> Any:
    """Resolve callbacks from configuration.

    Args:
      callbacks_config: List of callback configurations (CodeConfig objects).

    Returns:
      List of resolved callback objects.
    """
    callbacks = [
        cls._resolve_code_reference(config) for config in callbacks_config
    ]
    if not callbacks:
      return None
    elif len(callbacks) == 1:
      return callbacks[0]
    else:
      return callbacks

  @classmethod
  def _resolve_code_reference(cls, code_config: CodeConfig) -> Any:
    """Resolve a code reference to actual Python object.

    Args:
      code_config: The code configuration (CodeConfig).

    Returns:
      The resolved Python object.

    Raises:
      ValueError: If the code reference cannot be resolved.
    """
    if not code_config or not code_config.name:
      raise ValueError('CodeConfig must have a name')

    try:
      name = BUILTIN_TOOLS.get(code_config.name, code_config.name)

      # Try to import and get the object
      if '.' in name:
        module_path, obj_name = name.rsplit('.', 1)
        module = importlib.import_module(module_path)
        obj = getattr(module, obj_name)
      else:
        raise ValueError(f'Cannot resolve simple name: {name}')

      # Call with arguments if provided
      if code_config.args and callable(obj):
        kwargs = {arg.name: arg.value for arg in code_config.args if arg.name}
        positional_args = [
            arg.value for arg in code_config.args if not arg.name
        ]

        result = obj(*positional_args, **kwargs)
        if inspect.isawaitable(result):
          result = result
        return result

      return obj

    except (ValueError, ImportError, AttributeError) as e:
      raise ValueError(
          f'Could not resolve code reference: {code_config.name}'
      ) from e

  @classmethod
  def _register_agent_type(
      cls,
      name: str,
      agent_class: Type,
      config_class: Type,
  ):
    """Register a new agent type.

    Args:
      name: The name of the agent type (e.g., 'LlmAgent').
      agent_class: The agent class (e.g., LlmAgent).
      config_class: The config class (e.g., LlmAgentConfig).
    """
    cls._agent_registry[name] = agent_class
    cls._config_registry[name] = config_class
    logger.debug(f'Registered agent type: {name}')


# Auto-register built-in agent types when module is imported
def _register_builtin_types():
  """Register built-in agent and config types."""
  from .llm_agent import LlmAgent
  from .loop_agent import LoopAgent

  AgentFactory._register_agent_type('LlmAgent', LlmAgent, LlmAgentConfig)
  AgentFactory._register_agent_type('LoopAgent', LoopAgent, LoopAgentConfig)


# Register built-in types on module import
_register_builtin_types()
