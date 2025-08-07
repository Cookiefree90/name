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

from __future__ import annotations

import logging
from typing import Optional

from ..agents.callback_context import CallbackContext
from ..utils.feature_decorator import experimental
from .auth_credential import AuthCredential
from .auth_credential import AuthCredentialTypes
from .auth_schemes import AuthSchemeType
from .auth_tool import AuthConfig
from .exchanger.base_credential_exchanger import BaseCredentialExchanger
from .exchanger.credential_exchanger_registry import CredentialExchangerRegistry
from .refresher.base_credential_refresher import BaseCredentialRefresher
from .refresher.credential_refresher_registry import CredentialRefresherRegistry

logger = logging.getLogger("google_adk." + __name__)


@experimental
class CredentialManager:
  """Manages authentication credentials through a structured workflow.

  The CredentialManager orchestrates the complete lifecycle of authentication
  credentials, from initial loading to final preparation for use. It provides
  a centralized interface for handling various credential types and authentication
  schemes while maintaining proper credential hygiene (refresh, exchange, caching).

  This class is only for use by Agent Development Kit.

  Args:
      auth_config: Configuration containing authentication scheme and credentials

  Example:
      ```python
      auth_config = AuthConfig(
          auth_scheme=oauth2_scheme,
          raw_auth_credential=service_account_credential
      )
      manager = CredentialManager(auth_config)

      # Register custom exchanger if needed
      manager.register_credential_exchanger(
          AuthCredentialTypes.CUSTOM_TYPE,
          CustomCredentialExchanger()
      )

      # Register custom refresher if needed
      manager.register_credential_refresher(
          AuthCredentialTypes.CUSTOM_TYPE,
          CustomCredentialRefresher()
      )

      # Load and prepare credential
      credential = await manager.load_auth_credential(callback_context)
      ```
  """

  def __init__(
      self,
      auth_config: AuthConfig,
  ):
    self._auth_config = auth_config
    self._exchanger_registry = CredentialExchangerRegistry()
    self._refresher_registry = CredentialRefresherRegistry()

    # Register default exchangers and refreshers
    # Register OAuth2 exchanger for client credentials and authorization code flows
    from .exchanger.oauth2_credential_exchanger import OAuth2CredentialExchanger
    from .refresher.oauth2_credential_refresher import OAuth2CredentialRefresher

    oauth2_exchanger = OAuth2CredentialExchanger()
    self._exchanger_registry.register(AuthCredentialTypes.OAUTH2, oauth2_exchanger)
    self._exchanger_registry.register(AuthCredentialTypes.OPEN_ID_CONNECT, oauth2_exchanger)

    oauth2_refresher = OAuth2CredentialRefresher()
    self._refresher_registry.register(
        AuthCredentialTypes.OAUTH2, oauth2_refresher
    )
    self._refresher_registry.register(
        AuthCredentialTypes.OPEN_ID_CONNECT, oauth2_refresher
    )

    # TODO: support service account credential exchanger

  def register_credential_exchanger(
      self,
      credential_type: AuthCredentialTypes,
      exchanger_instance: BaseCredentialExchanger,
  ) -> None:
    """Register a credential exchanger for a credential type.

    Args:
        credential_type: The credential type to register for.
        exchanger_instance: The exchanger instance to register.
    """
    self._exchanger_registry.register(credential_type, exchanger_instance)

  async def request_credential(self, callback_context: CallbackContext) -> None:
    callback_context.request_credential(self._auth_config)

  async def get_auth_credential(
      self, callback_context: CallbackContext, verify_ssl: bool = True
  ) -> Optional[AuthCredential]:
    """Load and prepare authentication credential through a structured workflow.
    
    Args:
        callback_context: The callback context for credential operations.
        verify_ssl: Whether to verify SSL certificates during OAuth operations (default: True).
            Set to False for self-signed certificates.
    
    Returns:
        The prepared authentication credential, or None if unavailable.
    """
    
    logger.debug("🔄 CredentialManager.get_auth_credential() called")

    # Step 1: Validate credential configuration
    logger.debug("🔍 Step 1: Validating credential configuration")
    await self._validate_credential()
    logger.debug("✅ Step 1: Credential validation passed")

    # Step 2: Check if credential is already ready (no processing needed)
    logger.debug("🔍 Step 2: Checking if credential is already ready")
    if self._is_credential_ready():
      logger.debug("✅ Step 2: Credential is ready, returning raw credential")
      return self._auth_config.raw_auth_credential
    logger.debug("✅ Step 2: Credential needs processing")

    # Step 3: Try to load existing processed credential
    logger.debug("🔍 Step 3: Loading existing processed credential")
    credential = await self._load_existing_credential(callback_context)
    if credential:
      logger.debug("✅ Step 3: Found existing credential")
      if credential.oauth2 and credential.oauth2.access_token:
        logger.debug("✅ Existing credential has access_token, skipping exchange")
      else:
        logger.debug("⚠️ Existing credential has no access_token")
    else:
      logger.debug("✅ Step 3: No existing credential found")

    # Step 4: If no existing credential, load from auth response
    # TODO instead of load from auth response, we can store auth response in
    # credential service.
    was_from_auth_response = False
    if not credential:
      logger.debug("🔍 Step 4: Loading from auth response")
      credential = await self._load_from_auth_response(callback_context)
      if credential:
        logger.debug("✅ Step 4: Found credential from auth response")
        was_from_auth_response = True
      else:
        logger.debug("✅ Step 4: No credential from auth response")

    # Step 5: If still no credential available, return None
    if not credential:
      # For OAuth2 client credentials, fallback to raw credential
      if (self._auth_config.raw_auth_credential and 
          self._auth_config.raw_auth_credential.auth_type == AuthCredentialTypes.OAUTH2):
        logger.debug("✅ Step 5: Using raw OAuth2 credential for client credentials flow")
        credential = self._auth_config.raw_auth_credential
      else:
        logger.debug("❌ Step 5: No credential available, returning None")
        return None
    logger.debug("✅ Step 5: Credential available, proceeding to exchange")

    # Step 6: Exchange credential if needed (e.g., service account to access token)
    logger.debug("🔍 Step 6: Starting credential exchange")
    credential, was_exchanged = await self._exchange_credential(credential, verify_ssl)
    logger.debug(f"✅ Step 6: Exchange completed, was_exchanged={was_exchanged}")

    # Step 7: Refresh credential if expired
    logger.debug("🔍 Step 7: Checking if refresh needed")
    was_refreshed = False
    if not was_exchanged:
      credential, was_refreshed = await self._refresh_credential(credential)
      logger.debug(f"✅ Step 7: Refresh completed, was_refreshed={was_refreshed}")
    else:
      logger.debug("✅ Step 7: Skipping refresh since credential was exchanged")

    # Step 8: Save credential if it was modified
    if was_from_auth_response or was_exchanged or was_refreshed:
      logger.debug("🔍 Step 8: Saving modified credential")
      await self._save_credential(callback_context, credential)
      logger.debug("✅ Step 8: Credential saved")

    return credential

  async def _load_existing_credential(
      self, callback_context: CallbackContext
  ) -> Optional[AuthCredential]:
    """Load existing credential."""
    # First try to load from credential service
    credential = await self._load_from_credential_service(callback_context)
    if credential:
      return credential

    # Then try to load from context
    if hasattr(callback_context, "_auth_credential") and callback_context._auth_credential:
      return callback_context._auth_credential

    return None

  async def _load_from_credential_service(
      self, callback_context: CallbackContext
  ) -> Optional[AuthCredential]:
    """Load credential from credential service if available."""
    credential_service = callback_context._invocation_context.credential_service
    if credential_service:
      # Note: This should be made async in a future refactor
      # For now, assuming synchronous operation
      return await callback_context.load_credential(self._auth_config)
    return None

  async def _load_from_auth_response(
      self, callback_context: CallbackContext
  ) -> Optional[AuthCredential]:
    """Load credential from auth response in callback context."""
    return callback_context.get_auth_response(self._auth_config)

  async def _exchange_credential(
      self, credential: AuthCredential, verify_ssl: bool = True
  ) -> tuple[AuthCredential, bool]:
    """Exchange credential if needed and return the credential and whether it was exchanged.
    
    Args:
        credential: The credential to exchange.
        verify_ssl: Whether to verify SSL certificates during OAuth operations (default: True).
        
    Returns:
        Tuple of (exchanged_credential, was_exchanged).
    """
    logger.debug(f"🔄 _exchange_credential called for credential type: {credential.auth_type}")
    
    exchanger = self._exchanger_registry.get_exchanger(credential.auth_type)
    if not exchanger:
      logger.debug(f"❌ No exchanger found for credential type: {credential.auth_type}")
      return credential, False

    logger.debug(f"✅ Found exchanger: {type(exchanger).__name__}")
    logger.debug("🚀 Calling exchanger.exchange()")
    
    # Check if exchanger supports verify_ssl parameter (OAuth2CredentialExchanger does)
    from .exchanger.oauth2_credential_exchanger import OAuth2CredentialExchanger
    if isinstance(exchanger, OAuth2CredentialExchanger):
      exchanged_credential = await exchanger.exchange(
          credential, self._auth_config.auth_scheme, verify_ssl
      )
    else:
      # Fallback for other exchangers that don't support verify_ssl
      exchanged_credential = await exchanger.exchange(
          credential, self._auth_config.auth_scheme
      )
    
    logger.debug("✅ Exchanger.exchange() completed")
    return exchanged_credential, True

  async def _refresh_credential(
      self, credential: AuthCredential
  ) -> tuple[AuthCredential, bool]:
    """Refresh credential if expired and return the credential and whether it was refreshed."""
    refresher = self._refresher_registry.get_refresher(credential.auth_type)
    if not refresher:
      return credential, False

    if await refresher.is_refresh_needed(
        credential, self._auth_config.auth_scheme
    ):
      refreshed_credential = await refresher.refresh(
          credential, self._auth_config.auth_scheme
      )
      return refreshed_credential, True

    return credential, False

  def _is_credential_ready(self) -> bool:
    """Check if credential is ready to use without further processing."""
    raw_credential = self._auth_config.raw_auth_credential
    if not raw_credential:
      return False

    # Simple credentials that don't need exchange or refresh
    return raw_credential.auth_type in (
        AuthCredentialTypes.API_KEY,
        AuthCredentialTypes.HTTP,
        # Add other simple auth types as needed
    )

  async def _validate_credential(self) -> None:
    """Validate credential configuration and raise errors if invalid."""
    if not self._auth_config.raw_auth_credential:
      if self._auth_config.auth_scheme.type_ in (
          AuthSchemeType.oauth2,
          AuthSchemeType.openIdConnect,
      ):
        raise ValueError(
            "raw_auth_credential is required for auth_scheme type "
            f"{self._auth_config.auth_scheme.type_}"
        )

    raw_credential = self._auth_config.raw_auth_credential
    if raw_credential:
      if (
          raw_credential.auth_type
          in (
              AuthCredentialTypes.OAUTH2,
              AuthCredentialTypes.OPEN_ID_CONNECT,
          )
          and not raw_credential.oauth2
      ):
        raise ValueError(
            "auth_config.raw_credential.oauth2 required for credential type "
            f"{raw_credential.auth_type}"
        )
        # Additional validation can be added here

  async def _save_credential(
      self, callback_context: CallbackContext, credential: AuthCredential
  ) -> None:
    """Save credential to credential service if available."""
    # Update the exchanged credential in config
    self._auth_config.exchanged_auth_credential = credential

    credential_service = callback_context._invocation_context.credential_service
    if credential_service:
      await callback_context.save_credential(self._auth_config)
