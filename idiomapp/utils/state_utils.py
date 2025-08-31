import logging
import hashlib
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
import streamlit as st

logger = logging.getLogger(__name__)

class AppState(Enum):
    """Application states for managing the Streamlit app lifecycle"""
    INITIALIZING = "initializing"
    READY = "ready"
    TRANSLATING = "translating"
    ERROR = "error"
    CONFIG_CHANGING = "config_changing"
    LOADING = "loading"
    PROCESSING = "processing"

@dataclass
class AppStateData:
    """Data structure for application state"""
    state: AppState
    llm_provider: str
    model_name: str
    error_message: Optional[str] = None
    last_config_hash: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class StateManager:
    """Centralized state management for Streamlit applications"""
    
    def __init__(self, app_name: str = "default"):
        self.app_name = app_name
        self.state_key = f"{app_name}_app_state"
        self.cache_key = f"{app_name}_cache"
    
    def get_app_state(self) -> AppStateData:
        """Get current application state from session state"""
        if self.state_key not in st.session_state:
            # Initialize with default state
            from idiomapp.config import settings
            st.session_state[self.state_key] = AppStateData(
                state=AppState.INITIALIZING,
                llm_provider=settings.llm_provider.value,
                model_name=settings.current_model,
                last_config_hash=self._generate_config_hash(settings.llm_provider.value, settings.current_model)
            )
        return st.session_state[self.state_key]
    
    def transition_state(self, new_state: AppState, **kwargs) -> bool:
        """Transition to a new application state"""
        current = self.get_app_state()
        st.session_state[self.state_key] = AppStateData(
            state=new_state,
            llm_provider=kwargs.get("llm_provider", current.llm_provider),
            model_name=kwargs.get("model_name", current.model_name),
            error_message=kwargs.get("error_message"),
            last_config_hash=kwargs.get("last_config_hash", current.last_config_hash),
            metadata=kwargs.get("metadata", current.metadata)
        )
        logger.info(f"State transition: {current.state} -> {new_state}")
        return True
    
    def _generate_config_hash(self, provider: str, model: str) -> str:
        """Generate a hash for configuration to detect changes"""
        config_string = f"{provider}:{model}"
        return hashlib.md5(config_string.encode()).hexdigest()
    
    def has_config_changed(self, current_provider: str, current_model: str) -> bool:
        """Check if configuration has changed since last state update"""
        current_state = self.get_app_state()
        new_hash = self._generate_config_hash(current_provider, current_model)
        return current_state.last_config_hash != new_hash

def get_app_state():
    """Get current application state (convenience function)"""
    state_manager = StateManager()
    return state_manager.get_app_state()

def transition_state(new_state: AppState, **kwargs):
    """Transition to new state (convenience function)"""
    current = get_app_state()
    st.session_state["app_state"] = AppStateData(
        state=new_state,
        llm_provider=kwargs.get("llm_provider", current.llm_provider),
        model_name=kwargs.get("model_name", current.model_name),
        error_message=kwargs.get("error_message"),
        last_config_hash=kwargs.get("last_config_hash", current.last_config_hash)
    )
    logger.info(f"State transition: {current.state} -> {new_state}")
    return True

def get_cached_value(cache_key: str, cache_time_key: str, cache_duration: int, 
                    fetch_func: Callable, *args, **kwargs) -> Any:
    """
    Generic function to get a cached value with time-based expiration.
    
    Args:
        cache_key: Key for storing the cached value
        cache_time_key: Key for storing the cache timestamp
        cache_duration: Duration in seconds before cache expires
        fetch_func: Function to call to fetch the value if cache is invalid
        *args, **kwargs: Arguments to pass to fetch_func
        
    Returns:
        The cached or freshly fetched value
    """
    # Initialize cache if not exists
    if cache_key not in st.session_state:
        st.session_state[cache_key] = None
        st.session_state[cache_time_key] = 0
    
    current_time = time.time()
    # Check if cache is valid
    if (st.session_state[cache_key] is None or 
        current_time - st.session_state[cache_time_key] > cache_duration):
        
        # Fetch fresh value
        value = fetch_func(*args, **kwargs)
        st.session_state[cache_key] = value
        st.session_state[cache_time_key] = current_time
        logger.info(f"Updated cached value for {cache_key}")
        return value
    else:
        logger.info(f"Using cached value for {cache_key}")
        return st.session_state[cache_key]

def clear_cache(*cache_keys: str) -> None:
    """Clear specified cache keys from session state"""
    for key in cache_keys:
        if key in st.session_state:
            del st.session_state[key]
            logger.info(f"Cleared cache key: {key}")

def clear_all_caches() -> None:
    """Clear all cache-related keys from session state"""
    cache_keys = [key for key in st.session_state.keys() if 'cache' in key.lower()]
    for key in cache_keys:
        del st.session_state[key]
    logger.info(f"Cleared {len(cache_keys)} cache keys")

def safe_session_state_update(key: str, value: Any, force_update: bool = False) -> bool:
    """
    Safely update session state only if the value has changed.
    This prevents infinite loops in Streamlit.
    
    Args:
        key: Session state key to update
        value: New value to set
        force_update: If True, always update regardless of change
        
    Returns:
        True if updated, False if no change needed
    """
    if force_update or key not in st.session_state:
        st.session_state[key] = value
        return True
    elif st.session_state[key] != value:
        st.session_state[key] = value
        return True
    return False

def initialize_session_state(defaults: Dict[str, Any]) -> None:
    """
    Initialize session state with default values.
    
    Args:
        defaults: Dictionary of key-value pairs for session state initialization
    """
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            logger.debug(f"Initialized session state: {key} = {default_value}")

def get_session_state_safe(key: str, default: Any = None) -> Any:
    """
    Safely get a value from session state with a default fallback.
    
    Args:
        key: Session state key to retrieve
        default: Default value if key doesn't exist
        
    Returns:
        The session state value or default
    """
    return st.session_state.get(key, default)

def set_session_state_safe(key: str, value: Any) -> None:
    """
    Safely set a value in session state.
    
    Args:
        key: Session state key to set
        value: Value to set
    """
    st.session_state[key] = value

def reset_application_state() -> None:
    """Reset the application to initial state"""
    # Clear all session state except essential keys
    essential_keys = {'show_help_page'}  # Add any keys that should persist
    
    keys_to_remove = [key for key in st.session_state.keys() if key not in essential_keys]
    for key in keys_to_remove:
        del st.session_state[key]
    
    logger.info("Application state reset complete")

def validate_session_state(required_keys: list) -> bool:
    """
    Validate that required session state keys exist and have valid values.
    
    Args:
        required_keys: List of required session state keys
        
    Returns:
        True if all required keys exist and have non-None values
    """
    for key in required_keys:
        if key not in st.session_state or st.session_state[key] is None:
            logger.warning(f"Missing or invalid session state key: {key}")
            return False
    return True

def get_state_summary() -> Dict[str, Any]:
    """
    Get a summary of current session state for debugging.
    
    Returns:
        Dictionary containing session state summary
    """
    summary = {
        'total_keys': len(st.session_state),
        'cache_keys': len([k for k in st.session_state.keys() if 'cache' in k.lower()]),
        'state_keys': len([k for k in st.session_state.keys() if 'state' in k.lower()]),
        'app_state': get_app_state().state.value if 'app_state' in st.session_state else 'Not initialized'
    }
    return summary

def get_llm_client():
    """
    Get or create the LLM client based on current session state.
    Returns the cached client if available, otherwise creates a new one.
    """
    # Check if we have a cached client
    if "llm_client" in st.session_state and st.session_state["llm_client"] is not None:
        # Check if the cached client matches current provider/model
        from idiomapp.config import settings
        current_provider = st.session_state.get("llm_provider", settings.llm_provider.value)
        current_model = st.session_state.get("model_name", settings.current_model)
        
        try:
            # Get client status to check if it's still valid
            client_status = st.session_state["llm_client"].get_model_status()
            cached_provider = client_status.get("provider", "unknown")
            cached_model = client_status.get("model_name", "unknown")
            
            # If provider or model changed, we need a new client
            if cached_provider != current_provider or cached_model != current_model:
                logger.info(f"Provider/model changed, clearing cached client: {cached_provider}:{cached_model} -> {current_provider}:{current_model}")
                st.session_state["llm_client"] = None
            else:
                logger.info(f"Using cached LLM client instance for {current_provider}:{current_model}")
                return st.session_state["llm_client"]
        except Exception as e:
            logger.warning(f"Error checking cached client status: {str(e)}")
            st.session_state["llm_client"] = None
    
    # Create new client
    try:
        from idiomapp.config import settings
        from idiomapp.utils.llm_utils import LLMClient
        
        current_provider = st.session_state.get("llm_provider", settings.llm_provider.value)
        current_model = st.session_state.get("model_name", settings.current_model)
        
        # Get API key and organization from session state if using OpenAI
        api_key = None
        organization = None
        if current_provider == "openai":
            api_key = st.session_state.get("openai_api_key")
            organization = st.session_state.get("openai_organization")
        
        logger.info(f"Creating new LLM client for {current_provider}:{current_model}")
        client = LLMClient.create(
            provider=current_provider, 
            model_name=current_model,
            api_key=api_key,
            organization=organization
        )
        
        # Cache the client
        st.session_state["llm_client"] = client
        return client
        
    except Exception as e:
        logger.error(f"Error creating LLM client: {str(e)}")
        return None
