"""
Global configuration module for Idiomapp.

This module uses pydantic-settings to manage configuration from environment variables and defaults.
"""

import os
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable
from pydantic import Field, computed_field, model_validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define allowed LLM providers as an Enum for type safety
class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class IdiomaAppSettings(BaseSettings):
    """Global settings for the IdiomApp application."""
    
    # General configuration
    app_name: str = "IdiomApp"
    log_level: LogLevel = LogLevel.INFO
    
    # Streamlit configuration
    streamlit_server_port: int = 8503
    streamlit_server_headless: bool = True
    streamlit_server_enablecors: bool = False
    streamlit_server_enablexsrfprotection: bool = True
    streamlit_server_address: str = "0.0.0.0"
    streamlit_browser_gather_usage_stats: bool = False
    streamlit_ui_hide_sidebar_nav: bool = True
    streamlit_theme_base: str = "dark"
    streamlit_browser_server_address: str = "localhost"
    streamlit_client_toolbar_mode: str = "minimal"
    streamlit_client_show_error_details: bool = True
    streamlit_wide_mode: bool = True
    
    # LLM Provider configuration
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    
    # Ollama configuration
    ollama_host: str = "http://localhost:11434"
    default_model: str = "llama3.2:latest"
    
    # OpenAI configuration
    openai_api_key: str = ""
    openai_organization: str = ""
    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Controls randomness in OpenAI responses (0.0-2.0)")
    openai_max_tokens: int = Field(default=1024, ge=1, le=4096, description="Maximum tokens in OpenAI responses")
    
    # Model configurations by provider (will be parsed from comma-separated strings)
    ollama_models: str = "llama3.2:latest,llama3:latest,mistral:latest"
    openai_models: str = "gpt-3.5-turbo,gpt-4,gpt-4-turbo"
    
    # Language options (will be parsed from comma-separated strings)
    supported_languages: str = "en,es,ca"
    default_source_language: str = "en"
    default_target_languages: str = "es,ca"
    
    # Co-occurrence network defaults
    default_window_size: int = 2
    default_min_frequency: int = 1
    default_pos_filter: str = "NOUN,VERB,ADJ"
    
    @computed_field
    @property
    def ollama_models_list(self) -> List[str]:
        """Parse ollama_models from comma-separated string to list."""
        return [item.strip() for item in self.ollama_models.split(',') if item.strip()]
    
    @computed_field
    @property
    def openai_models_list(self) -> List[str]:
        """Parse openai_models from comma-separated string to list."""
        return [item.strip() for item in self.openai_models.split(',') if item.strip()]
    
    @computed_field
    @property
    def supported_languages_list(self) -> List[str]:
        """Parse supported_languages from comma-separated string to list."""
        return [item.strip() for item in self.supported_languages.split(',') if item.strip()]
    
    @computed_field
    @property
    def default_target_languages_list(self) -> List[str]:
        """Parse default_target_languages from comma-separated string to list."""
        return [item.strip() for item in self.default_target_languages.split(',') if item.strip()]
    
    @computed_field
    @property
    def default_pos_filter_list(self) -> List[str]:
        """Parse default_pos_filter from comma-separated string to list."""
        return [item.strip() for item in self.default_pos_filter.split(',') if item.strip()]
    
    @computed_field
    @property
    def current_model(self) -> str:
        """Determine the current model based on the selected provider."""
        if self.llm_provider == LLMProvider.OLLAMA:
            return self.default_model
        elif self.llm_provider == LLMProvider.OPENAI:
            return self.openai_model
        return self.default_model
    
    @computed_field
    @property
    def available_models(self) -> List[str]:
        """Return the available models for the current provider."""
        if self.llm_provider == LLMProvider.OLLAMA:
            return self.ollama_models_list
        elif self.llm_provider == LLMProvider.OPENAI:
            return self.openai_models_list
        return []
    
    @model_validator(mode='after')
    def validate_provider_specific_settings(self) -> 'IdiomaAppSettings':
        """Validate provider-specific settings."""
        if self.llm_provider == LLMProvider.OLLAMA:
            if not self.ollama_host:
                self.ollama_host = "http://localhost:11434"
            if not self.default_model:
                self.default_model = "llama3.2:latest"
        
        elif self.llm_provider == LLMProvider.OPENAI:
            if not self.openai_model:
                self.openai_model = "gpt-3.5-turbo"
        
        return self
    
    # Configure settings to use environment variables
    model_config = SettingsConfigDict(
        env_prefix="",  # Use environment variables without prefix
        env_file=".env",  # Read from .env file if it exists
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # Use double underscore for nested attributes
        extra="ignore",  # Ignore extra environment variables
        validate_default=True,  # Validate default values
    )


# Create a global settings instance
settings = IdiomaAppSettings()


# Helper functions for accessing settings
def get_settings() -> IdiomaAppSettings:
    """Return the global settings instance."""
    return settings

def get_model_by_provider(provider: str) -> str:
    """Return the appropriate model for the given provider."""
    if provider.lower() == LLMProvider.OLLAMA.value:
        return settings.default_model
    elif provider.lower() == LLMProvider.OPENAI.value:
        return settings.openai_model
    raise ValueError(f"Invalid LLM provider: {provider}")

def is_valid_provider(provider: str) -> bool:
    """Check if the given provider is valid."""
    return provider.lower() in [e.value for e in LLMProvider]


# Model capabilities configuration for OpenAI models
# This provides a centralized, maintainable way to configure model-specific behavior
MODEL_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    # GPT-5 models - strict parameter requirements
    "gpt-5": {
        "supports_max_completion_tokens": True,
        "supports_custom_temperature": False,
        "supports_custom_max_tokens": True,
        "description": "GPT-5 model with strict parameter requirements",
        "notes": "Only supports default temperature (1.0), uses max_completion_tokens"
    },
    "gpt-5-mini": {
        "supports_max_completion_tokens": True,
        "supports_custom_temperature": False,
        "supports_custom_max_tokens": True,
        "description": "GPT-5 mini model with strict parameter requirements",
        "notes": "Only supports default temperature (1.0), uses max_completion_tokens"
    },
    
    # GPT-4o models - modern parameter support
    "gpt-4o": {
        "supports_max_completion_tokens": True,
        "supports_custom_temperature": True,
        "supports_custom_max_tokens": True,
        "description": "GPT-4o model with modern parameter support",
        "notes": "Full parameter support including custom temperature"
    },
    "gpt-4o-mini": {
        "supports_max_completion_tokens": True,
        "supports_custom_temperature": True,
        "supports_custom_max_tokens": True,
        "description": "GPT-4o mini model with modern parameter support",
        "notes": "Full parameter support including custom temperature"
    },
    
    # Claude models
    "claude-3": {
        "supports_max_completion_tokens": True,
        "supports_custom_temperature": True,
        "supports_custom_max_tokens": True,
        "description": "Claude-3 model with modern parameter support",
        "notes": "Full parameter support including custom temperature"
    },
    "claude-3.5": {
        "supports_max_completion_tokens": True,
        "supports_custom_temperature": True,
        "supports_custom_max_tokens": True,
        "description": "Claude-3.5 model with modern parameter support",
        "notes": "Full parameter support including custom temperature"
    },
    
    # Standard GPT models
    "gpt-4": {
        "supports_max_completion_tokens": False,
        "supports_custom_temperature": True,
        "supports_custom_max_tokens": True,
        "description": "Standard GPT-4 model",
        "notes": "Uses max_tokens, supports custom temperature"
    },
    "gpt-3.5": {
        "supports_max_completion_tokens": False,
        "supports_custom_temperature": True,
        "supports_custom_max_tokens": True,
        "description": "Standard GPT-3.5 model",
        "notes": "Uses max_tokens, supports custom temperature"
    },
    
    # Legacy models
    "gpt-3": {
        "supports_max_completion_tokens": False,
        "supports_custom_temperature": True,
        "supports_custom_max_tokens": True,
        "description": "Legacy GPT-3 model",
        "notes": "Uses max_tokens, supports custom temperature"
    }
}

def get_model_capabilities(model_name: str) -> Dict[str, Any]:
    """
    Get capabilities for a specific model.
    
    Args:
        model_name: The name of the model to get capabilities for
        
    Returns:
        dict: Model capabilities including parameter support
    """
    # Try to find exact match first
    for model_pattern, capabilities in MODEL_CAPABILITIES.items():
        if model_pattern in model_name:
            return capabilities
    
    # Fallback: infer capabilities from model name patterns
    if any(prefix in model_name for prefix in ["gpt-5"]):
        return {
            "supports_max_completion_tokens": True,
            "supports_custom_temperature": False,
            "supports_custom_max_tokens": True,
            "description": "Inferred GPT-5 capabilities",
            "notes": "Inferred from model name pattern"
        }
    elif any(prefix in model_name for prefix in ["gpt-4o", "claude-3"]):
        return {
            "supports_max_completion_tokens": True,
            "supports_custom_temperature": True,
            "supports_custom_max_tokens": True,
            "description": "Inferred modern model capabilities",
            "notes": "Inferred from model name pattern"
        }
    else:
        # Default to legacy model capabilities
        return {
            "supports_max_completion_tokens": False,
            "supports_custom_temperature": True,
            "supports_custom_max_tokens": True,
            "description": "Default legacy model capabilities",
            "notes": "Default fallback for unknown models"
        }

def is_model_supported(model_name: str) -> bool:
    """
    Check if a model is supported by checking if we have capability information.
    
    Args:
        model_name: The name of the model to check
        
    Returns:
        bool: True if the model is supported
    """
    return any(pattern in model_name for pattern in MODEL_CAPABILITIES.keys())

def get_supported_models() -> list:
    """
    Get a list of all supported model patterns.
    
    Returns:
        list: List of supported model patterns
    """
    return list(MODEL_CAPABILITIES.keys()) 