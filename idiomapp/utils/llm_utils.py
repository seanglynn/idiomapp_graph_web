from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json

# For Ollama
import ollama

# For OpenAI
from openai import OpenAI
from openai.types.chat import ChatCompletion

from idiomapp.utils.logging_utils import get_logger
from idiomapp.config import settings, LLMProvider, get_model_capabilities
from idiomapp.utils.ollama_utils import get_valid_ollama_host, get_available_models, pull_model_if_needed

# Set up logging using the new cached logger
logger = get_logger("llm_utils")

class LLMClient(ABC):
    """Abstract base class for LLM clients (Ollama, OpenAI, etc.)"""
    
    @abstractmethod
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from a prompt"""
    
    @abstractmethod
    def get_model_status(self) -> Dict[str, Any]:
        """Get the status of the model"""
    
    @classmethod
    def create(cls, provider: str = None, model_name: str = None, api_key: str = None, organization: str = None) -> 'LLMClient':
        """Factory method to create the appropriate LLM client"""
        provider = provider or settings.llm_provider.value
        
        if provider == LLMProvider.OLLAMA.value:
            return OllamaClient(model_name or settings.default_model)
        elif provider == LLMProvider.OPENAI.value:
            return OpenAIClient(model_name or settings.openai_model, api_key, organization)
        else:
            logger.error(f"Unknown LLM provider: {provider}, falling back to Ollama")
            return OllamaClient(model_name or settings.default_model)


class OllamaClient(LLMClient):
    """Client for interacting with Ollama models."""
    
    # Class-level cache to prevent excessive model checks
    _model_available_cache = {}  # model_name -> bool
    
    def __init__(self, model_name=None):
        """
        Initialize the Ollama client.
        
        Args:
            model_name: The name of the model to use. If None, uses the DEFAULT_MODEL
                        from environment variables.
        """
        self.model_name = model_name or settings.default_model
        self.ollama_host = get_valid_ollama_host()
        
        # Set client configuration
        logger.info(f"Initializing Ollama client with model: {self.model_name}")
        logger.info(f"Using Ollama host: {self.ollama_host}")
        
        # Check if model is available
        self._check_model_availability()
    
    def _check_model_availability(self):
        """Check if the model is available, and try to pull it if not."""
        # Check cache first to avoid repeated API calls
        if self.model_name in self._model_available_cache:
            logger.info(f"Using cached model availability for {self.model_name}: {self._model_available_cache[self.model_name]}")
            return self._model_available_cache[self.model_name]
            
        logger.info(f"Checking availability of model: {self.model_name}")
        
        try:
            # Get list of available models
            available_models = get_available_models()
            model_available = self.model_name in available_models
            
            if model_available:
                logger.info(f"Model {self.model_name} is available")
                self._model_available_cache[self.model_name] = True
                return True
            else:
                logger.warning(f"Model {self.model_name} is not available, attempting to pull")
                pull_success = pull_model_if_needed(self.model_name)
                self._model_available_cache[self.model_name] = pull_success
                return pull_success
        except Exception as e:
            logger.error(f"Error checking model availability: {str(e)}")
            self._model_available_cache[self.model_name] = False
            return False
    
    def get_model_status(self):
        """
        Get the status of the Ollama model.
        
        Returns:
            dict: A dictionary with status information.
        """
        is_available = self._check_model_availability()
        return {
            "provider": LLMProvider.OLLAMA.value,
            "model_name": self.model_name,
            "available": is_available,
            "host": self.ollama_host
        }
    
    async def generate_text(self, prompt, system_prompt=None):
        """
        Generate text from the model.
        
        Args:
            prompt: The prompt to send to the model.
            system_prompt: Optional system prompt for context.
            
        Returns:
            str: The generated text response.
        """
        if not self._check_model_availability():
            logger.error(f"Model {self.model_name} is not available")
            return "Error: Model not available. Please check if Ollama is running and the model is installed."
            
        try:
            # Prepare the prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            logger.info(f"Generating text with Ollama model {self.model_name}")
            logger.debug(f"Prompt: {full_prompt[:100]}...")
            
            # Make API request
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            # Extract response text
            generated_text = response['message']['content'] or ""
            
            logger.info(f"Generated text length: {len(generated_text)}")
            logger.debug(f"Response: {generated_text[:100]}...")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Unexpected error generating text: {str(e)}")
            return f"Error: {str(e)}"


class OpenAIClient(LLMClient):
    """Client for interacting with OpenAI models."""
    
    def __init__(self, model_name=None, api_key: str = None, organization: str = None):
        """
        Initialize the OpenAI client.
        
        Args:
            model_name: The name of the model to use. If None, uses the OPENAI_MODEL
                        from settings.
            api_key: OpenAI API key. If None, uses the one from settings.
            organization: OpenAI organization ID. If None, uses the one from settings.
        """
        self.api_key = api_key or settings.openai_api_key
        self.organization = organization or settings.openai_organization
        self.model_name = model_name or settings.openai_model
        
        # Set client configuration
        logger.info(f"Initializing OpenAI client with model: {self.model_name}")
        if self.organization:
            logger.info(f"Using OpenAI organization: {self.organization}")
        
        # Initialize the OpenAI client
        if not self.api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
        else:
            client_kwargs = {"api_key": self.api_key}
            if self.organization:
                client_kwargs["organization"] = self.organization
            self.client = OpenAI(**client_kwargs)
    
    def get_model_status(self):
        """
        Get the status of the OpenAI model.
        
        Returns:
            dict: A dictionary with status information.
        """
        api_key_set = bool(self.api_key)
        
        return {
            "provider": LLMProvider.OPENAI.value,
            "model_name": self.model_name,
            "available": api_key_set,
            "api_key_set": api_key_set
        }
    
    async def generate_text(self, prompt, system_prompt=None):
        """
        Generate text from the OpenAI model.
        
        Args:
            prompt: The prompt to send to the model.
            system_prompt: Optional system prompt for context.
            
        Returns:
            str: The generated text response.
        """
        if not self.api_key:
            logger.error("OpenAI API key not set")
            return "Error: OpenAI API key not set. Please set the OPENAI_API_KEY environment variable."
            
        try:
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            # Add user prompt
            messages.append({"role": "user", "content": prompt})
            
            logger.info(f"Generating text with OpenAI model {self.model_name}")
            logger.debug(f"Prompt: {prompt[:100]}...")
            
            # Prepare API request parameters
            request_params = {
                "model": self.model_name,
                "messages": messages
            }
            
            # Get model capabilities and use appropriate parameters
            model_capabilities = get_model_capabilities(self.model_name)
            
            # Add token limit parameter based on model capabilities
            if model_capabilities.get("supports_max_completion_tokens", False):
                request_params["max_completion_tokens"] = settings.openai_max_tokens
                logger.debug(f"Using max_completion_tokens for model {self.model_name}")
            else:
                request_params["max_tokens"] = settings.openai_max_tokens
                logger.debug(f"Using max_tokens for model {self.model_name}")
            
            # Add temperature if supported by the model
            if model_capabilities.get("supports_custom_temperature", True):
                request_params["temperature"] = settings.openai_temperature
                logger.debug(f"Using custom temperature {settings.openai_temperature} for model {self.model_name}")
            else:
                logger.debug(f"Using default temperature for model {self.model_name}")
            
            # Make API request
            response: ChatCompletion = self.client.chat.completions.create(**request_params)
            
            # Extract response text
            generated_text = response.choices[0].message.content or ""
            
            logger.info(f"Generated text length: {len(generated_text)}")
            logger.debug(f"Response: {generated_text[:100]}...")
            
            return generated_text
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating text with OpenAI: {error_msg}")
            
            # Handle specific parameter errors and try fallback
            if "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
                logger.info(f"Attempting fallback with max_completion_tokens for model {self.model_name}")
                try:
                    # Try with max_completion_tokens instead
                    fallback_params = {
                        "model": self.model_name,
                        "messages": messages,
                        "max_completion_tokens": settings.openai_max_tokens
                    }
                    
                    # Only add temperature if it's supported by this model
                    model_capabilities = get_model_capabilities(self.model_name)
                    if model_capabilities.get("supports_custom_temperature", True):
                        fallback_params["temperature"] = settings.openai_temperature
                    
                    response: ChatCompletion = self.client.chat.completions.create(**fallback_params)
                    generated_text = response.choices[0].message.content or ""
                    
                    logger.info(f"Fallback successful with max_completion_tokens")
                    logger.info(f"Generated text length: {len(generated_text)}")
                    logger.debug(f"Response: {generated_text[:100]}...")
                    
                    return generated_text
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback attempt also failed: {str(fallback_error)}")

            raise e

    async def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a JSON response from the OpenAI model using structured output.

        Args:
            prompt: The prompt to send to the model.
            system_prompt: Optional system prompt for context.

        Returns:
            Dict: The parsed JSON response.
        """
        if not self.api_key:
            logger.error("OpenAI API key not set")
            return {"error": "OpenAI API key not set"}

        try:
            messages = []

            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add user prompt
            messages.append({"role": "user", "content": prompt})

            logger.info(f"Generating JSON with OpenAI model {self.model_name}")

            # Prepare API request parameters with JSON mode
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "response_format": {"type": "json_object"}
            }

            # Get model capabilities and use appropriate parameters
            model_capabilities = get_model_capabilities(self.model_name)

            # Add token limit parameter based on model capabilities
            if model_capabilities.get("supports_max_completion_tokens", False):
                request_params["max_completion_tokens"] = settings.openai_max_tokens
            else:
                request_params["max_tokens"] = settings.openai_max_tokens

            # Add temperature if supported by the model
            if model_capabilities.get("supports_custom_temperature", True):
                request_params["temperature"] = settings.openai_temperature

            # Make API request
            response: ChatCompletion = self.client.chat.completions.create(**request_params)

            # Extract and parse JSON response
            generated_text = response.choices[0].message.content or "{}"

            logger.info(f"Generated JSON length: {len(generated_text)}")
            logger.debug(f"JSON Response: {generated_text[:200]}...")

            return json.loads(generated_text)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {"error": f"JSON parse error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error generating JSON with OpenAI: {str(e)}")
            return {"error": str(e)}


# Helper functions from the original ollama_utils.py
def get_openai_available_models(api_key: str = None, organization: str = None) -> list:
    """
    Get a list of available OpenAI models from their API.
    
    Args:
        api_key: OpenAI API key. If None, tries to get from settings.
        organization: OpenAI organization ID. If None, tries to get from settings.
        
    Returns:
        list: List of available model names
    """
    if not api_key:
        api_key = settings.openai_api_key
    
    if not organization:
        organization = settings.openai_organization
    
    if not api_key:
        logger.warning("No OpenAI API key provided, cannot fetch available models")
        return ["gpt-3.5-turbo"]  # Fallback to a common model
    
    try:
        from openai import OpenAI
        
        client_kwargs = {"api_key": api_key}
        if organization:
            client_kwargs["organization"] = organization
        
        client = OpenAI(**client_kwargs)
        response = client.models.list()
        
        # Extract model IDs and filter for chat models
        available_models = []
        for model in response.data:
            model_id = model.id
            # Filter for common chat models (you can adjust this filtering)
            if any(prefix in model_id for prefix in ["gpt-", "claude-", "gemini-"]):
                available_models.append(model_id)
        
        # Sort models by name for better UX
        available_models.sort()
        
        logger.info(f"Successfully fetched {len(available_models)} available OpenAI models")
        return available_models
        
    except Exception as e:
        logger.error(f"Error fetching OpenAI models: {str(e)}")
        # Return fallback models if API call fails
        return ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]

# Ollama-specific functions have been moved to ollama_utils.py
