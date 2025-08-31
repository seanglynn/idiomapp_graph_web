import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json

# For Ollama
import ollama
import httpx
import requests

# For OpenAI
from openai import OpenAI
from openai.types.chat import ChatCompletion

from idiomapp.utils.logging_utils import get_logger
from idiomapp.config import settings, LLMProvider

# Set up logging using the new cached logger
logger = get_logger("llm_utils")

class LLMClient(ABC):
    """Abstract base class for LLM clients (Ollama, OpenAI, etc.)"""
    
    @abstractmethod
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from a prompt"""
        pass
    
    @abstractmethod
    def get_model_status(self) -> Dict[str, Any]:
        """Get the status of the model"""
        pass
    
    @classmethod
    def create(cls, provider: str = None, model_name: str = None) -> 'LLMClient':
        """Factory method to create the appropriate LLM client"""
        provider = provider or settings.llm_provider.value
        
        if provider == LLMProvider.OLLAMA.value:
            return OllamaClient(model_name or settings.default_model)
        elif provider == LLMProvider.OPENAI.value:
            return OpenAIClient(model_name or settings.openai_model)
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
            # Prepare the request - set stream to False for single response
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False  # This ensures we get a single JSON response instead of streaming
            }
            
            if system_prompt:
                request_data["system"] = system_prompt
                
            logger.info(f"Generating text with model {self.model_name}")
            logger.debug(f"Prompt: {prompt[:100]}...")
                
            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json=request_data,
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    error_msg = f"Error generating text: HTTP {response.status_code}"
                    logger.error(f"{error_msg} - {response.text}")
                    response.raise_for_status()
                    return f"Error: Failed to generate text. {error_msg}"
                # Parse response JSON safely
                try:
                    response_data = response.json()
                    generated_text = response_data.get("response", "")
                    
                    if not generated_text:
                        logger.warning("Empty response from model")
                        return "Error: Model returned empty response"
                    
                    logger.info(f"Generated text length: {len(generated_text)}")
                    logger.debug(f"Response: {generated_text[:100]}...")
                    
                    return generated_text
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {str(e)}")
                    logger.error(f"Raw response: {response.text[:200]}...")
                    return f"Error: Invalid response format from model"
                
        except httpx.TimeoutException:
            logger.error("Request timeout while generating text")
            return "Error: Request timeout. The model is taking too long to respond."
        except httpx.RequestError as e:
            logger.error(f"Network error while generating text: {str(e)}")
            return f"Error: Network error - {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error generating text: {str(e)}")
            return f"Error: {str(e)}"


class OpenAIClient(LLMClient):
    """Client for interacting with OpenAI models."""
    
    def __init__(self, model_name=None):
        """
        Initialize the OpenAI client.
        
        Args:
            model_name: The name of the model to use. If None, uses the OPENAI_MODEL
                        from environment variables.
        """
        self.api_key = settings.openai_api_key
        self.model_name = model_name or settings.openai_model
        
        # Set client configuration
        logger.info(f"Initializing OpenAI client with model: {self.model_name}")
        
        # Initialize the OpenAI client
        if not self.api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
        else:
            self.client = OpenAI(api_key=self.api_key)
    
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
            
            # Make API request
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=settings.openai_temperature,
                max_tokens=settings.openai_max_tokens
            )

            
            # Extract response text
            generated_text = response.choices[0].message.content or ""
            
            logger.info(f"Generated text length: {len(generated_text)}")
            logger.debug(f"Response: {generated_text[:100]}...")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            raise e

# Helper functions from the original ollama_utils.py
def is_ollama_running():
    """
    Check if Ollama is running and accessible through any of the potential hosts.
    
    Returns:
        tuple: (bool, str) - (is_running, host_url if running, None if not)
    """
    # Define fallback hosts to try if the primary host fails
    FALLBACK_HOSTS = [
        "http://localhost:11434",  # Local development
        "http://ollama:11434",     # Docker service name
        "http://host.docker.internal:11434"  # Mac/Windows Docker to host
    ]
    
    # First check the configured host
    configured_host = settings.ollama_host
    logger.info(f"Checking if Ollama is running at {configured_host}...")
    
    try:
        response = requests.get(f"{configured_host}/api/version", timeout=2.0)
        if response.status_code == 200:
            logger.info(f"Ollama is running at {configured_host}")
            return True, configured_host
    except Exception as e:
        logger.warning(f"Ollama not reachable at {configured_host}: {str(e)}")
    
    # If primary host failed, try fallbacks
    for host in FALLBACK_HOSTS:
        if host == configured_host:
            continue  # Already tried this one
            
        logger.info(f"Checking if Ollama is running at fallback {host}...")
        try:
            response = requests.get(f"{host}/api/version", timeout=2.0)
            if response.status_code == 200:
                logger.info(f"Ollama is running at fallback {host}")
                return True, host
        except Exception as e:
            logger.warning(f"Ollama not reachable at fallback {host}: {str(e)}")
    
    logger.error("Ollama service is not reachable at any configured host")
    return False, None

def get_valid_ollama_host():
    """
    Try to find a valid Ollama host by checking each potential host.
    
    Returns:
        str: A valid Ollama host URL, or the default one if none work
    """
    # Check if Ollama is running anywhere
    is_running, running_host = is_ollama_running()
    if is_running:
        return running_host
        
    # If all else fails, return the configured host (which will likely fail again, but it's our default)
    logger.error("Could not find any working Ollama host, returning configured host")
    return settings.ollama_host

def get_available_models():
    """
    Get a list of available Ollama models.
    
    Returns:
        list: List of available model names
    """
    models = []
    host = get_valid_ollama_host()
    
    try:
        # Try to use the API to get available models
        response = requests.get(f"{host}/api/tags", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            
            # Extract model names from response
            if "models" in data and isinstance(data["models"], list):
                for model in data["models"]:
                    if "name" in model and model["name"]:
                        models.append(model["name"])
            else:
                logger.warning("Unexpected response format when getting models")
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        
    return models

def pull_model_if_needed(model_name):
    """
    Check if a model needs to be pulled and pull it if it doesn't exist.
    
    Args:
        model_name: The name of the model to check/pull
        
    Returns:
        bool: True if the model is now available, False otherwise
    """
    logger.info(f"Checking if model {model_name} needs to be pulled...")
    
    # Get available models
    try:
        available_models = get_available_models()
        logger.info(f"Available models before pull attempt: {available_models}")
        
        # If model already exists, no need to pull
        if model_name in available_models:
            logger.info(f"Model {model_name} is already available")
            return True
            
        # Model doesn't exist, attempt to pull it
        logger.info(f"Model {model_name} not found, attempting to pull...")
        
        # Get a valid host to connect to
        ollama_host = get_valid_ollama_host()
        
        # Set OLLAMA_HOST environment variable temporarily for the native client
        original_env = os.environ.get("OLLAMA_HOST")
        os.environ["OLLAMA_HOST"] = ollama_host
        
        try:
            # Try to pull the model using the native client
            logger.info(f"Pulling model {model_name} using native client...")
            ollama.pull(model_name)
            logger.info(f"Successfully pulled model {model_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to pull model using native client: {str(e)}")
            
            # Try using the HTTP API if native client fails
            try:
                logger.info(f"Pulling model {model_name} using HTTP API...")
                response = requests.post(
                    f"{ollama_host}/api/pull",
                    json={"name": model_name},
                    timeout=300.0  # Longer timeout for model pulls
                )
                response.raise_for_status()
                logger.info(f"Successfully initiated pull for model {model_name}")
                
                # Check if model is now available
                available_models = get_available_models()
                if model_name in available_models:
                    logger.info(f"Model {model_name} is now available")
                    return True
                else:
                    logger.warning(f"Model {model_name} pull initiated but model not immediately available")
                    return False
            except Exception as e:
                logger.error(f"Failed to pull model using HTTP API: {str(e)}")
                return False
        finally:
            # Restore original environment
            if original_env:
                os.environ["OLLAMA_HOST"] = original_env
            elif "OLLAMA_HOST" in os.environ:
                del os.environ["OLLAMA_HOST"]
    except Exception as e:
        logger.error(f"Error checking/pulling model: {str(e)}")
        return False 