import ollama
import asyncio
import time
import os
from typing import List, Dict, Any, Optional
from idiomapp.utils.logging_utils import setup_logging
import httpx
import re
import requests
import threading
import logging

# Set up logging
logger = setup_logging("ollama_utils")

# Default values from environment
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "llama3.2:latest")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Define fallback hosts to try if the primary host fails
FALLBACK_HOSTS = [
    "http://localhost:11434",  # Local development
    "http://ollama:11434",     # Docker service name
    "http://host.docker.internal:11434"  # Mac/Windows Docker to host
]

def is_ollama_running():
    """
    Check if Ollama is running and accessible through any of the potential hosts.
    
    Returns:
        tuple: (bool, str) - (is_running, host_url if running, None if not)
    """
    # First check the configured host
    configured_host = OLLAMA_HOST
    logger.info(f"Checking if Ollama is running at {configured_host}...")
    
    try:
        response = requests.get(f"{configured_host}/api/version", timeout=2.0)
        if response.status_code == 200:
            logger.info(f"Ollama is running at {configured_host}")
            try:
                version_info = response.json()
                ollama_version = version_info.get("version", "unknown")
                logger.info(f"Ollama version: {ollama_version}")
            except Exception:
                logger.warning("Could not parse Ollama version information")
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
                try:
                    version_info = response.json()
                    ollama_version = version_info.get("version", "unknown")
                    logger.info(f"Ollama version: {ollama_version}")
                except Exception:
                    logger.warning("Could not parse Ollama version information")
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
    return OLLAMA_HOST

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

async def get_model_list(base_url):
    """
    Try different endpoints to get a list of available models.
    
    Args:
        base_url: The base URL of the Ollama server
        
    Returns:
        list: List of available model names, or empty list if none found
    """
    # Try multiple endpoints
    endpoints = [
        "/api/tags",   # Current 0.4.7 endpoint
        "/api/list",   # Old endpoint
        "/api/models"  # Alternative endpoint
    ]
    
    for endpoint in endpoints:
        try:
            logger.info(f"Checking models at {base_url}{endpoint}")
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}{endpoint}", timeout=5.0)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Use match statement to handle different API response structures
                    match data:
                        case {"models": models_list} if isinstance(models_list, list):
                            # Process models list from /api/list or /api/models endpoint
                            models = []
                            for model in models_list:
                                match model:
                                    case {"name": name} if name and isinstance(name, str):
                                        models.append(name)
                                    case _:
                                        continue
                            
                            if models:
                                logger.info(f"Found {len(models)} models using {endpoint}")
                                return models
                                
                        case {"models": None}:
                            logger.warning(f"API returned None for models list using {endpoint}")
                            
                        case {"tags": tags_list} if isinstance(tags_list, list):
                            # Process tags list from /api/tags endpoint
                            models = []
                            for tag in tags_list:
                                match tag:
                                    case {"name": name} if name and isinstance(name, str):
                                        models.append(name)
                                    case _:
                                        continue
                                        
                            if models:
                                logger.info(f"Found {len(models)} models using {endpoint}")
                                return models
                                
                        case {"tags": None}:
                            logger.warning(f"API returned None for tags list using {endpoint}")
                            
                        case _:
                            logger.warning(f"Unknown API response format from {endpoint}: {data}")
                            
        except Exception as e:
            logger.warning(f"Failed to get models from {endpoint}: {str(e)}")
    
    # Didn't find any models
    return []

class OllamaClient:
    """
    Client for interacting with Ollama models.
    Supports text generation and simple graph analysis.
    """
    # Class-level cache to prevent excessive model checks
    _model_available_cache = {}  # model_name -> bool
    
    def __init__(self, model_name=None):
        """
        Initialize the Ollama client with a specific model.
        
        Args:
            model_name: The name of the Ollama model to use
        """
        # Set the model name
        self.model_name = model_name or DEFAULT_MODEL
        self.base_url = OLLAMA_HOST
        
        # Set up the Ollama client with the correct host
        os.environ["OLLAMA_HOST"] = self.base_url
        
        # Status tracking
        self.model_status = "unknown"  # Possible values: "unknown", "available", "downloading", "not_found"
        self.model_download_progress = 0
        self.model_error = None
        
        logger.info(f"Initialized OllamaClient with model: {self.model_name} at {self.base_url}")
        
        # Check model availability only if not already known
        if self.model_name not in OllamaClient._model_available_cache:
            self._check_model_availability()
        else:
            is_available = OllamaClient._model_available_cache[self.model_name]
            if is_available:
                self.model_status = "available"
                logger.info(f"Model {self.model_name} is available (from cache)")
            else:
                # If cache says not available, check again in case it's been added
                self._check_model_availability()
    
    def _check_model_availability(self):
        """
        Check if the model is available and update the status accordingly.
        """
        try:
            # First, try a direct check with the model name
            try:
                # Using the "show" command is more reliable than parsing the model list
                model_info = ollama.show(self.model_name)
                
                if model_info:
                    self.model_status = "available"
                    logger.info(f"Model {self.model_name} is available (verified with show)")
                    OllamaClient._model_available_cache[self.model_name] = True
                    return
            except Exception as show_error:
                logger.debug(f"Model show check failed: {str(show_error)}")
                # Continue to fallback method
            
            # Get list of available models
            models_response = ollama.list()
            model_names = []
            
            # Extract model names using a flexible approach that works with Ollama 0.4.8+
            try:
                # Handle the new Ollama 0.4.8+ response format (Model objects)
                if hasattr(models_response, 'models'):
                    for model in models_response.models:
                        if hasattr(model, 'model'):  # New structure uses 'model' attribute
                            model_names.append(model.model)
                        elif hasattr(model, 'name'):  # Older structure used 'name' attribute
                            model_names.append(model.name)
                
                # Handle older dictionary-based response format
                elif isinstance(models_response, dict) and 'models' in models_response:
                    for model in models_response['models']:
                        if isinstance(model, dict):
                            if 'model' in model:
                                model_names.append(model['model'])
                            elif 'name' in model:
                                model_names.append(model['name'])
                
                logger.info(f"Available models: {model_names}")
                
                # If we couldn't parse any model names but got a response, try another approach
                if not model_names and models_response:
                    # This is a more flexible approach for any future API changes
                    try:
                        # Try to extract model names using string conversion for Model objects
                        if hasattr(models_response, 'models') and models_response.models:
                            model_names = [str(model).split("'")[1] if "'" in str(model) else str(model) 
                                         for model in models_response.models]
                            logger.info(f"Extracted model names using string conversion: {model_names}")
                    except Exception as string_error:
                        logger.error(f"Failed to convert models to strings: {string_error}")
            except Exception as parse_error:
                logger.error(f"Error parsing model list: {str(parse_error)}")
                logger.debug(f"Raw model list data: {models_response}")
            
            # Check model availability and handle accordingly
            if self.model_name in model_names:
                # Model is already available
                self.model_status = "available"
                logger.info(f"Model {self.model_name} is available")
                OllamaClient._model_available_cache[self.model_name] = True
            else:
                # Model needs to be downloaded
                self.model_status = "downloading"
                logger.warning(f"Model {self.model_name} not found. Attempting to pull it now...")
                try:
                    # Try pulling the model directly 
                    logger.info(f"Pulling model {self.model_name}...")
                    # Pull with streaming to update progress
                    self._pull_model_with_progress()
                    self.model_status = "available"
                    logger.info(f"Successfully pulled model {self.model_name}")
                    OllamaClient._model_available_cache[self.model_name] = True
                except Exception as pull_error:
                    # If pulling fails, inform the user how to do it manually
                    self.model_status = "not_found"
                    self.model_error = str(pull_error)
                    logger.error(f"Failed to automatically pull model: {str(pull_error)}")
                    logger.warning(f"Model {self.model_name} not available. You can pull it manually with: 'docker exec -it idiomapp-ollama ollama pull {self.model_name}'")
                    OllamaClient._model_available_cache[self.model_name] = False
        except Exception as e:
            self.model_status = "unknown"
            self.model_error = str(e)
            logger.error(f"Error checking model availability: {str(e)}")
            logger.warning(f"Could not determine if model {self.model_name} is available.")
    
    def _pull_model_with_progress(self):
        """
        Pull the model and track progress.
        """
        try:
            # Try to pull with progress tracking
            self.model_download_progress = 0
            
            # Start a background thread to update progress
            threading_event = threading.Event()
            
            def pull_model():
                try:
                    # Pull the model
                    for progress in ollama.pull(self.model_name, stream=True):
                        if 'completed' in progress and 'total' in progress:
                            if progress['total'] > 0:  # Avoid division by zero
                                percentage = (progress['completed'] / progress['total']) * 100
                                self.model_download_progress = percentage
                                logger.info(f"Download progress: {percentage:.1f}%")
                    self.model_status = "available"
                    self.model_download_progress = 100
                except Exception as e:
                    self.model_status = "not_found"
                    self.model_error = str(e)
                    logger.error(f"Error pulling model: {str(e)}")
                finally:
                    threading_event.set()
            
            # Start the thread
            thread = threading.Thread(target=pull_model)
            thread.daemon = True
            thread.start()
            
            # Wait briefly for any immediate errors
            threading_event.wait(timeout=1.0)
            
        except Exception as e:
            self.model_status = "not_found"
            self.model_error = str(e)
            logger.error(f"Error setting up model pull: {str(e)}")
            raise
    
    def get_model_status(self):
        """
        Get the current status of the model.
        
        Returns:
            dict: Status information about the model
        """
        return {
            "model_name": self.model_name,
            "status": self.model_status,
            "download_progress": self.model_download_progress,
            "error": self.model_error
        }
    
    async def generate_text(self, prompt, system_prompt=None):
        """
        Generate text using the Ollama model.
        
        Args:
            prompt: The prompt to generate text from
            system_prompt: Optional system prompt for context
            
        Returns:
            str: The generated text
        """
        logger.info(f"Sending request to model {self.model_name} (prompt length: {len(prompt)})")
        start_time = time.time()
        
        try:
            # Use the official Python client
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                
            )
            
            response_text = response['message']['content']
            elapsed_time = time.time() - start_time
            logger.info(f"Received response in {elapsed_time:.2f} seconds")
            return response_text
            
        except ollama.ResponseError as e:
            error_msg = f"Failed to generate text: {str(e)}"
            if e.status_code == 404:
                # Model not found
                error_msg += f"\n\nThe model '{self.model_name}' needs to be downloaded first."
                error_msg += f"\nRun these commands in your terminal:"
                error_msg += f"\n  1. docker exec -it idiomapp-ollama /bin/bash"
                error_msg += f"\n  2. ollama pull {self.model_name}"
            
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Error generating text: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

# Simple function to get available models
def get_available_models():
    """
    Get a list of available Ollama models.
    
    Returns:
        list: List of model names
    """
    logger.info("Fetching available models...")
    try:
        # Set OLLAMA_HOST environment variable for the client
        original_host = os.environ.get("OLLAMA_HOST")
        os.environ["OLLAMA_HOST"] = OLLAMA_HOST
        
        try:
            # Use the official client
            models_response = ollama.list()
            
            model_names = []
            
            # Try to extract model names in different ways since the API structure can vary
            # First, check if we have the standard models key
            if hasattr(models_response, 'models'):
                # Ollama 0.4.8+ returns Model objects in models list
                for model in models_response.models:
                    if hasattr(model, 'model'):  # New structure uses 'model' attribute
                        model_names.append(model.model)
                    elif hasattr(model, 'name'):  # Old structure used 'name' attribute
                        model_names.append(model.name)
            # Fallback to dictionary-style access if needed (older versions)
            elif isinstance(models_response, dict) and 'models' in models_response:
                for model in models_response['models']:
                    if isinstance(model, dict):
                        if 'model' in model:
                            model_names.append(model['model'])
                        elif 'name' in model:
                            model_names.append(model['name'])
            
            # If we got any models, return them
            if model_names:
                logger.info(f"Found {len(model_names)} models: {', '.join(model_names)}")
                return model_names
            
            # If no models were found but we got a response, log a warning
            if models_response:
                logger.warning(f"Could not extract model names from response: {models_response}")
                
                # Last resort: try to directly access the first model if it exists
                try:
                    if hasattr(models_response, 'models') and models_response.models:
                        # If the model object is string-convertible, use that
                        model_names = [str(model).split("'")[1] if "'" in str(model) else str(model) 
                                      for model in models_response.models]
                        logger.info(f"Extracted model names using string conversion: {model_names}")
                        return model_names
                except Exception as string_error:
                    logger.error(f"Failed to convert models to strings: {string_error}")
            
            logger.warning("No models found, returning default model")
            return [DEFAULT_MODEL]
            
        finally:
            # Restore original environment
            if original_host:
                os.environ["OLLAMA_HOST"] = original_host
            elif "OLLAMA_HOST" in os.environ:
                del os.environ["OLLAMA_HOST"]
                
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return [DEFAULT_MODEL]  # Return default model as fallback 