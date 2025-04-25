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
        
        # Try to check if the model exists, if not, try to pull it
        try:
            models = ollama.list()
            model_names = []
            
            # Extract model names safely using match
            match models:
                case {"models": model_list} if isinstance(model_list, list):
                    for model in model_list:
                        match model:
                            case {"name": name} if name and isinstance(name, str):
                                model_names.append(name)
                            case _:
                                continue
                
                case _:
                    logger.warning(f"Unexpected model list structure: {models}")
            
            # Check model availability and handle accordingly
            match (self.model_name in model_names, self.model_name):
                case (False, model_name):
                    self.model_status = "downloading"
                    logger.warning(f"Model {model_name} not found. Attempting to pull it now...")
                    try:
                        # Try pulling the model directly 
                        logger.info(f"Pulling model {model_name}...")
                        # Pull with streaming to update progress
                        self._pull_model_with_progress()
                        self.model_status = "available"
                        logger.info(f"Successfully pulled model {model_name}")
                    except Exception as pull_error:
                        # If pulling fails, inform the user how to do it manually
                        self.model_status = "not_found"
                        self.model_error = str(pull_error)
                        logger.error(f"Failed to automatically pull model: {str(pull_error)}")
                        logger.warning(f"Model {model_name} not available. You can pull it manually with: 'docker exec -it idiomapp-ollama ollama pull {model_name}'")
                
                case (True, _):
                    self.model_status = "available"
                    logger.info(f"Model {self.model_name} is available")
                
                case _:
                    # Should not reach here, but handle as unknown
                    self.model_status = "unknown"
                    logger.warning(f"Unexpected model status check result")
                    
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
    
    async def analyze_graph(self, graph_description: str) -> Dict[str, Any]:
        """
        Analyze a graph based on its description.
        
        Args:
            graph_description: Description of the graph structure
            
        Returns:
            dict: Analysis results with structure and insights
        """
        try:
            # Generate the analysis using the Ollama model
            analysis_text = await self.generate_text(prompt=f"""
        Analyze this graph:
        
        {graph_description}
        
        Consider the structure, degree distribution, centrality, and any interesting patterns. 
        Provide a clear and concise analysis of what this graph represents and its notable characteristics.
        
        Your response should be structured as follows:
        1. Analysis: Detailed observations about the graph.
        2. Summary: A brief 1-2 sentence summary of key findings.
            """)
            
            # Split the analysis into sections
            analysis_parts = analysis_text.split("Summary:")
            
            if len(analysis_parts) > 1:
                detailed_analysis = analysis_parts[0].replace("Analysis:", "").strip()
                summary = analysis_parts[1].strip()
            else:
                # In case the model didn't follow the formatting
                detailed_analysis = analysis_text
                summary = "Analysis completed. See details for more information."
            
            return {
                "analysis": detailed_analysis,
                "summary": summary
            }
        except Exception as e:
            logger.error(f"Error analyzing graph: {str(e)}")
            return {
                "analysis": f"Error analyzing graph: {str(e)}",
                "summary": "Analysis failed"
            }
    
    async def suggest_graph_improvements(self, graph_description: str) -> List[str]:
        """
        Suggest improvements for a graph based on its description.
        
        Args:
            graph_description: Description of the graph structure
            
        Returns:
            list: List of actionable improvement suggestions
        """
        try:
            # Generate improvement suggestions
            suggestions_text = await self.generate_text(prompt=f"""
        Review this graph:
        
        {graph_description}
        
        Suggest 3-5 specific, actionable improvements that could be made to this graph to:
        1. Improve its structure
        2. Make it more robust
        3. Optimize for better information flow
        4. Address any weaknesses you identify
        
        Format your response as a numbered list of suggestions only, with each item starting with an action verb.
        For example:
        1. Add edges between nodes X and Y to...
        2. Redistribute connections from the most central node to...
            """)
            
            # Parse the numbered list from the response
            lines = suggestions_text.strip().split('\n')
            suggestions = []
            
            for line in lines:
                # Look for numbered list items
                line = line.strip()
                if re.match(r'^\d+\.', line) or re.match(r'^-', line):
                    # Remove the number/bullet and whitespace
                    suggestion = re.sub(r'^\d+\.|-\s*', '', line).strip()
                    if suggestion:
                        suggestions.append(suggestion)
            
            # If no structured list was found, return the whole text
            if not suggestions and suggestions_text.strip():
                return [suggestions_text.strip()]
                
            return suggestions
        except Exception as e:
            logger.error(f"Error generating graph improvement suggestions: {str(e)}")
            return [f"Error generating suggestions: {str(e)}"]

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
            models = ollama.list()
            
            # Use match statement to handle different response structures
            match models:
                case {"models": model_list} if isinstance(model_list, list):
                    # Standard case: we have a models list
                    model_names = []
                    for model in model_list:
                        # Safe extraction of model name
                        match model:
                            case {"name": name} if name and isinstance(name, str):
                                model_names.append(name)
                            case _:
                                logger.warning(f"Skipping invalid model entry: {model}")
                    
                    logger.info(f"Found {len(model_names)} models: {', '.join(model_names) if model_names else 'none'}")
                    return model_names if model_names else [DEFAULT_MODEL]
                
                case {"models": None}:
                    # Handle the case where models key exists but is None
                    logger.warning("API returned None for models list")
                    return [DEFAULT_MODEL]
                
                case {}:
                    # Empty response
                    logger.warning("API returned empty response")
                    return [DEFAULT_MODEL]
                
                case _:
                    # Unknown structure
                    logger.warning(f"Unexpected API response structure: {models}")
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