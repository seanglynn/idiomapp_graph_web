import ollama
import asyncio
import time
import os
from typing import List, Dict, Any, Optional
from idiomapp.logging_utils import setup_logging

# Set up logging
logger = setup_logging("ollama_utils")

# Default model constant - can be overridden by environment variable
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2:latest")

class OllamaClient:
    """
    A utility class for interacting with Ollama models.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the Ollama client with a specific model.
        
        Args:
            model_name (str): The name of the Ollama model to use. Defaults to DEFAULT_MODEL.
        """
        self.model_name = model_name or DEFAULT_MODEL
        logger.info(f"Initialized OllamaClient with model: {self.model_name}")
    
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using the Ollama model.
        
        Args:
            prompt (str): The prompt to send to the model.
            system_prompt (Optional[str]): Optional system prompt to guide the model.
            
        Returns:
            str: The generated text response.
        """
        try:
            logger.info(f"Sending request to model {self.model_name} (prompt length: {len(prompt)})")
            start_time = time.time()
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Received response from Ollama in {elapsed_time:.2f} seconds")
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {e}")
            return f"Error: Unable to generate text with model {self.model_name}. Please ensure Ollama is running and the model is available."
    
    async def analyze_graph(self, graph_description: str) -> Dict[str, Any]:
        """
        Analyze a graph using the Ollama model.
        
        Args:
            graph_description (str): Description of the graph to analyze.
            
        Returns:
            Dict[str, Any]: Analysis results as a dictionary.
        """
        logger.info(f"Starting graph analysis with model {self.model_name}")
        system_prompt = """
        You are a graph analysis expert. Given a description of a graph,
        provide analytical insights about its structure, properties, and potential applications.
        Return your analysis in a concise, structured format.
        """
        
        prompt = f"Analyze this graph: {graph_description}"
        
        start_time = time.time()
        analysis = await self.generate_text(prompt, system_prompt)
        elapsed_time = time.time() - start_time
        logger.info(f"Completed graph analysis in {elapsed_time:.2f} seconds")
        
        # Return analysis in a structured format
        return {
            "analysis": analysis,
            "summary": analysis.split("\n")[0] if "\n" in analysis else analysis[:100]
        }
    
    async def suggest_graph_improvements(self, graph_description: str) -> List[str]:
        """
        Suggest improvements or modifications to a graph.
        
        Args:
            graph_description (str): Description of the current graph.
            
        Returns:
            List[str]: A list of suggested improvements.
        """
        logger.info(f"Starting improvement suggestions for graph with model {self.model_name}")
        system_prompt = """
        You are a graph optimization expert. Given a description of a graph,
        suggest specific improvements or modifications that could enhance its properties
        or make it more suitable for particular applications.
        Keep your suggestions concise and actionable.
        List exactly 3 suggestions, one per line, starting with a dash (-).
        """
        
        prompt = f"Suggest improvements for this graph: {graph_description}"
        
        start_time = time.time()
        suggestions_text = await self.generate_text(prompt, system_prompt)
        elapsed_time = time.time() - start_time
        logger.info(f"Completed improvement suggestions in {elapsed_time:.2f} seconds")
        
        # Parse suggestions into a list
        suggestions = [s.strip()[2:].strip() for s in suggestions_text.split("\n") 
                      if s.strip().startswith("-")]
        
        logger.info(f"Extracted {len(suggestions)} improvement suggestions")
        return suggestions if suggestions else ["Increase connectivity", 
                                               "Add centrality measures", 
                                               "Consider community detection"]

# Helper function to get available Ollama models
async def list_available_models() -> List[str]:
    """
    Get a list of available Ollama models.
    
    Returns:
        List[str]: A list of model names.
    """
    try:
        logger.info("Requesting list of available Ollama models...")
        models_response = ollama.list()
        available_models = []
        
        # Handle new Ollama API response format (Python object with Model class)
        if hasattr(models_response, 'models') and models_response.models:
            for model in models_response.models:
                if hasattr(model, 'model'):
                    available_models.append(model.model)
        # Handle older dictionary-based response format
        elif isinstance(models_response, dict) and 'models' in models_response:
            for model in models_response['models']:
                if isinstance(model, dict) and 'name' in model:
                    available_models.append(model['name'])
        
        if not available_models:
            logger.warning(f"No models found in Ollama response: {models_response}")
            # Return the default model from environment
            return [DEFAULT_MODEL]
            
        logger.info(f"Found {len(available_models)} Ollama models: {', '.join(available_models)}")
        return available_models
    except Exception as e:
        logger.error(f"Error getting Ollama models: {e}")
        return [DEFAULT_MODEL]  # Default from environment variable

# Synchronous wrapper for the async function
def get_available_models() -> List[str]:
    """
    Synchronous wrapper to get available Ollama models.
    
    Returns:
        List[str]: A list of model names.
    """
    try:
        logger.debug("Fetching available Ollama models...")
        models = list_available_models()
        if asyncio.iscoroutine(models):
            models = asyncio.run(models)
        
        # Return default models if no models were found
        if not models:
            logger.warning("No models found, returning default models")
            return [DEFAULT_MODEL]
        return models
    except Exception as e:
        logger.error(f"Error in get_available_models: {e}")
        return [DEFAULT_MODEL] 