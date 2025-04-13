import ollama
import asyncio
import time
import os
from typing import List, Dict, Any, Optional
from idiomapp.utils.logging_utils import setup_logging
import httpx
import re

# Set up logging
logger = setup_logging("ollama_utils")

# Default values from environment
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "llama3.2:latest")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

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
        self.model_name = model_name or DEFAULT_MODEL
        self.base_url = OLLAMA_HOST
        logger.info(f"Initialized OllamaClient with model: {self.model_name}")
    
    async def generate_text(self, prompt, system_prompt=None):
        """
        Generate text using the Ollama model.
        
        Args:
            prompt: The prompt to generate text from
            system_prompt: Optional system prompt for context
            
        Returns:
            str: The generated text
        """
        start_time = time.time()
        
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        logger.info(f"Sending request to model {self.model_name} (prompt length: {len(prompt)})")
        
        try:
            # Make the API request to Ollama
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=120.0  # Longer timeout for larger responses
                )
                response.raise_for_status()
                
                # Extract the response text
                data = response.json()
                response_text = data.get("response", "")
                
                elapsed_time = time.time() - start_time
                logger.info(f"Received response from Ollama in {elapsed_time:.2f} seconds")
                
                return response_text
                
        except Exception as e:
            logger.error(f"Error generating text from Ollama: {str(e)}")
            raise
    
    async def analyze_graph(self, graph_description: str) -> Dict[str, Any]:
        """
        Analyze a graph based on its description.
        
        Args:
            graph_description: Description of the graph structure
            
        Returns:
            dict: Analysis results with structure and insights
        """
        prompt = f"""
        Analyze this graph:
        
        {graph_description}
        
        Consider the structure, degree distribution, centrality, and any interesting patterns. 
        Provide a clear and concise analysis of what this graph represents and its notable characteristics.
        
        Your response should be structured as follows:
        1. Analysis: Detailed observations about the graph.
        2. Summary: A brief 1-2 sentence summary of key findings.
        """
        
        try:
            # Generate the analysis using the Ollama model
            analysis_text = await self.generate_text(prompt)
            
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
        prompt = f"""
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
        """
        
        try:
            # Generate improvement suggestions
            suggestions_text = await self.generate_text(prompt)
            
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

# Helper function to get available Ollama models
async def list_available_models():
    """
    Get a list of models available in Ollama.
    
    Returns:
        list: List of available model names
    """
    logger.info("Requesting list of available Ollama models...")
    ollama_host = OLLAMA_HOST
    
    try:
        # Make API request to Ollama
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ollama_host}/api/tags")
            response.raise_for_status()
            
            data = response.json()
            
            # Extract model names from the response
            # The format can be different depending on the Ollama version
            if "models" in data:
                # Newer Ollama versions
                models = [model.get("name") for model in data.get("models", [])]
            else:
                # Older Ollama versions
                models = [model.get("name") for model in data.get("models", [])]
            
            logger.info(f"Found {len(models)} Ollama models: {', '.join(models) if models else 'none'}")
            
            # If no models are found, return the default
            if not models:
                logger.info(f"No models found, returning default: {DEFAULT_MODEL}")
                return [DEFAULT_MODEL]
                
            return models
    except Exception as e:
        logger.error(f"Error listing Ollama models: {str(e)}")
        logger.info(f"Returning default model: {DEFAULT_MODEL}")
        return [DEFAULT_MODEL]

# Synchronous wrapper for the async function
def get_available_models() -> List[str]:
    """
    Synchronous wrapper to get available Ollama models.
    
    Returns:
        List[str]: A list of model names.
    """
    try:
        logger.info("Fetching available Ollama models...")
        models = list_available_models()
        if asyncio.iscoroutine(models):
            models = asyncio.run(models)
        
        # Return default models if no models were found
        if not models:
            logger.warning("No models found, returning default model")
            return [DEFAULT_MODEL]
        return models
    except Exception as e:
        logger.error(f"Error in get_available_models: {e}")
        return [DEFAULT_MODEL] 