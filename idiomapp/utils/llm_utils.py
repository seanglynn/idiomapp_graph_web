"""
LLM provider clients for IdiomApp.

Three providers sit behind one small interface: Ollama (local), OpenAI, and
Anthropic (Claude). Every client is genuinely async - the SDK clients are the async
variants, so several calls can be issued concurrently with `asyncio.gather`.

Two details worth knowing before editing:

* The underlying SDK clients are built lazily and memoised against the running
  event loop (see `_loop_bound_client`). They hold connection pools bound to one
  loop, so a client object cached in Streamlit's session state must be able to
  rebind if it is used from a different loop.
* Claude models reject unsupported request parameters with a 400 rather than
  ignoring them, so `AnthropicClient` builds its kwargs through the capability
  gate in `idiomapp.config`.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

# For Ollama
import ollama

# For OpenAI
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

# For Anthropic (Claude)
import anthropic
from anthropic import AsyncAnthropic
from pydantic import BaseModel

from idiomapp.utils.logging_utils import get_logger
from idiomapp.utils.json_utils import extract_json
from idiomapp.utils.async_utils import loop_key
from idiomapp.config import (
    settings,
    LLMProvider,
    get_model_capabilities,
    anthropic_supports_effort,
)
from idiomapp.utils.ollama_utils import get_valid_ollama_host, get_available_models, pull_model_if_needed

# Set up logging using the new cached logger
logger = get_logger("llm_utils")


class LLMClient(ABC):
    """Abstract base class for LLM clients (Ollama, OpenAI, Anthropic)."""

    @abstractmethod
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from a prompt"""

    @abstractmethod
    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a JSON object from a prompt.

        Args:
            prompt: The prompt to send to the model.
            system_prompt: Optional system prompt for context.
            schema: Optional Pydantic model describing the expected shape. Providers
                that support structured output use it to guarantee the response
                parses; others fall back to prompting plus tolerant extraction.

        Returns:
            The parsed object, or {"error": "..."} on failure.
        """

    @abstractmethod
    def get_model_status(self) -> Dict[str, Any]:
        """Get the status of the model"""

    @classmethod
    def create(
        cls,
        provider: str = None,
        model_name: str = None,
        api_key: str = None,
        organization: str = None,
    ) -> 'LLMClient':
        """Factory method to create the appropriate LLM client"""
        provider = provider or settings.llm_provider.value

        if provider == LLMProvider.OLLAMA.value:
            return OllamaClient(model_name or settings.default_model)
        elif provider == LLMProvider.OPENAI.value:
            return OpenAIClient(model_name or settings.openai_model, api_key, organization)
        elif provider == LLMProvider.ANTHROPIC.value:
            return AnthropicClient(model_name or settings.anthropic_model, api_key)
        else:
            logger.error(f"Unknown LLM provider: {provider}, falling back to Ollama")
            return OllamaClient(model_name or settings.default_model)

    def _loop_bound_client(self, factory):
        """
        Build (and memoise) the underlying async SDK client for the running loop.

        Async SDK clients wrap an httpx connection pool tied to the event loop that
        created them. This client object is cached in Streamlit's session state and
        outlives any single loop, so the transport is rebuilt whenever the loop
        changes rather than being pinned at construction time.
        """
        key = loop_key()
        if getattr(self, "_sdk_loop_key", None) != key or getattr(self, "_sdk_client", None) is None:
            self._sdk_client = factory()
            self._sdk_loop_key = key
            logger.debug(f"Built a new SDK client for event loop {key}")
        return self._sdk_client


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
        self._sdk_client = None
        self._sdk_loop_key = None

        # Set client configuration
        logger.info(f"Initializing Ollama client with model: {self.model_name}")
        logger.info(f"Using Ollama host: {self.ollama_host}")

        # Check if model is available
        self._check_model_availability()

    def _client(self) -> ollama.AsyncClient:
        """
        Get the async Ollama client for this host.

        The host is passed to the client directly. Earlier versions set
        os.environ["OLLAMA_HOST"] around each call and restored it afterwards, which
        mutates process-global state from Streamlit's per-session script threads -
        two sessions pointed at different hosts could read each other's value.
        """
        return self._loop_bound_client(lambda: ollama.AsyncClient(host=self.ollama_host))

    def _check_model_availability(self):
        """Check if the model is available, and try to pull it if not."""
        # Check cache first to avoid repeated API calls
        if self.model_name in self._model_available_cache:
            logger.debug(f"Using cached model availability for {self.model_name}: {self._model_available_cache[self.model_name]}")
            return self._model_available_cache[self.model_name]

        logger.debug(f"Checking availability of model: {self.model_name}")

        try:
            # Get list of available models
            available_models = get_available_models()
            model_available = self.model_name in available_models

            if model_available:
                logger.debug(f"Model {self.model_name} is available")
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

    def _build_messages(self, prompt, system_prompt):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

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
            logger.debug(f"Generating text with Ollama model {self.model_name}, prompt={len(prompt)} chars")

            response = await self._client().chat(
                model=self.model_name,
                messages=self._build_messages(prompt, system_prompt),
            )

            generated_text = self._extract_content(response)

            logger.debug(f"Generated text length: {len(generated_text)}")
            if not generated_text:
                logger.info(f"Empty response from Ollama model {self.model_name}")

            return generated_text

        except Exception as e:
            logger.error(f"Unexpected error generating text: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"

    async def generate_json(self, prompt, system_prompt=None, schema=None):
        """
        Generate JSON using Ollama's native JSON mode.

        Ollama has no schema-constrained output, so `schema` is used only as a hint
        appended to the prompt; the response is parsed tolerantly.
        """
        if not self._check_model_availability():
            return {"error": "Model not available. Please check if Ollama is running."}

        try:
            logger.debug(f"Generating JSON with Ollama model {self.model_name}")

            response = await self._client().chat(
                model=self.model_name,
                messages=self._build_messages(prompt, system_prompt),
                format="json",
            )

            generated_text = self._extract_content(response)
            if not generated_text:
                return {"error": "Empty response from Ollama"}

            parsed = extract_json(generated_text)
            if parsed is None:
                return {"error": "Could not parse JSON from Ollama response"}
            return parsed

        except Exception as e:
            logger.error(f"Error generating JSON with Ollama: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def _extract_content(response) -> str:
        """Pull the message text out of an Ollama response (dict or ChatResponse)."""
        try:
            return response['message']['content']
        except (KeyError, TypeError):
            try:
                return response['response']
            except (KeyError, TypeError):
                logger.warning(f"Unexpected response structure: {type(response)}")
                return str(response)


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
        self._sdk_client = None
        self._sdk_loop_key = None

        # Set client configuration
        logger.info(f"Initializing OpenAI client with model: {self.model_name}")
        if self.organization:
            logger.info(f"Using OpenAI organization: {self.organization}")

        if not self.api_key:
            logger.error("OPENAI_API_KEY environment variable not set")

    def _client(self) -> AsyncOpenAI:
        """Get the async OpenAI client, rebuilt if the event loop changed."""
        def build():
            kwargs = {"api_key": self.api_key}
            if self.organization:
                kwargs["organization"] = self.organization
            return AsyncOpenAI(**kwargs)
        return self._loop_bound_client(build)

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

    def _build_request_params(self, prompt, system_prompt) -> Dict[str, Any]:
        """Assemble request params, honouring this model's parameter support."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_params = {"model": self.model_name, "messages": messages}

        model_capabilities = get_model_capabilities(self.model_name)

        if model_capabilities.get("supports_max_completion_tokens", False):
            request_params["max_completion_tokens"] = settings.openai_max_tokens
        else:
            request_params["max_tokens"] = settings.openai_max_tokens

        if model_capabilities.get("supports_custom_temperature", True):
            request_params["temperature"] = settings.openai_temperature

        return request_params

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

        request_params = self._build_request_params(prompt, system_prompt)

        try:
            logger.debug(f"Generating text with OpenAI model {self.model_name}")

            response: ChatCompletion = await self._client().chat.completions.create(**request_params)

            generated_text = response.choices[0].message.content or ""

            logger.debug(f"Generated text length: {len(generated_text)}")
            if not generated_text:
                logger.info(
                    f"Empty response from OpenAI model {self.model_name}. "
                    f"Finish reason: {response.choices[0].finish_reason}, "
                    f"Refusal: {getattr(response.choices[0].message, 'refusal', None)}"
                )

            return generated_text

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating text with OpenAI: {error_msg}")

            # Some models only accept max_completion_tokens; retry once if that is why we failed.
            if "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
                logger.info(f"Attempting fallback with max_completion_tokens for model {self.model_name}")
                try:
                    fallback_params = dict(request_params)
                    fallback_params.pop("max_tokens", None)
                    fallback_params["max_completion_tokens"] = settings.openai_max_tokens

                    response = await self._client().chat.completions.create(**fallback_params)
                    generated_text = response.choices[0].message.content or ""

                    logger.info("Fallback successful with max_completion_tokens")
                    return generated_text

                except Exception as fallback_error:
                    logger.error(f"Fallback attempt also failed: {str(fallback_error)}")

            raise e

    async def generate_json(self, prompt, system_prompt=None, schema=None):
        """
        Generate a JSON response from the OpenAI model using JSON mode.

        `schema` is accepted for interface compatibility; OpenAI JSON mode does not
        constrain to it, so the response is parsed tolerantly.
        """
        if not self.api_key:
            logger.error("OpenAI API key not set")
            return {"error": "OpenAI API key not set"}

        try:
            logger.debug(f"Generating JSON with OpenAI model {self.model_name}")

            request_params = self._build_request_params(prompt, system_prompt)
            request_params["response_format"] = {"type": "json_object"}

            response: ChatCompletion = await self._client().chat.completions.create(**request_params)

            generated_text = response.choices[0].message.content or "{}"
            logger.debug(f"Generated JSON length: {len(generated_text)}")

            return json.loads(generated_text)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {"error": f"JSON parse error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error generating JSON with OpenAI: {str(e)}")
            return {"error": str(e)}


class AnthropicClient(LLMClient):
    """
    Client for interacting with Anthropic's Claude models.

    Claude rejects unsupported request parameters with a 400 instead of ignoring
    them, so requests are assembled through a capability gate rather than sending a
    fixed parameter set:

      * temperature / top_p are never sent - they 400 on Opus 5 and Sonnet 5.
        Response shaping uses `output_config={"effort": ...}` instead.
      * `effort` itself is only sent for models that accept it. Claude Haiku 4.5 -
        the default model - predates it and would reject the request.
      * `system` is a top-level parameter, not a message with role="system".
    """

    def __init__(self, model_name=None, api_key: str = None):
        """
        Initialize the Anthropic client.

        Args:
            model_name: Claude model id. Defaults to settings.anthropic_model.
            api_key: Anthropic API key. If None, uses the one from settings.
        """
        self.model_name = model_name or settings.anthropic_model
        self.api_key = api_key or settings.anthropic_api_key
        self._sdk_client = None
        self._sdk_loop_key = None

        logger.info(f"Initializing Anthropic client with model: {self.model_name}")
        if not self.api_key:
            logger.error("ANTHROPIC_API_KEY environment variable not set")

    def _client(self) -> AsyncAnthropic:
        """Get the async Anthropic client, rebuilt if the event loop changed."""
        return self._loop_bound_client(lambda: AsyncAnthropic(api_key=self.api_key))

    def get_model_status(self):
        """
        Get the status of the Claude model.

        Returns:
            dict: A dictionary with status information.
        """
        api_key_set = bool(self.api_key)

        return {
            "provider": LLMProvider.ANTHROPIC.value,
            "model_name": self.model_name,
            "available": api_key_set,
            "api_key_set": api_key_set
        }

    def _build_request_params(self, prompt, system_prompt) -> Dict[str, Any]:
        """
        Assemble request params for this model.

        Only parameters this model actually accepts are included - see the class
        docstring and ANTHROPIC_MODEL_CAPABILITIES in idiomapp.config.
        """
        params: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": settings.anthropic_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            params["system"] = system_prompt

        if anthropic_supports_effort(self.model_name):
            params["output_config"] = {"effort": settings.anthropic_effort}

        return params

    @staticmethod
    def _refusal_message(response) -> Optional[str]:
        """Return an error string if the model declined, else None."""
        if response.stop_reason != "refusal":
            return None
        category = getattr(response.stop_details, "category", None)
        logger.warning(f"Claude declined the request (category: {category})")
        return f"Error: request declined by the model ({category or 'unspecified'})"

    @staticmethod
    def _describe_error(e: Exception) -> str:
        """Map an SDK exception to the 'Error: ...' convention the UI expects."""
        if isinstance(e, anthropic.AuthenticationError):
            return "Error: Invalid Anthropic API key."
        if isinstance(e, anthropic.PermissionDeniedError):
            return "Error: Anthropic API key lacks permission for this model."
        if isinstance(e, anthropic.NotFoundError):
            return "Error: Unknown Claude model. Check ANTHROPIC_MODEL."
        if isinstance(e, anthropic.RateLimitError):
            return "Error: Anthropic rate limit reached. Please retry shortly."
        if isinstance(e, anthropic.APIStatusError):
            if e.status_code >= 500:
                return f"Error: Anthropic server error ({e.status_code}). Please retry."
            return f"Error: Anthropic API error ({e.status_code}): {e.message}"
        if isinstance(e, anthropic.APIConnectionError):
            return "Error: Could not reach the Anthropic API. Check your connection."
        return f"Error: {str(e)}"

    async def generate_text(self, prompt, system_prompt=None):
        """
        Generate text from the Claude model.

        Args:
            prompt: The prompt to send to the model.
            system_prompt: Optional system prompt for context.

        Returns:
            str: The generated text response, or an "Error: ..." string.
        """
        if not self.api_key:
            logger.error("Anthropic API key not set")
            return "Error: Anthropic API key not set. Please set the ANTHROPIC_API_KEY environment variable."

        try:
            logger.debug(f"Generating text with Claude model {self.model_name}")

            response = await self._client().messages.create(
                **self._build_request_params(prompt, system_prompt)
            )

            # Check the stop reason before touching content.
            refusal = self._refusal_message(response)
            if refusal:
                return refusal

            generated_text = "".join(
                block.text for block in response.content if block.type == "text"
            )

            logger.debug(f"Generated text length: {len(generated_text)}")
            if not generated_text:
                logger.info(
                    f"Empty response from Claude model {self.model_name}. "
                    f"Stop reason: {response.stop_reason}"
                )

            return generated_text

        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {str(e)}")
            return self._describe_error(e)

    async def generate_json(self, prompt, system_prompt=None, schema=None):
        """
        Generate a JSON object from the Claude model.

        With a Pydantic `schema`, structured output constrains the response so it is
        guaranteed to parse. The SDK derives the API schema from the model itself
        (stripping defaults, setting additionalProperties: false), so the tolerant
        model can be passed as-is.

        Structured output has not been exercised against every model this app offers,
        so a schema rejected as a bad request falls back to an unconstrained call
        once rather than failing the feature. Callers validate the result themselves,
        which keeps every provider's output on the same path.
        """
        if not self.api_key:
            logger.error("Anthropic API key not set")
            return {"error": "Anthropic API key not set"}

        try:
            params = self._build_request_params(prompt, system_prompt)

            if schema is not None:
                try:
                    logger.debug(f"Generating structured JSON with Claude model {self.model_name}")
                    response = await self._client().messages.parse(**params, output_format=schema)

                    refusal = self._refusal_message(response)
                    if refusal:
                        return {"error": refusal}

                    parsed = response.parsed_output
                    if parsed is not None:
                        return parsed.model_dump(exclude_none=True, by_alias=True)

                    logger.warning("Claude returned no structured output; retrying without a schema")
                except anthropic.BadRequestError as e:
                    logger.warning(
                        f"Structured output rejected for {self.model_name} ({e.message}); "
                        f"retrying without a schema"
                    )

            logger.debug(f"Generating JSON with Claude model {self.model_name}")
            response = await self._client().messages.create(**params)

            refusal = self._refusal_message(response)
            if refusal:
                return {"error": refusal}

            text = "".join(block.text for block in response.content if block.type == "text")
            if not text:
                return {"error": "Empty response from Claude"}

            result = extract_json(text)
            if result is None:
                return {"error": "Could not parse JSON from Claude response"}
            return result

        except Exception as e:
            logger.error(f"Error generating JSON with Anthropic: {str(e)}")
            return {"error": self._describe_error(e)}


# Helper functions for populating the model dropdowns.
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
        return list(settings.openai_models_list)

    try:
        from openai import OpenAI

        client_kwargs = {"api_key": api_key}
        if organization:
            client_kwargs["organization"] = organization

        client = OpenAI(**client_kwargs)
        response = client.models.list()

        available_models = sorted(
            model.id for model in response.data if model.id.startswith("gpt-")
        )

        logger.info(f"Successfully fetched {len(available_models)} available OpenAI models")
        return available_models or list(settings.openai_models_list)

    except Exception as e:
        logger.error(f"Error fetching OpenAI models: {str(e)}")
        return list(settings.openai_models_list)


def get_anthropic_available_models(api_key: str = None) -> list:
    """
    Get a list of available Claude models from the Anthropic API.

    Args:
        api_key: Anthropic API key. If None, tries to get from settings.

    Returns:
        list: List of available model ids, falling back to the configured list.
    """
    if not api_key:
        api_key = settings.anthropic_api_key

    if not api_key:
        logger.warning("No Anthropic API key provided, using configured model list")
        return list(settings.anthropic_models_list)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.models.list(limit=100)
        available_models = [model.id for model in response.data]

        logger.info(f"Successfully fetched {len(available_models)} available Claude models")
        return available_models or list(settings.anthropic_models_list)

    except Exception as e:
        logger.error(f"Error fetching Anthropic models: {str(e)}")
        return list(settings.anthropic_models_list)

# Ollama-specific functions have been moved to ollama_utils.py
