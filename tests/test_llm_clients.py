"""
Tests for the LLM provider abstraction.

These focus on the contract the Streamlit app depends on - the factory, the status
dict shape, and JSON output - plus the Claude request-parameter rules, which fail as
a 400 at runtime rather than anything a type checker would catch.

Every provider SDK is mocked; nothing here makes a network call.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from idiomapp.config import anthropic_supports_effort
from idiomapp.utils.llm_utils import (
    AnthropicClient,
    LLMClient,
    OllamaClient,
    OpenAIClient,
)


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "provider, expected",
    [
        ("openai", OpenAIClient),
        ("anthropic", AnthropicClient),
    ],
)
def test_create_returns_the_right_client(provider, expected):
    client = LLMClient.create(provider=provider, api_key="test-key")
    assert isinstance(client, expected)


def test_create_builds_ollama_client():
    with patch("idiomapp.utils.llm_utils.get_available_models", return_value=["llama3.2:latest"]):
        client = LLMClient.create(provider="ollama", model_name="llama3.2:latest")
    assert isinstance(client, OllamaClient)


def test_create_falls_back_to_ollama_for_unknown_provider():
    with patch("idiomapp.utils.llm_utils.get_available_models", return_value=[]), \
         patch("idiomapp.utils.llm_utils.pull_model_if_needed", return_value=False):
        client = LLMClient.create(provider="not-a-provider")
    assert isinstance(client, OllamaClient)


# --------------------------------------------------------------------------
# Status dict shape - display_model_status() and get_llm_client() rely on it
# --------------------------------------------------------------------------
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_status_dict_shape(provider):
    status = LLMClient.create(provider=provider, api_key="test-key").get_model_status()
    assert {"provider", "model_name", "available"} <= set(status)
    assert status["provider"] == provider
    assert status["available"] is True


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_status_reports_unavailable_without_a_key(provider):
    status = LLMClient.create(provider=provider, api_key="").get_model_status()
    assert status["available"] is False


# --------------------------------------------------------------------------
# Claude request parameters
#
# Claude rejects unsupported parameters with a 400 rather than ignoring them, so
# these assertions guard against a silent runtime break.
# --------------------------------------------------------------------------
def test_claude_never_sends_temperature():
    """temperature/top_p are rejected on Opus 5 and Sonnet 5."""
    for model in ("claude-haiku-4-5", "claude-opus-5", "claude-sonnet-5"):
        params = AnthropicClient(model, api_key="k")._build_request_params("hi", None)
        assert "temperature" not in params
        assert "top_p" not in params


def test_claude_omits_effort_on_haiku():
    """Haiku 4.5 predates output_config.effort and 400s if it is sent."""
    params = AnthropicClient("claude-haiku-4-5", api_key="k")._build_request_params("hi", None)
    assert "output_config" not in params


def test_claude_sends_effort_on_current_models():
    for model in ("claude-opus-5", "claude-sonnet-5", "claude-opus-4-8"):
        params = AnthropicClient(model, api_key="k")._build_request_params("hi", None)
        assert "output_config" in params, model
        assert params["output_config"]["effort"]


def test_claude_system_prompt_is_top_level():
    """`system` is a request parameter, not a message with role="system"."""
    params = AnthropicClient("claude-opus-5", api_key="k")._build_request_params("hi", "be terse")
    assert params["system"] == "be terse"
    assert all(m["role"] != "system" for m in params["messages"])


def test_effort_capability_gate():
    assert anthropic_supports_effort("claude-opus-5") is True
    assert anthropic_supports_effort("claude-haiku-4-5") is False
    # Unknown ids are assumed current generation - that is where new models land.
    assert anthropic_supports_effort("claude-something-new") is True


# --------------------------------------------------------------------------
# Response handling
# --------------------------------------------------------------------------
def _fake_message(text="hola", stop_reason="end_turn", category=None):
    block = MagicMock()
    block.type = "text"
    block.text = text

    message = MagicMock()
    message.content = [block]
    message.stop_reason = stop_reason
    message.stop_details = MagicMock(category=category)
    return message


@pytest.mark.asyncio
async def test_claude_generate_text_returns_content():
    client = AnthropicClient("claude-haiku-4-5", api_key="k")
    sdk = MagicMock()
    sdk.messages.create = AsyncMock(return_value=_fake_message("hola"))
    client._sdk_client, client._sdk_loop_key = sdk, "fixed"

    with patch("idiomapp.utils.llm_utils.loop_key", return_value="fixed"):
        assert await client.generate_text("hello") == "hola"


@pytest.mark.asyncio
async def test_claude_refusal_becomes_an_error_string_not_an_exception():
    """The UI tests `translation.startswith("Error:")`, so refusals must not raise."""
    client = AnthropicClient("claude-opus-5", api_key="k")
    sdk = MagicMock()
    sdk.messages.create = AsyncMock(
        return_value=_fake_message(stop_reason="refusal", category="cyber")
    )
    client._sdk_client, client._sdk_loop_key = sdk, "fixed"

    with patch("idiomapp.utils.llm_utils.loop_key", return_value="fixed"):
        result = await client.generate_text("...")

    assert result.startswith("Error:")
    assert "cyber" in result


@pytest.mark.asyncio
async def test_claude_missing_key_returns_error_string():
    result = await AnthropicClient("claude-opus-5", api_key="").generate_text("hi")
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_claude_generate_json_uses_structured_output_when_given_a_schema():
    from pydantic import BaseModel

    class Translation(BaseModel):
        translation: str

    client = AnthropicClient("claude-opus-5", api_key="k")
    parsed = MagicMock()
    parsed.stop_reason = "end_turn"
    parsed.parsed_output = Translation(translation="hola")

    sdk = MagicMock()
    sdk.messages.parse = AsyncMock(return_value=parsed)
    client._sdk_client, client._sdk_loop_key = sdk, "fixed"

    with patch("idiomapp.utils.llm_utils.loop_key", return_value="fixed"):
        result = await client.generate_json("hi", schema=Translation)

    assert result == {"translation": "hola"}
    assert sdk.messages.parse.await_args.kwargs["output_format"] is Translation


@pytest.mark.asyncio
async def test_claude_generate_json_without_schema_parses_text():
    client = AnthropicClient("claude-haiku-4-5", api_key="k")
    sdk = MagicMock()
    sdk.messages.create = AsyncMock(return_value=_fake_message('{"translation": "hola"}'))
    client._sdk_client, client._sdk_loop_key = sdk, "fixed"

    with patch("idiomapp.utils.llm_utils.loop_key", return_value="fixed"):
        assert await client.generate_json("hi") == {"translation": "hola"}


@pytest.mark.asyncio
async def test_openai_generate_json_returns_a_dict():
    client = OpenAIClient("gpt-4o", api_key="k")
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content='{"translation": "hola"}'))]

    sdk = MagicMock()
    sdk.chat.completions.create = AsyncMock(return_value=completion)
    client._sdk_client, client._sdk_loop_key = sdk, "fixed"

    with patch("idiomapp.utils.llm_utils.loop_key", return_value="fixed"):
        assert await client.generate_json("hi") == {"translation": "hola"}

    assert sdk.chat.completions.create.await_args.kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_ollama_generate_json_uses_json_format():
    with patch("idiomapp.utils.llm_utils.get_available_models", return_value=["llama3.2:latest"]):
        client = OllamaClient("llama3.2:latest")

    sdk = MagicMock()
    sdk.chat = AsyncMock(return_value={"message": {"content": '{"translation": "hola"}'}})
    client._sdk_client, client._sdk_loop_key = sdk, "fixed"

    with patch("idiomapp.utils.llm_utils.loop_key", return_value="fixed"), \
         patch.object(OllamaClient, "_check_model_availability", return_value=True):
        assert await client.generate_json("hi") == {"translation": "hola"}

    assert sdk.chat.await_args.kwargs["format"] == "json"


def test_all_clients_implement_the_full_interface():
    """generate_json is part of the ABC, so no call site needs to duck-type it."""
    for cls in (OllamaClient, OpenAIClient, AnthropicClient):
        assert not getattr(cls, "__abstractmethods__", None), cls


# --------------------------------------------------------------------------
# Wire format
#
# _build_request_params covers what we assemble; this covers what the SDK
# actually puts on the wire, which is what the API would reject.
# --------------------------------------------------------------------------
def _capture_request_body(model: str) -> dict:
    """Run a generate_text call against a mock transport and return the request body."""
    import asyncio
    import json

    import httpx
    from anthropic import AsyncAnthropic

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    async def run():
        client = AnthropicClient(model, api_key="sk-ant-test")
        client._sdk_client = AsyncAnthropic(
            api_key="sk-ant-test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        client._sdk_loop_key = "fixed"
        with patch("idiomapp.utils.llm_utils.loop_key", return_value="fixed"):
            await client.generate_text("hello", "be terse")

    asyncio.run(run())
    return captured


def test_wire_body_for_haiku_has_no_effort_or_temperature():
    body = _capture_request_body("claude-haiku-4-5")
    assert "output_config" not in body
    assert "temperature" not in body
    assert body["system"] == "be terse"
    assert [m["role"] for m in body["messages"]] == ["user"]


def test_wire_body_for_opus_carries_effort():
    body = _capture_request_body("claude-opus-5")
    assert body["output_config"]["effort"]
    assert "temperature" not in body
