"""
Tests for the LLM provider abstraction.

These focus on the contract the Streamlit app depends on - the factory, the status
dict shape, and JSON output - plus the Claude request-parameter rules, which fail as
a 400 at runtime rather than anything a type checker would catch.

Every provider SDK is mocked; nothing here makes a network call.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import BadRequestError

from idiomapp.config import anthropic_supports_effort, settings
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
    with patch(
        "idiomapp.utils.llm_utils.get_available_models",
        return_value=["llama3.2:latest"],
    ):
        client = LLMClient.create(provider="ollama", model_name="llama3.2:latest")
    assert isinstance(client, OllamaClient)


def test_create_falls_back_to_ollama_for_unknown_provider():
    with patch("idiomapp.utils.llm_utils.get_available_models", return_value=[]), patch(
        "idiomapp.utils.llm_utils.pull_model_if_needed", return_value=False
    ):
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
def test_status_reports_unavailable_without_a_key(provider, monkeypatch):
    """
    An explicit empty api_key must mean "no key", even if a real one happens to be
    configured in the environment - `LLMClient.create` falls back to `settings.*`
    only when the caller does not supply a key at all.
    """
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
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
    params = AnthropicClient("claude-haiku-4-5", api_key="k")._build_request_params(
        "hi", None
    )
    assert "output_config" not in params


def test_claude_sends_effort_on_current_models():
    for model in ("claude-opus-5", "claude-sonnet-5", "claude-opus-4-8"):
        params = AnthropicClient(model, api_key="k")._build_request_params("hi", None)
        assert "output_config" in params, model
        assert params["output_config"]["effort"]


def test_claude_system_prompt_is_top_level():
    """`system` is a request parameter, not a message with role="system"."""
    params = AnthropicClient("claude-opus-5", api_key="k")._build_request_params(
        "hi", "be terse"
    )
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
async def test_claude_missing_key_returns_error_string(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
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
    sdk.messages.create = AsyncMock(
        return_value=_fake_message('{"translation": "hola"}')
    )
    client._sdk_client, client._sdk_loop_key = sdk, "fixed"

    with patch("idiomapp.utils.llm_utils.loop_key", return_value="fixed"):
        assert await client.generate_json("hi") == {"translation": "hola"}


@pytest.mark.asyncio
async def test_openai_generate_json_returns_a_dict():
    client = OpenAIClient("gpt-4o", api_key="k")
    completion = MagicMock()
    completion.choices = [
        MagicMock(message=MagicMock(content='{"translation": "hola"}'))
    ]

    sdk = MagicMock()
    sdk.chat.completions.create = AsyncMock(return_value=completion)
    client._sdk_client, client._sdk_loop_key = sdk, "fixed"

    with patch("idiomapp.utils.llm_utils.loop_key", return_value="fixed"):
        assert await client.generate_json("hi") == {"translation": "hola"}

    assert sdk.chat.completions.create.await_args.kwargs["response_format"] == {
        "type": "json_object"
    }


@pytest.mark.asyncio
async def test_openai_generate_text_retries_when_capabilities_are_unconfirmed():
    """An unlisted model's first 400 is retried once with the other token param."""
    import httpx

    client = OpenAIClient("some-mystery-model-9000", api_key="k")
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content="hola"))]

    sdk = MagicMock()
    sdk.chat.completions.create = AsyncMock(
        side_effect=[
            BadRequestError(
                message="Unsupported parameter: 'max_tokens'",
                response=httpx.Response(
                    400, request=httpx.Request("POST", "https://x")
                ),
                body=None,
            ),
            completion,
        ]
    )
    client._sdk_client, client._sdk_loop_key = sdk, "fixed"

    with patch("idiomapp.utils.llm_utils.loop_key", return_value="fixed"):
        result = await client.generate_text("hi")

    assert result == "hola"
    assert sdk.chat.completions.create.await_count == 2
    second_call_kwargs = sdk.chat.completions.create.await_args_list[1].kwargs
    assert "max_completion_tokens" in second_call_kwargs
    assert "max_tokens" not in second_call_kwargs


@pytest.mark.asyncio
async def test_openai_generate_text_does_not_retry_when_capabilities_are_confirmed():
    """A known model's 400 is a real error, not a guess to second-guess."""
    import httpx

    client = OpenAIClient("gpt-4o", api_key="k")
    sdk = MagicMock()
    sdk.chat.completions.create = AsyncMock(
        side_effect=BadRequestError(
            message="some unrelated error",
            response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
            body=None,
        )
    )
    client._sdk_client, client._sdk_loop_key = sdk, "fixed"

    with patch("idiomapp.utils.llm_utils.loop_key", return_value="fixed"):
        with pytest.raises(BadRequestError):
            await client.generate_text("hi")

    assert sdk.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_openai_generate_json_also_gets_the_retry_safety_net():
    """generate_json shares _create_chat_completion, so it is no longer asymmetric
    with generate_text - previously it had no retry safety net at all."""
    import httpx

    client = OpenAIClient("some-mystery-model-9000", api_key="k")
    completion = MagicMock()
    completion.choices = [
        MagicMock(message=MagicMock(content='{"translation": "hola"}'))
    ]

    sdk = MagicMock()
    sdk.chat.completions.create = AsyncMock(
        side_effect=[
            BadRequestError(
                message="Unsupported parameter: 'max_tokens'",
                response=httpx.Response(
                    400, request=httpx.Request("POST", "https://x")
                ),
                body=None,
            ),
            completion,
        ]
    )
    client._sdk_client, client._sdk_loop_key = sdk, "fixed"

    with patch("idiomapp.utils.llm_utils.loop_key", return_value="fixed"):
        result = await client.generate_json("hi")

    assert result == {"translation": "hola"}
    assert sdk.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_ollama_generate_json_uses_json_format():
    with patch(
        "idiomapp.utils.llm_utils.get_available_models",
        return_value=["llama3.2:latest"],
    ):
        client = OllamaClient("llama3.2:latest")

    sdk = MagicMock()
    sdk.chat = AsyncMock(
        return_value={"message": {"content": '{"translation": "hola"}'}}
    )
    client._sdk_client, client._sdk_loop_key = sdk, "fixed"

    with patch("idiomapp.utils.llm_utils.loop_key", return_value="fixed"), patch.object(
        OllamaClient, "_check_model_availability", return_value=True
    ):
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
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

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


# --------------------------------------------------------------------------
# Schema handling and provider parity
# --------------------------------------------------------------------------
def test_claude_401_maps_to_error_string():
    """Pins the mapping confirmed live against api.anthropic.com with a dummy key."""
    import anthropic
    import httpx

    err = anthropic.AuthenticationError(
        message="API key is invalid.",
        response=httpx.Response(401, request=httpx.Request("POST", "https://x")),
        body=None,
    )
    assert AnthropicClient._describe_error(err) == "Error: Invalid Anthropic API key."


@pytest.mark.asyncio
async def test_claude_falls_back_when_schema_is_rejected():
    """A schema the API refuses must not kill the feature - retry unconstrained once."""
    import anthropic
    import httpx

    from idiomapp.utils.schemas import WordAnalysis

    client = AnthropicClient("claude-haiku-4-5", api_key="k")
    sdk = MagicMock()
    sdk.messages.parse = AsyncMock(
        side_effect=anthropic.BadRequestError(
            message="output_config.format is not supported",
            response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
            body=None,
        )
    )
    sdk.messages.create = AsyncMock(
        return_value=_fake_message('{"definition": "a cat"}')
    )
    client._sdk_client, client._sdk_loop_key = sdk, "fixed"

    with patch("idiomapp.utils.llm_utils.loop_key", return_value="fixed"):
        result = await client.generate_json("analyse", schema=WordAnalysis)

    assert result == {"definition": "a cat"}
    assert sdk.messages.parse.await_count == 1
    assert sdk.messages.create.await_count == 1


@pytest.mark.asyncio
async def test_claude_schema_path_returns_canonical_dict():
    """
    Uses `Usage`, not `WordAnalysis`, because that is what real word-analysis calls
    actually send now - see the `WordAnalysis` docstring in schemas.py for why the
    combined model is never itself passed as a structured-output schema.
    """
    from idiomapp.utils.schemas import Usage

    client = AnthropicClient("claude-opus-5", api_key="k")
    parsed = MagicMock()
    parsed.stop_reason = "end_turn"
    parsed.parsed_output = Usage.model_validate({"idioms": {"dar la lata": "to annoy"}})

    sdk = MagicMock()
    sdk.messages.parse = AsyncMock(return_value=parsed)
    client._sdk_client, client._sdk_loop_key = sdk, "fixed"

    with patch("idiomapp.utils.llm_utils.loop_key", return_value="fixed"):
        result = await client.generate_json("analyse", schema=Usage)

    assert result["idioms"] == [{"term": "dar la lata", "gloss": "to annoy"}]
    assert sdk.messages.parse.await_args.kwargs["output_format"] is Usage


@pytest.mark.asyncio
async def test_providers_agree_on_shape():
    """
    The same awkward payload through Ollama and Claude produces the same analysis.

    `_get_llm_word_analysis` fires one call per WordAnalysis group concurrently and
    nests each group's raw response under its own group key before validation
    (`merged[group_name] = result`). Ollama is not schema-aware, so it is mocked to
    answer with the same flat payload every time regardless of which group's
    prompt it received - each group's submodel (`extra="ignore"`) simply keeps the
    fields it declares and drops the rest via ordinary field-name matching, no
    validator does cross-group routing. Claude *is* schema-constrained per call,
    so its mock must answer according to whichever group schema each call
    actually asked for.
    """
    from idiomapp.utils.nlp_utils import _get_llm_word_analysis
    from idiomapp.utils.schemas import (
        Grammar,
        LearnerNotes,
        Meaning,
        Pronunciation,
        Usage,
    )

    payload = (
        '{"definition":"a cat","idioms":{"dar la lata":"to annoy"},"synonyms":"felino"}'
    )

    with patch("idiomapp.utils.llm_utils.get_available_models", return_value=["m"]):
        ollama_client = OllamaClient("m")
    ollama_sdk = MagicMock()
    ollama_sdk.chat = AsyncMock(return_value={"message": {"content": payload}})
    ollama_client._sdk_client, ollama_client._sdk_loop_key = ollama_sdk, "fixed"

    # What each group schema should plausibly be asked to return for this payload.
    group_payloads = {
        Meaning: {"definition": "a cat", "synonyms": "felino"},
        Usage: {"idioms": {"dar la lata": "to annoy"}},
        Grammar: {},
        Pronunciation: {},
        LearnerNotes: {},
    }

    async def fake_parse(**kwargs):
        schema = kwargs["output_format"]
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.parsed_output = schema.model_validate(group_payloads[schema])
        return response

    claude_client = AnthropicClient("claude-opus-5", api_key="k")
    claude_sdk = MagicMock()
    claude_sdk.messages.parse = AsyncMock(side_effect=fake_parse)
    claude_client._sdk_client, claude_client._sdk_loop_key = claude_sdk, "fixed"

    with patch("idiomapp.utils.llm_utils.loop_key", return_value="fixed"), patch.object(
        OllamaClient, "_check_model_availability", return_value=True
    ):
        via_ollama = await _get_llm_word_analysis("gato", "es", "NOUN", ollama_client)
        via_claude = await _get_llm_word_analysis("gato", "es", "NOUN", claude_client)

    assert via_ollama == via_claude
    assert via_ollama["idioms"] == [{"term": "dar la lata", "gloss": "to annoy"}]
    assert via_ollama["synonyms"] == ["felino"]


@pytest.mark.asyncio
async def test_spacy_keys_survive_llm_merge():
    """
    The LLM half is merged onto spaCy's, not substituted for it.

    spaCy is stubbed with a blank pipeline so the suite never reaches for a model
    download - CI has no pretrained models installed and must stay offline.
    """
    import spacy

    from idiomapp.utils.analysis_cache import InMemoryWordAnalysisCache
    from idiomapp.utils.nlp_utils import analyze_word_linguistics

    client = MagicMock()
    client.generate_json = AsyncMock(return_value={"definition": "a cat"})
    client.get_model_status = MagicMock(
        return_value={"available": True, "provider": "anthropic", "model_name": "m"}
    )

    with patch(
        "idiomapp.utils.nlp_utils.load_spacy_model", return_value=spacy.blank("es")
    ), patch(
        "idiomapp.utils.nlp_utils.get_word_analysis_cache",
        return_value=InMemoryWordAnalysisCache(),
    ):
        analysis = await analyze_word_linguistics("gato", "es", client)

    assert analysis["word"] == "gato"
    assert "pos" in analysis and "lemma" in analysis  # from spaCy
    assert analysis["definition"] == "a cat"  # from the LLM
