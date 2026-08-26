"""
Tests for idiomapp.config settings parsing.
"""

import pytest

from idiomapp.config import IdiomaAppSettings, LLMProvider, resolve_anthropic_model


@pytest.mark.parametrize("raw", ["claude", "Claude", "CLAUDE", " claude "])
def test_llm_provider_accepts_claude_as_an_alias_for_anthropic(raw, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", raw)
    assert IdiomaAppSettings().llm_provider is LLMProvider.ANTHROPIC


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("anthropic", LLMProvider.ANTHROPIC),
        ("ollama", LLMProvider.OLLAMA),
        ("openai", LLMProvider.OPENAI),
    ],
)
def test_llm_provider_accepts_its_canonical_values(raw, expected, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", raw)
    assert IdiomaAppSettings().llm_provider is expected


def test_llm_provider_rejects_an_unknown_value(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
    with pytest.raises(Exception):
        IdiomaAppSettings()


def test_resolve_anthropic_model_prefers_an_exact_match():
    available = ["claude-haiku-4-5", "claude-opus-5"]
    assert resolve_anthropic_model("claude-haiku-4-5", available) == "claude-haiku-4-5"


def test_resolve_anthropic_model_finds_a_dated_snapshot_of_a_configured_alias():
    # This is the real shape `models.list()` returns: the configured rolling
    # alias isn't itself in the list, only its current dated snapshot is.
    available = [
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-5-20251101",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-5-20250929",
    ]
    assert (
        resolve_anthropic_model("claude-haiku-4-5", available)
        == "claude-haiku-4-5-20251001"
    )


def test_resolve_anthropic_model_returns_none_when_nothing_matches():
    assert resolve_anthropic_model("claude-haiku-4-5", ["claude-opus-5"]) is None


def test_resolve_anthropic_model_does_not_match_a_different_generation():
    # "claude-opus-4" must not match "claude-opus-4-5-20251101" - that's a later
    # generation's snapshot (Opus 4.5), not a snapshot of plain Opus 4.
    available = ["claude-opus-4-5-20251101"]
    assert resolve_anthropic_model("claude-opus-4", available) is None
