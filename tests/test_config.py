"""
Tests for idiomapp.config settings parsing.
"""

import pytest

from idiomapp.config import IdiomaAppSettings, LLMProvider


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
