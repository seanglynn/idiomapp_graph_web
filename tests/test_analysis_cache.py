"""
Tests for the persisted word-analysis cache (idiomapp/utils/analysis_cache.py)
and its wiring into nlp_utils.analyze_word_linguistics.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from idiomapp.utils.analysis_cache import (
    FileWordAnalysisCache,
    InMemoryWordAnalysisCache,
)


# --------------------------------------------------------------------------
# Cache backends
# --------------------------------------------------------------------------
def test_in_memory_cache_round_trips():
    cache = InMemoryWordAnalysisCache()
    assert cache.get("gato", "es", "anthropic", "claude-haiku-4-5") is None

    cache.set("gato", "es", "anthropic", "claude-haiku-4-5", {"definition": "cat"})
    assert cache.get("gato", "es", "anthropic", "claude-haiku-4-5") == {
        "definition": "cat"
    }


def test_cache_key_is_case_insensitive_on_the_word():
    cache = InMemoryWordAnalysisCache()
    cache.set("Gato", "es", "anthropic", "m", {"definition": "cat"})
    assert cache.get("gato", "es", "anthropic", "m") == {"definition": "cat"}
    assert cache.get("GATO", "es", "anthropic", "m") == {"definition": "cat"}


def test_cache_key_distinguishes_provider_and_model():
    """A different provider/model must not serve another's cached result -
    they can produce meaningfully different analyses."""
    cache = InMemoryWordAnalysisCache()
    cache.set("gato", "es", "anthropic", "claude-haiku-4-5", {"definition": "a"})
    assert cache.get("gato", "es", "anthropic", "claude-opus-5") is None
    assert cache.get("gato", "es", "ollama", "claude-haiku-4-5") is None


def test_cache_key_distinguishes_language():
    cache = InMemoryWordAnalysisCache()
    cache.set("gato", "es", "anthropic", "m", {"definition": "cat"})
    assert cache.get("gato", "ca", "anthropic", "m") is None


def test_clear_all_empties_the_cache():
    cache = InMemoryWordAnalysisCache()
    cache.set("gato", "es", "anthropic", "m", {"definition": "cat"})
    assert cache.clear_all() is True
    assert cache.get("gato", "es", "anthropic", "m") is None
    assert cache.get_stats()["total_entries"] == 0


def test_file_cache_persists_across_instances(tmp_path):
    """A fresh FileWordAnalysisCache pointed at the same directory must see
    entries written by a previous instance - this is the whole point of a
    persisted cache (surviving process restarts)."""
    first = FileWordAnalysisCache(storage_dir=str(tmp_path))
    first.set(
        "hablar", "es", "anthropic", "claude-haiku-4-5", {"definition": "to speak"}
    )

    second = FileWordAnalysisCache(storage_dir=str(tmp_path))
    assert second.get("hablar", "es", "anthropic", "claude-haiku-4-5") == {
        "definition": "to speak"
    }
    assert second.get_stats()["total_entries"] == 1


def test_file_cache_creates_its_storage_directory(tmp_path):
    target = tmp_path / "does_not_exist_yet"
    FileWordAnalysisCache(storage_dir=str(target))
    assert target.is_dir()
    assert (target / "cache.json").exists()


# --------------------------------------------------------------------------
# Wiring into analyze_word_linguistics
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_analyze_word_linguistics_caches_a_successful_llm_result():
    import spacy

    from idiomapp.utils.nlp_utils import analyze_word_linguistics

    cache = InMemoryWordAnalysisCache()
    client = MagicMock()
    client.generate_json = AsyncMock(return_value={"definition": "a cat"})
    client.get_model_status = MagicMock(
        return_value={"available": True, "provider": "anthropic", "model_name": "m"}
    )

    with patch(
        "idiomapp.utils.nlp_utils.load_spacy_model", return_value=spacy.blank("es")
    ), patch("idiomapp.utils.nlp_utils.get_word_analysis_cache", return_value=cache):
        await analyze_word_linguistics("gato", "es", client)

    assert cache.get("gato", "es", "anthropic", "m") is not None
    # One call per WordAnalysis group (meaning/usage/grammar/pronunciation/
    # learner_notes) - see nlp_utils._get_llm_word_analysis.
    assert client.generate_json.await_count == 5


@pytest.mark.asyncio
async def test_analyze_word_linguistics_serves_a_cache_hit_without_calling_the_llm():
    import spacy

    from idiomapp.utils.nlp_utils import analyze_word_linguistics

    cache = InMemoryWordAnalysisCache()
    cache.set(
        "gato",
        "es",
        "anthropic",
        "m",
        {"word": "gato", "language": "es", "pos": "NOUN", "definition": "cached cat"},
    )
    client = MagicMock()
    client.generate_json = AsyncMock(return_value={"definition": "should not be used"})
    client.get_model_status = MagicMock(
        return_value={"available": True, "provider": "anthropic", "model_name": "m"}
    )

    with patch(
        "idiomapp.utils.nlp_utils.load_spacy_model", return_value=spacy.blank("es")
    ) as mock_load_spacy, patch(
        "idiomapp.utils.nlp_utils.get_word_analysis_cache", return_value=cache
    ):
        analysis = await analyze_word_linguistics("gato", "es", client)

    assert analysis["definition"] == "cached cat"
    assert client.generate_json.await_count == 0
    mock_load_spacy.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_word_linguistics_does_not_cache_an_llm_error():
    """A transient LLM failure must not get "stuck" cached - the next call
    should retry the LLM rather than silently reusing a degraded result."""
    import spacy

    from idiomapp.utils.nlp_utils import analyze_word_linguistics

    cache = InMemoryWordAnalysisCache()
    client = MagicMock()
    client.generate_json = AsyncMock(return_value={"error": "rate limited"})
    client.get_model_status = MagicMock(
        return_value={"available": True, "provider": "anthropic", "model_name": "m"}
    )

    with patch(
        "idiomapp.utils.nlp_utils.load_spacy_model", return_value=spacy.blank("es")
    ), patch("idiomapp.utils.nlp_utils.get_word_analysis_cache", return_value=cache):
        await analyze_word_linguistics("gato", "es", client)

    assert cache.get("gato", "es", "anthropic", "m") is None


@pytest.mark.asyncio
async def test_analyze_word_linguistics_without_a_client_never_touches_the_cache():
    import spacy

    from idiomapp.utils.nlp_utils import analyze_word_linguistics

    cache = InMemoryWordAnalysisCache()

    with patch(
        "idiomapp.utils.nlp_utils.load_spacy_model", return_value=spacy.blank("es")
    ), patch("idiomapp.utils.nlp_utils.get_word_analysis_cache", return_value=cache):
        await analyze_word_linguistics("gato", "es", client=None)

    assert cache.get_stats()["total_entries"] == 0
