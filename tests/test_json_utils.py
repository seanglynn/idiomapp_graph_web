"""
Tests for `idiomapp.utils.json_utils.extract_json`.

This is the salvage path used only when a provider's own JSON guarantee wasn't
enough (Ollama wrapping output despite `format="json"`, or Claude's unconstrained
fallback) - it is not the primary parsing path for any provider.
"""

import json

from idiomapp.utils.json_utils import extract_json


def test_parses_plain_json():
    assert extract_json('{"translation": "hola"}') == {"translation": "hola"}


def test_parses_json_fenced_block_with_language_tag():
    response = (
        'Sure, here you go:\n```json\n{"translation": "hola"}\n```\n'
        "Let me know if you need anything else."
    )
    assert extract_json(response) == {"translation": "hola"}


def test_parses_bare_fenced_block_without_language_tag():
    response = '```\n{"a": 1}\n```'
    assert extract_json(response) == {"a": 1}


def test_returns_none_for_empty_response():
    assert extract_json("") is None


def test_returns_none_for_unparseable_text():
    assert extract_json("not json at all") is None


def test_prose_wrapped_json_without_a_fence_is_not_recovered():
    """
    Pins the intentionally narrowed contract: only the fenced case is a supported
    fallback now, not arbitrary prose-wrapping (the dropped balanced-brace
    scanner used to recover this, at the cost of a large, untested regex pass).
    """
    assert extract_json('Sure! {"a": 1} Hope that helps.') is None


def test_ignores_valid_json_that_is_not_an_object():
    assert extract_json("[1, 2, 3]") is None
    assert extract_json('"just a string"') is None


def test_does_not_corrupt_apostrophes():
    """Regression pin for the dropped quote-rewrite stage's known corruption bug."""
    response = json.dumps({"text": "don't stop"})
    assert extract_json(response) == {"text": "don't stop"}
