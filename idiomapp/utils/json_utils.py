"""
Tolerant JSON extraction from LLM responses.

Providers differ in how reliably they honour "reply with JSON only": Anthropic can
be given a schema and will return exactly that, OpenAI has a JSON mode, and a local
Ollama model may wrap its answer in prose or a markdown fence. This module holds the
one salvage routine all providers share, so the recovery logic lives in a single
place instead of being re-implemented at each call site.
"""

import json
import re
from typing import Any, Dict, Optional

from idiomapp.utils.logging_utils import get_logger

logger = get_logger("json_utils")


def extract_json(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse a JSON object out of a raw LLM response.

    Tries the whole response first - what every well-behaved provider actually
    returns (OpenAI's JSON mode, Claude's unconstrained fallback, Ollama's
    grammar-constrained ``format="json"``). Falls back to the contents of a
    ```json ... ``` or bare ``` ... ``` fenced block - the one failure mode this
    app has actually needed a fallback for (a local Ollama model wrapping its
    answer in a fence despite ``format="json"``).

    An earlier version also scanned surrounding prose for the first balanced
    ``{...}`` and rewrote quotes/trailing commas before retrying. Both are
    dropped: neither had any test coverage, and the quote rewrite corrupted
    ordinary contractions in translated text (``"don't"`` -> ``"don"t"``).

    Args:
        response: Raw text returned by the model.

    Returns:
        The parsed object, or None if nothing usable was found.
    """
    if not response:
        return None

    parsed = _parse_object(response)
    if parsed is not None:
        return parsed

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
    if fenced:
        parsed = _parse_object(fenced.group(1))
        if parsed is not None:
            return parsed
        logger.debug("Fenced block was not valid JSON either")

    logger.warning("Could not parse a JSON object from the LLM response")
    return None


def _parse_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse *text* as JSON, returning it only if the result is an object."""
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None
