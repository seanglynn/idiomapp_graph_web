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
    Pull a JSON object out of a raw LLM response.

    Tries, in order: a ```json fenced block, the first balanced {...} object, and
    finally that same object with the two mistakes models actually make (trailing
    commas, single-quoted strings) repaired.

    Args:
        response: Raw text returned by the model.

    Returns:
        The parsed object, or None if nothing usable was found.
    """
    if not response:
        return None

    # 1. Fenced code block.
    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            logger.debug("Fenced block was not valid JSON, falling through")

    # 2. First balanced object in the text.
    json_str = _first_balanced_object(response)
    if json_str is None:
        logger.warning("No JSON object found in LLM response")
        return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 3. Repair the common malformations and retry once.
    repaired = re.sub(r',\s*([}\]])', r'\1', json_str)   # trailing commas
    repaired = repaired.replace("'", '"')                # single-quoted strings
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        logger.warning(f"Could not parse JSON from LLM response: {e}")
        return None


def _first_balanced_object(text: str) -> Optional[str]:
    """Return the first brace-balanced {...} substring, or None."""
    depth = 0
    start = -1
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start != -1:
                return text[start:i + 1]
    return None
