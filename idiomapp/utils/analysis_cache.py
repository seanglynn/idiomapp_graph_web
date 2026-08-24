"""
Persisted cache for LLM word-analysis results.

`analyze_word_linguistics()` in `nlp_utils.py` combines a fast, local spaCy pass
with a real LLM call for the rich linguistic detail (definition, etymology,
grammar, ...). The LLM call is the expensive, billed part - previously it ran on
every single "Analyze Selected Word" click, including re-analyzing a word already
analyzed earlier in the same session, or in a previous one. This module persists
that result to disk, keyed on the word plus the provider/model that produced it
(different models can give meaningfully different analyses, so a cache hit must
match all four), so the same word/language/provider/model combination is only
ever sent to the LLM once.

Mirrors the ABC + file-backed + in-memory pattern already used by
`graph_storage.py`, for the same reasons: a small, human-inspectable JSON store,
and an in-memory variant for tests.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from idiomapp.utils.logging_utils import get_logger

logger = get_logger("analysis_cache")


def _cache_key(word: str, language: str, provider: str, model: str) -> str:
    """Build a stable cache key. Word matching is case-insensitive."""
    return "|".join([word.strip().lower(), language, provider or "", model or ""])


class WordAnalysisCache(ABC):
    """Abstract base class for word-analysis cache backends."""

    @abstractmethod
    def get(
        self, word: str, language: str, provider: str, model: str
    ) -> Optional[Dict[str, Any]]:
        """Return the cached analysis dict, or None on a cache miss."""

    @abstractmethod
    def set(
        self,
        word: str,
        language: str,
        provider: str,
        model: str,
        analysis: Dict[str, Any],
    ) -> None:
        """Persist an analysis result for later lookup."""

    @abstractmethod
    def clear_all(self) -> bool:
        """Clear every cached entry. Returns True on success."""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return a small summary dict (entry count, storage size, ...)."""


class FileWordAnalysisCache(WordAnalysisCache):
    """File-persisted cache: one JSON file, one entry per cache key."""

    def __init__(self, storage_dir: str = "./word_analysis_cache"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.cache_file = self.storage_dir / "cache.json"
        if not self.cache_file.exists():
            self._save({})

    def _load(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading word analysis cache: {e}")
            return {}

    def _save(self, data: Dict[str, Dict[str, Any]]) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Error saving word analysis cache: {e}")

    def get(
        self, word: str, language: str, provider: str, model: str
    ) -> Optional[Dict[str, Any]]:
        key = _cache_key(word, language, provider, model)
        entry = self._load().get(key)
        return entry["analysis"] if entry else None

    def set(
        self,
        word: str,
        language: str,
        provider: str,
        model: str,
        analysis: Dict[str, Any],
    ) -> None:
        key = _cache_key(word, language, provider, model)
        data = self._load()
        data[key] = {
            "word": word,
            "language": language,
            "provider": provider,
            "model": model,
            "cached_at": datetime.now().isoformat(),
            "analysis": analysis,
        }
        self._save(data)
        logger.info(
            f"Cached word analysis for '{word}' ({language}, {provider}:{model})"
        )

    def clear_all(self) -> bool:
        try:
            self._save({})
            logger.info("Cleared word analysis cache")
            return True
        except Exception as e:
            logger.error(f"Error clearing word analysis cache: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        data = self._load()
        size_mb = 0.0
        if self.cache_file.exists():
            size_mb = round(self.cache_file.stat().st_size / (1024 * 1024), 2)
        return {"total_entries": len(data), "storage_size_mb": size_mb}


class InMemoryWordAnalysisCache(WordAnalysisCache):
    """In-memory cache for testing; data is lost when the process exits."""

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}

    def get(
        self, word: str, language: str, provider: str, model: str
    ) -> Optional[Dict[str, Any]]:
        key = _cache_key(word, language, provider, model)
        entry = self._data.get(key)
        return entry["analysis"] if entry else None

    def set(
        self,
        word: str,
        language: str,
        provider: str,
        model: str,
        analysis: Dict[str, Any],
    ) -> None:
        key = _cache_key(word, language, provider, model)
        self._data[key] = {
            "word": word,
            "language": language,
            "provider": provider,
            "model": model,
            "cached_at": datetime.now().isoformat(),
            "analysis": analysis,
        }

    def clear_all(self) -> bool:
        self._data.clear()
        return True

    def get_stats(self) -> Dict[str, Any]:
        return {"total_entries": len(self._data), "storage_size_mb": 0.0}


def get_word_analysis_cache() -> WordAnalysisCache:
    """Get the default word-analysis cache instance (file-persisted)."""
    return FileWordAnalysisCache()


def get_in_memory_word_analysis_cache() -> WordAnalysisCache:
    """Get an in-memory cache instance for testing."""
    return InMemoryWordAnalysisCache()
