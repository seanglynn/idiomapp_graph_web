"""Small value types the other models are built from."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WordKey:
    """The identity of one (word, language) pair, used everywhere this app
    needs to remember "have I already analyzed this word" - normalized the
    same way `analysis_cache`'s own on-disk cache key is (lowercased,
    whitespace-trimmed), so the two stay in sync without extra bookkeeping.
    """

    word: str
    language: str

    @classmethod
    def of(cls, word: str, language: str) -> "WordKey":
        """Build a WordKey from raw (possibly mixed-case/padded) input."""
        return cls(word.strip().lower(), language)


@dataclass(frozen=True, slots=True)
class Position:
    """A node's on-screen coordinates, captured from a click so it can be
    held in place (pinned) the next time the graph is redrawn - see
    `graph_viz.apply_pinned_positions` for why that matters."""

    x: float
    y: float
