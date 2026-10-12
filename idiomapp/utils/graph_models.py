"""The typed models the app's graphs are made of: graph elements (what a node
or edge *is*) and requests (what a click or an expansion *asks for*).

Kept apart from `graph_viz.py`, which builds graphs out of these models and
renders them for ECharts. Nothing here imports networkx or any rendering
code, so this module stays a plain description of what the graph knows.

Payloads that cross the browser boundary as JSON are Pydantic models;
everything that only lives inside Python is a plain dataclass. See
`docs/graph_data_model.md` for why, and for a diagram of how these types
relate.
"""

from dataclasses import dataclass
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from idiomapp.utils.logging_utils import get_logger

logger = get_logger("graph_models")


# ===========================================================================
# Graph elements
# ===========================================================================


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


# ---------------------------------------------------------------------------
# Node payloads: what a click on a node actually means. Every variant carries
# a `kind` tag so Pydantic can tell them apart when parsing a click event back
# from the browser, and so app.py's click-handling code can pattern-match on
# "what kind of thing was clicked" instead of guessing from loose dict keys.
# ---------------------------------------------------------------------------


class SemanticWordPayload(BaseModel):
    """A word node from the main translation graph (built from the sentence
    the user typed in, before any exploration happens) - the app's plain,
    ordinary "here's a word" node. Clicking one triggers word analysis."""

    kind: Literal["word"] = "word"
    id: str
    label: str
    language: str
    pos: str
    details: str
    node_type: str
    group: str
    sentence_group: str = ""


class CategoryPayload(BaseModel):
    """A category hub attached to an analyzed word - e.g. "Synonyms" or
    "Etymology". Clicking it shows or hides that category's results
    (a simple on/off toggle, not an analysis trigger)."""

    kind: Literal["category"] = "category"
    word_key: WordKey
    category: str
    label: str


# Which categories are safe to recurse into when a leaf is clicked: synonyms
# and antonyms are always a single word in the *same* language (see
# idiomapp.utils.schemas). Cognates look like single words too, but their
# language is a free-text gloss ("English", "French", ...) this app can't map
# to a supported language code, so they're deliberately excluded. Everything
# else (idioms, proverbs, examples, collocations, conjugations, forms) is a
# phrase or sentence, not a word to re-analyze.
RECURSIVE_LEAF_CATEGORIES = frozenset({"synonyms", "antonyms"})


class RecursiveLeafPayload(BaseModel):
    """A leaf result that is itself a real word in the same language (a
    synonym or antonym - see RECURSIVE_LEAF_CATEGORIES) - clicking it
    analyzes *that* word too and lets exploration continue from there, the
    same way the original word did."""

    kind: Literal["item"] = "item"
    word_key: WordKey
    category: str


class LeafNodePayload(BaseModel):
    """A leaf result that is just information to read - an idiom, an
    example sentence, an etymology detail, and so on. Clicking it does
    nothing; it isn't a word the app can meaningfully re-analyze.

    `text` is the term itself (or the whole item, for a category that's just
    a flat list of strings to begin with). `gloss` is set only for a category
    where the LLM returned each item as a separate term/gloss pair (e.g. a
    cognate's word and which language it's a cognate in) - None otherwise."""

    kind: Literal["leaf_node"] = "leaf_node"
    category: str
    text: str
    gloss: Optional[str] = None


class CooccurrenceWordPayload(BaseModel):
    """A word node from the (unrelated, unchanged) Co-occurrence Network
    graph. Kept as its own typed payload for consistency, even though this
    graph isn't part of the new exploration feature."""

    kind: Literal["cooccurrence_word"] = "cooccurrence_word"
    word: str
    language: Optional[str] = None
    degree: int


NodePayload = Annotated[
    Union[
        SemanticWordPayload,
        CategoryPayload,
        RecursiveLeafPayload,
        LeafNodePayload,
        CooccurrenceWordPayload,
    ],
    Field(discriminator="kind"),
]
_node_payload_adapter: TypeAdapter = TypeAdapter(NodePayload)


class EdgePayload(BaseModel):
    """What a click on an edge/connection means - always just informational
    (no edge triggers an action), so unlike nodes every edge kind shares one
    shape rather than needing its own class."""

    kind: str
    label: str = ""
    description: str = ""
    strength: Optional[float] = None
    weight: Optional[int] = None


def parse_node_payload(raw: dict) -> Optional[NodePayload]:
    """Reconstruct a typed NodePayload from the plain dict a browser click
    hands back. Returns None for anything unrecognized rather than raising -
    a click that can't be understood should be silently ignored from the
    user's point of view (no crash), but is logged as a warning: by the time
    this is called a real click already happened, so a validation failure
    here means the frontend sent something this app's own code didn't
    produce (or a payload variant genuinely doesn't match its own schema),
    which is worth knowing about even though it isn't fatal.
    """
    try:
        return _node_payload_adapter.validate_python(raw)
    except ValidationError as exc:
        logger.warning(f"Could not parse a clicked node's data: {exc}")
        return None


def parse_edge_payload(raw: dict) -> Optional[EdgePayload]:
    """Reconstruct a typed EdgePayload from the plain dict a browser click
    hands back, or None (logged as a warning, for the same reason as
    `parse_node_payload`) if it doesn't look like a valid edge payload."""
    try:
        return EdgePayload.model_validate(raw)
    except ValidationError as exc:
        logger.warning(f"Could not parse a clicked edge's data: {exc}")
        return None


# ===========================================================================
# Requests
# ===========================================================================


@dataclass(frozen=True)
class NodeSelection:
    """What a click on a node resolved to: the typed payload describing what
    was clicked, plus its on-screen position if the browser reported one
    (used to pin the node in place - see `graph_viz.apply_pinned_positions`)."""

    payload: NodePayload
    position: Optional[Position] = None


@dataclass(frozen=True)
class EdgeSelection:
    """What a click on an edge resolved to: its descriptive payload, plus the
    ids of the two nodes it connects (so the panel can show "X -> Y")."""

    payload: EdgePayload
    source_id: str
    target_id: str


GraphSelection = Union[NodeSelection, EdgeSelection]


@dataclass(frozen=True)
class SourcedSelection:
    """A GraphSelection tagged with which graph it came from (the Semantic
    Graph or the Co-occurrence Network) - kept together so switching between
    the two graphs' tabs doesn't show a leftover selection from the other
    one, which would otherwise be confusing rather than just visually off."""

    selection: GraphSelection
    source: Literal["semantic", "cooccurrence"]


@dataclass(frozen=True, slots=True)
class WordExpansionRequest:
    """What to expand for one already-analyzed word: which word, which
    existing graph node its category hubs attach to, and which of its
    categories are currently toggled open (only those show their leaves).
    See `graph_viz.build_word_expansion`."""

    word_key: WordKey
    attach_node_id: str
    open_categories: frozenset[str]
