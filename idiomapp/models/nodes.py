"""Node payloads: what a click on a graph node actually means.

Every variant carries a `kind` tag so Pydantic can tell them apart when
parsing a click event back from the browser, and so app.py's click-handling
code can pattern-match on "what kind of thing was clicked" instead of
guessing from loose dict keys.
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from idiomapp.models.identity import WordKey
from idiomapp.utils.logging_utils import get_logger

logger = get_logger("models.nodes")


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
# idiomapp.models.llm). Cognates look like single words too, but their
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
