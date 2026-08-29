"""Adapters that turn this app's graph data into ECharts graph-series options,
plus the typed data model the three graphs (Semantic, Co-occurrence, and the
in-graph word-exploration nodes) are all built from.

Pure Python - no `streamlit` or `streamlit_echarts` import here, so every
function is directly unit-testable. The graph itself is represented as a real
`networkx.MultiDiGraph` (not a hand-rolled pair of node/edge lists) while it's
being built and grown; `graph_to_echarts_data` is the *only* place that graph
gets flattened into the plain JSON dict a browser-based JS component needs.
See `docs/graph_data_model.md` for a diagram of how all these types relate.

Two kinds of typed object are used, on purpose:
- Plain `@dataclass`es (`WordKey`, `Position`, `StyledNode`, `StyledEdge`,
  `NodeSelection`, `EdgeSelection`) for things that only ever live inside
  Python and never need to be re-validated from the outside.
- Pydantic models (every `NodePayload`/`EdgePayload` variant) for the pieces
  that get sent to the browser as JSON and come back from a click as JSON too
  - Pydantic's discriminated unions turn "which kind of thing did the user
  click?" into real, validated parsing instead of hand-written dict digging.
"""

import re
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal, Optional, Union

import networkx as nx
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from idiomapp.config import (
    GROUP_COLORS,
    LANGUAGE_MAP,
    POS_BORDER_COLORS,
    RELATION_COLORS,
)
from idiomapp.utils.logging_utils import get_logger
from idiomapp.utils.nlp_utils import get_language_color

logger = get_logger("graph_viz")

# Graph category config for word-exploration hubs: (data_key, label, color)
GRAPH_CATEGORIES = [
    ("cognates", "🌍 Cognates", "#E91E63"),
    ("synonyms", "≈ Synonyms", "#3498DB"),
    ("antonyms", "≠ Antonyms", "#E74C3C"),
    ("idioms", "🎭 Idioms", "#F39C12"),
    ("collocations", "🔗 Collocations", "#00BCD4"),
]

# Which categories are safe to recurse into when a leaf is clicked: synonyms
# and antonyms are always a single word in the *same* language (see
# idiomapp.utils.schemas). Cognates look like single words too, but their
# language is a free-text gloss ("English", "French", ...) this app can't map
# to a supported language code, so they're deliberately excluded. Everything
# else (idioms, proverbs, examples, collocations, conjugations, forms) is a
# phrase or sentence, not a word to re-analyze.
RECURSIVE_LEAF_CATEGORIES = frozenset({"synonyms", "antonyms"})

# Registered on the frontend via st_echarts(..., events={"click": GRAPH_CLICK_JS}).
# `params.data` is whatever dict a StyledNode/StyledEdge's to_echarts_dict()
# produced, including our nested "raw" payload - ECharts passes custom keys
# through untouched, confirmed empirically (a live click round-tripped it intact).
GRAPH_CLICK_JS = (
    "function(params) { return {dataType: params.dataType, ...params.data}; }"
)


# ===========================================================================
# Typed data model
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
    `apply_pinned_positions` for why that matters."""

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


class RecursiveLeafPayload(BaseModel):
    """A leaf result that is itself a real word in the same language (a
    synonym or antonym) - clicking it analyzes *that* word too and lets
    exploration continue from there, the same way the original word did."""

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


def _parse_node_payload(raw: dict) -> Optional[NodePayload]:
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


def _parse_edge_payload(raw: dict) -> Optional[EdgePayload]:
    """Reconstruct a typed EdgePayload from the plain dict a browser click
    hands back, or None (logged as a warning, for the same reason as
    `_parse_node_payload`) if it doesn't look like a valid edge payload."""
    try:
        return EdgePayload.model_validate(raw)
    except ValidationError as exc:
        logger.warning(f"Could not parse a clicked edge's data: {exc}")
        return None


# ---------------------------------------------------------------------------
# Styled nodes/edges: a NodePayload/EdgePayload plus everything needed to
# actually draw it (color, size, tooltip text). These never themselves need
# to round-trip through JSON as a whole object - only their to_echarts_dict()
# output does - so they're plain dataclasses, not Pydantic models.
# ---------------------------------------------------------------------------


@dataclass
class StyledNode:
    """One node, ready to draw: its visual styling plus the typed payload
    that says what it represents and what clicking it should do."""

    id: str
    label: str
    symbol_size: int
    symbol: Literal["circle", "diamond"]
    color: str
    border_color: Optional[str]
    tooltip: str
    payload: NodePayload
    x: Optional[float] = None
    y: Optional[float] = None
    fixed: bool = False

    def to_echarts_dict(self) -> dict:
        """Flatten this node into the plain JSON shape ECharts needs. This is
        the one point where a StyledNode stops being a typed Python object -
        everything upstream of this call works with real objects instead."""
        item_style: dict = {"color": self.color}
        if self.border_color:
            item_style["borderColor"] = self.border_color
            item_style["borderWidth"] = 2
        data: dict = {
            "name": self.id,
            "value": self.label,
            "symbolSize": self.symbol_size,
            "symbol": self.symbol,
            "itemStyle": item_style,
            "label": {"show": True, "formatter": self.label},
            "tooltip": {"formatter": self.tooltip},
            "raw": self.payload.model_dump(),
        }
        if self.x is not None and self.y is not None:
            data["x"] = self.x
            data["y"] = self.y
            data["fixed"] = self.fixed
        return data


@dataclass
class StyledEdge:
    """One edge/connection, ready to draw: its visual styling plus the typed
    payload describing what it represents."""

    source: str
    target: str
    color: str
    width: float
    dashed: bool
    curveness: float
    tooltip: str
    payload: EdgePayload

    def to_echarts_dict(self) -> dict:
        """Flatten this edge into the plain JSON shape ECharts needs."""
        return {
            "source": self.source,
            "target": self.target,
            "lineStyle": {
                "color": self.color,
                "width": self.width,
                "type": "dashed" if self.dashed else "solid",
                "curveness": self.curveness,
            },
            "tooltip": {"formatter": self.tooltip},
            "raw": self.payload.model_dump(),
        }


@dataclass(frozen=True)
class NodeSelection:
    """What a click on a node resolved to: the typed payload describing what
    was clicked, plus its on-screen position if the browser reported one
    (used to pin the node in place - see `apply_pinned_positions`)."""

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


# ===========================================================================
# Small, shared helpers (unchanged from the earlier click-to-analyze work)
# ===========================================================================


def sanitize_tooltip_text(text: str) -> str:
    """Strip HTML tags from tooltip text and escape any remaining angle brackets."""
    text = re.sub(r"<[^>]*>", "", text)
    return text.replace("<", "&lt;").replace(">", "&gt;")


def format_entry(item: Any) -> str:
    """Render one canonical {term, gloss} entry (or a plain value) for display."""
    if isinstance(item, dict):
        term = item.get("term", "")
        gloss = item.get("gloss")
        return f"{term}: {gloss}" if gloss else str(term)
    return str(item)


def format_entries(data: dict, key: str, *, limit: int = 8) -> list:
    """
    Render data[key] as a list of display strings, truncated to *limit*.

    Analysis data reaches this point already canonicalised by
    WordAnalysis.to_display_dict(): every Entries field is a list of
    {"term", "gloss"} dicts and every StrList field is a flat list of strings -
    never a bare mapping.
    """
    items = data.get(key) or []
    return [format_entry(item) for item in items[:limit]]


def raw_entries(data: dict, key: str, *, limit: int = 8) -> list:
    """
    Like format_entries, but without flattening: data[key] truncated to
    *limit*, with each item left exactly as WordAnalysis.to_display_dict()
    canonicalised it - a {"term", "gloss"} dict for an Entries field, a plain
    string for a StrList field.

    Used when building graph nodes, which keep a leaf's term and gloss as
    separate fields (see LeafNodePayload) instead of joining them into one
    string the way the text side panel (which uses format_entries) does.
    """
    items = data.get(key) or []
    return list(items[:limit])


def adjust_color(hex_color: str, amount: int) -> str:
    """Lighten (positive amount) or darken (negative) a hex color."""
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    adjusted = [max(0, min(255, channel + amount)) for channel in rgb]
    return "#%02x%02x%02x" % tuple(adjusted)


def _ensure_sentence_group_colors() -> None:
    """Memoize per-sentence GROUP_COLORS variants onto the shared config dict.

    Mirrors the old inline behavior: a one-time computation cached on the
    module-level constant itself rather than recomputed on every render.
    """
    if "en-s9" in GROUP_COLORS:
        return
    for i in range(1, 10):
        suffix = f"-s{i}"
        for base in ("en", "es", "ca", "en-related", "es-related", "ca-related"):
            GROUP_COLORS[f"{base}{suffix}"] = adjust_color(GROUP_COLORS[base], i * 10)


def filter_invalid_nodes(graph_data: dict) -> dict:
    """Drop error-placeholder nodes (and any edge referencing one) from a
    translation graph. Returns a new dict; does not mutate the input. Runs
    before the raw graph_data is turned into a typed graph, since this is a
    data-quality filter, not part of the graph's shape."""
    error_keywords = ("translation", "failed", "error", "try", "again")
    valid_nodes = [
        n
        for n in graph_data.get("nodes", [])
        if not any(kw in n.get("label", "").lower() for kw in error_keywords)
    ]
    valid_ids = {n["id"] for n in valid_nodes}
    valid_edges = [
        e
        for e in graph_data.get("edges", [])
        if e.get("from") in valid_ids and e.get("to") in valid_ids
    ]
    return {**graph_data, "nodes": valid_nodes, "edges": valid_edges}


# ===========================================================================
# Building the graph itself: an nx.MultiDiGraph, not a pair of lists.
#
# MultiDiGraph (not a plain Graph/DiGraph) on purpose: two words in this
# app's translation graph can legitimately be connected by more than one
# edge at once (e.g. a translation edge and a cognate edge between the same
# pair) - nothing upstream de-duplicates that, so a simple graph would
# silently drop one of them. Node/edge attributes hold the typed
# StyledNode/StyledEdge objects; graph_to_echarts_data (at the bottom of
# this file) is the only function that turns the graph into plain dicts.
# ===========================================================================


def _style_semantic_node(node: dict) -> StyledNode:
    """Style one node from the main translation graph: fill color by
    language/sentence group, border color by part of speech - matching the
    colors this app has always used for these graphs."""
    group = node.get("group", "default")
    color = GROUP_COLORS.get(group, "#4CC9F0")
    pos = node.get("pos", "unknown")
    border_color = POS_BORDER_COLORS.get(pos.lower(), "#4361EE")
    size = 30 if node.get("node_type") == "primary" else 20
    lang_code = node.get("language", "unknown")
    lang_name = LANGUAGE_MAP.get(lang_code, {}).get("name", lang_code.upper())
    details = sanitize_tooltip_text(node.get("details", ""))
    tooltip = f"{node['label']} ({lang_name}); Part of speech: {pos}"
    if details:
        tooltip += f"; Details: {details}"
    return StyledNode(
        id=node["id"],
        label=node["label"],
        symbol_size=size,
        symbol="circle",
        color=color,
        border_color=border_color,
        tooltip=tooltip,
        payload=SemanticWordPayload(
            id=node["id"],
            label=node["label"],
            language=lang_code,
            pos=pos,
            details=node.get("details", ""),
            node_type=node.get("node_type", "related"),
            group=group,
            sentence_group=node.get("sentence_group", ""),
        ),
    )


def _style_semantic_edge(edge: dict) -> StyledEdge:
    """Style one edge from the main translation graph: color/width/dashing
    by relation type, using a two-tier scheme - a handful of relation types
    (translation, cognate, ...) get their own fixed look, everything else
    falls back to the shared RELATION_COLORS table."""
    relation = edge.get("relation", "related")
    strength = edge.get("strength", 0.5)

    if relation in ("direct_translation", "translation"):
        width, color, dashed = 3 * strength, "#FFFFFF", False
    elif relation == "cognate":
        width, color, dashed = 2.5 * strength, "#FFD700", False
    elif relation == "semantic_equivalent":
        width, color, dashed = 2.5 * strength, "#00B8D4", False
    elif relation == "cross_sentence":
        width, color, dashed = 1.5 * strength, edge.get("color", "#AA44BB"), True
    elif "semantic_similarity" in relation:
        width, color, dashed = 2 * strength, edge.get("color", "#FFAA00"), True
    else:
        width, color, dashed = (
            1 + strength,
            RELATION_COLORS.get(relation, "#AAAAAA"),
            False,
        )

    # Matches the original pyvis-era rule exactly: an edge that merely HAS a
    # "dashes" key renders dashed regardless of that key's actual value, same
    # as relation being exactly cross_sentence/semantic_similarity.
    if relation in ("cross_sentence", "semantic_similarity") or "dashes" in edge:
        dashed = True

    label = sanitize_tooltip_text(edge.get("label", relation.replace("_", " ")))
    if "title" in edge:
        tooltip = sanitize_tooltip_text(edge["title"])
    elif "description" in edge:
        tooltip = sanitize_tooltip_text(edge["description"])
    else:
        tooltip = f"{label} ({strength:.2f})"

    return StyledEdge(
        source=edge["from"],
        target=edge["to"],
        color=color,
        width=width,
        dashed=dashed,
        curveness=0.15,
        tooltip=tooltip,
        payload=EdgePayload(
            kind=relation,
            label=label,
            description=edge.get("description", edge.get("title", "")),
            strength=strength,
        ),
    )


def build_semantic_graph(graph_data: dict) -> nx.MultiDiGraph:
    """Build the typed graph for the main Semantic Graph view from a
    (pre-filtered) translation-graph {"nodes","edges"} dict - the graph a
    click can then be layered onto via `compose_semantic_graph_with_expansions`."""
    _ensure_sentence_group_colors()
    g = nx.MultiDiGraph()
    for node in graph_data.get("nodes", []):
        styled = _style_semantic_node(node)
        g.add_node(styled.id, styled=styled)
    for edge in graph_data.get("edges", []):
        styled = _style_semantic_edge(edge)
        g.add_edge(styled.source, styled.target, styled=styled)
    return g


def build_cooccurrence_graph(graph, lang_code: Optional[str] = None) -> nx.MultiDiGraph:
    """Build the typed graph for the (unchanged) Co-occurrence Network view
    from a networkx graph of raw word co-occurrence counts."""
    color = get_language_color(lang_code) if lang_code else "#4CC9F0"
    g = nx.MultiDiGraph()
    for node in graph.nodes():
        node_id = str(node)
        degree = graph.degree(node)
        g.add_node(
            node_id,
            styled=StyledNode(
                id=node_id,
                label=node_id,
                symbol_size=20 + degree * 3,
                symbol="circle",
                color=color,
                border_color="#4361EE",
                tooltip=f"Word: {sanitize_tooltip_text(node_id)}; Co-occurrences: {degree}",
                payload=CooccurrenceWordPayload(
                    word=node_id, language=lang_code, degree=degree
                ),
            ),
        )
    for source, target, data in graph.edges(data=True):
        weight = data.get("weight", 1)
        g.add_edge(
            str(source),
            str(target),
            styled=StyledEdge(
                source=str(source),
                target=str(target),
                color="#FFFFFF" if weight > 2 else "#AAAAAA",
                width=1 + weight / 2,
                dashed=False,
                curveness=0.1,
                tooltip=f"Co-occurrence: {weight}",
                payload=EdgePayload(kind="cooccurrence_edge", weight=weight),
            ),
        )
    return g


# ===========================================================================
# In-graph word exploration: growing the Semantic Graph with a clicked
# word's analysis, in place, instead of showing it in a separate graph.
# ===========================================================================


def build_word_expansion(
    word_key: WordKey,
    analysis_data: dict,
    *,
    attach_node_id: str,
    expanded_categories: set,
    word_id_index: dict,
) -> tuple:
    """Build the category hubs (and, for expanded categories, their leaf
    results) for one already-analyzed word, attached to its existing node
    in the graph instead of a free-standing "main" node.

    This is the same hub-and-spoke idea the old standalone Knowledge Graph
    used (one hub per non-empty category, up to 8 leaves each, long text
    truncated) - just pointed at a real place in the main graph instead of
    building its own separate graph.

    A category hub is always shown once its word is analyzed; its leaves
    only appear once that category has been toggled open (its
    (word_key, category) pair is in `expanded_categories`). A leaf that's
    itself a recursable word (see RECURSIVE_LEAF_CATEGORIES) reuses an
    existing node for that word if one is already in `word_id_index` -
    e.g. two different words that share a synonym end up pointing at one
    shared node, not two separate copies - and registers a freshly-made one
    otherwise, so later expansions can find and reuse it too.

    Returns (new_nodes, new_edges); does not mutate the graph itself -
    that's `compose_semantic_graph_with_expansions`'s job.
    """
    nodes: list = []
    edges: list = []
    counter = 0

    def leaf_id(hub_id: str) -> str:
        nonlocal counter
        counter += 1
        return f"item::{hub_id}::{counter}"

    def add_category(
        category: str,
        label: str,
        color: str,
        items: list,
        *,
        hub_tooltip: Optional[str] = None,
    ) -> None:
        if not items and not hub_tooltip:
            return
        hub_id = f"cat::{attach_node_id}::{category}"
        nodes.append(
            StyledNode(
                id=hub_id,
                label=label,
                symbol_size=30,
                symbol="diamond",
                color=color,
                border_color=None,
                tooltip=hub_tooltip or label,
                payload=CategoryPayload(
                    word_key=word_key, category=category, label=label
                ),
            )
        )
        edges.append(
            StyledEdge(
                source=attach_node_id,
                target=hub_id,
                color=color,
                width=3,
                dashed=False,
                curveness=0.1,
                tooltip=label,
                payload=EdgePayload(kind="category_edge", label=label),
            )
        )
        if (word_key, category) not in expanded_categories:
            return
        for item in items[:8]:
            # An Entries-backed category (e.g. cognates, idioms) hands back
            # {"term", "gloss"} dicts here - raw_entries() doesn't flatten
            # them the way format_entries() does for the text panel. A
            # StrList category (e.g. synonyms) hands back plain strings, so
            # this just becomes term=item, gloss=None - unchanged from before.
            if isinstance(item, dict):
                term, gloss = str(item.get("term", "")), item.get("gloss")
            else:
                term, gloss = item, None
            display = term[:27] + "..." if len(term) > 30 else term
            tooltip = f"{term}: {gloss}" if gloss else term
            if category in RECURSIVE_LEAF_CATEGORIES:
                leaf_key = WordKey.of(term, word_key.language)
                node_id = word_id_index.get(leaf_key)
                if node_id is None:
                    node_id = f"word::{leaf_key.language}::{leaf_key.word}"
                    word_id_index[leaf_key] = node_id
                    nodes.append(
                        StyledNode(
                            id=node_id,
                            label=display,
                            symbol_size=20,
                            symbol="circle",
                            color=color,
                            border_color=None,
                            tooltip=tooltip,
                            payload=RecursiveLeafPayload(
                                word_key=leaf_key, category=category
                            ),
                        )
                    )
            else:
                node_id = leaf_id(hub_id)
                nodes.append(
                    StyledNode(
                        id=node_id,
                        label=display,
                        symbol_size=20,
                        symbol="circle",
                        color=color,
                        border_color=None,
                        tooltip=tooltip,
                        payload=LeafNodePayload(
                            category=category, text=term, gloss=gloss
                        ),
                    )
                )
            edges.append(
                StyledEdge(
                    source=hub_id,
                    target=node_id,
                    color=color,
                    width=1,
                    dashed=False,
                    curveness=0.1,
                    tooltip=tooltip,
                    payload=EdgePayload(kind="leaf_edge", label=term),
                )
            )

    etym_color = "#9B59B6"
    etymology_items = []
    for sub_key, prefix in (("language_origin", "Origin"), ("root", "Root")):
        if sub_key in analysis_data:
            value = analysis_data[sub_key]
            etymology_items.append(
                f"{prefix}: {value}" if sub_key == "root" else str(value)
            )
    etymology_text = analysis_data.get("etymology")
    if etymology_items or etymology_text:
        add_category(
            "etymology",
            "📜 Etymology",
            etym_color,
            etymology_items,
            hub_tooltip=etymology_text,
        )

    for key, label, color in GRAPH_CATEGORIES:
        add_category(key, label, color, raw_entries(analysis_data, key))

    conj = raw_entries(analysis_data, "conjugations", limit=6)
    if conj:
        add_category("conjugations", "📝 Conjugations", "#2ECC71", conj)

    examples = analysis_data.get("examples", [])
    if isinstance(examples, list) and examples:
        short = [ex[:40] + "..." if len(ex) > 40 else ex for ex in examples[:4]]
        add_category("examples", "💬 Examples", "#1ABC9C", short)

    forms = []
    for fkey, flabel in (
        ("plural", "Plural"),
        ("gender", "Gender"),
        ("infinitive", "Infinitive"),
        ("verb_type", "Verb type"),
    ):
        val = analysis_data.get(fkey)
        if val:
            forms.append(f"{flabel}: {val}")
    for entries_key in ("gender_forms", "related_forms"):
        forms.extend(raw_entries(analysis_data, entries_key))
    if forms:
        add_category("forms", "📋 Forms", "#4CAF50", forms)

    return nodes, edges


def _index_semantic_words(g: nx.MultiDiGraph) -> dict:
    """Scan a graph's existing SemanticWordPayload nodes into a {WordKey: node
    id} lookup, preferring a "primary" node over a "related" one for the same
    word if both exist. Used to seed word_id_index before layering expansions
    on, so a clicked word reuses its real base-graph node instead of minting
    a duplicate."""
    index: dict = {}
    for node_id, data in g.nodes(data=True):
        payload = data["styled"].payload
        if not isinstance(payload, SemanticWordPayload):
            continue
        key = WordKey.of(payload.label, payload.language)
        if key not in index or payload.node_type == "primary":
            index[key] = node_id
    return index


def compose_semantic_graph_with_expansions(
    base: nx.MultiDiGraph,
    analyses: dict,
    expanded_categories: set,
) -> nx.MultiDiGraph:
    """Layer every already-analyzed word's category hubs (and any expanded
    categories' leaves) onto a copy of the base translation graph.

    Words are added in a small bounded loop rather than a single pass:
    a word can only be attached once we know *where* in the graph to attach
    it, and that place might itself be a leaf added by a different word's
    expansion (recursive drill-down) - so this repeats until nothing new
    gets attached, however many passes that takes, capped at
    len(analyses) + 1 so it can never loop forever even in a bug scenario.
    A word whose parent category is currently collapsed is simply skipped
    for this rerun; its subtree comes back instantly (already cached in
    `analyses`) the moment that category is re-expanded.
    """
    g = base.copy()
    word_id_index = _index_semantic_words(g)

    built: set = set()
    for _ in range(len(analyses) + 1):
        progressed = False
        for word_key, analysis_data in analyses.items():
            if word_key in built:
                continue
            attach_node_id = word_id_index.get(word_key)
            if attach_node_id is None:
                continue
            new_nodes, new_edges = build_word_expansion(
                word_key,
                analysis_data,
                attach_node_id=attach_node_id,
                expanded_categories=expanded_categories,
                word_id_index=word_id_index,
            )
            for node in new_nodes:
                g.add_node(node.id, styled=node)
            for edge in new_edges:
                g.add_edge(edge.source, edge.target, styled=edge)
            built.add(word_key)
            progressed = True
        if not progressed:
            break
    return g


def apply_pinned_positions(g: nx.MultiDiGraph, positions: dict) -> nx.MultiDiGraph:
    """Hold already-seen nodes still the next time the graph is drawn.

    ECharts' force layout re-simulates from scratch whenever a graph's node
    or edge list changes, so without this every existing node can visibly
    jump each time a new one is added by a click. Any node whose id has a
    captured screen position (see NodeSelection.position, set from a
    previous click) gets that position pinned; brand-new nodes are left for
    the simulation to place naturally.
    """
    g = g.copy()
    for node_id, position in positions.items():
        if node_id not in g.nodes:
            continue
        styled = g.nodes[node_id]["styled"]
        g.nodes[node_id]["styled"] = replace(
            styled, x=position.x, y=position.y, fixed=True
        )
    return g


def graph_to_echarts_data(g: nx.MultiDiGraph) -> dict:
    """Flatten a typed graph into the plain {"nodes": [...], "edges": [...]}
    dict `build_graph_echarts_options` needs. The only place in this module
    that produces an untyped dict - everything before this point works with
    real StyledNode/StyledEdge objects, and this is where that stops
    mattering because the data is about to leave Python as JSON anyway."""
    return {
        "nodes": [g.nodes[n]["styled"].to_echarts_dict() for n in g.nodes],
        "edges": [d["styled"].to_echarts_dict() for _, _, d in g.edges(data=True)],
    }


# Simple "+"/"-" glyphs for the zoom toolbox buttons, as ECharts `path://` icons
# (a 1024x1024 unit box - ECharts scales them to the toolbox's itemSize). Roam
# already lets a mouse/trackpad/touch user zoom by gesture; these two buttons plus
# "restore" make that discoverable without one, particularly on mobile.
_ZOOM_IN_ICON = (
    "path://M448,128 L576,128 L576,448 L896,448 L896,576 L576,576 "
    "L576,896 L448,896 L448,576 L128,576 L128,448 L448,448 Z"
)
_ZOOM_OUT_ICON = "path://M128,448 L896,448 L896,576 L128,576 Z"

# streamlit_echarts.JsCode marks a string as "real JS, not a JSON string" by
# wrapping it in this exact placeholder, which its frontend bundle recognises
# and evaluates as a function. Reproduced by hand instead of importing JsCode
# here: streamlit_echarts registers a Streamlit v2 component *at import time*,
# which needs a live script-run context to resolve - true inside app.py (which
# already imports it) or under AppTest, but not for a plain, direct import of
# this module, which is exactly how tests/test_graph_viz.py's 60+ tests import
# it. Keeping graph_viz.py free of that import is what keeps those importable
# and runnable without a Streamlit context at all.
_JS_PLACEHOLDER = "--x_x--0_0--"


def _js_handler(code: str) -> str:
    """Wrap a JS function body the same way streamlit_echarts.JsCode does, so
    it survives options serialization as real JS instead of a JSON string."""
    return f"{_JS_PLACEHOLDER}{code}{_JS_PLACEHOLDER}"


def build_graph_echarts_options(graph_data: dict, *, layout: str = "force") -> dict:
    """Wrap already-ECharts-shaped node/edge dicts (from graph_to_echarts_data)
    in the outer `option` object st_echarts expects: one force-directed graph
    series, draggable and zoomable, with hovering over a node dimming
    everything not adjacent to it, plus a zoom in/out/reset toolbox so zooming
    doesn't rely on an undiscoverable scroll-wheel/pinch gesture alone.
    """
    return {
        "tooltip": {},
        "toolbox": {
            "show": True,
            "right": 20,
            "top": 10,
            "itemSize": 18,
            "itemGap": 10,
            "feature": {
                "restore": {"title": "Reset view"},
                "myZoomIn": {
                    "show": True,
                    "title": "Zoom in",
                    "icon": _ZOOM_IN_ICON,
                    "onclick": _js_handler(
                        "function(event, api) {"
                        " api.dispatchAction({type: 'graphRoam', zoom: 1.2}); "
                        "}"
                    ),
                },
                "myZoomOut": {
                    "show": True,
                    "title": "Zoom out",
                    "icon": _ZOOM_OUT_ICON,
                    "onclick": _js_handler(
                        "function(event, api) {"
                        " api.dispatchAction({type: 'graphRoam', zoom: 0.8}); "
                        "}"
                    ),
                },
            },
        },
        "series": [
            {
                "type": "graph",
                "layout": layout,
                "roam": True,
                "draggable": True,
                "label": {"show": True, "position": "right"},
                "data": graph_data.get("nodes", []),
                "links": graph_data.get("edges", []),
                "force": {"repulsion": 250, "edgeLength": 90},
                "emphasis": {"focus": "adjacency"},
            }
        ],
    }


def resolve_graph_click(chart_event: Optional[dict]) -> Optional[GraphSelection]:
    """Turn the raw dict the browser hands back on a click into a typed
    NodeSelection or EdgeSelection.

    Returns None in two very different situations, deliberately handled
    differently:
    - **Nothing to resolve** - no click happened on this rerun. This is the
      ordinary case on almost every rerun (Streamlit reruns the whole script
      far more often than the user actually clicks the graph), so it's
      silent: no log line, nothing went wrong.
    - **A click happened but this app couldn't make sense of it** - the
      event's shape doesn't match what this app's own JS handler
      (GRAPH_CLICK_JS) and StyledNode/StyledEdge ever produce. That's
      unexpected regardless of cause (a future ECharts/streamlit-echarts
      upgrade changing the event shape, a bug elsewhere in this module, a
      stale browser tab from before a code change), so it's logged as a
      warning rather than swallowed - still doesn't crash the page, but
      leaves a trail instead of silently doing nothing.
    """
    if not chart_event:
        return None
    if not isinstance(chart_event, dict):
        logger.warning(f"Graph click event was not a dict: {type(chart_event)!r}")
        return None

    data_type = chart_event.get("dataType")
    raw = chart_event.get("raw")
    if not isinstance(raw, dict):
        logger.warning(
            f"Graph click event had no usable 'raw' payload: {chart_event!r}"
        )
        return None

    if data_type == "node":
        payload = _parse_node_payload(raw)
        if payload is None:
            return None  # already logged inside _parse_node_payload
        position = None
        x, y = chart_event.get("x"), chart_event.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            position = Position(x=x, y=y)
        return NodeSelection(payload=payload, position=position)

    if data_type == "edge":
        payload = _parse_edge_payload(raw)
        source_id, target_id = chart_event.get("source"), chart_event.get("target")
        if payload is None:
            return None  # already logged inside _parse_edge_payload
        if not isinstance(source_id, str) or not isinstance(target_id, str):
            logger.warning(
                f"Clicked edge was missing a source/target id: {chart_event!r}"
            )
            return None
        return EdgeSelection(payload=payload, source_id=source_id, target_id=target_id)

    logger.warning(f"Graph click event had an unrecognized dataType: {data_type!r}")
    return None
