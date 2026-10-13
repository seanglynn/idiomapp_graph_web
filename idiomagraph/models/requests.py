"""Requests: what a click on the graph, or an expansion of a word, asks for."""

from dataclasses import dataclass
from typing import Literal, Optional, Union

from idiomagraph.models.edges import EdgePayload
from idiomagraph.models.identity import Position, WordKey
from idiomagraph.models.nodes import NodePayload


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
