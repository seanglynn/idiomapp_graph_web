"""Every typed model the app is built on, one file per kind of model:

- `identity` - `WordKey` and `Position`, the small value types the others use
- `nodes` - what a graph node is: the `NodePayload` variants
- `edges` - what a graph edge is: `EdgePayload`
- `requests` - what a click or an expansion asks for: selections and
  `WordExpansionRequest`
- `llm` - the structured output the LLM providers return (`WordAnalysis`,
  `Translation`, ...)

None of these import networkx, Streamlit, or any rendering code; graph
building and drawing live in `idiomagraph.utils.graph_viz`. Payloads that cross
the browser boundary as JSON are Pydantic models, and everything that only
lives inside Python is a plain dataclass - see `docs/graph_data_model.md`.

Import from this package (`from idiomagraph.models import WordKey`) rather than
from the individual files.
"""

from idiomagraph.models.edges import EdgePayload, parse_edge_payload
from idiomagraph.models.identity import Position, WordKey
from idiomagraph.models.llm import (
    WORD_ANALYSIS_GROUPS,
    AlignmentPair,
    Entry,
    Grammar,
    LearnerNotes,
    Meaning,
    Pronunciation,
    Translation,
    Usage,
    WordAnalysis,
    prompt_example,
)
from idiomagraph.models.nodes import (
    RECURSIVE_LEAF_CATEGORIES,
    CategoryPayload,
    CooccurrenceWordPayload,
    LeafNodePayload,
    NodePayload,
    RecursiveLeafPayload,
    SemanticWordPayload,
    parse_node_payload,
)
from idiomagraph.models.requests import (
    EdgeSelection,
    GraphSelection,
    NodeSelection,
    SourcedSelection,
    WordExpansionRequest,
)

__all__ = [
    "RECURSIVE_LEAF_CATEGORIES",
    "WORD_ANALYSIS_GROUPS",
    "AlignmentPair",
    "CategoryPayload",
    "CooccurrenceWordPayload",
    "EdgePayload",
    "EdgeSelection",
    "Entry",
    "Grammar",
    "GraphSelection",
    "LeafNodePayload",
    "LearnerNotes",
    "Meaning",
    "NodePayload",
    "NodeSelection",
    "Position",
    "Pronunciation",
    "RecursiveLeafPayload",
    "SemanticWordPayload",
    "SourcedSelection",
    "Translation",
    "Usage",
    "WordAnalysis",
    "WordExpansionRequest",
    "WordKey",
    "parse_edge_payload",
    "parse_node_payload",
    "prompt_example",
]
