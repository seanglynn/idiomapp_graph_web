# Graph Architecture Roadmap

## Goal

Keep IdiomApp a single Streamlit app while making the graph's data model explicit: what the graph contains should be defined in one place, separate from how it is built and drawn.

## Where we are

- Graph element models live inside `idiomapp/utils/graph_viz.py`, mixed with graph building, styling, and ECharts rendering (see [the graph data model](graph_data_model.md)):
  - identity/geometry: `WordKey`, `Position`
  - node payloads (Pydantic discriminated union): `SemanticWordPayload`, `CategoryPayload`, `RecursiveLeafPayload`, `LeafNodePayload`, `CooccurrenceWordPayload`
  - edge payload: `EdgePayload`
  - click/request objects: `NodeSelection`, `EdgeSelection`, `SourcedSelection`
- Styling objects (`StyledNode`, `StyledEdge`) are renderer concerns and also live there.
- LLM response models (`WordAnalysis`, `Translation`, `AlignmentPair`, ...) live in `idiomapp/utils/schemas.py`.

## Next: a dedicated graph models file

- Create `idiomapp/utils/graph_models.py` holding:
  - **Graph elements:** `WordKey`, `Position`, the `NodePayload` variants and union, `EdgePayload`, and their parse helpers.
  - **Requests:** the selection objects a click produces (`NodeSelection`, `EdgeSelection`, `SourcedSelection`) and an explicit word-expansion request (the inputs `build_word_expansion` takes today).
- Keep `StyledNode`/`StyledEdge`, graph building, and ECharts code in `graph_viz.py`, importing from `graph_models.py`.
- Leave `schemas.py` as the home of LLM response models.
- No behaviour change: existing call sites and tests keep working (re-export from `graph_viz.py` only if needed during the move).

**Done when:** `graph_viz.py` defines no Pydantic models; `graph_models.py` has no rendering or networkx code; tests cover JSON round-trip for each node variant, the edge payload, and the expansion request; `graph_data_model.md` points to the new file.

## Later (not now)

- Stable IDs for nodes and edges (word + language, plus sense where needed).
- Provenance (LLM, heuristic, stored) on edges.
- A result limit with a `has_more` flag on expansion, so a limit never looks like the end of the graph.

## Design principle

**Models define what the graph knows; `graph_viz.py` decides how to build and show it.** One app, clearly separated layers.
