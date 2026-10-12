# Graph Architecture Roadmap

## Goal

Make IdiomApp's graph an open-ended language exploration experience: users can keep following words, phrases, and sentences, while each backend operation stays bounded and the domain model remains explicit and extensible.

## Priorities

### 1. Establish canonical typed graph models
- Audit and build on `idiomapp/utils/schemas.py`, `graph_viz.py`, `graph_storage.py`, and [the current graph data model](graph_data_model.md).
- Define validated, discriminated models for node types (initially words, phrases, sentences, meanings, and grammar concepts) and edge types (translation/alignment, usage, grammar, semantic, and etymology).
- Give nodes and edges stable IDs; distinguish word senses where needed.
- Preserve multiple distinct relationships between the same pair of nodes.
- Record provenance and, where appropriate, confidence and timestamps.
- Separate canonical domain data from renderer-specific properties such as coordinates, colours, and sizes. Use the etymology project's [Pydantic models](https://github.com/ofcRS/etymology/blob/main/backend/models.py) as a reference, not as a schema to copy wholesale.

**Done when:** graph payloads are validated and round-trip through JSON; invalid endpoints and unknown types are handled safely; tests cover node variants, edge variants, and parallel relationships.

### 2. Make traversal feel unbounded
- Users should be able to keep clicking newly discovered words, phrases, and sentences without hitting a visible lifetime depth or node cap.
- Expand on demand instead of loading the full reachable graph.
- Apply server-side per-request budgets for result count, traversal work, time, and concurrency as appropriate.
- Support pagination/cursors for high-degree nodes and return explicit continuation metadata (for example, `has_more` and `next_cursor`).
- Keep the user's exploration history and graph state when expanding; focusing or hiding branches must not delete explored knowledge.

**Done when:** reaching a backend budget never silently appears to be the end of the graph; the UI can request more results and continue exploring from any returned node.

### 3. Standardise graph expansion
- Define a typed expansion request/response contract with a node ID, optional relationship filters, a bounded page size, and continuation state.
- Centralise expansion logic behind a service/API boundary, independent of the renderer and underlying data provider.
- Validate requested limits server-side; add timeouts, caching, and safe handling of partial results.
- Allow expansion to combine stored graph data with provider-generated discoveries (including etymology) without changing the client contract.

**Done when:** all graph expansion uses the same contract and returns typed nodes/edges plus clear continuation status.

### 4. Render progressively
- Merge new nodes and edges incrementally, deduplicating by stable ID.
- Preserve selection, layout, and exploration state as the graph grows.
- Provide ways to focus on a selected node and reduce visual clutter without imposing a traversal dead end.
- Keep presentation concerns out of canonical node and edge models.

**Done when:** repeated expansion does not duplicate entities or reset the user's exploration, and larger graphs remain navigable.

## Suggested delivery order

1. Audit existing schemas and graph call sites; document identity and relationship conventions.
2. Implement canonical node/edge models and integrity/round-trip tests.
3. Introduce the shared expansion contract, backend budgets, and continuation support.
4. Update the UI for incremental expansion and preserved exploration state.
5. Add further node/edge types and external providers incrementally.

## Design principle

**The domain model defines what the graph knows; the traversal layer decides what to fetch; the renderer decides how to show it.** The graph should feel unbounded to the learner, while each operation remains predictable and safe.
