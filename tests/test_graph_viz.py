"""
Tests for idiomapp/utils/graph_viz.py - the typed graph data model and the
adapters that turn it into ECharts graph-series options.

Pure pytest throughout: no Streamlit runtime, no ECharts JS involved. These
pin the *data* each function produces; the live click round-trip itself was
verified empirically against a running st_echarts component (see the
graph-viz migration plans) and isn't something a unit test can exercise.
"""

import networkx as nx

from idiomapp.utils.graph_viz import (
    GRAPH_CATEGORIES,
    GRAPH_CLICK_JS,
    RECURSIVE_LEAF_CATEGORIES,
    CategoryPayload,
    CooccurrenceWordPayload,
    EdgePayload,
    EdgeSelection,
    LeafNodePayload,
    NodeSelection,
    Position,
    RecursiveLeafPayload,
    SemanticWordPayload,
    SourcedSelection,
    WordKey,
    adjust_color,
    apply_pinned_positions,
    build_cooccurrence_graph,
    build_graph_echarts_options,
    build_semantic_graph,
    build_word_expansion,
    compose_semantic_graph_with_expansions,
    filter_invalid_nodes,
    format_entries,
    format_entry,
    graph_to_echarts_data,
    raw_entries,
    resolve_graph_click,
    sanitize_tooltip_text,
)


# --------------------------------------------------------------------------
# Small pure helpers - unchanged API
# --------------------------------------------------------------------------
def test_format_entry_with_gloss():
    assert format_entry({"term": "gato", "gloss": "cat"}) == "gato: cat"


def test_format_entry_without_gloss():
    assert format_entry({"term": "gato", "gloss": None}) == "gato"


def test_format_entry_plain_value():
    assert format_entry("gato") == "gato"


def test_format_entries_missing_key_is_empty():
    assert format_entries({}, "synonyms") == []


def test_format_entries_truncates_at_limit():
    data = {"synonyms": [{"term": str(i), "gloss": None} for i in range(10)]}
    assert len(format_entries(data, "synonyms", limit=3)) == 3


def test_raw_entries_missing_key_is_empty():
    assert raw_entries({}, "cognates") == []


def test_raw_entries_truncates_at_limit():
    data = {"cognates": [{"term": str(i), "gloss": None} for i in range(10)]}
    assert len(raw_entries(data, "cognates", limit=3)) == 3


def test_raw_entries_does_not_flatten_dict_items():
    # Unlike format_entries, term/gloss stay a dict - graph nodes keep them as
    # separate fields instead of joining them into one display string.
    data = {"cognates": [{"term": "similar", "gloss": "English"}]}
    assert raw_entries(data, "cognates") == [{"term": "similar", "gloss": "English"}]


def test_raw_entries_leaves_plain_strings_unchanged():
    data = {"collocations": ["salir a la luna"]}
    assert raw_entries(data, "collocations") == ["salir a la luna"]


def test_sanitize_tooltip_text_strips_tags_and_escapes():
    assert sanitize_tooltip_text("<b>bold</b> & <i>x</i>") == "bold & x"


def test_adjust_color_lightens_and_clamps():
    assert adjust_color("#000000", 10) == "#0a0a0a"
    assert adjust_color("#fefefe", 10) == "#ffffff"  # clamped at 255


def test_graph_categories_and_click_js_are_defined():
    assert len(GRAPH_CATEGORIES) == 5
    assert "dataType" in GRAPH_CLICK_JS


def test_recursive_leaf_categories_are_the_two_bare_word_ones():
    # Deliberately just these two - see the module docstring/comment next to
    # RECURSIVE_LEAF_CATEGORIES for why cognates are excluded despite also
    # being structurally single words.
    assert RECURSIVE_LEAF_CATEGORIES == {"synonyms", "antonyms"}


# --------------------------------------------------------------------------
# WordKey
# --------------------------------------------------------------------------
def test_word_key_of_normalizes_like_the_analysis_cache():
    assert WordKey.of("  Luna ", "es") == WordKey(word="luna", language="es")


def test_word_key_is_hashable_for_use_as_a_dict_or_set_member():
    a = WordKey.of("luna", "es")
    b = WordKey.of("LUNA", "es")
    assert a == b
    assert {a, b} == {a}  # same key, so the set collapses to one


# --------------------------------------------------------------------------
# resolve_graph_click - the browser-click parsing boundary
# --------------------------------------------------------------------------
def test_resolve_graph_click_no_event_is_silently_none():
    assert resolve_graph_click(None) is None
    assert resolve_graph_click({}) is None


def test_resolve_graph_click_non_dict_event_is_none():
    assert resolve_graph_click("not a dict") is None  # type: ignore[arg-type]


def test_resolve_graph_click_missing_raw_is_none():
    assert resolve_graph_click({"dataType": "node"}) is None
    assert resolve_graph_click({"dataType": "node", "raw": "oops"}) is None


def test_resolve_graph_click_unknown_datatype_is_none():
    assert resolve_graph_click({"dataType": "series", "raw": {"kind": "word"}}) is None


def test_resolve_graph_click_malformed_node_payload_is_none():
    # Missing every required field of every NodePayload variant.
    assert resolve_graph_click({"dataType": "node", "raw": {"kind": "word"}}) is None


def test_resolve_graph_click_edge_missing_source_target_is_none():
    raw = {"kind": "translation"}
    assert resolve_graph_click({"dataType": "edge", "raw": raw}) is None


def test_resolve_graph_click_node_without_position():
    raw = {
        "kind": "word",
        "id": "cat_en",
        "label": "cat",
        "language": "en",
        "pos": "noun",
        "details": "",
        "node_type": "primary",
        "group": "en",
    }
    selection = resolve_graph_click({"dataType": "node", "raw": raw})
    assert isinstance(selection, NodeSelection)
    assert isinstance(selection.payload, SemanticWordPayload)
    assert selection.payload.id == "cat_en"
    assert selection.position is None


def test_resolve_graph_click_node_with_position():
    raw = {
        "kind": "word",
        "id": "cat_en",
        "label": "cat",
        "language": "en",
        "pos": "noun",
        "details": "",
        "node_type": "primary",
        "group": "en",
    }
    selection = resolve_graph_click(
        {"dataType": "node", "raw": raw, "x": 12.5, "y": -3.0}
    )
    assert selection.position == Position(x=12.5, y=-3.0)


def test_resolve_graph_click_edge():
    raw = {"kind": "cognate"}
    selection = resolve_graph_click(
        {"dataType": "edge", "raw": raw, "source": "a", "target": "b"}
    )
    assert isinstance(selection, EdgeSelection)
    assert selection.payload == EdgePayload(kind="cognate")
    assert selection.source_id == "a"
    assert selection.target_id == "b"


def test_sourced_selection_pairs_a_selection_with_its_graph():
    edge_selection = EdgeSelection(
        payload=EdgePayload(kind="cognate"), source_id="a", target_id="b"
    )
    sourced = SourcedSelection(selection=edge_selection, source="cooccurrence")
    assert sourced.source == "cooccurrence"
    assert sourced.selection is edge_selection


# --------------------------------------------------------------------------
# build_graph_echarts_options - wraps already-ECharts-shaped data
# --------------------------------------------------------------------------
def test_build_graph_echarts_options_shape():
    graph_data = {
        "nodes": [{"name": "a", "value": "a", "raw": {"id": "a"}}],
        "edges": [{"source": "a", "target": "b", "raw": {"from": "a", "to": "b"}}],
    }
    options = build_graph_echarts_options(graph_data)
    series = options["series"][0]
    assert series["type"] == "graph"
    assert series["data"] == graph_data["nodes"]
    assert series["links"] == graph_data["edges"]


def test_build_graph_echarts_options_empty_graph():
    options = build_graph_echarts_options({"nodes": [], "edges": []})
    series = options["series"][0]
    assert series["data"] == []
    assert series["links"] == []


def test_build_graph_echarts_options_has_a_zoom_toolbox():
    options = build_graph_echarts_options({"nodes": [], "edges": []})
    feature = options["toolbox"]["feature"]
    assert set(feature.keys()) == {"restore", "myZoomIn", "myZoomOut"}
    # Custom feature onclick handlers must be real JS, not a JSON string - the
    # streamlit_echarts frontend only evaluates values wrapped in this exact
    # placeholder (see graph_viz._js_handler).
    for key in ("myZoomIn", "myZoomOut"):
        onclick = feature[key]["onclick"]
        assert onclick.startswith("--x_x--0_0--")
        assert onclick.endswith("--x_x--0_0--")
        assert "dispatchAction" in onclick


# --------------------------------------------------------------------------
# filter_invalid_nodes - unchanged
# --------------------------------------------------------------------------
def test_filter_invalid_nodes_drops_error_nodes_and_their_edges():
    graph_data = {
        "nodes": [
            {"id": "ok", "label": "gato"},
            {"id": "bad", "label": "Translation failed"},
        ],
        "edges": [{"from": "ok", "to": "bad", "relation": "related_term"}],
    }
    result = filter_invalid_nodes(graph_data)
    assert [n["id"] for n in result["nodes"]] == ["ok"]
    assert result["edges"] == []


def test_filter_invalid_nodes_does_not_mutate_input():
    graph_data = {"nodes": [{"id": "ok", "label": "gato"}], "edges": []}
    filter_invalid_nodes(graph_data)
    assert graph_data["nodes"] == [{"id": "ok", "label": "gato"}]


# --------------------------------------------------------------------------
# build_semantic_graph - the main translation graph, now an nx.MultiDiGraph
# --------------------------------------------------------------------------
def _semantic_node(**overrides) -> dict:
    node = {
        "id": "cat_en",
        "label": "cat",
        "language": "en",
        "pos": "noun",
        "details": "keyterm",
        "node_type": "primary",
        "group": "en",
    }
    node.update(overrides)
    return node


def test_semantic_node_styled_by_pos_and_group():
    g = build_semantic_graph({"nodes": [_semantic_node()], "edges": []})
    styled = g.nodes["cat_en"]["styled"]
    assert styled.symbol_size == 30  # primary
    assert styled.border_color == "#FF9500"  # noun border
    assert "cat (English); Part of speech: noun" in styled.tooltip
    assert isinstance(styled.payload, SemanticWordPayload)
    assert styled.payload.id == "cat_en"


def test_semantic_related_node_is_smaller():
    g = build_semantic_graph(
        {"nodes": [_semantic_node(node_type="related")], "edges": []}
    )
    assert g.nodes["cat_en"]["styled"].symbol_size == 20


def test_semantic_edge_two_tier_styling_hardcoded_relation():
    edge = {"from": "a", "to": "b", "relation": "semantic_equivalent", "strength": 1.0}
    g = build_semantic_graph(
        {"nodes": [_semantic_node(id="a"), _semantic_node(id="b")], "edges": [edge]}
    )
    styled = next(d["styled"] for _, _, d in g.edges(data=True))
    assert styled.color == "#00B8D4"
    assert styled.dashed is False


def test_semantic_edge_two_tier_styling_fallback_relation():
    edge = {"from": "a", "to": "b", "relation": "related_term", "strength": 0.5}
    g = build_semantic_graph(
        {"nodes": [_semantic_node(id="a"), _semantic_node(id="b")], "edges": [edge]}
    )
    styled = next(d["styled"] for _, _, d in g.edges(data=True))
    # related_term isn't one of the hardcoded relations, so it falls back to
    # RELATION_COLORS - confirms the fallback tier is actually reachable.
    assert styled.color == "#A0FFFF"


def test_semantic_edge_dashes_key_present_forces_dashed_regardless_of_value():
    edge = {
        "from": "a",
        "to": "b",
        "relation": "related_term",
        "strength": 0.5,
        "dashes": False,
    }
    g = build_semantic_graph(
        {"nodes": [_semantic_node(id="a"), _semantic_node(id="b")], "edges": [edge]}
    )
    styled = next(d["styled"] for _, _, d in g.edges(data=True))
    # Pins the quirky-but-intentional original rule: presence of a "dashes"
    # key forces dashed styling regardless of its actual value.
    assert styled.dashed is True


def test_semantic_edge_prefers_title_over_description_for_tooltip():
    edge = {
        "from": "a",
        "to": "b",
        "relation": "related_term",
        "title": "the title",
        "description": "the description",
    }
    g = build_semantic_graph(
        {"nodes": [_semantic_node(id="a"), _semantic_node(id="b")], "edges": [edge]}
    )
    styled = next(d["styled"] for _, _, d in g.edges(data=True))
    assert styled.tooltip == "the title"


def test_semantic_graph_preserves_parallel_edges():
    # A MultiDiGraph, not a plain Graph - two edges between the same pair
    # (e.g. a translation edge and a cognate edge) must both survive.
    edges = [
        {"from": "a", "to": "b", "relation": "translation", "strength": 1.0},
        {"from": "a", "to": "b", "relation": "cognate", "strength": 0.5},
    ]
    g = build_semantic_graph(
        {"nodes": [_semantic_node(id="a"), _semantic_node(id="b")], "edges": edges}
    )
    assert g.number_of_edges("a", "b") == 2


# --------------------------------------------------------------------------
# build_cooccurrence_graph - unrelated graph, unchanged behavior, new shape
# --------------------------------------------------------------------------
def test_cooccurrence_adapter_sizes_nodes_by_degree():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1)
    graph.add_edge("a", "c", weight=3)
    g = build_cooccurrence_graph(graph, "es")
    assert g.nodes["a"]["styled"].symbol_size == 20 + 2 * 3  # degree 2
    assert g.nodes["b"]["styled"].symbol_size == 20 + 1 * 3  # degree 1


def test_cooccurrence_adapter_edge_color_by_weight():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=5)
    graph.add_edge("c", "d", weight=1)
    g = build_cooccurrence_graph(graph)
    by_pair = {(u, v): d["styled"] for u, v, d in g.edges(data=True)}
    assert by_pair[("a", "b")].color == "#FFFFFF"
    assert by_pair[("c", "d")].color == "#AAAAAA"


def test_cooccurrence_adapter_empty_graph():
    g = build_cooccurrence_graph(nx.Graph())
    assert list(g.nodes) == []
    assert list(g.edges) == []


def test_cooccurrence_node_payload_is_typed():
    graph = nx.Graph()
    graph.add_node("luna")
    g = build_cooccurrence_graph(graph, "es")
    payload = g.nodes["luna"]["styled"].payload
    assert isinstance(payload, CooccurrenceWordPayload)
    assert payload.word == "luna"
    assert payload.language == "es"


# --------------------------------------------------------------------------
# build_word_expansion - category hubs + leaves attached to an existing node
# --------------------------------------------------------------------------
LUNA = WordKey.of("luna", "es")


def test_word_expansion_hub_shown_without_expansion():
    # A category hub appears as soon as its word is analyzed, even if the
    # category itself hasn't been toggled open yet - only its leaves wait.
    nodes, edges = build_word_expansion(
        LUNA,
        {"synonyms": [{"term": "astro", "gloss": None}]},
        attach_node_id="luna_es",
        expanded_categories=set(),
        word_id_index={},
    )
    hub_ids = {n.id for n in nodes}
    assert "cat::luna_es::synonyms" in hub_ids
    assert not any(isinstance(n.payload, RecursiveLeafPayload) for n in nodes)


def test_word_expansion_leaves_shown_once_expanded():
    nodes, edges = build_word_expansion(
        LUNA,
        {"synonyms": [{"term": "astro", "gloss": None}]},
        attach_node_id="luna_es",
        expanded_categories={(LUNA, "synonyms")},
        word_id_index={},
    )
    leaf = next(n for n in nodes if isinstance(n.payload, RecursiveLeafPayload))
    assert leaf.payload.word_key == WordKey.of("astro", "es")
    assert leaf.payload.category == "synonyms"


def test_word_expansion_recursive_leaf_reuses_existing_word_id_index_entry():
    existing_id = "some_other_node_id_for_astro"
    word_id_index = {WordKey.of("astro", "es"): existing_id}
    nodes, edges = build_word_expansion(
        LUNA,
        {"synonyms": [{"term": "astro", "gloss": None}]},
        attach_node_id="luna_es",
        expanded_categories={(LUNA, "synonyms")},
        word_id_index=word_id_index,
    )
    # No new node minted for "astro" - the edge points at the pre-existing id.
    assert not any(n.id == "word::es::astro" for n in nodes)
    assert any(e.target == existing_id for e in edges)


def test_word_expansion_non_recursive_leaf_is_a_plain_leaf_node():
    nodes, edges = build_word_expansion(
        LUNA,
        {"idioms": [{"term": "estar en la luna", "gloss": "to be spacing out"}]},
        attach_node_id="luna_es",
        expanded_categories={(LUNA, "idioms")},
        word_id_index={},
    )
    leaf = next(n for n in nodes if n.payload.kind == "leaf_node")
    assert isinstance(leaf.payload, LeafNodePayload)
    assert leaf.payload.category == "idioms"
    # The term and its gloss stay separate fields instead of being joined
    # into one string - the side panel renders them as term + a caption.
    assert leaf.payload.text == "estar en la luna"
    assert leaf.payload.gloss == "to be spacing out"


def test_word_expansion_cognate_leaf_keeps_word_and_language_separate():
    nodes, _ = build_word_expansion(
        LUNA,
        {"cognates": [{"term": "similar", "gloss": "English"}]},
        attach_node_id="luna_es",
        expanded_categories={(LUNA, "cognates")},
        word_id_index={},
    )
    leaf = next(n for n in nodes if isinstance(n.payload, LeafNodePayload))
    assert leaf.payload.text == "similar"
    assert leaf.payload.gloss == "English"
    assert leaf.label == "similar"


def test_word_expansion_str_list_leaf_has_no_gloss():
    # collocations is a plain StrList upstream - never a {term, gloss} pair -
    # so this must stay exactly as it worked before this change.
    nodes, _ = build_word_expansion(
        LUNA,
        {"collocations": ["salir a la luna"]},
        attach_node_id="luna_es",
        expanded_categories={(LUNA, "collocations")},
        word_id_index={},
    )
    leaf = next(n for n in nodes if isinstance(n.payload, LeafNodePayload))
    assert leaf.payload.text == "salir a la luna"
    assert leaf.payload.gloss is None


def test_word_expansion_caps_leaves_at_eight_per_category():
    analysis = {"synonyms": [{"term": str(i), "gloss": None} for i in range(12)]}
    nodes, _ = build_word_expansion(
        LUNA,
        analysis,
        attach_node_id="luna_es",
        expanded_categories={(LUNA, "synonyms")},
        word_id_index={},
    )
    leaves = [n for n in nodes if isinstance(n.payload, RecursiveLeafPayload)]
    assert len(leaves) == 8


def test_word_expansion_truncates_long_leaf_labels_but_keeps_full_tooltip():
    long_synonym = "a" * 40
    nodes, _ = build_word_expansion(
        LUNA,
        {"synonyms": [{"term": long_synonym, "gloss": None}]},
        attach_node_id="luna_es",
        expanded_categories={(LUNA, "synonyms")},
        word_id_index={},
    )
    leaf = next(n for n in nodes if isinstance(n.payload, RecursiveLeafPayload))
    assert leaf.label == "a" * 27 + "..."
    assert leaf.tooltip == long_synonym


def test_word_expansion_empty_category_produces_no_hub():
    nodes, _ = build_word_expansion(
        LUNA,
        {"synonyms": []},
        attach_node_id="luna_es",
        expanded_categories=set(),
        word_id_index={},
    )
    assert nodes == []


def test_word_expansion_etymology_is_a_toggleable_category_like_any_other():
    # Confirms the deliberate departure from the old standalone graph, where
    # etymology was always shown: it's now a normal CategoryPayload hub, and
    # its language_origin/root sub-items are inert leaves that only appear
    # once expanded, exactly like synonyms/idioms/etc. The etymology
    # narrative itself stays on the hub's own tooltip (not a leaf) - matching
    # how the hub always carried its own description before this migration.
    analysis = {
        "etymology": "from Latin luna",
        "language_origin": "Latin",
        "root": "luna",
    }
    nodes_collapsed, _ = build_word_expansion(
        LUNA,
        analysis,
        attach_node_id="luna_es",
        expanded_categories=set(),
        word_id_index={},
    )
    hub = next(n for n in nodes_collapsed if isinstance(n.payload, CategoryPayload))
    assert hub.payload.category == "etymology"
    assert hub.tooltip == "from Latin luna"
    assert len(nodes_collapsed) == 1  # hub only, no leaves yet

    nodes_expanded, _ = build_word_expansion(
        LUNA,
        analysis,
        attach_node_id="luna_es",
        expanded_categories={(LUNA, "etymology")},
        word_id_index={},
    )
    leaves = [n for n in nodes_expanded if isinstance(n.payload, LeafNodePayload)]
    assert len(leaves) == 2  # language_origin + root, both inert


def test_word_expansion_etymology_hub_shows_with_narrative_but_no_subfields():
    # A hub with a tooltip but zero leaves is still worth showing - the LLM
    # doesn't always break etymology out into language_origin/root.
    nodes, _ = build_word_expansion(
        LUNA,
        {"etymology": "uncertain origin"},
        attach_node_id="luna_es",
        expanded_categories={(LUNA, "etymology")},
        word_id_index={},
    )
    assert len(nodes) == 1
    assert nodes[0].tooltip == "uncertain origin"


def test_word_expansion_examples_truncated_to_forty_chars_then_leaf_label_to_27():
    # examples get truncated to 40 chars first, then that (already-truncated)
    # string goes through the same 27-char leaf-label truncation every other
    # category gets - a real double-truncation, faithfully carried over from
    # the original pyvis-era behavior.
    long_example = "x" * 50
    nodes, _ = build_word_expansion(
        LUNA,
        {"examples": [long_example]},
        attach_node_id="luna_es",
        expanded_categories={(LUNA, "examples")},
        word_id_index={},
    )
    leaf = next(n for n in nodes if isinstance(n.payload, LeafNodePayload))
    assert leaf.label == "x" * 27 + "..."
    assert leaf.tooltip == "x" * 40 + "..."


def test_word_expansion_forms_aggregated_from_multiple_keys():
    analysis = {
        "plural": "lunas",
        "gender_forms": [{"term": "luna", "gloss": "feminine"}],
    }
    nodes, _ = build_word_expansion(
        LUNA,
        analysis,
        attach_node_id="luna_es",
        expanded_categories={(LUNA, "forms")},
        word_id_index={},
    )
    forms_leaves = [n for n in nodes if isinstance(n.payload, LeafNodePayload)]
    labels = [n.label for n in forms_leaves]
    # A plain scalar field (plural) still becomes one pre-formatted label with
    # no gloss - it was never a {term, gloss} pair to begin with.
    assert "Plural: lunas" in labels
    plural_leaf = next(n for n in forms_leaves if n.label == "Plural: lunas")
    assert plural_leaf.payload.gloss is None
    # An Entries field (gender_forms) keeps its term and gloss as separate
    # fields on the payload instead of being joined into the label string.
    assert "luna" in labels
    gender_leaf = next(n for n in forms_leaves if n.label == "luna")
    assert gender_leaf.payload.text == "luna"
    assert gender_leaf.payload.gloss == "feminine"


def test_word_expansion_empty_analysis_produces_nothing():
    nodes, edges = build_word_expansion(
        LUNA, {}, attach_node_id="luna_es", expanded_categories=set(), word_id_index={}
    )
    assert nodes == []
    assert edges == []


# --------------------------------------------------------------------------
# compose_semantic_graph_with_expansions - layering expansions onto the base
# --------------------------------------------------------------------------
def _base_graph() -> nx.MultiDiGraph:
    return build_semantic_graph(
        {
            "nodes": [
                _semantic_node(id="luna_es", label="luna", language="es"),
                _semantic_node(id="moon_en", label="moon", language="en"),
            ],
            "edges": [{"from": "moon_en", "to": "luna_es", "relation": "translation"}],
        }
    )


def test_compose_attaches_a_word_to_its_matching_base_node():
    base = _base_graph()
    composed = compose_semantic_graph_with_expansions(
        base, {LUNA: {"synonyms": [{"term": "astro", "gloss": None}]}}, set()
    )
    assert composed.has_edge("luna_es", "cat::luna_es::synonyms")


def test_compose_skips_a_word_with_no_matching_base_node():
    base = _base_graph()
    ghost_key = WordKey.of("ghost", "es")
    composed = compose_semantic_graph_with_expansions(
        base, {ghost_key: {"synonyms": [{"term": "spirit", "gloss": None}]}}, set()
    )
    # Nothing crashes, and nothing new is attached - there's no node for
    # "ghost" anywhere in the base graph to hang an expansion off of.
    assert set(composed.nodes) == set(base.nodes)


def test_compose_does_not_mutate_the_base_graph():
    base = _base_graph()
    before = set(base.nodes)
    compose_semantic_graph_with_expansions(
        base, {LUNA: {"synonyms": [{"term": "astro", "gloss": None}]}}, set()
    )
    assert set(base.nodes) == before


def test_compose_dedups_a_synonym_shared_by_two_expanded_words():
    base = build_semantic_graph(
        {
            "nodes": [
                _semantic_node(id="luna_es", label="luna", language="es"),
                _semantic_node(id="sol_es", label="sol", language="es"),
            ],
            "edges": [],
        }
    )
    sol_key = WordKey.of("sol", "es")
    analyses = {
        LUNA: {"synonyms": [{"term": "astro", "gloss": None}]},
        sol_key: {"synonyms": [{"term": "astro", "gloss": None}]},
    }
    expanded = {(LUNA, "synonyms"), (sol_key, "synonyms")}
    composed = compose_semantic_graph_with_expansions(base, analyses, expanded)

    astro_nodes = [
        n
        for n in composed.nodes
        if isinstance(composed.nodes[n]["styled"].payload, RecursiveLeafPayload)
        and composed.nodes[n]["styled"].payload.word_key == WordKey.of("astro", "es")
    ]
    assert len(astro_nodes) == 1  # one shared node, not two
    assert composed.in_degree(astro_nodes[0]) == 2  # an edge in from each hub


def test_compose_resolves_two_level_recursion_regardless_of_dict_order():
    base = _base_graph()
    astro_key = WordKey.of("astro", "es")
    # astro's analysis is inserted BEFORE luna's - the fixed-point loop must
    # still resolve it, since astro's attach point (a node under luna) only
    # exists once luna's own expansion has been built.
    analyses = {
        astro_key: {"synonyms": [{"term": "estrella", "gloss": None}]},
        LUNA: {"synonyms": [{"term": "astro", "gloss": None}]},
    }
    expanded = {(LUNA, "synonyms"), (astro_key, "synonyms")}
    composed = compose_semantic_graph_with_expansions(base, analyses, expanded)

    estrella_key = WordKey.of("estrella", "es")
    assert any(
        isinstance(composed.nodes[n]["styled"].payload, RecursiveLeafPayload)
        and composed.nodes[n]["styled"].payload.word_key == estrella_key
        for n in composed.nodes
    )


def test_compose_skips_a_word_whose_parent_category_is_collapsed():
    base = _base_graph()
    astro_key = WordKey.of("astro", "es")
    analyses = {
        LUNA: {"synonyms": [{"term": "astro", "gloss": None}]},
        astro_key: {"synonyms": [{"term": "estrella", "gloss": None}]},
    }
    # luna's synonyms category is NOT expanded, so astro's own node (and
    # therefore its own expansion) never gets attached this rerun.
    composed = compose_semantic_graph_with_expansions(base, analyses, set())
    estrella_key = WordKey.of("estrella", "es")
    assert not any(
        isinstance(composed.nodes[n]["styled"].payload, RecursiveLeafPayload)
        and composed.nodes[n]["styled"].payload.word_key == estrella_key
        for n in composed.nodes
    )


# --------------------------------------------------------------------------
# apply_pinned_positions
# --------------------------------------------------------------------------
def test_apply_pinned_positions_sets_xy_fixed_on_known_nodes():
    base = _base_graph()
    positions = {"luna_es": Position(x=1.0, y=2.0)}
    pinned = apply_pinned_positions(base, positions)
    styled = pinned.nodes["luna_es"]["styled"]
    assert (styled.x, styled.y, styled.fixed) == (1.0, 2.0, True)


def test_apply_pinned_positions_ignores_unknown_node_ids():
    base = _base_graph()
    positions = {"no_such_node": Position(x=1.0, y=2.0)}
    pinned = apply_pinned_positions(base, positions)  # should not raise
    assert pinned.nodes["luna_es"]["styled"].x is None


def test_apply_pinned_positions_does_not_mutate_the_input_graph():
    base = _base_graph()
    apply_pinned_positions(base, {"luna_es": Position(x=1.0, y=2.0)})
    assert base.nodes["luna_es"]["styled"].x is None


# --------------------------------------------------------------------------
# graph_to_echarts_data - the one dict-producing boundary
# --------------------------------------------------------------------------
def test_graph_to_echarts_data_flattens_nodes_and_edges():
    base = _base_graph()
    data = graph_to_echarts_data(base)
    names = {n["name"] for n in data["nodes"]}
    assert names == {"luna_es", "moon_en"}
    assert data["edges"][0]["source"] == "moon_en"
    assert data["edges"][0]["target"] == "luna_es"


def test_graph_to_echarts_data_includes_pinned_position():
    base = _base_graph()
    pinned = apply_pinned_positions(base, {"luna_es": Position(x=5.0, y=6.0)})
    data = graph_to_echarts_data(pinned)
    luna = next(n for n in data["nodes"] if n["name"] == "luna_es")
    assert luna["x"] == 5.0 and luna["y"] == 6.0 and luna["fixed"] is True
    moon = next(n for n in data["nodes"] if n["name"] == "moon_en")
    assert "x" not in moon  # never clicked, left for the layout to place


def test_graph_to_echarts_data_feeds_build_graph_echarts_options_end_to_end():
    base = _base_graph()
    options = build_graph_echarts_options(graph_to_echarts_data(base))
    assert options["series"][0]["type"] == "graph"
    assert len(options["series"][0]["data"]) == 2
