"""
Tests for idiomapp/utils/graph_viz.py - the pyvis-to-ECharts adapter layer.

Pure pytest throughout: no Streamlit runtime, no ECharts JS involved. These
pin the *data* each adapter produces; the live click round-trip itself was
verified empirically against a running st_echarts component (see the
graph-viz migration plan) and isn't something a unit test can exercise.
"""

import networkx as nx

from idiomapp.utils.graph_viz import (
    GRAPH_CATEGORIES,
    GRAPH_CLICK_JS,
    format_entries,
    format_entry,
    adjust_color,
    build_graph_echarts_options,
    cooccurrence_graph_to_echarts_data,
    filter_invalid_nodes,
    resolve_graph_click,
    sanitize_tooltip_text,
    semantic_graph_to_echarts_data,
    word_analysis_to_echarts_data,
)


# --------------------------------------------------------------------------
# Small pure helpers
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


def test_sanitize_tooltip_text_strips_tags_and_escapes():
    assert sanitize_tooltip_text("<b>bold</b> & <i>x</i>") == "bold & x"


def test_adjust_color_lightens_and_clamps():
    assert adjust_color("#000000", 10) == "#0a0a0a"
    assert adjust_color("#fefefe", 10) == "#ffffff"  # clamped at 255


def test_resolve_graph_click_none_event_is_none():
    assert resolve_graph_click(None) is None


def test_resolve_graph_click_unknown_datatype_is_none():
    assert resolve_graph_click({"dataType": "series", "raw": {}}) is None


def test_resolve_graph_click_node():
    raw = {"id": "cat_en", "label": "cat"}
    event = {"dataType": "node", "raw": raw, "name": "cat_en"}
    assert resolve_graph_click(event) == {"kind": "node", "data": raw}


def test_resolve_graph_click_edge():
    raw = {"from": "a", "to": "b", "relation": "cognate"}
    event = {"dataType": "edge", "raw": raw}
    assert resolve_graph_click(event) == {"kind": "edge", "data": raw}


# --------------------------------------------------------------------------
# build_graph_echarts_options - the shared assembly function
# --------------------------------------------------------------------------
def test_build_graph_echarts_options_shape():
    graph_data = {
        "nodes": [
            {
                "id": "a",
                "label": "a",
                "symbolSize": 30,
                "symbol": "circle",
                "itemStyle": {"color": "#fff"},
                "tooltip": "tip",
                "raw": {"id": "a"},
            }
        ],
        "edges": [
            {
                "from": "a",
                "to": "b",
                "style": {"color": "#000", "width": 2},
                "tooltip": "edge tip",
                "raw": {"from": "a", "to": "b"},
            }
        ],
    }
    options = build_graph_echarts_options(graph_data)
    series = options["series"][0]
    assert series["type"] == "graph"
    assert series["data"][0]["name"] == "a"
    assert series["data"][0]["raw"] == {"id": "a"}
    assert series["links"][0]["source"] == "a"
    assert series["links"][0]["target"] == "b"
    assert series["links"][0]["raw"] == {"from": "a", "to": "b"}


def test_build_graph_echarts_options_empty_graph():
    options = build_graph_echarts_options({"nodes": [], "edges": []})
    series = options["series"][0]
    assert series["data"] == []
    assert series["links"] == []


# --------------------------------------------------------------------------
# Semantic / translation graph
# --------------------------------------------------------------------------
def test_filter_invalid_nodes_drops_error_nodes_and_their_edges():
    graph_data = {
        "nodes": [
            {"id": "ok", "label": "gato"},
            {"id": "bad", "label": "Translation failed"},
        ],
        "edges": [
            {"from": "ok", "to": "bad", "relation": "related_term"},
        ],
    }
    result = filter_invalid_nodes(graph_data)
    assert [n["id"] for n in result["nodes"]] == ["ok"]
    assert result["edges"] == []


def test_filter_invalid_nodes_does_not_mutate_input():
    graph_data = {
        "nodes": [{"id": "ok", "label": "gato"}],
        "edges": [],
    }
    filter_invalid_nodes(graph_data)
    assert graph_data["nodes"] == [{"id": "ok", "label": "gato"}]


def test_semantic_node_styled_by_pos_and_group():
    node = {
        "id": "cat_en",
        "label": "cat",
        "language": "en",
        "pos": "noun",
        "details": "keyterm",
        "node_type": "primary",
        "group": "en",
    }
    styled = semantic_graph_to_echarts_data({"nodes": [node], "edges": []})
    echarts_node = styled["nodes"][0]
    assert echarts_node["symbolSize"] == 30  # primary
    assert echarts_node["itemStyle"]["borderColor"] == "#FF9500"  # noun border
    assert "cat (English); Part of speech: noun" in echarts_node["tooltip"]
    assert echarts_node["raw"] is node


def test_semantic_related_node_is_smaller():
    node = {"id": "x", "label": "x", "node_type": "related", "group": "en"}
    styled = semantic_graph_to_echarts_data({"nodes": [node], "edges": []})
    assert styled["nodes"][0]["symbolSize"] == 20


def test_semantic_edge_two_tier_styling_hardcoded_relation():
    edge = {"from": "a", "to": "b", "relation": "semantic_equivalent", "strength": 1.0}
    styled = semantic_graph_to_echarts_data({"nodes": [], "edges": [edge]})
    assert styled["edges"][0]["style"]["color"] == "#00B8D4"
    assert styled["edges"][0]["style"]["type"] == "solid"


def test_semantic_edge_two_tier_styling_fallback_relation():
    edge = {"from": "a", "to": "b", "relation": "related_term", "strength": 0.5}
    styled = semantic_graph_to_echarts_data({"nodes": [], "edges": [edge]})
    # related_term isn't one of the hardcoded relations, so it falls back to
    # RELATION_COLORS - confirms the fallback tier is actually reachable.
    assert styled["edges"][0]["style"]["color"] == "#A0FFFF"


def test_semantic_edge_dashes_key_present_forces_dashed_regardless_of_value():
    edge = {
        "from": "a",
        "to": "b",
        "relation": "related_term",
        "strength": 0.5,
        "dashes": False,
    }
    styled = semantic_graph_to_echarts_data({"nodes": [], "edges": [edge]})
    # Pins the quirky-but-intentional original pyvis rule: presence of a
    # "dashes" key forces dashed styling regardless of its actual value.
    assert styled["edges"][0]["style"]["type"] == "dashed"


def test_semantic_edge_prefers_title_over_description_for_tooltip():
    edge = {
        "from": "a",
        "to": "b",
        "relation": "related_term",
        "title": "the title",
        "description": "the description",
    }
    styled = semantic_graph_to_echarts_data({"nodes": [], "edges": [edge]})
    assert styled["edges"][0]["tooltip"] == "the title"


# --------------------------------------------------------------------------
# Co-occurrence network
# --------------------------------------------------------------------------
def test_cooccurrence_adapter_sizes_nodes_by_degree():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1)
    graph.add_edge("a", "c", weight=3)
    data = cooccurrence_graph_to_echarts_data(graph, "es")
    by_id = {n["id"]: n for n in data["nodes"]}
    assert by_id["a"]["symbolSize"] == 20 + 2 * 3  # degree 2
    assert by_id["b"]["symbolSize"] == 20 + 1 * 3  # degree 1


def test_cooccurrence_adapter_edge_color_by_weight():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=5)
    graph.add_edge("c", "d", weight=1)
    data = cooccurrence_graph_to_echarts_data(graph)
    by_pair = {(e["from"], e["to"]): e for e in data["edges"]}
    assert by_pair[("a", "b")]["style"]["color"] == "#FFFFFF"
    assert by_pair[("c", "d")]["style"]["color"] == "#AAAAAA"


def test_cooccurrence_adapter_empty_graph():
    data = cooccurrence_graph_to_echarts_data(nx.Graph())
    assert data == {"nodes": [], "edges": []}


# --------------------------------------------------------------------------
# Per-word knowledge graph
# --------------------------------------------------------------------------
def test_word_analysis_graph_always_has_main_node():
    data = word_analysis_to_echarts_data("gato", "es", {"pos": "NOUN"})
    assert data["nodes"][0]["id"] == "main"
    assert data["nodes"][0]["label"] == "gato"


def test_word_analysis_graph_category_hub_and_leaves():
    analysis = {"synonyms": [{"term": "felino", "gloss": None}]}
    data = word_analysis_to_echarts_data("gato", "es", analysis)
    ids = {n["id"] for n in data["nodes"]}
    assert "cat_synonyms" in ids
    assert "item_1" in ids
    leaf = next(n for n in data["nodes"] if n["id"] == "item_1")
    assert leaf["label"] == "felino"
    assert leaf["raw"]["word"] == "felino"


def test_word_analysis_graph_caps_leaves_at_eight_per_category():
    analysis = {"synonyms": [{"term": str(i), "gloss": None} for i in range(12)]}
    data = word_analysis_to_echarts_data("w", "es", analysis)
    leaves = [n for n in data["nodes"] if n["id"].startswith("item_")]
    assert len(leaves) == 8


def test_word_analysis_graph_truncates_long_leaf_labels():
    long_synonym = "a" * 40
    analysis = {"synonyms": [{"term": long_synonym, "gloss": None}]}
    data = word_analysis_to_echarts_data("w", "es", analysis)
    leaf = next(n for n in data["nodes"] if n["id"] == "item_1")
    assert leaf["label"] == "a" * 27 + "..."
    assert leaf["tooltip"] == long_synonym  # full text preserved in tooltip


def test_word_analysis_graph_empty_category_produces_no_hub():
    data = word_analysis_to_echarts_data("w", "es", {"synonyms": []})
    ids = {n["id"] for n in data["nodes"]}
    assert "cat_synonyms" not in ids


def test_word_analysis_graph_etymology_subnodes():
    analysis = {"etymology": "from Latin", "language_origin": "Latin", "root": "cattus"}
    data = word_analysis_to_echarts_data("gato", "es", analysis)
    ids = {n["id"] for n in data["nodes"]}
    assert "etymology" in ids
    assert any(i.startswith("language_origin_") for i in ids)
    assert any(i.startswith("root_") for i in ids)


def test_word_analysis_graph_examples_truncated_to_forty_chars():
    # examples get truncated to 40 chars by add_category's caller, then that
    # (already-truncated) string goes through the *same* 27-char leaf-label
    # truncation every other category gets - a real double-truncation, faithfully
    # reproduced from the original pyvis code's add_category closure.
    long_example = "x" * 50
    data = word_analysis_to_echarts_data("w", "es", {"examples": [long_example]})
    leaf = next(n for n in data["nodes"] if n["id"] == "item_1")
    assert leaf["label"] == "x" * 27 + "..."
    assert leaf["tooltip"] == "x" * 40 + "..."


def test_word_analysis_graph_forms_aggregated_from_multiple_keys():
    analysis = {
        "plural": "gatos",
        "gender_forms": [{"term": "gata", "gloss": "female"}],
    }
    data = word_analysis_to_echarts_data("gato", "es", analysis)
    leaves = [n["label"] for n in data["nodes"] if n["id"].startswith("item_")]
    assert "Plural: gatos" in leaves
    assert "gata: female" in leaves


def test_word_analysis_graph_has_no_leaves_for_empty_analysis():
    data = word_analysis_to_echarts_data("w", "es", {})
    assert len(data["nodes"]) == 1  # just "main"
    assert data["edges"] == []


# --------------------------------------------------------------------------
# Module-level constants sanity
# --------------------------------------------------------------------------
def test_graph_categories_and_click_js_are_defined():
    assert len(GRAPH_CATEGORIES) == 5
    assert "dataType" in GRAPH_CLICK_JS
