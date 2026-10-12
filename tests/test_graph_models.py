"""
Tests for idiomapp/utils/graph_models.py - the typed graph element and request
models, kept apart from the graph building and rendering in graph_viz.py.
"""

import ast
import json
from pathlib import Path

import pytest

from idiomapp.utils import graph_models
from idiomapp.utils.graph_models import (
    RECURSIVE_LEAF_CATEGORIES,
    CategoryPayload,
    CooccurrenceWordPayload,
    EdgePayload,
    EdgeSelection,
    LeafNodePayload,
    RecursiveLeafPayload,
    SemanticWordPayload,
    SourcedSelection,
    WordKey,
    parse_edge_payload,
    parse_node_payload,
)

LUNA = WordKey.of("luna", "es")


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


def test_recursive_leaf_categories_are_the_two_bare_word_ones():
    # Deliberately just these two - see the comment next to
    # RECURSIVE_LEAF_CATEGORIES for why cognates are excluded despite also
    # being structurally single words.
    assert RECURSIVE_LEAF_CATEGORIES == {"synonyms", "antonyms"}


# --------------------------------------------------------------------------
# Node/edge payloads round-trip through JSON
# --------------------------------------------------------------------------
NODE_PAYLOADS = [
    SemanticWordPayload(
        id="luna_es",
        label="luna",
        language="es",
        pos="NOUN",
        details="moon",
        node_type="primary",
        group="1",
        sentence_group="s1",
    ),
    CategoryPayload(word_key=LUNA, category="synonyms", label="≈ Synonyms"),
    RecursiveLeafPayload(word_key=WordKey.of("astro", "es"), category="synonyms"),
    LeafNodePayload(category="cognates", text="lunar", gloss="English"),
    LeafNodePayload(category="collocations", text="salir a la luna"),
    CooccurrenceWordPayload(word="luna", language="es", degree=3),
]


@pytest.mark.parametrize("payload", NODE_PAYLOADS, ids=lambda p: p.kind)
def test_node_payload_round_trips_through_json(payload):
    # Dump to JSON and parse back the way a browser click does: the
    # discriminator has to rebuild the exact same variant.
    assert parse_node_payload(json.loads(payload.model_dump_json())) == payload


def test_parse_node_payload_unknown_kind_is_none():
    assert parse_node_payload({"kind": "not_a_kind"}) is None


def test_parse_node_payload_malformed_is_none():
    assert parse_node_payload({"kind": "word", "id": "x"}) is None


def test_edge_payload_round_trips_through_json():
    edge = EdgePayload(kind="translation", label="moon", strength=0.9, weight=2)
    assert parse_edge_payload(json.loads(edge.model_dump_json())) == edge


def test_parse_edge_payload_malformed_is_none():
    assert parse_edge_payload({"label": "missing kind"}) is None


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------
def test_sourced_selection_pairs_a_selection_with_its_graph():
    edge = EdgeSelection(payload=EdgePayload(kind="x"), source_id="a", target_id="b")
    sourced = SourcedSelection(selection=edge, source="semantic")
    assert sourced.selection is edge
    assert sourced.source == "semantic"


# --------------------------------------------------------------------------
# Layering: models know nothing about graph building or rendering
# --------------------------------------------------------------------------
def test_graph_models_does_not_import_graph_building_or_rendering():
    tree = ast.parse(Path(graph_models.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {
        "networkx",
        "streamlit",
        "streamlit_echarts",
        "idiomapp.utils.graph_viz",
    }
    assert not imported & forbidden
