"""
Tests for idiomapp/utils/nlp_utils.py's cross-language relationship detection.

`detect_cognate` is pure string logic - no spaCy involved. `process_sentence_pair`
does call spaCy (via `analyze_parts_of_speech`) to tokenize each sentence first,
so those tests patch `load_spacy_model` to return a blank (untrained) pipeline,
matching the pattern already established in tests/test_analysis_cache.py and
tests/test_llm_clients.py - tokenization only, no model download required.
"""

from unittest.mock import patch

import spacy

from idiomapp.utils.nlp_utils import detect_cognate, process_sentence_pair


# --------------------------------------------------------------------------
# detect_cognate - pure function, no spaCy
# --------------------------------------------------------------------------
def test_detect_cognate_prefix_and_suffix_match():
    assert detect_cognate("nation", "nation") == 0.9


def test_detect_cognate_prefix_only():
    assert detect_cognate("computer", "computable") == 0.7


def test_detect_cognate_suffix_only():
    assert detect_cognate("luna", "lluna") == 0.6


def test_detect_cognate_high_edit_distance_fallback():
    # Constructed, not a real word pair: different first and last character but
    # otherwise identical, so neither the prefix nor suffix rule fires and only
    # the edit-distance fallback can - real cognates almost always share a
    # prefix or suffix too (see test_detect_cognate_prefix_and_suffix_match),
    # which made this branch genuinely hard to reach with real vocabulary.
    assert detect_cognate("xlephantasq", "zlephantasr") == 0.8


def test_detect_cognate_short_words_are_never_cognates():
    # Both the length-gate case and words that just don't match at all.
    assert detect_cognate("es", "és") is None
    assert detect_cognate("moon", "noche") is None


def test_detect_cognate_real_translation_pair_is_not_a_cognate():
    # The exact case that motivated separating these two signals: "moon" and
    # "luna" are a correct translation but share no visible string structure.
    assert detect_cognate("moon", "luna") is None


# --------------------------------------------------------------------------
# process_sentence_pair - translation and cognate edges are independent
# --------------------------------------------------------------------------
def _blank_pipeline(language):
    return spacy.blank(language)


def _edges_between(graph_data, relation):
    return [e for e in graph_data["edges"] if e.get("relation") == relation]


def test_process_sentence_pair_draws_a_translation_edge_from_alignment():
    graph_data = {"nodes": [], "edges": []}
    with patch(
        "idiomapp.utils.nlp_utils.load_spacy_model", side_effect=_blank_pipeline
    ):
        process_sentence_pair(
            "The moon is beautiful",
            "La luna es hermosa",
            "en",
            "es",
            graph_data,
            set(),
            {},
            alignment_pairs=frozenset({("moon", "luna")}),
        )

    translations = _edges_between(graph_data, "translation")
    assert len(translations) == 1
    edge = translations[0]
    assert edge["from"] == "moon_en"
    assert edge["to"] == "luna_es"
    assert edge["strength"] == 1.0


def test_process_sentence_pair_draws_a_cognate_edge_independent_of_alignment():
    graph_data = {"nodes": [], "edges": []}
    with patch(
        "idiomapp.utils.nlp_utils.load_spacy_model", side_effect=_blank_pipeline
    ):
        process_sentence_pair(
            "Some nation",
            "Alguna nation",
            "en",
            "es",
            graph_data,
            set(),
            {},
            # No alignment data at all - cognate detection must still fire.
        )

    cognates = _edges_between(graph_data, "cognate")
    assert any(e["from"] == "nation_en" and e["to"] == "nation_es" for e in cognates)
    assert all(e["dashes"] is True for e in cognates)


def test_process_sentence_pair_a_pair_can_be_both_translation_and_cognate():
    graph_data = {"nodes": [], "edges": []}
    with patch(
        "idiomapp.utils.nlp_utils.load_spacy_model", side_effect=_blank_pipeline
    ):
        process_sentence_pair(
            "Some nation",
            "Alguna nation",
            "en",
            "es",
            graph_data,
            set(),
            {},
            alignment_pairs=frozenset({("nation", "nation")}),
        )

    kinds = {
        e["relation"]
        for e in graph_data["edges"]
        if e["from"] == "nation_en" and e["to"] == "nation_es"
    }
    assert kinds == {"translation", "cognate"}


def test_process_sentence_pair_unrelated_pair_gets_no_edge():
    graph_data = {"nodes": [], "edges": []}
    with patch(
        "idiomapp.utils.nlp_utils.load_spacy_model", side_effect=_blank_pipeline
    ):
        process_sentence_pair(
            "The moon",
            "La luna",
            "en",
            "es",
            graph_data,
            set(),
            {},
        )

    # Neither aligned nor a cognate - no relationship between these two.
    assert not any(
        e["from"] == "moon_en" and e["to"] == "luna_es" for e in graph_data["edges"]
    )
