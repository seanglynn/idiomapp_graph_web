"""
Tests for the word-analysis display layer in idiomapp/streamlit/app.py, and
for the Semantic Graph's click-to-explore behavior (the selection panel and
the click dispatcher that drives auto-analysis/category toggling).

`streamlit.testing.v1.AppTest`-based tests throughout - Streamlit's official
headless test harness. Importing from `idiomapp.streamlit.app` does not run
`main()`: Streamlit only executes a script's `if __name__ == "__main__"` block
when that script is the actual run target, which a plain import does not set -
so this never touches Ollama, spaCy, or any network/session-dependent code,
confirmed empirically to run in ~3s.

A function that ends by calling `st.rerun()` (several of the click-dispatch
functions below do, on purpose - see graph_data_model docs) makes AppTest
re-execute the whole script from the top. Every test script below that calls
one of these guards the call behind `if "already_dispatched" not in
st.session_state`, so the second (rerun-triggered) execution reads the
already-updated state instead of calling the function - and, in the toggle
case, flipping it right back - again. Confirmed empirically: without this
guard, a script that unconditionally calls a toggle function on every run
never converges and the test times out.

(`_format_entry`/`_format_entries` moved to `idiomapp.utils.graph_viz` and are
covered by `tests/test_graph_viz.py` now, along with the typed selection/
payload model these tests build fixtures out of.)
"""

from streamlit.testing.v1 import AppTest


# --------------------------------------------------------------------------
# Rendering, via AppTest
# --------------------------------------------------------------------------
RICH_ANALYSIS = {
    "pos": "VERB",
    "definition": "to speak",
    "register": "informal",
    "frequency": "common",
    "ipa": "aβlaɾ",
    "syllables": "ha-blar",
    "stress": "final",
    "etymology": "from Latin fabulari",
    "language_origin": "Latin",
    "root": "fab-",
    "cognates": [{"term": "fable", "gloss": "English"}],
    "synonyms": ["decir", "conversar"],
    "antonyms": ["callar"],
    "examples": ["Ella habla español."],
    "idioms": [
        {"term": "hablar por hablar", "gloss": "to talk for the sake of talking"}
    ],
    "infinitive": "hablar",
    "verb_type": "regular",
    "conjugations": [{"term": "present", "gloss": "hablo"}],
    "related_forms": [{"term": "hablante", "gloss": "speaker"}],
    "tips": ["Practice the -ar conjugation pattern"],
}


def _run(analysis_data: dict, word: str = "hablar", language: str = "es") -> AppTest:
    script = f"""
from idiomapp.streamlit.app import display_word_analysis
display_word_analysis({word!r}, {language!r}, {analysis_data!r})
"""
    at = AppTest.from_string(script)
    at.run(timeout=30)
    return at


def test_rich_analysis_renders_without_exception():
    at = _run(RICH_ANALYSIS)
    assert not at.exception


def test_llm_error_shows_warning_and_skips_tabs():
    """The early-return path (analysis_data has "llm_error") runs before the tab
    layout is ever built - this is the branch most likely to regress from
    removing the dead grammar_data/pron_data blocks, since it executes first."""
    at = _run({"llm_error": "Anthropic rate limit reached"})
    assert not at.exception
    assert any("LLM unavailable" in w.value for w in at.warning)
    assert len(at.tabs) == 0


def test_sparse_analysis_shows_only_the_matching_tab():
    """Regression check for tab-visibility after removing the dead
    `pron_data`-based clause in _display_analysis_panels."""
    at = _run({"ipa": "test", "syllables": "test"})
    assert not at.exception
    assert len(at.tabs) == 1


def test_no_data_shows_info_message_not_empty_tabs():
    at = _run({"pos": "NOUN"})
    assert not at.exception
    assert len(at.tabs) == 0
    assert any("No detailed data available" in i.value for i in at.info)


def test_grammar_tab_renders_for_verb_without_exception():
    at = _run(RICH_ANALYSIS)
    assert not at.exception


def test_grammar_tab_renders_for_noun_without_exception():
    data = dict(
        RICH_ANALYSIS,
        pos="NOUN",
        gender="masculine",
        plural="gatos",
        articles=[{"term": "el", "gloss": None}],
    )
    at = _run(data, word="gato")
    assert not at.exception


def test_grammar_tab_renders_for_adjective_without_exception():
    data = dict(
        RICH_ANALYSIS,
        pos="ADJ",
        gender_forms=[{"term": "m", "gloss": "hablador"}],
        comparison=[{"term": "more", "gloss": "más hablador"}],
        position="after noun",
    )
    at = _run(data, word="hablador")
    assert not at.exception


def test_grammar_tab_renders_for_unknown_pos_without_exception():
    data = dict(RICH_ANALYSIS, pos="OTHER", related_words=["hablante"])
    at = _run(data)
    assert not at.exception


# --------------------------------------------------------------------------
# The Semantic Graph's selection panel (_render_graph_selection_panel)
# --------------------------------------------------------------------------
_PANEL_IMPORTS = """
import streamlit as st
from idiomapp.utils.graph_viz import (
    CategoryPayload, EdgePayload, EdgeSelection, LeafNodePayload,
    NodeSelection, SemanticWordPayload, SourcedSelection, WordKey,
)
from idiomapp.streamlit.app import _render_graph_selection_panel
"""


def _run_panel(selection_setup: str, *, source: str = "semantic") -> AppTest:
    script = f"""
{_PANEL_IMPORTS}
st.session_state.setdefault("graph_word_analyses", {{}})
{selection_setup}
_render_graph_selection_panel({source!r})
"""
    at = AppTest.from_string(script)
    at.run(timeout=30)
    return at


def test_panel_empty_state_prompts_to_click():
    at = _run_panel("st.session_state['graph_selection'] = None")
    assert not at.exception
    assert any("Click a node or edge" in c.value for c in at.caption)


def test_panel_ignores_a_selection_from_a_different_graph():
    """A co-occurrence-graph selection shouldn't show up while looking at the
    semantic graph's panel, or vice versa - each graph gets its own empty
    state instead of a confusing leftover from the other one."""
    setup = """
st.session_state['graph_selection'] = SourcedSelection(
    selection=NodeSelection(payload=CategoryPayload(
        word_key=WordKey.of('luna', 'es'), category='synonyms', label='Synonyms')),
    source='cooccurrence',
)
"""
    at = _run_panel(setup, source="semantic")
    assert not at.exception
    assert any("Click a node or edge" in c.value for c in at.caption)


def test_panel_shows_category_label():
    setup = """
st.session_state['graph_selection'] = SourcedSelection(
    selection=NodeSelection(payload=CategoryPayload(
        word_key=WordKey.of('luna', 'es'), category='synonyms', label='≈ Synonyms')),
    source='semantic',
)
"""
    at = _run_panel(setup)
    assert not at.exception
    assert any("Synonyms" in m.value for m in at.markdown)


def test_panel_shows_leaf_node_category_and_text():
    setup = """
st.session_state['graph_selection'] = SourcedSelection(
    selection=NodeSelection(payload=LeafNodePayload(
        category='idioms', text='estar en la luna')),
    source='semantic',
)
"""
    at = _run_panel(setup)
    assert not at.exception
    assert any("Idioms" in m.value for m in at.markdown)
    assert any("estar en la luna" in m.value for m in at.markdown)
    # No gloss on this leaf - no caption should render for it.
    assert not any("English" in c.value for c in at.caption)


def test_panel_shows_leaf_node_gloss_as_a_caption():
    setup = """
st.session_state['graph_selection'] = SourcedSelection(
    selection=NodeSelection(payload=LeafNodePayload(
        category='cognates', text='night', gloss='English')),
    source='semantic',
)
"""
    at = _run_panel(setup)
    assert not at.exception
    assert any("night" in m.value for m in at.markdown)
    assert any("English" in c.value for c in at.caption)


def test_panel_shows_analyzing_placeholder_before_analysis_arrives():
    setup = """
st.session_state['graph_selection'] = SourcedSelection(
    selection=NodeSelection(payload=SemanticWordPayload(
        id='luna_es', label='luna', language='es', pos='noun', details='moon',
        node_type='primary', group='es')),
    source='semantic',
)
"""
    at = _run_panel(setup)
    assert not at.exception
    assert any("Analyzing" in c.value for c in at.caption)
    assert any("luna" in m.value for m in at.markdown)


def test_panel_shows_full_analysis_once_cached():
    setup = """
st.session_state['graph_word_analyses'][WordKey.of('luna', 'es')] = {
    'pos': 'NOUN', 'definition': 'the moon', 'synonyms': ['astro'],
}
st.session_state['graph_selection'] = SourcedSelection(
    selection=NodeSelection(payload=SemanticWordPayload(
        id='luna_es', label='luna', language='es', pos='noun', details='moon',
        node_type='primary', group='es')),
    source='semantic',
)
"""
    at = _run_panel(setup)
    assert not at.exception
    # display_word_analysis's own tabs, proving the full analysis rendered
    # inline in the panel rather than just the bare node info.
    assert len(at.tabs) > 0


def test_panel_shows_semantic_edge_relation_and_strength():
    setup = """
st.session_state['graph_selection'] = SourcedSelection(
    selection=EdgeSelection(
        payload=EdgePayload(kind='cognate', description='shared Latin root', strength=0.8),
        source_id='moon_en', target_id='luna_es',
    ),
    source='semantic',
)
"""
    at = _run_panel(setup)
    assert not at.exception
    assert any("moon_en" in m.value and "luna_es" in m.value for m in at.markdown)
    assert any("0.80" in c.value for c in at.caption)


# --------------------------------------------------------------------------
# Click dispatch (_dispatch_semantic_graph_click and its helpers)
# --------------------------------------------------------------------------
def test_dispatch_toggles_a_category_on_then_off():
    # Each script run is allowed exactly one dispatch call (matching the real
    # app's rule: a click's chart_event is only truthy for the rerun right
    # after it - see the module-level RECURSIVE_LEAF_CATEGORIES/chart_event
    # note in graph_viz.py). "clicks" counts how many dispatches have
    # actually happened across both AppTest runs below, driving click 1 (on)
    # then click 2 (off) rather than repeating the same click.
    script = """
import streamlit as st
from idiomapp.utils.graph_viz import WordKey, CategoryPayload, NodeSelection
from idiomapp.streamlit.app import _dispatch_semantic_graph_click

st.session_state.setdefault("graph_expanded_categories", set())
st.session_state.setdefault("clicks", 0)
if "dispatched_this_run" not in st.session_state:
    st.session_state["dispatched_this_run"] = True
    st.session_state["clicks"] += 1
    wk = WordKey.of("luna", "es")
    selection = NodeSelection(
        payload=CategoryPayload(word_key=wk, category="synonyms", label="Synonyms")
    )
    _dispatch_semantic_graph_click(selection)
st.write("expanded=" + str(st.session_state["graph_expanded_categories"]))
"""
    at = AppTest.from_string(script)
    at.run(timeout=30)  # click 1: toggles the category on
    assert not at.exception
    on_texts = [e.value for e in at.get("markdown")]
    assert any("synonyms" in t for t in on_texts)

    del at.session_state["dispatched_this_run"]  # simulate a second click
    at.run(timeout=30)  # click 2: toggles it back off
    assert not at.exception
    off_texts = [e.value for e in at.get("markdown")]
    assert all("synonyms" not in t for t in off_texts)


def test_dispatch_leaf_node_click_does_nothing():
    """Clicking an idiom/example leaf isn't a word to analyze - dispatch is a
    deliberate no-op, confirmed by nothing in session state changing."""
    script = """
import streamlit as st
from idiomapp.utils.graph_viz import LeafNodePayload, NodeSelection
from idiomapp.streamlit.app import _dispatch_semantic_graph_click

st.session_state.setdefault("graph_word_analyses", {})
selection = NodeSelection(payload=LeafNodePayload(category="idioms", text="x"))
_dispatch_semantic_graph_click(selection)
st.write("done, analyses=" + str(st.session_state["graph_word_analyses"]))
"""
    at = AppTest.from_string(script)
    at.run(timeout=30)
    assert not at.exception
    assert any("done, analyses={}" in e.value for e in at.get("markdown"))


def test_dispatch_analyzes_a_fresh_word_through_the_shared_cache_path(monkeypatch):
    """The one call path for graph-driven analysis: analyze_word_linguistics,
    the exact function the dropdown+button flow already uses - confirms no
    separate/duplicate LLM-calling path was introduced for the graph."""
    import idiomapp.streamlit.app as app_module

    calls = []

    async def fake_analyze(word, language, client):
        calls.append((word, language))
        return {"pos": "NOUN", "definition": f"fake analysis of {word}"}

    monkeypatch.setattr(app_module, "analyze_word_linguistics", fake_analyze)
    monkeypatch.setattr(app_module, "get_llm_client", lambda: object())

    script = """
import streamlit as st
from idiomapp.utils.graph_viz import WordKey, SemanticWordPayload, NodeSelection
from idiomapp.streamlit.app import _dispatch_semantic_graph_click

st.session_state.setdefault("graph_word_analyses", {})
st.session_state["model_available"] = True
if "already_dispatched" not in st.session_state:
    st.session_state["already_dispatched"] = True
    payload = SemanticWordPayload(
        id="luna_es", label="luna", language="es", pos="noun", details="moon",
        node_type="primary", group="es",
    )
    _dispatch_semantic_graph_click(NodeSelection(payload=payload))
analyses = st.session_state["graph_word_analyses"]
st.write(str(list(analyses.items())))
"""
    at = AppTest.from_string(script)
    at.run(timeout=30)
    assert not at.exception
    assert calls == [("luna", "es")]
    assert any("fake analysis of luna" in e.value for e in at.get("markdown"))


def test_dispatch_skips_an_already_analyzed_word_entirely(monkeypatch):
    """The explicit "everything is cachable" requirement, in its strongest
    form: re-clicking an already-analyzed word doesn't just hit a fast cache -
    it doesn't call analyze_word_linguistics at all."""
    import idiomapp.streamlit.app as app_module

    async def should_not_be_called(word, language, client):
        raise AssertionError("analyze_word_linguistics should not be called")

    monkeypatch.setattr(app_module, "analyze_word_linguistics", should_not_be_called)

    script = """
import streamlit as st
from idiomapp.utils.graph_viz import WordKey, SemanticWordPayload, NodeSelection
from idiomapp.streamlit.app import _dispatch_semantic_graph_click

st.session_state.setdefault("graph_word_analyses", {})
st.session_state["graph_word_analyses"][WordKey.of("luna", "es")] = {"pos": "NOUN"}
payload = SemanticWordPayload(
    id="luna_es", label="luna", language="es", pos="noun", details="moon",
    node_type="primary", group="es",
)
_dispatch_semantic_graph_click(NodeSelection(payload=payload))
st.write("no crash, no rerun loop")
"""
    at = AppTest.from_string(script)
    at.run(timeout=30)
    assert not at.exception
    assert any("no crash" in e.value for e in at.get("markdown"))


# --------------------------------------------------------------------------
# _filter_edges - the Semantic Graph's strength + relationship-kind filter
# --------------------------------------------------------------------------
_FILTER_EDGES_SCRIPT = """
import streamlit as st
from idiomapp.streamlit.app import _filter_edges

edges = [
    {{"relation": "translation", "strength": 1.0}},
    {{"relation": "cognate", "strength": 0.6}},
    {{"relation": "cross_sentence", "strength": 0.4}},
]
st.session_state["result"] = _filter_edges(edges, {min_strength}, {selected_kinds!r})
"""


def _run_filter_edges(min_strength: float, selected_kinds) -> list:
    at = AppTest.from_string(
        _FILTER_EDGES_SCRIPT.format(
            min_strength=min_strength, selected_kinds=selected_kinds
        )
    )
    at.run(timeout=30)
    assert not at.exception
    return at.session_state["result"]


def test_filter_edges_keeps_everything_by_default():
    result = _run_filter_edges(0.0, ["translation", "cognate", "cross_sentence"])
    assert len(result) == 3


def test_filter_edges_strength_threshold_excludes_weaker_edges():
    result = _run_filter_edges(0.5, ["translation", "cognate", "cross_sentence"])
    assert {e["relation"] for e in result} == {"translation", "cognate"}


def test_filter_edges_kind_selection_excludes_deselected_kinds():
    result = _run_filter_edges(0.0, ["translation"])
    assert len(result) == 1
    assert result[0]["relation"] == "translation"


def test_filter_edges_strength_and_kind_combine():
    result = _run_filter_edges(0.5, ["cognate"])
    assert len(result) == 1
    assert result[0]["relation"] == "cognate"
