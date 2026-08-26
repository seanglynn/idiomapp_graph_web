"""
Tests for the word-analysis display layer in idiomapp/streamlit/app.py.

`streamlit.testing.v1.AppTest`-based tests for `display_word_analysis` and its
tab renderers - Streamlit's official headless test harness. Importing
`display_word_analysis` from `idiomapp.streamlit.app` does not run `main()`:
Streamlit only executes a script's `if __name__ == "__main__"` block when that
script is the actual run target, which a plain import does not set - so this
never touches Ollama, spaCy, or any network/session-dependent code, confirmed
empirically to run in ~3s.

(`_format_entry`/`_format_entries` moved to `idiomapp.utils.graph_viz` and are
covered by `tests/test_graph_viz.py` now.)
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


def test_knowledge_graph_renders_an_echarts_component_containing_the_word():
    """The knowledge graph renders via st_echarts, a components.v2 widget - AppTest
    surfaces it as a "bidi_component" node whose proto.json is the full ECharts
    `options` payload (confirmed empirically), not an iframe/srcdoc like the old
    pyvis embed. component_name pins that it's genuinely the echarts component,
    not some other bidi component elsewhere on the page."""
    at = _run(RICH_ANALYSIS)
    assert not at.exception
    echarts_nodes = [
        n
        for n in at.main
        if n.type == "bidi_component"
        and n.proto.component_name == "streamlit-echarts.streamlit_echarts"
    ]
    assert echarts_nodes
    assert "hablar" in echarts_nodes[0].proto.json


def test_knowledge_graph_includes_conjugations_and_related_forms():
    """Pins the behavior fix: these fields were silently invisible in the graph
    before this cleanup (dead isinstance(..., dict) checks against data that is
    always a list post-schema); now they should actually appear."""
    at = _run(RICH_ANALYSIS)
    assert not at.exception
    echarts_nodes = [n for n in at.main if n.type == "bidi_component"]
    payload = echarts_nodes[0].proto.json
    assert "Conjugations" in payload
    assert "Forms" in payload


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
