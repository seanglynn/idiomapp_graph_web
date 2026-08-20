"""
Tests for the word-analysis schemas.

The point of `WordAnalysis` is that the three providers disagree about shape - Claude
is schema-constrained, Ollama and OpenAI are merely asked for JSON - and the model
normalises all of it before the UI sees it. These tests pin that normalisation, since
a regression here shows up as a stringified dict in the UI rather than an exception.
"""

import pytest

from idiomapp.utils.schemas import (
    Entry,
    Grammar,
    Pronunciation,
    Translation,
    WordAnalysis,
)

# Every shape a provider has been seen to emit for a "list or mapping" field.
SHAPES = {
    "mapping": ({"dar la lata": "to annoy"}, [("dar la lata", "to annoy")]),
    "list_of_str": (
        ["dar la lata", "ni idea"],
        [("dar la lata", None), ("ni idea", None)],
    ),
    "list_of_dict": ([{"dar la lata": "to annoy"}], [("dar la lata", "to annoy")]),
    "entry_dicts": (
        [{"term": "dar la lata", "gloss": "to annoy"}],
        [("dar la lata", "to annoy")],
    ),
    "bare_string": ("dar la lata", [("dar la lata", None)]),
    "null": (None, []),
}

# Fields declared as Entries on WordAnalysis.
ENTRY_FIELDS = [
    "idioms",
    "false_friends",
    "cognates",
    "conjugations",
    "articles",
    "gender_forms",
    "comparison",
    "related_forms",
]


@pytest.mark.parametrize("field", ENTRY_FIELDS)
@pytest.mark.parametrize("shape", list(SHAPES))
def test_entry_fields_normalise_every_shape(field, shape):
    """Whatever a provider sends, the field ends up as a list of Entry."""
    raw, expected = SHAPES[shape]
    model = WordAnalysis.model_validate({field: raw})
    got = [(e.term, e.gloss) for e in getattr(model, field)]
    assert got == expected


def test_missing_entry_field_is_empty_list():
    assert WordAnalysis.model_validate({}).idioms == []


@pytest.mark.parametrize(
    "raw, expected",
    [
        (["a", "b"], ["a", "b"]),
        ("a", ["a"]),  # bare string, not chars
        ({"x": "y"}, ["x: y"]),  # mapping flattened readably
        (None, []),
        ([1, 2], ["1", "2"]),  # non-strings coerced
    ],
)
def test_str_list_fields_normalise(raw, expected):
    assert WordAnalysis.model_validate({"synonyms": raw}).synonyms == expected


def test_sparse_response_validates():
    """A model that answers with almost nothing must not fail the analysis."""
    model = WordAnalysis.model_validate({})
    assert model.definition is None
    assert model.to_display_dict() == {}


def test_unmodelled_field_is_ignored_not_an_error():
    model = WordAnalysis.model_validate({"definition": "x", "invented_field": "y"})
    assert model.definition == "x"
    assert "invented_field" not in model.to_display_dict()


def test_register_alias_round_trips():
    """`register` shadows a BaseModel attribute, so it is stored as usage_register."""
    model = WordAnalysis.model_validate({"register": "informal"})
    assert model.usage_register == "informal"
    assert model.to_display_dict()["register"] == "informal"


def test_nested_blocks_flatten_onto_top_level():
    model = WordAnalysis.model_validate(
        {
            "grammar": {"gender": "masculine", "conjugations": {"present": "es"}},
            "pronunciation": {"ipa": "ˈɡa.to"},
        }
    )
    display = model.to_display_dict()
    assert display["gender"] == "masculine"
    assert display["ipa"] == "ˈɡa.to"
    assert display["conjugations"] == [{"term": "present", "gloss": "es"}]
    # The nested containers themselves are gone once flattened.
    assert "grammar" not in display and "pronunciation" not in display


def test_top_level_wins_over_nested():
    model = WordAnalysis.model_validate(
        {
            "gender": "feminine",
            "grammar": {"gender": "masculine"},
        }
    )
    assert model.to_display_dict()["gender"] == "feminine"


def test_display_dict_drops_empties():
    """Tab gating keys off presence, so empty values must not appear."""
    display = WordAnalysis.model_validate(
        {"definition": "x", "synonyms": []}
    ).to_display_dict()
    assert "synonyms" not in display
    assert display["definition"] == "x"


def test_display_dict_is_idempotent():
    """Re-validating a display dict yields the same thing - Claude's output goes round twice."""
    first = WordAnalysis.model_validate(
        {"idioms": {"a": "b"}, "register": "informal", "synonyms": ["s"]}
    ).to_display_dict()
    second = WordAnalysis.model_validate(first).to_display_dict()
    assert first == second


def test_entry_rejects_unknown_keys():
    with pytest.raises(Exception):
        Entry.model_validate({"term": "a", "nope": 1})


def test_translation_schema():
    assert Translation.model_validate({"translation": "hola"}).translation == "hola"


def test_grammar_and_pronunciation_tolerate_partial_data():
    assert Grammar.model_validate({"gender": "masculine"}).plural is None
    assert Pronunciation.model_validate({"ipa": "x"}).stress is None
