"""
Tests for the word-analysis schemas.

The point of `WordAnalysis` is that the three providers disagree about shape - Claude
is schema-constrained (per group - see the `WordAnalysis` docstring for why), Ollama
and OpenAI are merely asked for JSON - and the model normalises all of it before the
UI sees it. These tests pin that normalisation, since a regression here shows up as a
stringified dict in the UI rather than an exception.
"""

import json

import pytest

from idiomapp.utils.schemas import (
    Entry,
    Grammar,
    LearnerNotes,
    Meaning,
    Pronunciation,
    Translation,
    Usage,
    WORD_ANALYSIS_GROUPS,
    WordAnalysis,
    prompt_example,
)


def _get_field(model: WordAnalysis, field: str):
    """Find a WordAnalysis leaf field wherever its group put it."""
    for group_name, group_model in WORD_ANALYSIS_GROUPS.items():
        if field in group_model.model_fields:
            return getattr(getattr(model, group_name), field)
    raise AssertionError(f"{field!r} is not declared on any WordAnalysis group")


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

# Fields declared as Entries somewhere on WordAnalysis (across all five groups).
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
    got = [(e.term, e.gloss) for e in _get_field(model, field)]
    assert got == expected


def test_missing_entry_field_is_empty_list():
    assert WordAnalysis.model_validate({}).usage.idioms == []


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
    assert WordAnalysis.model_validate({"synonyms": raw}).meaning.synonyms == expected


def test_sparse_response_validates():
    """A model that answers with almost nothing must not fail the analysis."""
    model = WordAnalysis.model_validate({})
    assert model.meaning.definition is None
    assert model.to_display_dict() == {}


def test_unmodelled_field_is_ignored_not_an_error():
    model = WordAnalysis.model_validate({"definition": "x", "invented_field": "y"})
    assert model.meaning.definition == "x"
    assert "invented_field" not in model.to_display_dict()


def test_register_alias_round_trips():
    """`register` shadows a BaseModel attribute, so it is stored as usage_register."""
    model = WordAnalysis.model_validate({"register": "informal"})
    assert model.usage.usage_register == "informal"
    assert model.to_display_dict()["register"] == "informal"


def test_flat_fields_regroup_by_owning_group():
    """
    A flat-answering provider's fields land in the right group, not just at any
    group (extra="ignore" would otherwise silently drop them).
    """
    model = WordAnalysis.model_validate(
        {
            "definition": "a small carnivore",  # -> meaning
            "gender": "masculine",  # -> grammar
            "ipa": "ˈɡa.to",  # -> pronunciation
            "tips": ["remember the gender"],  # -> learner_notes
        }
    )
    assert model.meaning.definition == "a small carnivore"
    assert model.grammar.gender == "masculine"
    assert model.pronunciation.ipa == "ˈɡa.to"
    assert model.learner_notes.tips == ["remember the gender"]


def test_already_grouped_input_is_left_alone():
    """A provider that nests correctly (Claude) should not be second-guessed."""
    model = WordAnalysis.model_validate(
        {
            "grammar": {"gender": "masculine", "conjugations": {"present": "es"}},
            "pronunciation": {"ipa": "ˈɡa.to"},
        }
    )
    assert model.grammar.gender == "masculine"
    assert model.grammar.conjugations == [Entry(term="present", gloss="es")]
    assert model.pronunciation.ipa == "ˈɡa.to"


def test_display_dict_flattens_all_five_groups():
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
    # The group containers themselves never appear, only their leaf fields.
    for group_name in WORD_ANALYSIS_GROUPS:
        assert group_name not in display


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


# --------------------------------------------------------------------------
# Group registry and structured-output sizing
# --------------------------------------------------------------------------
def test_word_analysis_has_exactly_five_root_fields():
    """
    Regression guard for the bug this grouping fixes: a single `messages.parse`
    call against the old 41-field flat model was rejected outright by Claude
    (compiled grammar too large). WordAnalysis itself is never sent to Claude in
    one piece any more (see `_get_llm_word_analysis`), but if a root field ever
    gets added directly to WordAnalysis instead of into one of the five groups,
    this is the tripwire.
    """
    assert set(WordAnalysis.model_fields) == set(WORD_ANALYSIS_GROUPS)


def test_group_field_names_are_disjoint():
    """`to_display_dict` unions the five groups with no precedence rule - that is
    only safe if no field name is declared in two groups."""
    seen: dict[str, str] = {}
    for group_name, model in WORD_ANALYSIS_GROUPS.items():
        for field_name in model.model_fields:
            assert field_name not in seen, (
                f"{field_name!r} declared in both {seen.get(field_name)!r} and "
                f"{group_name!r}"
            )
            seen[field_name] = group_name


@pytest.mark.parametrize(
    "group_model", [Meaning, Usage, Grammar, Pronunciation, LearnerNotes]
)
def test_prompt_example_is_valid_json_covering_every_field(group_model):
    example = json.loads(prompt_example(group_model))
    expected_keys = {
        info.alias or name for name, info in group_model.model_fields.items()
    }
    assert set(example) == expected_keys


def test_prompt_example_marks_list_fields_as_lists():
    example = json.loads(prompt_example(Meaning))
    assert isinstance(example["synonyms"], list)
    assert isinstance(example["definition"], str)
