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


def _owning_group(field: str) -> str:
    """Which of the five WordAnalysis groups declares *field*."""
    for group_name, group_model in WORD_ANALYSIS_GROUPS.items():
        if field in group_model.model_fields:
            return group_name
    raise AssertionError(f"{field!r} is not declared on any WordAnalysis group")


def _nested(field: str, raw):
    """Build the group-nested input `model_validate` actually expects for *field*."""
    return {_owning_group(field): {field: raw}}


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
    model = WordAnalysis.model_validate(_nested(field, raw))
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
    model = WordAnalysis.model_validate({"meaning": {"synonyms": raw}})
    assert model.meaning.synonyms == expected


def test_sparse_response_validates():
    """A model that answers with almost nothing must not fail the analysis."""
    model = WordAnalysis.model_validate({})
    assert model.meaning.definition is None
    assert model.to_display_dict() == {}


def test_unmodelled_field_is_ignored_not_an_error():
    """
    A provider volunteering a field we do not model must not fail the analysis -
    Meaning's own extra="ignore" is what protects this, once the field is inside
    the right group.
    """
    model = WordAnalysis.model_validate(
        {"meaning": {"definition": "x", "invented_field": "y"}}
    )
    assert model.meaning.definition == "x"
    assert "invented_field" not in model.to_display_dict()


def test_register_alias_round_trips():
    """`register` shadows a BaseModel attribute, so it is stored as usage_register."""
    model = WordAnalysis.model_validate({"usage": {"register": "informal"}})
    assert model.usage.usage_register == "informal"
    assert model.to_display_dict()["register"] == "informal"


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
        {"meaning": {"definition": "x", "synonyms": []}}
    ).to_display_dict()
    assert "synonyms" not in display
    assert display["definition"] == "x"


def test_flat_display_dict_no_longer_round_trips_through_model_validate():
    """
    `to_display_dict()`'s output is flat; re-validating it used to round-trip via
    a `model_validator` that folded flat keys back into their group. That
    validator was dead code - no production call site ever passed flat input to
    `model_validate` (`_get_llm_word_analysis`'s merge always nests each group's
    response under its own group key first) - and has been removed. This pins the
    resulting, narrower contract explicitly rather than silently losing coverage:
    a flat dict at the root is accepted (`extra="ignore"`) but populates nothing.
    """
    first = WordAnalysis.model_validate(
        {"usage": {"idioms": {"a": "b"}, "register": "informal"}}
    ).to_display_dict()
    second = WordAnalysis.model_validate(first).to_display_dict()
    assert first  # the original had real content
    assert second == {}  # flat input is now a no-op


def test_nested_mapping_gloss_flattens_readably_not_as_python_repr():
    """Pins the _stringify fix: a doubly-nested value must render as text, not repr."""
    model = WordAnalysis.model_validate(
        {
            "usage": {
                "idioms": {"dar la lata": {"lit": "give the can", "fig": "to annoy"}}
            }
        }
    )
    [entry] = model.usage.idioms
    assert entry.term == "dar la lata"
    assert entry.gloss == "lit: give the can, fig: to annoy"


def test_nested_list_value_in_str_list_flattens_readably():
    model = WordAnalysis.model_validate(
        {"meaning": {"synonyms": {"animal": ["gato", "felino"]}}}
    )
    assert model.meaning.synonyms == ["animal: gato, felino"]


def test_list_of_multi_key_dict_expands_each_pair_into_its_own_entry():
    """A dict without a "term" key expands into one Entry per pair, not one entry."""
    model = WordAnalysis.model_validate(
        {"meaning": {"cognates": [{"chat": "cat", "gato": "cat (es)"}]}}
    )
    got = [(e.term, e.gloss) for e in model.meaning.cognates]
    assert got == [("chat", "cat"), ("gato", "cat (es)")]


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
