"""
Typed schemas for LLM structured output.

These models are **tolerant on input and canonical on output**. That split matters
because the three providers behave differently:

* Claude is given the model as a schema (``messages.parse``), so it returns the
  canonical shape directly.
* Ollama and OpenAI are only *asked* for JSON, so they return whatever they like -
  most commonly a field as either ``{"expr": "meaning"}`` or ``["expr", ...]``.

Rather than pushing that variation into the display layer (which previously carried
an ``isinstance`` branch per field), every provider's output is validated through the
same model and the ``BeforeValidator`` hooks normalise the shapes on the way in. The
UI then only ever sees one shape.

``extra="ignore"`` is deliberate: a model that volunteers a field we do not know about
should not fail the whole analysis.

Field ``description``\\ s are not decoration - they are the single source of truth
for what each field means. They travel two ways from here: into the JSON schema
Claude receives for structured output, and into ``prompt_example()`` below, which
renders the same schema as illustrative JSON for providers that only take a prompt.
Nothing about a field's meaning is hand-duplicated as separate prompt text.
"""

import json
from typing import Annotated, Any, Optional, get_origin

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


class Entry(BaseModel):
    """
    One term, optionally glossed.

    Covers every "either a list or a mapping" field. ``{"dar la lata": "to annoy"}``
    becomes ``Entry(term="dar la lata", gloss="to annoy")``; a bare ``"dar la lata"``
    becomes ``Entry(term="dar la lata", gloss=None)``. Renderers show ``term: gloss``
    when a gloss is present and ``term`` alone when it is not.
    """

    model_config = ConfigDict(extra="forbid")

    term: str
    gloss: Optional[str] = None


def _to_entries(value: Any) -> Any:
    """Normalise a mapping / list / scalar into a list of Entry-shaped dicts."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [{"term": str(k), "gloss": _stringify(v)} for k, v in value.items()]
    if isinstance(value, str):
        return [{"term": value, "gloss": None}]
    if isinstance(value, list):
        entries = []
        for item in value:
            if isinstance(item, dict):
                if "term" in item:
                    entries.append(item)
                else:
                    # A dict without "term" is treated as term:gloss pairs to
                    # expand - not just the single-pair case shown above, but any
                    # dict: two pairs in one list item become two separate Entry
                    # objects, not one malformed one.
                    entries.extend(
                        {"term": str(k), "gloss": _stringify(v)}
                        for k, v in item.items()
                    )
            else:
                entries.append({"term": str(item), "gloss": None})
        return entries
    return [{"term": str(value), "gloss": None}]


def _to_str_list(value: Any) -> Any:
    """Normalise a mapping / list / scalar into a flat list of strings."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [f"{k}: {_stringify(v)}" for k, v in value.items()]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [_stringify(item) for item in value]
    return [str(value)]


def _stringify(value: Any) -> str:
    """Render a leaf value for display, flattening nested containers readably at any depth."""
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_stringify(v)}" for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(_stringify(item) for item in value)
    return str(value)


# Fields that arrive as either a mapping or a list of terms.
Entries = Annotated[list[Entry], BeforeValidator(_to_entries)]
# Fields that are conceptually a flat list of strings.
StrList = Annotated[list[str], BeforeValidator(_to_str_list)]


def prompt_example(model_cls: type[BaseModel]) -> str:
    """
    Render a schema's fields as illustrative JSON for a prompt-based provider.

    Ollama and OpenAI are not schema-constrained - they need an example in the
    prompt text to know what shape and fields are expected. Rather than hand-write
    that example (and have it silently drift from the real model the way the old
    flat ``WordAnalysis`` prompt did), this reads it straight off the model: each
    field's own ``description`` becomes its placeholder value, and list-typed
    fields render as a one-item list so the expected shape is obvious.

    Claude does not need this - it receives ``model_cls`` directly as
    ``output_format`` and is constrained to match it exactly.
    """
    example: dict[str, Any] = {}
    for name, info in model_cls.model_fields.items():
        key = info.alias or name
        hint = info.description or name
        example[key] = [hint] if get_origin(info.annotation) is list else hint
    return json.dumps(example, indent=2)


class Meaning(BaseModel):
    """Meaning, origin, and the word's place in the wider vocabulary."""

    model_config = ConfigDict(extra="ignore")

    definition: Optional[str] = Field(
        default=None, description="Clear, concise definition of the word"
    )
    etymology: Optional[str] = Field(
        default=None, description="Origin and history of the word"
    )
    language_origin: Optional[str] = Field(
        default=None,
        description="Source language the word derives from (e.g. Latin, Greek, Arabic)",
    )
    root: Optional[str] = Field(
        default=None, description="Root morpheme the word is built from"
    )
    historical_evolution: Optional[str] = Field(
        default=None, description="How the word's form or meaning has changed over time"
    )
    hypernym: Optional[str] = Field(
        default=None, description="A broader category term this word is a kind of"
    )
    cognates: Entries = Field(
        default_factory=list,
        description="Related words in other languages (English, French, Italian, Portuguese)",
    )
    synonyms: StrList = Field(
        default_factory=list, description="Words with a similar meaning"
    )
    antonyms: StrList = Field(
        default_factory=list, description="Words with the opposite meaning"
    )
    hyponyms: StrList = Field(
        default_factory=list,
        description="More specific words that fall under this word",
    )
    semantic_field: StrList = Field(
        default_factory=list,
        description="Other words in the same conceptual/topical area",
    )
    related_words: StrList = Field(
        default_factory=list,
        description="Words that share a root or are otherwise closely related",
    )


class Usage(BaseModel):
    """How the word is actually used: examples, register, collocations."""

    # populate_by_name lets `usage_register` be filled from the field name too;
    # see the field's comment below for why it isn't just called `register`.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    examples: StrList = Field(
        default_factory=list, description="Example sentences using the word naturally"
    )
    collocations: StrList = Field(
        default_factory=list,
        description="Common word combinations this word appears in",
    )
    idioms: Entries = Field(
        default_factory=list, description="Idiomatic expressions that use this word"
    )
    proverbs: StrList = Field(
        default_factory=list, description="Proverbs or sayings that use this word"
    )
    # Named `usage_register` because a bare `register` shadows a BaseModel attribute
    # and makes Pydantic warn. The alias keeps the wire/display key as "register".
    usage_register: Optional[str] = Field(
        default=None,
        alias="register",
        serialization_alias="register",
        description="Register the word is used in: formal, informal, or colloquial",
    )
    frequency: Optional[str] = Field(
        default=None, description="How common the word is: common, uncommon, or rare"
    )
    regional_variations: Optional[str] = Field(
        default=None,
        description="Differences in usage or meaning between regions/dialects",
    )
    slang_usage: Optional[str] = Field(
        default=None, description="Slang or colloquial usage of the word, if any"
    )


class Grammar(BaseModel):
    """Grammatical properties: inflection, agreement, word-class specifics."""

    model_config = ConfigDict(extra="ignore")

    infinitive: Optional[str] = Field(
        default=None, description="Base/infinitive form, for verbs"
    )
    verb_type: Optional[str] = Field(
        default=None, description="Whether the verb is regular or irregular"
    )
    gender: Optional[str] = Field(
        default=None, description="Grammatical gender, for nouns: masculine or feminine"
    )
    plural: Optional[str] = Field(default=None, description="Plural form, for nouns")
    position: Optional[str] = Field(
        default=None,
        description="Typical position in a sentence, e.g. for adjectives: before/after the noun",
    )
    grammar_notes: Optional[str] = Field(
        default=None, description="Any other notable grammatical behavior"
    )
    conjugations: Entries = Field(
        default_factory=list,
        description="Key conjugated forms (present, past, future), for verbs",
    )
    articles: Entries = Field(
        default_factory=list,
        description="Definite/indefinite articles used with this word, for nouns",
    )
    gender_forms: Entries = Field(
        default_factory=list,
        description="Masculine/feminine forms of this word, if it varies by gender",
    )
    comparison: Entries = Field(
        default_factory=list,
        description="Comparative and superlative forms, for adjectives/adverbs",
    )
    related_forms: Entries = Field(
        default_factory=list,
        description="Other grammatically related word forms (e.g. adjective from a noun)",
    )


class Pronunciation(BaseModel):
    """How the word sounds."""

    model_config = ConfigDict(extra="ignore")

    ipa: Optional[str] = Field(
        default=None, description="IPA (International Phonetic Alphabet) transcription"
    )
    syllables: Optional[str] = Field(
        default=None, description="Syllable breakdown, e.g. syl-la-bles"
    )
    stress: Optional[str] = Field(
        default=None, description="Which syllable carries the stress"
    )
    pronunciation_notes: Optional[str] = Field(
        default=None, description="Any other notable pronunciation guidance"
    )


class LearnerNotes(BaseModel):
    """Tips and pitfalls aimed at a language learner rather than a dictionary reader."""

    model_config = ConfigDict(extra="ignore")

    tips: StrList = Field(
        default_factory=list,
        description="Practical tips for learners memorizing or using this word",
    )
    common_mistakes: StrList = Field(
        default_factory=list, description="Common mistakes learners make with this word"
    )
    false_friends: Entries = Field(
        default_factory=list,
        description="Similar-looking words in another language with a different meaning",
    )
    cultural_notes: Optional[str] = Field(
        default=None, description="Cultural context relevant to how this word is used"
    )


# The WordAnalysis groups, in the order they are presented - as its five field
# names. Shared by nlp_utils, which issues one call per group concurrently rather
# than sending WordAnalysis itself to a provider in one piece (see its docstring).
WORD_ANALYSIS_GROUPS: dict[str, type[BaseModel]] = {
    "meaning": Meaning,
    "usage": Usage,
    "grammar": Grammar,
    "pronunciation": Pronunciation,
    "learner_notes": LearnerNotes,
}


class WordAnalysis(BaseModel):
    """
    The LLM half of a word analysis, normalised into five cohesive groups.

    This shape exists because of a real constraint, not just for tidiness:
    Anthropic's structured-output endpoint rejects a schema once its *compiled
    grammar* gets too large, and that is roughly proportional to total field count
    and nesting, not just how many fields sit at the root. The previous shape was
    41 fields flat at the top level - partly because ``grammar``/``pronunciation``
    were represented twice, as both a nested block and duplicate flattened fields -
    and a single ``messages.parse`` call using it was rejected outright (verified
    live against the API). Nesting those 41 fields under five groups in one schema
    still gets rejected for the same reason: the total complexity does not shrink
    just by regrouping it.

    What actually works (also verified live): each group's schema *alone* is well
    within the limit. So ``nlp_utils._get_llm_word_analysis`` does not send this
    whole model to Claude in one call - it fires one ``messages.parse`` call per
    group in ``WORD_ANALYSIS_GROUPS``, concurrently, each schema-constrained on its
    own small model, and merges the five results into this one. This model is what
    that merge is validated through, and what every provider's output - Claude's
    structured, Ollama/OpenAI's tolerant-parsed - ultimately normalises to.

    ``model_validate`` therefore expects group-nested input only - the shape
    ``_get_llm_word_analysis``'s merge always produces (``{"meaning": {...},
    "usage": {...}, ...}``). A flat dict such as ``{"definition": "x"}`` at the
    root populates nothing: this model has ``extra="ignore"`` and ``"definition"``
    is not one of its five field names. An earlier version folded flat keys into
    their owning group via a ``model_validator``, but no call site ever passed it
    flat input - it was dead code, and has been removed.

    Deliberately does NOT include the spaCy-derived keys (``pos``, ``lemma``,
    ``is_alpha``, ``vector_norm``, ...). Those are set by
    ``nlp_utils.analyze_word_linguistics`` before the LLM is called, and the validated
    fields here are merged on top of them.
    """

    model_config = ConfigDict(extra="ignore")

    meaning: Meaning = Field(default_factory=Meaning)
    usage: Usage = Field(default_factory=Usage)
    grammar: Grammar = Field(default_factory=Grammar)
    pronunciation: Pronunciation = Field(default_factory=Pronunciation)
    learner_notes: LearnerNotes = Field(default_factory=LearnerNotes)

    def to_display_dict(self) -> dict:
        """
        Flatten into the dict the Streamlit display layer consumes.

        Empty values are dropped so the existing "is this key present?" tab gating
        keeps working. The five groups have disjoint field names by construction,
        so this is a plain union - no precedence rule is needed between them.
        """
        data: dict = {}
        for group_name in WORD_ANALYSIS_GROUPS:
            group = getattr(self, group_name)
            data.update(group.model_dump(exclude_none=True, by_alias=True))

        return {k: v for k, v in data.items() if v not in (None, [], {}, "")}


class Translation(BaseModel):
    """Schema for a single translation response."""

    model_config = ConfigDict(extra="ignore")

    translation: str = Field(description="The translated text")
