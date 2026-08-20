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
"""

from typing import Annotated, Any, Optional

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
                    # A single-pair dict inside a list, e.g. [{"expr": "meaning"}]
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
    """Render a leaf value for display, flattening nested containers readably."""
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


# Fields that arrive as either a mapping or a list of terms.
Entries = Annotated[list[Entry], BeforeValidator(_to_entries)]
# Fields that are conceptually a flat list of strings.
StrList = Annotated[list[str], BeforeValidator(_to_str_list)]


class Pronunciation(BaseModel):
    """Nested pronunciation block. Also mirrored at the top level of WordAnalysis."""

    model_config = ConfigDict(extra="ignore")

    ipa: Optional[str] = None
    syllables: Optional[str] = None
    stress: Optional[str] = None
    pronunciation_notes: Optional[str] = None


class Grammar(BaseModel):
    """Nested grammar block. Flattened onto the top level before display."""

    model_config = ConfigDict(extra="ignore")

    infinitive: Optional[str] = None
    verb_type: Optional[str] = None
    gender: Optional[str] = None
    plural: Optional[str] = None
    conjugations: Entries = Field(default_factory=list)


class WordAnalysis(BaseModel):
    """
    The LLM half of a word analysis.

    Deliberately does NOT include the spaCy-derived keys (``pos``, ``lemma``,
    ``is_alpha``, ``vector_norm``, ...). Those are set by
    ``nlp_utils.analyze_word_linguistics`` before the LLM is called, and the validated
    fields here are merged on top of them.
    """

    # populate_by_name lets `usage_register` be filled from the JSON key "register";
    # see the field's comment below.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # --- meaning and origin ---
    definition: Optional[str] = None
    etymology: Optional[str] = None
    language_origin: Optional[str] = None
    root: Optional[str] = None
    historical_evolution: Optional[str] = None
    hypernym: Optional[str] = None

    cognates: Entries = Field(default_factory=list)
    synonyms: StrList = Field(default_factory=list)
    antonyms: StrList = Field(default_factory=list)
    hyponyms: StrList = Field(default_factory=list)
    semantic_field: StrList = Field(default_factory=list)
    related_words: StrList = Field(default_factory=list)

    # --- usage ---
    examples: StrList = Field(default_factory=list)
    collocations: StrList = Field(default_factory=list)
    idioms: Entries = Field(default_factory=list)
    proverbs: StrList = Field(default_factory=list)
    # Named `usage_register` because a bare `register` shadows a BaseModel attribute
    # and makes Pydantic warn. The alias keeps the wire/display key as "register".
    usage_register: Optional[str] = Field(
        default=None, alias="register", serialization_alias="register"
    )
    frequency: Optional[str] = None
    regional_variations: Optional[str] = None
    slang_usage: Optional[str] = None

    # --- grammar (nested, plus the flattened equivalents the UI reads) ---
    grammar: Optional[Grammar] = None
    infinitive: Optional[str] = None
    verb_type: Optional[str] = None
    gender: Optional[str] = None
    plural: Optional[str] = None
    position: Optional[str] = None
    grammar_notes: Optional[str] = None
    conjugations: Entries = Field(default_factory=list)
    articles: Entries = Field(default_factory=list)
    gender_forms: Entries = Field(default_factory=list)
    comparison: Entries = Field(default_factory=list)
    related_forms: Entries = Field(default_factory=list)

    # --- pronunciation (nested, plus the flattened equivalents) ---
    pronunciation: Optional[Pronunciation] = None
    ipa: Optional[str] = None
    syllables: Optional[str] = None
    stress: Optional[str] = None
    pronunciation_notes: Optional[str] = None

    # --- learner notes ---
    tips: StrList = Field(default_factory=list)
    common_mistakes: StrList = Field(default_factory=list)
    false_friends: Entries = Field(default_factory=list)
    cultural_notes: Optional[str] = None

    def to_display_dict(self) -> dict:
        """
        Flatten into the dict the Streamlit display layer consumes.

        Empty values are dropped so the existing "is this key present?" tab gating
        keeps working, and the nested grammar/pronunciation blocks are merged onto
        the top level (top-level values win, matching the previous behaviour).
        """
        data = self.model_dump(exclude_none=True, by_alias=True)

        for block in ("grammar", "pronunciation"):
            nested = data.pop(block, None) or {}
            for key, value in nested.items():
                if key not in data or not data[key]:
                    data[key] = value

        return {k: v for k, v in data.items() if v not in (None, [], {}, "")}


class Translation(BaseModel):
    """Schema for a single translation response."""

    model_config = ConfigDict(extra="ignore")

    translation: str
