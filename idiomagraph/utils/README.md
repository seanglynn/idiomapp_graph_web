# Idiomagraph utility modules

This directory holds the modules behind the Streamlit app in
`idiomagraph/streamlit/app.py`: LLM clients, NLP, graph building and
rendering, persistence, and session state.

## LLM and output schemas

### `llm_utils.py`

Async clients for each LLM provider, all behind one interface:

- `LLMClient`: abstract base class (`generate_text`, `generate_json`)
- `OllamaClient`, `OpenAIClient`, `AnthropicClient`: one client per provider
- `get_openai_available_models` / `get_anthropic_available_models`: populate the sidebar model dropdowns

Each client builds its SDK client lazily, once per event loop.

### `ollama_utils.py`

Ollama server helpers:

- `is_ollama_running` / `get_valid_ollama_host`: find a reachable Ollama host
- `pull_model_if_needed`: download the configured model if it isn't installed
- `get_available_models` / `get_model_list`: list installed models

### `schemas.py`

Pydantic models that every provider's JSON output is validated against:

- `Translation` and `AlignmentPair`: a translation plus its word-level alignment
- `WordAnalysis`: a structured word analysis, made of `Meaning`, `Usage`, `Grammar`, `Pronunciation` and `LearnerNotes`
- `Entry`: the shared `{term, gloss}` item used inside analysis lists
- `prompt_example`: render a model as an example JSON block for a prompt

### `json_utils.py`

- `extract_json`: tolerantly pull a JSON object out of a raw LLM reply

## NLP

### `nlp_utils.py`

spaCy and [textacy](https://textacy.readthedocs.io/) processing, plus graph construction from sentences:

- `load_spacy_model` / `ensure_models_available`: load spaCy models, downloading them if missing
- `detect_language`, `split_into_sentences`, `analyze_parts_of_speech`
- `calculate_similarity` / `detect_cognate`: string-similarity and cognate heuristics
- `process_sentence_pair`, `add_cross_language_relationships`, `merge_language_graphs`: build the semantic graph data
- `build_word_cooccurrence_network` / `get_network_stats`: the co-occurrence network and its stats
- `analyze_word_linguistics`: run the LLM word analysis for one word

## Graph

### `graph_viz.py`

Typed graph building, click-to-explore expansion, and the ECharts adapter. See
[`docs/graph_data_model.md`](../../docs/graph_data_model.md) for the full model.

- **Identity:** `WordKey`, `Position`
- **Node payloads** (a Pydantic discriminated union on `kind`): `SemanticWordPayload`,
  `CategoryPayload`, `RecursiveLeafPayload`, `LeafNodePayload`, `CooccurrenceWordPayload`
- **Edge payload:** `EdgePayload`
- **Rendering:** `StyledNode` and `StyledEdge`, which pair visual styling with a typed payload
- **Click results:** `NodeSelection`, `EdgeSelection`, `SourcedSelection`
- `build_semantic_graph` / `build_cooccurrence_graph`: build a `networkx.MultiDiGraph` for each view
- `build_word_expansion`: category hubs and leaves for one analyzed word. Synonym and antonym
  leaves (`RECURSIVE_LEAF_CATEGORIES`) can be analyzed in turn.
- `compose_semantic_graph_with_expansions`: layer every expansion onto the base graph
- `apply_pinned_positions`: keep already-seen nodes in place between reruns
- `graph_to_echarts_data` / `build_graph_echarts_options`: turn the graph into ECharts options
- `resolve_graph_click`: turn a raw browser click into a typed `NodeSelection` or `EdgeSelection`

## Persistence

### `graph_storage.py`

Saved graph history:

- `GraphStorage`: abstract base class
- `StreamlitGraphStorage`: JSON files on disk, under `graph_storage/`
- `InMemoryGraphStorage`: for tests
- `get_graph_storage`, `get_in_memory_storage`, `get_storage_by_type`

### `analysis_cache.py`

Word-analysis cache, so analyzing the same word again is instant:

- `WordAnalysisCache`: abstract base class
- `FileWordAnalysisCache`: one JSON file on disk, under `word_analysis_cache/`
- `InMemoryWordAnalysisCache`: for tests
- `get_word_analysis_cache`, `get_in_memory_word_analysis_cache`

## App plumbing

### `state_utils.py`

Streamlit session state and client caching:

- `StateManager`, `AppState`, `get_app_state`, `transition_state`: the app lifecycle state
- `initialize_session_state`, `get_session_state_safe`, `set_session_state_safe`, `reset_application_state`
- `get_cached_value`, `clear_cache`, `clear_all_caches`: time-based caching in session state
- `get_llm_client` / `get_provider_credentials`: build the LLM client for the selected provider

### `async_utils.py`

Bridges Streamlit's synchronous reruns and the async LLM clients:

- `run_async`: run a coroutine on this session's long-lived event loop
- `get_event_loop`: get or create that loop
- `loop_key`: identify the running loop, for keying loop-bound resources

### `audio_utils.py`

Text-to-speech through gTTS:

- `generate_audio` / `text_to_speech`: produce audio for text in a given language
- `process_translation_audio`: extract a translation's text and generate its audio
- `extract_translation_text` / `clean_text_for_tts`: prepare text for speech

### `logging_utils.py`

- `get_logger`: get a cached logger for a module (recommended)
- `setup_logging`: legacy wrapper around `get_logger`
- `get_recent_logs` / `clear_logs`: the in-app debug log buffer

## Usage

```python
from idiomagraph.utils.logging_utils import get_logger
from idiomagraph.utils.nlp_utils import split_into_sentences
from idiomagraph.utils.graph_viz import build_semantic_graph, resolve_graph_click
from idiomagraph.utils.state_utils import get_llm_client

logger = get_logger(__name__)
```

Dependencies are declared in [`pyproject.toml`](../../pyproject.toml).
