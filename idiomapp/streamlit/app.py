import os
import re
import asyncio
import html
from dataclasses import dataclass, field
from typing import Optional

# Third-party imports
import streamlit as st
from streamlit_echarts import st_echarts

# Internal imports
from idiomapp.utils.llm_utils import (
    get_openai_available_models,
    get_anthropic_available_models,
)
from idiomapp.utils.ollama_utils import get_available_models
from idiomapp.utils.async_utils import run_async
from idiomapp.utils.schemas import AlignmentPair, Translation
from idiomapp.utils.state_utils import get_llm_client, get_provider_credentials
from idiomapp.utils.logging_utils import get_logger, get_recent_logs, clear_logs
from idiomapp.utils.graph_storage import get_graph_storage
from idiomapp.utils.graph_viz import (
    GRAPH_CLICK_JS,
    CategoryPayload,
    CooccurrenceWordPayload,
    EdgeSelection,
    LeafNodePayload,
    NodeSelection,
    RecursiveLeafPayload,
    SemanticWordPayload,
    SourcedSelection,
    WordKey,
    apply_pinned_positions,
    build_cooccurrence_graph,
    build_graph_echarts_options,
    build_semantic_graph,
    compose_semantic_graph_with_expansions,
    filter_invalid_nodes,
    format_entries,
    graph_to_echarts_data,
    resolve_graph_click,
)
from idiomapp.config import (
    settings,
    LANGUAGE_MAP,
    LLMProvider,
    resolve_anthropic_model,
)
from idiomapp.utils.nlp_utils import (
    split_into_sentences,
    build_word_cooccurrence_network,
    detect_language,
    analyze_word_linguistics,
    process_sentence_pair,
    add_cross_sentence_relationships,
    add_cross_language_relationships,
    merge_language_graphs,
)
from idiomapp.utils.audio_utils import generate_audio, process_translation_audio

# Set up logging
logger = get_logger("streamlit_app")

# Language configuration imported from central config
# TODO: Add more TTS; Add language detection

# Set up page configuration - use Streamlit's native theming
st.set_page_config(
    page_title="Idiomapp",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Add minimal custom styling that works with dark theme
st.markdown(
    """
<style>
    /* Improve chat message readability for dark theme. A second, conflicting copy
       of these two rules used to be injected on every rerun inside main()'s chat
       sidebar; these are the values that actually rendered once both were merged
       by the browser's CSS cascade (padding kept its !important from this block,
       the rest - border-radius/border-left width/background-color/font-size -
       came from the later, more specific block and are folded in here). */
    .chat-message-user, .chat-message-ai {
        padding: 15px !important;
        margin-bottom: 15px;
        border-radius: 5px;
        font-size: 0.9em;
        white-space: pre-wrap;
    }
    .chat-message-user {
        border-left: 3px solid #4361EE;
        background-color: rgba(67, 97, 238, 0.1);
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .chat-message-ai {
        border-left: 3px solid #4CC9F0;
        background-color: rgba(76, 201, 240, 0.1);
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    audio::-webkit-media-controls-panel {
        background-color: #333333;
    }
    audio::-webkit-media-controls-play-button {
        background-color: #4361EE;
        border-radius: 50%;
    }
    /* Style the TTS button for dark theme */
    .stButton button[data-testid^="tts_"] {
        border-radius: 50%;
        width: 40px;
        height: 40px;
        padding: 6px;
        font-size: 18px;
        margin-top: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.4);
        background-color: #3A90EE;
        color: white;
    }
    /* Highlight the TTS button on hover */
    .stButton button[data-testid^="tts_"]:hover {
        background-color: #4CC9F0;
        transform: scale(1.05);
        transition: all 0.2s;
    }
    /* Custom audio player button style */
    .audio-play-button {
        background-color: #4361EE;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 5px 10px;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.2s;
    }
    .audio-play-button:hover {
        background-color: #4CC9F0;
        transform: scale(1.05);
    }
    /* Chat container styling */
    .stChatContainer {
        height: 500px;
        overflow-y: auto;
        border: 1px solid #4361EE;
        border-radius: 10px;
        padding: 15px;
        background-color: #1E1E1E;
    }
    /* The sidebar starts collapsed so the graph gets most of the screen.
       Reskin its native toggle as a hamburger icon, the same way whether
       it's currently collapsed (stExpandSidebarButton) or open
       (stSidebarCollapseButton) - both already toggle Streamlit's own
       already-responsive sidebar (it overlays as a drawer on narrow/mobile
       viewports); only the glyph changes here. Streamlit renders that glyph
       as a Material Symbols icon-font ligature (literal text like
       "keyboard_double_arrow_right" rendered as a chevron glyph by the
       font), so swapping it means hiding that text and rendering the
       "menu" ligature - the standard hamburger glyph - in its place. */
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {
        font-size: 0;
    }
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"]::after,
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]::after {
        content: "menu";
        font-family: "Material Symbols Rounded";
        font-size: 24px;
        font-weight: 400;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Helper functions for language selection UI
def get_index(item_list: list, target: str, default: int = 0) -> int:
    """
    Safely get the index of an item in a list (language, model, or provider).

    Args:
        item_list: List of items (language codes, model names, or provider names)
        target: Item to find
        default: Default index to return if not found

    Returns:
        Index of target in item_list, or default if not found
    """
    try:
        if target in item_list:
            return item_list.index(target)
    except (ValueError, TypeError, AttributeError):
        pass
    return default


def get_language_name(lang_code: str, fallback: str = None) -> str:
    """
    Safely get the display name for a language code.

    Args:
        lang_code: Language code (e.g., 'en', 'es', 'ca')
        fallback: Fallback value if language not found (defaults to lang_code.upper())

    Returns:
        Language display name, or fallback if not found
    """
    if lang_code in LANGUAGE_MAP:
        return LANGUAGE_MAP[lang_code]["name"]
    return fallback if fallback is not None else lang_code.upper()


def get_language_display(lang_code: str, include_flag: bool = True) -> str:
    """
    Get formatted language display string with name and optionally flag.

    Args:
        lang_code: Language code (e.g., 'en', 'es', 'ca')
        include_flag: Whether to include flag emoji

    Returns:
        Formatted string: "Name 🏳️" or "Name" depending on include_flag
    """
    if lang_code in LANGUAGE_MAP:
        lang_info = LANGUAGE_MAP[lang_code]
        if include_flag:
            return f"{lang_info['name']} {lang_info['flag']}"
        return lang_info["name"]
    return lang_code.upper()


# Betweenness centrality is O(V*E) and was recomputed on every rerun of the
# co-occurrence tab. Keyed on a hashable edge signature because NetworkX graphs are
# neither hashable nor picklable in a way st.cache_data can key on.
@st.cache_data(ttl=600, show_spinner=False)
def _cached_centrality(node_signature: tuple, edge_signature: tuple) -> tuple:
    """Compute (degree, betweenness) centrality for a graph given its nodes and edges."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(
        node_signature
    )  # keeps isolated nodes, which affect degree centrality
    graph.add_edges_from(edge_signature)
    return nx.degree_centrality(graph), nx.betweenness_centrality(graph)


def compute_centrality(graph) -> tuple:
    """Get (degree, betweenness) centrality for a co-occurrence graph, cached."""
    return _cached_centrality(
        tuple(sorted(graph.nodes())),
        tuple(sorted(tuple(sorted(e)) for e in graph.edges())),
    )


# Remote model lists are fetched over the network. Without caching these were
# re-fetched on every Streamlit rerun - i.e. on every keystroke in the sidebar.
@st.cache_data(ttl=3600, show_spinner=False)
def cached_openai_models(api_key: str, organization: str) -> list:
    """Available OpenAI models, cached for an hour."""
    return get_openai_available_models(api_key, organization)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_anthropic_models(api_key: str) -> list:
    """Available Claude models, cached for an hour."""
    return get_anthropic_available_models(api_key)


def reset_llm_client_state() -> None:
    """
    Drop the cached LLM client so the next run rebuilds it with new credentials.

    Shared by every provider's credential handler in the sidebar.
    """
    st.session_state["model_available"] = False
    st.session_state["needs_client_reinit"] = True
    st.session_state["llm_client"] = None
    st.session_state["cached_model_status"] = None
    st.session_state["cached_available_models"] = None


# Providers whose display name is not just a capitalisation of the internal key.
PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "anthropic": "Claude",
}


def format_provider_name(provider: str) -> str:
    """
    Format provider name for display.

    Args:
        provider: Provider name (e.g., 'ollama', 'openai', 'anthropic')

    Returns:
        Human-readable provider name ('anthropic' renders as 'Claude')
    """
    return PROVIDER_DISPLAY_NAMES.get(provider, provider.title())


def format_pos_option(pos_tag: str, pos_options: list) -> str:
    """
    Format part-of-speech tag for display.

    Args:
        pos_tag: POS tag (e.g., 'NOUN', 'VERB')
        pos_options: List of (name, tag) tuples

    Returns:
        Display name for the POS tag, or the tag itself if not found
    """
    for name, tag in pos_options:
        if tag == pos_tag:
            return name
    return pos_tag


def format_documentation_file(index: int, file_names: list) -> str:
    """
    Format documentation file name for display.

    Args:
        index: Index in the file_names list
        file_names: List of file names

    Returns:
        File name at the given index
    """
    if 0 <= index < len(file_names):
        return file_names[index]
    return ""


def get_centrality_sort_key(item: tuple) -> float:
    """
    Get sort key for centrality items (sort by value, descending).

    Args:
        item: Tuple of (word, centrality_score)

    Returns:
        Centrality score (negated for descending sort)
    """
    return -item[1]  # Negate for descending sort


def build_model_label(model_type: str, is_available: bool) -> str:
    """
    Build label for model selectbox with optional warning indicator.

    Args:
        model_type: Type of model (e.g., 'Ollama Model', 'OpenAI Model')
        is_available: Whether the model is available

    Returns:
        Label string with optional warning emoji
    """
    warning = "" if is_available else "⚠️"
    return f"{model_type} {warning}".strip()


def extract_translation_text(message_content: str) -> tuple[str, str]:
    """
    Extract the actual translation text from a message that may contain language labels.

    Args:
        message_content: Full message content (e.g., "English 🇬🇧: Hello world")

    Returns:
        Tuple of (translation_text, language_label) or (message_content, "") if no label found
    """
    # Check if content has language label format: "Language Name 🏳️: translation text"
    for lang_code in LANGUAGE_MAP.keys():
        lang_display = get_language_display(lang_code)
        if f"{lang_display}:" in message_content:
            # Extract text after the language label
            parts = message_content.split(f"{lang_display}:", 1)
            if len(parts) == 2:
                translation_text = parts[1].strip()
                return translation_text, lang_display
    # If no label found, return content as-is
    return message_content, ""


def parse_translation_message(message: dict) -> tuple[str, str, str]:
    """
    Parse a translation message to extract translation text, language code, and display name.

    Args:
        message: Message dictionary with 'content' and optionally 'target_lang'

    Returns:
        Tuple of (translation_text, lang_code, lang_display)
    """
    content = message.get("content", "")
    target_lang = message.get("target_lang")

    # Extract translation text (remove language label if present)
    translation_text, lang_display = extract_translation_text(content)

    # If we don't have target_lang, try to find it from the display name
    if not target_lang and lang_display:
        for lang_code, lang_info in LANGUAGE_MAP.items():
            if lang_display == get_language_display(lang_code):
                target_lang = lang_code
                break

    return translation_text, target_lang or "", lang_display


def render_chat_message(message, role, target_lang=None, source_lang="en"):
    """Render a chat message with TTS capability."""
    # Add debug logging to track message rendering
    logger.debug(f"Rendering message with role: {role}, content: {message[:50]}...")

    # Use match statement to determine message style based on role
    match role:
        case "user":
            css_class = "chat-message-user"
            prefix = "You"

            # Process and render user messages
            formatted_content = process_message_content(message)
            st.markdown(
                f"""<div class='{css_class}'>
                <strong>{prefix}:</strong> {formatted_content}
                </div>""",
                unsafe_allow_html=True,
            )

        case "assistant" | "ai":
            css_class = "chat-message-ai"
            prefix = "LLM"

            # Check if this is a translation message (contains language name + flag)
            is_translation = False
            for lang_code, lang_info in LANGUAGE_MAP.items():
                if f"{lang_info['name']} {lang_info['flag']}:" in message:
                    is_translation = True
                    break

            # For LLM responses, create container with message and TTS button
            formatted_content = process_message_content(message)

            # Display message
            if is_translation:
                st.markdown(
                    f"""<div class='{css_class}'>
                    {formatted_content}
                    </div>""",
                    unsafe_allow_html=True,
                )

                # If we have multiple language translations in one message,
                # create separate audio players for each language segment
                model_available = st.session_state.get("model_available", False)
                if model_available:
                    # Split the message by recognizable language headers
                    translation_segments = {}

                    # Check for each language pattern in the message
                    for lang_code, lang_info in LANGUAGE_MAP.items():
                        pattern = f"{lang_info['name']} {lang_info['flag']}:"
                        if pattern in message:
                            # Find all instances of this language pattern
                            segments = message.split(pattern)

                            logger.info(
                                f"Found {len(segments)} segments for {lang_code} with pattern '{pattern}'"
                            )
                            logger.info(f"Message content: {message[:200]}...")

                            if len(segments) > 1:
                                # The content is after the pattern, might need to clean up
                                # Get segments that follow the pattern
                                for i in range(1, len(segments)):
                                    content = segments[i].strip()
                                    logger.info(
                                        f"Segment {i} content: '{content[:100]}...'"
                                    )

                                    # If this is not the last segment, need to extract up to the next language
                                    if i < len(segments) - 1:
                                        # Find the next language marker
                                        next_lang_marker = None
                                        for (
                                            check_lang,
                                            check_info,
                                        ) in LANGUAGE_MAP.items():
                                            check_pattern = f"\n\n{check_info['name']} {check_info['flag']}:"
                                            if check_pattern in content:
                                                next_lang_marker = content.find(
                                                    check_pattern
                                                )
                                                break

                                        if next_lang_marker is not None:
                                            content = content[:next_lang_marker].strip()
                                            logger.info(
                                                f"Content after next marker removal: '{content[:100]}...'"
                                            )

                                    # Store the translation content for this language
                                    # Use a list to handle multiple segments per language
                                    if lang_code not in translation_segments:
                                        translation_segments[lang_code] = []

                                    translation_segments[lang_code].append(
                                        {
                                            "content": content,
                                            "lang_name": lang_info["name"],
                                            "flag": lang_info["flag"],
                                        }
                                    )

                                    logger.info(
                                        f"Stored segment for {lang_code}: '{content[:100]}...'"
                                    )

                    # Generate audio players for each language segment
                    if translation_segments:
                        for lang_code, segments in translation_segments.items():
                            for segment in segments:
                                try:
                                    # Log attempt to generate audio
                                    logger.info(
                                        f"Generating audio for {lang_code} segment"
                                    )

                                    # Prepare the translation text with the language header
                                    translation_text = f"{segment['lang_name']} {segment['flag']}: {segment['content']}"

                                    # Generate the audio HTML for this segment, caching
                                    # by content+language so an unchanged segment does
                                    # not re-hit the gTTS network call on every rerun.
                                    cache_key = (
                                        f"audio_{hash(translation_text + lang_code)}"
                                    )
                                    if cache_key in st.session_state:
                                        audio_html = st.session_state[cache_key]
                                    else:
                                        audio_html = process_translation_audio(
                                            translation_text, source_lang, lang_code
                                        )
                                        st.session_state[cache_key] = audio_html

                                    # Show the audio player with a clear label
                                    st.markdown(
                                        f"<p style='margin: 5px 0; color: #CCCCCC; font-size: 12px;'>Audio for {segment['lang_name']} {segment['flag']}</p>",
                                        unsafe_allow_html=True,
                                    )
                                    st.markdown(audio_html, unsafe_allow_html=True)
                                    logger.info(
                                        f"Audio player displayed for {lang_code}"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Error generating audio for {lang_code}: {str(e)}"
                                    )
                                    st.error(f"Audio error: {str(e)}")
                else:
                    st.warning("Audio unavailable - LLM model not ready")
            else:
                st.markdown(
                    f"""<div class='{css_class}'>
                    <strong>{prefix}:</strong> {formatted_content}
                    </div>""",
                    unsafe_allow_html=True,
                )

                # Show audio for non-translation messages
                model_available = st.session_state.get("model_available", False)
                if model_available and role == "assistant" and target_lang:
                    try:
                        # Generate audio for this message, caching by content+language
                        # so an unchanged message does not re-hit the gTTS network
                        # call on every rerun.
                        cache_key = f"audio_{hash(message + target_lang)}"
                        if cache_key in st.session_state:
                            audio_html = st.session_state[cache_key]
                        else:
                            audio_html = process_translation_audio(
                                message, source_lang, target_lang
                            )
                            st.session_state[cache_key] = audio_html
                        st.markdown(audio_html, unsafe_allow_html=True)
                    except Exception as e:
                        logger.error(f"Error generating audio: {str(e)}")
                        st.error(f"Audio error: {str(e)}")

        case _:
            # Default handling for unknown roles
            st.markdown(
                f"**Message ({role}):** {html.escape(message)}", unsafe_allow_html=True
            )


def process_message_content(message):
    """Process message content to handle code blocks and HTML escaping"""
    content = []
    lines = message.split("\n")

    # Simple code block detection
    in_code_block = False
    for line in lines:
        # Use match statement to handle different line content types
        match line.strip():
            case code_start if code_start.startswith("```"):
                in_code_block = not in_code_block
                content.append(f"<pre>{line}</pre>" if in_code_block else "</pre>")
            case _ if in_code_block:
                # Don't escape inside code blocks
                content.append(line)
            case _:
                # Escape HTML outside code blocks
                content.append(html.escape(line))

    # Join lines with line breaks
    return "<br>".join(content)


def text_to_speech(text, lang_code=None, message_key=None):
    """
    Convert text to speech and return the audio player HTML
    Uses Google Text-to-Speech (gTTS)
    """
    try:
        # First check if text is too short
        if len(text.strip()) < 2:
            return ""

        # If language code not provided, detect it
        if not lang_code:
            logger.info("Language not specified, detecting language")
            detected_lang = detect_language(text)
            logger.info(f"Detected language: {detected_lang}")
            lang_code = detected_lang

        # Use message_key for caching if provided
        cache_key = f"audio_{message_key}" if message_key else None

        # Check cache first if we have a key
        if cache_key and cache_key in st.session_state:
            logger.info(f"Using cached audio for {cache_key}")
            return st.session_state[cache_key]

        # Generate the audio
        audio_html = generate_audio(text, lang_code)

        # Cache the result if we have a key
        if cache_key:
            logger.info(f"Caching audio with key {cache_key}")
            st.session_state[cache_key] = audio_html

        return audio_html

    except Exception as e:
        logger.error(f"Text-to-speech error: {str(e)}")
        return f"<div style='color: red; padding: 5px;'>TTS Error: {str(e)}</div>"


# Add this function to display model status
def display_model_status(client):
    """
    Display the status of the LLM model and check if it's available.

    Args:
        client: The LLM client instance.

    Returns:
        bool: True if the model is available, False otherwise.
    """
    # Display status once per session
    if "model_status_displayed_once" in st.session_state:
        # Return cached result without displaying again
        return st.session_state.get("last_model_available", False)

    # Mark that we've displayed the status
    st.session_state["model_status_displayed_once"] = True

    # Get status information from the client
    status = client.get_model_status()
    provider = status.get("provider", "unknown")
    model_name = status.get("model_name", "unknown")
    is_available = status.get("available", False)

    # Create a container for the status message
    status_container = st.empty()

    # Display status message based on availability
    if is_available:
        status_container.success(
            f"✅ {provider.title()} model '{model_name}' is available"
        )
        st.session_state["last_model_available"] = True
        return True
    else:
        # Different message based on provider
        if provider == "ollama":
            host = status.get("host", "unknown")
            status_container.error(
                f"⚠️ Ollama model '{model_name}' is not available. " f"Host: {host}"
            )
        elif provider in ("openai", "anthropic"):
            status_container.error(
                f"⚠️ {format_provider_name(provider)} model '{model_name}' is not available. "
                f"API key {'is not set' if not status.get('api_key_set') else 'may be invalid'}"
            )
        else:
            status_container.error(f"⚠️ LLM provider '{provider}' is not available.")

        st.session_state["last_model_available"] = False
        return False


def display_translation_error(error_message: str, target_lang: str):
    """
    Display translation errors in a consistent, user-friendly way.

    Args:
        error_message: The error message to display
        target_lang: Target language code for context
    """
    lang_info = LANGUAGE_MAP.get(target_lang, {})
    lang_name = lang_info.get("name", target_lang)
    flag = lang_info.get("flag", "🌐")

    st.error(f"{lang_name} {flag}: {error_message}")


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """A translated segment of text, plus any word-level alignment the LLM
    reported. `error` is set instead of `text` when translation failed -
    callers check that rather than sniffing `text` for an "Error:" prefix."""

    text: str = ""
    alignment: list[AlignmentPair] = field(default_factory=list)
    error: Optional[str] = None


async def translate_text(
    client, source_text, source_lang, target_lang
) -> TranslationResult:
    """
    Translate text using the LLM client.

    Every provider implements generate_json, so there is one path here rather than
    a per-provider branch. Only Claude's client actually validates its response
    against `schema` internally (see llm_utils.py); Ollama/OpenAI return raw
    parsed JSON, so the result is re-validated through `Translation` here for
    every provider alike - that's also what makes `Translation.alignment`'s
    tolerant parsing (schemas.py's `_to_alignment_pairs`) actually run.

    Args:
        client: The LLM client (Ollama, OpenAI or Anthropic)
        source_text: Text to translate
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        A TranslationResult - `.text`/`.alignment` on success, `.error` on failure.
    """
    logger.info(f"Translating from {source_lang} to {target_lang}: {source_text}")

    source_name = LANGUAGE_MAP[source_lang]["name"]
    target_name = LANGUAGE_MAP[target_lang]["name"]

    system_prompt = (
        f"You are a professional translator. Translate text from {source_name} to "
        f'{target_name}. Return ONLY a JSON object with keys "translation" (the '
        f'translated text) and "alignment" (a list of {{"source_word": ..., '
        f'"target_word": ...}} pairs, one per content word - skip function words '
        f"with no clear one-to-one match). Preserve the original formatting "
        f"(line breaks, punctuation)."
    )

    prompt = f"""Translate this text from {source_name} to {target_name}:

{source_text}

Return JSON: {{"translation": "your translation here", "alignment": [{{"source_word": "...", "target_word": "..."}}]}}"""

    try:
        result = await client.generate_json(
            prompt, system_prompt=system_prompt, schema=Translation
        )

        if "error" in result:
            logger.error(f"Translation error: {result['error']}")
            error = result["error"]
            message = error if str(error).startswith("Error:") else f"Error: {error}"
            return TranslationResult(error=message)

        parsed = Translation.model_validate(result)
        translation = parsed.translation.strip()
        if not translation:
            logger.error("Empty translation in response")
            return TranslationResult(error="Error: Empty translation received")

        logger.info(f"Translation result: {translation[:100]}...")
        return TranslationResult(text=translation, alignment=parsed.alignment)

    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return TranslationResult(error=f"Error: {str(e)}")


async def analyze_translation(source_text, target_texts, target_langs, alignments=None):
    """
    Analyze translation and generate graph with related words.

    Args:
        source_text: Source text to translate
        target_texts: List of translations
        target_langs: List of target languages
        alignments: Optional list of AlignmentPair lists, one per target_lang
            (same order/length as target_texts) - see TranslationResult.
            Covers the whole translated text, not per-sentence; scoped down to
            each sentence pair automatically, since a pair only ever matches
            words that sentence actually contains.

    Returns:
        Dictionary with nodes and edges for the graph
    """
    logger.info(f"Analyzing translation: {source_text}")

    # Initialize graph data
    graph_data = {
        "nodes": [],
        "edges": [],
        "metadata": {
            "source_lang": "en",  # Default source language is English
            "target_langs": target_langs,
            "source_text": source_text,
            "translations": target_texts,
        },
    }

    # Detect source language if not English
    detected_source_lang = detect_language(source_text)
    graph_data["metadata"]["source_lang"] = detected_source_lang

    # Split texts into sentences
    source_sentences = split_into_sentences(source_text)
    target_sentences_by_lang = {}

    for lang, text in zip(target_langs, target_texts):
        target_sentences_by_lang[lang] = split_into_sentences(text)

    # One lowercased (source_word, target_word) lookup set per language, built
    # once - process_sentence_pair's existing per-word-pair loop naturally
    # scopes this to whichever words a given sentence pair actually contains.
    alignment_pairs_by_lang = {}
    for lang, pairs in zip(target_langs, alignments or []):
        alignment_pairs_by_lang[lang] = frozenset(
            (pair.source_word.lower(), pair.target_word.lower()) for pair in pairs
        )

    # Set of nodes already added to avoid duplicates
    added_nodes = set()
    # Cache for word relations to avoid redundant API calls
    word_relations_cache = {}

    # Process each sentence pair
    for sentence_idx, source_sentence in enumerate(source_sentences):
        sentence_group = f"-s{sentence_idx + 1}" if len(source_sentences) > 1 else ""

        # Process each target language for this sentence
        for lang in target_langs:
            # Get corresponding target sentence if available
            if sentence_idx < len(target_sentences_by_lang.get(lang, [])):
                target_sentence = target_sentences_by_lang[lang][sentence_idx]
                process_sentence_pair(
                    source_sentence,
                    target_sentence,
                    detected_source_lang,
                    lang,
                    graph_data,
                    added_nodes,
                    word_relations_cache,
                    sentence_group,
                    alignment_pairs_by_lang.get(lang, frozenset()),
                )

    # Add cross-sentence relationships if multiple sentences
    if len(source_sentences) > 1:
        add_cross_sentence_relationships(graph_data)

    # Add cross-language relationships if multiple target languages
    if len(target_langs) > 1:
        add_cross_language_relationships(graph_data, target_langs)

    return graph_data


def visualize_translation_graph(graph_data):
    """Visualize translation and related words as an interactive ECharts graph,
    with a click-to-select side panel (see _render_graph_selection_panel)."""
    logger.info("Visualizing translation graph")

    if not graph_data or not isinstance(graph_data, dict):
        logger.warning("Invalid or empty graph data provided")
        st.warning("No graph data available to visualize.")
        return

    if graph_data.get("metadata", {}).get("error"):
        logger.warning("Graph data contains error, skipping visualization")
        st.warning("Unable to generate graph due to translation errors.")
        return

    if not graph_data.get("nodes"):
        logger.warning("No nodes in graph data")
        st.info("No words to display in the graph. Try translating more text.")
        return

    filtered = filter_invalid_nodes(graph_data)
    if not filtered["nodes"]:
        logger.warning("All nodes filtered out as error-related")
        st.warning("Unable to generate meaningful graph from the translation results.")
        return

    base = build_semantic_graph(filtered)
    composed = compose_semantic_graph_with_expansions(
        base,
        st.session_state["graph_word_analyses"],
        st.session_state["graph_expanded_categories"],
    )
    pinned = apply_pinned_positions(composed, st.session_state["graph_node_positions"])
    options = build_graph_echarts_options(graph_to_echarts_data(pinned))

    graph_col, panel_col = st.columns([2, 1], gap="medium")
    with graph_col:
        result = st_echarts(
            options=options,
            events={"click": GRAPH_CLICK_JS},
            height="400px",
            key="echarts_semantic_graph",
        )
    if result:
        selection = resolve_graph_click(result.get("chart_event"))
        if selection:
            _remember_click_position(selection)
            st.session_state["graph_selection"] = SourcedSelection(
                selection=selection, source="semantic"
            )
            _dispatch_semantic_graph_click(selection)
    with panel_col:
        _render_graph_selection_panel("semantic")


def _remember_click_position(selection) -> None:
    """Capture a clicked node's on-screen position (if the frontend reported
    one) so `apply_pinned_positions` can hold it still on the next redraw -
    without this, adding a word's category hubs makes the whole graph
    re-layout and every existing node can visibly jump, not just grow."""
    if not isinstance(selection, NodeSelection) or selection.position is None:
        return
    node_id = getattr(selection.payload, "id", None)
    if node_id:
        st.session_state["graph_node_positions"][node_id] = selection.position


def visualize_cooccurrence_network(graph, lang_code=None):
    """Visualize a word co-occurrence network as an interactive ECharts graph,
    with a click-to-select side panel (see _render_graph_selection_panel)."""
    if len(graph.nodes()) == 0:
        logger.warning("Co-occurrence graph is empty - no nodes to display")
        st.info(
            "No co-occurrence data available for this text. Try a longer text "
            "or adjust co-occurrence settings."
        )
        return

    options = build_graph_echarts_options(
        graph_to_echarts_data(build_cooccurrence_graph(graph, lang_code))
    )

    graph_col, panel_col = st.columns([2, 1], gap="medium")
    with graph_col:
        result = st_echarts(
            options=options,
            events={"click": GRAPH_CLICK_JS},
            height="600px",
            key="echarts_cooccurrence_graph",
        )
    if result:
        selection = resolve_graph_click(result.get("chart_event"))
        if selection:
            st.session_state["graph_selection"] = SourcedSelection(
                selection=selection, source="cooccurrence"
            )
    with panel_col:
        _render_graph_selection_panel("cooccurrence")


def _dispatch_semantic_graph_click(selection) -> None:
    """React to a click in the Semantic Graph - this is what makes the graph
    an *exploration* tool instead of just a picture: clicking a word analyzes
    it automatically, clicking one of its category hubs (Synonyms, Etymology,
    ...) shows or hides that category's results, and clicking a result that's
    itself a word (a synonym) analyzes and explores *that* word too, the same
    way, letting exploration continue as far as the user wants to go.

    Edge clicks and clicks on plain, non-clickable leaf nodes (an idiom, an
    example sentence) do nothing here - the side panel already shows their
    details; there's no further action attached to them.
    """
    if isinstance(selection, EdgeSelection):
        return
    payload = selection.payload

    if isinstance(payload, CategoryPayload):
        _toggle_expanded_category(payload.word_key, payload.category)
        st.rerun()
        return

    if isinstance(payload, LeafNodePayload):
        return

    if isinstance(payload, (SemanticWordPayload, RecursiveLeafPayload)):
        if isinstance(payload, SemanticWordPayload):
            word_key = WordKey.of(payload.label, payload.language)
        else:
            word_key = payload.word_key
        _analyze_word_for_graph(word_key)


def _toggle_expanded_category(word_key: WordKey, category: str) -> None:
    """Flip one category hub between shown-with-results and collapsed."""
    expanded = st.session_state["graph_expanded_categories"]
    key = (word_key, category)
    if key in expanded:
        expanded.discard(key)
    else:
        expanded.add(key)


def _analyze_word_for_graph(word_key: WordKey) -> None:
    """Run (cache-backed) word analysis for a word clicked in the Semantic
    Graph, unless it's already been analyzed this session - in which case
    this does nothing at all, not even a cache lookup, so re-clicking an
    already-explored word is instant.

    Uses the exact same `analyze_word_linguistics` call - and therefore the
    exact same on-disk cache - as the dropdown+button flow, so an LLM is
    never called twice for the same word regardless of which path asked for
    it first.
    """
    analyses = st.session_state["graph_word_analyses"]
    if word_key in analyses:
        return
    if not st.session_state.get("model_available", False):
        st.error("⚠️ LLM model not available. Please check the model status above.")
        return
    client = get_llm_client()
    if client is None:
        st.error("⚠️ Could not create an LLM client. Check the provider settings.")
        return
    with st.spinner(f"Analyzing '{word_key.word}'..."):
        result = run_async(
            analyze_word_linguistics(word_key.word, word_key.language, client)
        )
    if "error" not in result:
        analyses[word_key] = result
    st.rerun()


def show_language_graphs_help():
    """
    Display the language graphs help page using markdown files from the docs/ folder.
    """
    st.title("Understanding Language Graphs")

    # Path to docs directory
    docs_path = "docs/"

    # Find all markdown files recursively
    md_files = []

    try:
        # Walk through docs directory and find all .md files
        for root, dirs, files in os.walk(docs_path):
            for file in files:
                if file.endswith(".md"):
                    full_path = os.path.join(root, file)
                    md_files.append(full_path)

        if md_files:
            # Sort files alphabetically
            md_files.sort()

            # Create a dropdown to select between multiple files if there are more than one
            if len(md_files) > 1:
                file_names = [os.path.basename(f).replace(".md", "") for f in md_files]
                selected_index = st.selectbox(
                    "Select documentation:",
                    range(len(file_names)),
                    format_func=lambda i: format_documentation_file(i, file_names),
                )
                selected_file = md_files[selected_index]
            else:
                selected_file = md_files[0]

            # Read and display the selected file
            with open(selected_file, "r", encoding="utf-8") as f:
                md_content = f.read()
                st.markdown(md_content)
                logger.info(f"Displaying documentation from: {selected_file}")
        else:
            # No markdown files found
            st.error(f"No documentation files found in {docs_path}")
            st.markdown(get_fallback_help_content())
    except Exception as e:
        # Error accessing docs directory
        st.error(f"Error accessing documentation: {str(e)}")
        st.markdown(get_fallback_help_content())

    # Add a button to go back to the main application
    if st.button("← Back to Translation App", use_container_width=True):
        # Set the session state to indicate we're returning to the main app
        st.session_state["show_help_page"] = False
        st.rerun()


def get_fallback_help_content():
    """
    Return fallback help content as a markdown string.
    This is used when the documentation files can't be found.
    """
    return """
    # Language Graphs in Idiomapp

    Visualizing language connections through powerful interactive networks!

    ---

    ## Semantic Graphs: Revealing Translation Meaning

    - **Nodes** = words in different languages
    - **Edges** = translation relationships and meaning connections

    When you translate text in Idiomapp, it creates a semantic network that shows:

    - Direct translations between languages (e.g., "dog" → "perro")
    - Related words within each language
    - Part-of-speech connections

    **Key Features:**
    - Words are color-coded by language (🇬🇧 blue, 🇪🇸 pink, 🏴󠁥󠁳󠁣󠁴󠁿 purple)
    - Edge thickness shows translation strength
    - Node size indicates word importance

    This visualization helps you see how concepts map across languages, revealing both similarities and differences in expression.

    ---

    ## Co-occurrence Networks: Words That Travel Together

    - **Nodes** = individual words
    - **Edges** = words that appear near each other in text

    The co-occurrence view shows:

    - Which words frequently appear together
    - Key terms in a text by their connections
    - Natural word groupings based on usage
    """


def handle_translation_error(
    error_message: str, source_lang: str, target_lang: str
) -> str:
    """
    Handle translation errors gracefully and return user-friendly error messages.

    Args:
        error_message: The raw error message from the LLM client
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        str: User-friendly error message
    """
    # Extract error details if it's an OpenAI API error
    if "Error code:" in error_message:
        try:
            # Parse the error structure
            if "model_not_found" in error_message or "404" in error_message:
                return "⚠️ Model not available. Please select a different model from the sidebar."
            elif "invalid_api_key" in error_message:
                return "⚠️ Invalid API key. Please check your OpenAI API key in the sidebar."
            elif (
                "insufficient_quota" in error_message
                or "quota_exceeded" in error_message
            ):
                return "⚠️ API quota exceeded. Please check your OpenAI billing and usage limits."
            elif "rate_limit" in error_message or "429" in error_message:
                return "⚠️ Rate limit exceeded. Please wait a moment and try again."
            elif "401" in error_message:
                return "⚠️ Authentication failed. Please check your OpenAI API key."
            elif "403" in error_message:
                return "⚠️ Access denied. Please check your OpenAI account permissions."
            else:
                return "⚠️ API error occurred. Please try again or check your OpenAI account."
        except Exception:
            return "⚠️ Translation service error. Please try again."

    # Handle other types of errors
    if "Error:" in error_message:
        return "⚠️ Translation service error. Please try again."

    return f"⚠️ Unable to translate to {LANGUAGE_MAP.get(target_lang, {}).get('name', target_lang)}. Please try again."


def _badge(text: str, color: str) -> str:
    """Return an HTML badge span, or empty string if text is falsy."""
    if not text:
        return ""
    return (
        f'<span style="background:{color};color:#fff;'
        f'padding:2px 10px;border-radius:12px;font-size:13px;">{text}</span>'
    )


def display_word_analysis(word: str, language: str, analysis_data: dict):
    """Display detailed linguistic analysis of a word in a compact, visual layout."""
    lang_name = LANGUAGE_MAP.get(language, {}).get("name", language.title())

    # ── Compact header bar ──────────────────────────────────────────────────
    pos = analysis_data.get("pos", "")
    definition = analysis_data.get("definition", "")
    ipa = analysis_data.get("ipa", "")

    badges = (
        _badge(pos, "#4361EE")
        + _badge(analysis_data.get("register", ""), "#2ECC71")
        + _badge(analysis_data.get("frequency", ""), "#F39C12")
    )
    ipa_html = (
        f'<span style="color:#aaa;font-size:14px;font-style:italic;">/{ipa}/</span>'
        if ipa
        else ""
    )
    def_html = (
        f'<p style="color:#ccc;margin:10px 0 0;font-size:14px;">{definition}</p>'
        if definition
        else ""
    )

    st.markdown(
        f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid #4361EE;
                padding:16px 20px;border-radius:12px;margin-bottom:16px;">
      <div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;">
        <span style="font-size:28px;font-weight:bold;color:#fff;">{word}</span>
        <span style="color:#aaa;font-size:14px;">{lang_name}</span>
        {ipa_html}
      </div>
      <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;">{badges}</div>
      {def_html}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── LLM error notice (compact) ───────────────────────────────────────────
    if "llm_error" in analysis_data:
        st.warning(
            f"LLM unavailable — showing basic data only. ({analysis_data['llm_error']})"
        )
        with st.expander("🔧 Raw Data", expanded=False):
            st.json(analysis_data)
        return

    # The tabbed text detail (Origins/Meaning/Grammar/Usage/Idioms/Tips/Sound) -
    # this app's per-word Knowledge Graph used to live in a column next to this;
    # it's gone now that the Semantic Graph shows the same category/leaf data
    # inline, in place, when a word is clicked there. This text stays exactly
    # as it was, just full width instead of sharing the row with that graph.
    _display_analysis_panels(word, language, analysis_data)

    # Raw data at the very bottom, collapsed
    with st.expander("🔧 Raw Data", expanded=False):
        st.json(analysis_data)


def _render_origins_tab(d: dict):
    """Render the Origins tab content."""
    if "etymology" in d:
        st.markdown(f"**Origin:** {d['etymology']}")
    if "language_origin" in d:
        st.markdown(f"**Source language:** {d['language_origin']}")
    if "root" in d:
        st.markdown(f"**Root:** `{d['root']}`")
    if "historical_evolution" in d:
        st.caption(d["historical_evolution"])
    if "cognates" in d:
        st.markdown("**Cognates in other languages:**")
        st.markdown(
            "  " + " / ".join(f"**{c}**" for c in format_entries(d, "cognates"))
        )


def _render_meaning_tab(d: dict):
    """Render the Meaning tab content."""
    col1, col2 = st.columns(2)
    with col1:
        if "synonyms" in d:
            st.markdown("**Synonyms**")
            for s in format_entries(d, "synonyms"):
                st.markdown(f"  - {s}")
        if "hypernym" in d:
            st.markdown(f"**Broader:** {d['hypernym']}")
    with col2:
        if "antonyms" in d:
            st.markdown("**Antonyms**")
            for a in format_entries(d, "antonyms"):
                st.markdown(f"  - {a}")
        if "hyponyms" in d:
            st.markdown(
                f"**Specific:** {', '.join(format_entries(d, 'hyponyms', limit=5))}"
            )
    if "semantic_field" in d:
        st.caption(
            "Related concept words: " + " / ".join(format_entries(d, "semantic_field"))
        )


def _render_grammar_tab(d: dict):
    """Render the Grammar tab content with POS-specific logic."""
    pos = d.get("pos", "")
    if pos == "VERB":
        _display_verb_analysis(d)
    elif pos == "NOUN":
        _display_noun_analysis(d)
    elif pos == "ADJ":
        _display_adjective_analysis(d)
    else:
        _display_generic_analysis(d)
    if "grammar_notes" in d:
        st.caption(f"📖 {d['grammar_notes']}")


def _render_usage_tab(d: dict):
    """Render the Usage tab content."""
    for i, ex in enumerate(format_entries(d, "examples", limit=5), 1):
        st.markdown(f"{i}. *{ex}*")
    collocs = format_entries(d, "collocations", limit=10)
    if collocs:
        st.markdown("**Common collocations:** " + " / ".join(f"`{c}`" for c in collocs))
    if "regional_variations" in d:
        st.info(f"🌍 {d['regional_variations']}")


def _render_idioms_tab(d: dict):
    """Render the Idioms tab content."""
    for item in format_entries(d, "idioms", limit=6):
        st.markdown(f"- *{item}*")
    for p in format_entries(d, "proverbs", limit=3):
        st.markdown(f"- *{p}*")
    if "slang_usage" in d:
        st.caption(f"🗣️ Slang: {d['slang_usage']}")


def _render_tips_tab(d: dict):
    """Render the Tips tab content."""
    if "cultural_notes" in d:
        st.info(d["cultural_notes"])
    false_friends = format_entries(d, "false_friends")
    if false_friends:
        st.warning("⚠️ False friends: " + ", ".join(false_friends))
    for m in format_entries(d, "common_mistakes"):
        st.error(f"  ✗ {m}", icon=None)
    for t in format_entries(d, "tips", limit=4):
        st.success(f"💡 {t}")


def _render_sound_tab(d: dict):
    """Render the Sound tab content."""
    ipa = d.get("ipa")
    syllables = d.get("syllables")
    stress = d.get("stress")
    notes = d.get("pronunciation_notes")
    if ipa:
        st.markdown(f"**IPA:** `/{ipa}/`")
    if syllables:
        st.markdown(f"**Syllables:** {syllables}")
    if stress:
        st.markdown(f"**Stress:** {stress}")
    if notes:
        st.caption(notes)


# Tab configuration: (label, required_keys, render_function)
_TAB_DEFS = [
    (
        "📜 Origins",
        ["etymology", "root", "language_origin", "cognates"],
        _render_origins_tab,
    ),
    (
        "🔗 Meaning",
        ["synonyms", "antonyms", "semantic_field", "hypernym"],
        _render_meaning_tab,
    ),
    (
        "📝 Grammar",
        [
            "infinitive",
            "conjugations",
            "gender",
            "plural",
            "gender_forms",
            "comparison",
            "verb_type",
        ],
        _render_grammar_tab,
    ),
    ("💬 Usage", ["examples", "collocations", "regional_variations"], _render_usage_tab),
    ("🎭 Idioms", ["idioms", "proverbs", "slang_usage"], _render_idioms_tab),
    (
        "📚 Tips",
        ["cultural_notes", "false_friends", "common_mistakes", "tips"],
        _render_tips_tab,
    ),
    ("🔊 Sound", ["ipa", "syllables", "stress"], _render_sound_tab),
]


def _display_analysis_panels(word: str, language: str, analysis_data: dict):
    """Render tabbed detail panels, showing only tabs that have data."""
    active = [
        (label, fn)
        for label, keys, fn in _TAB_DEFS
        if any(k in analysis_data for k in keys)
    ]
    if not active:
        st.info(
            "No detailed data available — the LLM may not have returned structured results."
        )
        return

    tabs = st.tabs([label for label, _ in active])
    for tab, (_, render_fn) in zip(tabs, active):
        with tab:
            render_fn(analysis_data)


def _render_graph_selection_panel(source: str) -> None:
    """Render the persistent detail panel for the currently-selected node/edge
    in one of the two graphs that have one (Semantic Graph, Co-occurrence
    Network). Scoped to `source` so switching graphs doesn't show a stale
    selection left over from the other one.
    """
    sourced = st.session_state.get("graph_selection")
    if not isinstance(sourced, SourcedSelection) or sourced.source != source:
        st.caption("Click a node or edge in the graph to see details here.")
        return

    selection = sourced.selection
    if isinstance(selection, EdgeSelection):
        _render_edge_selection(selection)
        return

    payload = selection.payload

    if isinstance(payload, CategoryPayload):
        # Just a label - clicking this hub is what shows/hides its results
        # (see _dispatch_semantic_graph_click); there's nothing more to show
        # here than what the hub's own label already says.
        st.markdown(f"**{payload.label}**")
        return

    if isinstance(payload, LeafNodePayload):
        st.markdown(f"**{payload.category.replace('_', ' ').title()}**")
        st.write(payload.text)
        if payload.gloss:
            st.caption(payload.gloss)
        return

    if isinstance(payload, CooccurrenceWordPayload):
        # The co-occurrence graph is unrelated to the new exploration feature -
        # unchanged manual "Analyze this word" button + singular session keys.
        st.markdown(f"### {payload.word}")
        if payload.language:
            st.caption(
                LANGUAGE_MAP.get(payload.language, {}).get("name", payload.language)
            )
        st.caption(f"Co-occurrences: {payload.degree}")
        _render_analyze_button(
            payload.word, payload.language or "", key=f"panel_analyze_{source}"
        )
        analysis = st.session_state.get("current_word_analysis")
        if (
            analysis
            and st.session_state.get("current_word") == payload.word
            and "error" not in analysis
        ):
            display_word_analysis(payload.word, payload.language or "", analysis)
        return

    # SemanticWordPayload or RecursiveLeafPayload: a real word from the
    # Semantic Graph. Clicking it already triggered (or found cached)
    # analysis in _dispatch_semantic_graph_click, so there's no button here -
    # just show what's known, plus the full analysis once it's ready.
    if isinstance(payload, SemanticWordPayload):
        word, language = payload.label, payload.language
    else:
        word, language = payload.word_key.word, payload.word_key.language
    word_key = WordKey.of(word, language)

    st.markdown(f"### {word}")
    st.caption(LANGUAGE_MAP.get(language, {}).get("name", language))
    if isinstance(payload, SemanticWordPayload):
        if payload.pos:
            st.markdown(_badge(payload.pos, "#4361EE"), unsafe_allow_html=True)
        if payload.details:
            st.write(payload.details)

    analysis = st.session_state["graph_word_analyses"].get(word_key)
    if analysis and "error" not in analysis:
        display_word_analysis(word, language, analysis)
    else:
        st.caption("Analyzing...")


def _render_edge_selection(selection: EdgeSelection) -> None:
    """Render the panel content for a clicked edge - always just descriptive,
    since no edge triggers an action of its own."""
    payload = selection.payload
    if payload.kind in ("category_edge", "leaf_edge"):
        st.caption("A connection in the word-exploration graph.")
        return
    if payload.kind == "cooccurrence_edge":
        st.markdown(f"**{selection.source_id} ↔ {selection.target_id}**")
        st.caption(f"Co-occurrence count: {payload.weight}")
        return

    # A semantic-graph relation edge (translation, cognate, ...).
    st.markdown(f"**{selection.source_id} → {selection.target_id}**")
    st.caption(payload.kind.replace("_", " "))
    if payload.description:
        st.write(payload.description)
    if payload.strength is not None:
        st.caption(f"Strength: {payload.strength:.2f}")


def _render_analyze_button(word: str, language: str, *, key: str) -> None:
    """Render an "Analyze this word" button that runs the same cache-backed LLM
    analysis the dropdown+button flow uses, writing to the same session-state
    keys so both paths stay in sync."""
    if not st.button("🔍 Analyze this word", key=key):
        return
    if not st.session_state.get("model_available", False):
        st.error("⚠️ LLM model not available. Please check the model status above.")
        return
    client = get_llm_client()
    if client is None:
        st.error("⚠️ Could not create an LLM client. Check the provider settings.")
        return
    with st.spinner(f"Analyzing '{word}' using LLM..."):
        analysis_data = run_async(analyze_selected_word(word, language, client))
    st.session_state["current_word_analysis"] = analysis_data
    st.session_state["current_word"] = word
    st.session_state["current_word_lang"] = language
    st.rerun()


def _show_entries(d: dict, key: str, heading: str):
    """Render an Entries or StrList field as a titled bullet list."""
    items = format_entries(d, key)
    if not items:
        return
    st.markdown(f"**{heading}:**")
    for item in items:
        st.markdown(f"- {item}")


def _show_examples(d: dict):
    """Render numbered usage examples."""
    items = format_entries(d, "examples", limit=5)
    if items:
        st.markdown("**Usage Examples:**")
        for i, ex in enumerate(items, 1):
            st.markdown(f"{i}. {ex}")


def _display_verb_analysis(d: dict):
    """Display verb-specific analysis."""
    col1, col2 = st.columns(2)
    with col1:
        if "infinitive" in d:
            st.markdown(f"**Infinitive:** {d['infinitive']}")
        if "verb_type" in d:
            st.markdown(f"**Verb Type:** {d['verb_type']}")
        _show_entries(d, "conjugations", "Key Conjugations")
    with col2:
        _show_entries(d, "related_forms", "Related Forms")
        _show_entries(d, "synonyms", "Synonyms")
    _show_examples(d)
    if "grammar_notes" in d:
        st.info(f"**Grammar Notes:** {d['grammar_notes']}")


def _display_noun_analysis(d: dict):
    """Display noun-specific analysis."""
    col1, col2 = st.columns(2)
    with col1:
        if "gender" in d:
            st.markdown(f"**Gender:** {d['gender']}")
        if "plural" in d:
            st.markdown(f"**Plural:** {d['plural']}")
        _show_entries(d, "articles", "Articles")
    with col2:
        _show_entries(d, "related_forms", "Related Forms")
        _show_entries(d, "synonyms", "Synonyms")
    _show_examples(d)
    if "cultural_notes" in d:
        st.info(f"**Cultural Notes:** {d['cultural_notes']}")


def _display_adjective_analysis(d: dict):
    """Display adjective-specific analysis."""
    col1, col2 = st.columns(2)
    with col1:
        _show_entries(d, "gender_forms", "Gender Forms")
        _show_entries(d, "comparison", "Comparison Forms")
    with col2:
        _show_entries(d, "synonyms", "Synonyms")
        _show_entries(d, "antonyms", "Antonyms")
    _show_examples(d)
    if "position" in d:
        st.info(f"**Position Rule:** {d['position']}")


def _display_generic_analysis(d: dict):
    """Display generic analysis for other parts of speech."""
    if "definition" in d:
        st.markdown(f"**Definition:** {d['definition']}")
    if "related_words" in d:
        st.markdown("**Related Words:**")
        for w in format_entries(d, "related_words"):
            st.markdown(f"- {w}")
    _show_examples(d)
    if "grammar_notes" in d:
        st.info(f"**Grammar Notes:** {d['grammar_notes']}")


async def analyze_selected_word(word: str, language: str, client):
    """
    Analyze a selected word using the LLM client.

    Args:
        word: Word to analyze
        language: Language code
        client: LLM client

    Returns:
        Analysis data dictionary
    """
    try:
        analysis = await analyze_word_linguistics(word, language, client)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing word {word}: {e}")
        return {"error": f"Analysis failed: {str(e)}"}


def _report_error(action: str, e: Exception) -> None:
    """Show and log an error for a failed sidebar action, e.g. "loading graph history"."""
    st.error(f"Error {action}: {e}")
    logger.error(f"Error {action}: {e}")


def _render_sidebar() -> tuple[str, list[str]]:
    """Render the sidebar (language/LLM/graph settings) and return the source
    language and target languages the user selected."""
    # Add a sidebar with translation settings
    with st.sidebar:
        st.header("Translation Settings")

        # Help button at the top of the sidebar with improved styling
        st.markdown(
            """
        <style>
        .doc-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background-color: #4361EE;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            text-decoration: none;
            margin-bottom: 1rem;
            width: 100%;
            font-weight: bold;
            cursor: pointer;
        }
        .doc-button:hover {
            background-color: #3A56D4;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Hidden button that will be triggered by the custom HTML button
        if st.button(
            "📚 Understanding Language Graphs",
            key="doc_button_hidden",
            help="Learn about language graphs and how to use them",
            use_container_width=True,
        ):
            st.session_state["show_help_page"] = True
            st.rerun()

        # Add a visual separator
        st.markdown("<hr>", unsafe_allow_html=True)

        # Language selection - KEEP VISIBLE (priority)
        st.header("Language Settings")
        source_lang = st.selectbox(
            "Source Language",
            settings.supported_languages_list,
            index=get_index(
                settings.supported_languages_list, settings.default_source_language
            ),
            format_func=get_language_display,
            help="Select the source language",
        )

        # Multiple target languages selection
        target_langs = st.multiselect(
            "Target Languages",
            settings.supported_languages_list,
            default=settings.default_target_languages_list,
            format_func=get_language_display,
            help="Select one or more target languages",
        )

        # Ensure at least one target language is selected
        if not target_langs:
            st.warning("Please select at least one target language")
            # Use first available default target language, or first supported language as fallback
            if settings.default_target_languages_list:
                target_langs = [settings.default_target_languages_list[0]]
            elif settings.supported_languages_list:
                target_langs = [settings.supported_languages_list[0]]
            else:
                target_langs = []

        # Model availability status
        if not st.session_state["model_available"]:
            st.error("⚠️ Selected model is not available. LLM features are disabled.")
        else:
            st.success("✅ LLM model is ready to use")

        # LLM Settings - Move to collapsible expander
        with st.expander("⚙️ LLM Settings", expanded=False):
            # LLM Provider selection
            st.subheader("LLM Provider")
            provider_options = [p.value for p in LLMProvider]
            provider_index = get_index(
                provider_options, st.session_state["llm_provider"]
            )
            selected_provider = st.selectbox(
                "LLM Provider",
                provider_options,
                index=provider_index,
                format_func=format_provider_name,
                help="Select the LLM provider to use for translation",
            )

            model_name = st.session_state["model_name"]

            # Show provider-specific options. Credentials live inside their own
            # provider branch so an unrelated provider's key box is never shown.
            if selected_provider == "ollama":
                # Use cached available models to prevent repeated API calls
                if "cached_available_models" not in st.session_state:
                    st.session_state["cached_available_models"] = (
                        get_available_models()
                        if st.session_state["llm_provider"] == "ollama"
                        else ["llama3.2:latest"]
                    )
                available_models = st.session_state["cached_available_models"]
                is_ollama_available = (
                    st.session_state["model_available"]
                    and st.session_state["llm_provider"] == "ollama"
                )
                model_label = build_model_label("Ollama Model", is_ollama_available)
                model_index = get_index(
                    available_models, st.session_state["model_name"]
                )
                is_disabled = not is_ollama_available

                model_name = st.selectbox(
                    model_label,
                    available_models,
                    index=model_index,
                    help="Select the Ollama model to use for translation",
                    disabled=is_disabled,
                )

            elif selected_provider == "openai":
                openai_models = cached_openai_models(
                    *get_provider_credentials("openai")
                )
                is_openai_available = (
                    st.session_state["model_available"]
                    and st.session_state["llm_provider"] == "openai"
                )
                model_label = build_model_label("OpenAI Model", is_openai_available)
                model_index = get_index(openai_models, st.session_state["model_name"])
                is_disabled = not is_openai_available

                model_name = st.selectbox(
                    model_label,
                    openai_models,
                    index=model_index,
                    help="Select the OpenAI model to use for translation",
                    disabled=is_disabled,
                )

                openai_api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    value=settings.openai_api_key,
                    help="Enter your OpenAI API key to use ChatGPT",
                )

                openai_organization = st.text_input(
                    "OpenAI Organization ID (Optional)",
                    value=settings.openai_organization,
                    help="Enter your OpenAI organization ID if you're part of an organization",
                )

                if (
                    openai_api_key != settings.openai_api_key
                    or openai_organization != settings.openai_organization
                ):
                    # Credentials are held in session state only - never written to
                    # environment variables.
                    if openai_api_key:
                        st.success(
                            "OpenAI credentials updated. Reinitializing client..."
                        )
                        st.session_state["openai_api_key"] = openai_api_key
                        if openai_organization:
                            st.session_state[
                                "openai_organization"
                            ] = openai_organization
                        elif "openai_organization" in st.session_state:
                            del st.session_state["openai_organization"]
                        reset_llm_client_state()
                    else:
                        for key in ("openai_api_key", "openai_organization"):
                            if key in st.session_state:
                                del st.session_state[key]
                        st.warning(
                            "API key cleared. Please enter a valid API key to use OpenAI."
                        )

            elif selected_provider == "anthropic":
                anthropic_api_key, _ = get_provider_credentials("anthropic")
                anthropic_models = cached_anthropic_models(anthropic_api_key)
                is_anthropic_available = (
                    st.session_state["model_available"]
                    and st.session_state["llm_provider"] == "anthropic"
                )
                model_label = build_model_label("Claude Model", is_anthropic_available)
                # The configured model is often a rolling alias ("claude-haiku-4-5")
                # while the live API lists a dated snapshot ("claude-haiku-4-5-
                # 20251001") - resolve one against the other before doing the
                # exact-match lookup, or the dropdown loses track of it and
                # silently falls back to whatever the API listed first.
                configured_model = resolve_anthropic_model(
                    st.session_state["model_name"], anthropic_models
                )
                model_index = get_index(
                    anthropic_models, configured_model or st.session_state["model_name"]
                )
                is_disabled = not is_anthropic_available

                model_name = st.selectbox(
                    model_label,
                    anthropic_models,
                    index=model_index,
                    help="Select the Claude model to use for translation",
                    disabled=is_disabled,
                )

                anthropic_api_key = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    value=settings.anthropic_api_key,
                    help="Enter your Anthropic API key to use Claude",
                )

                if anthropic_api_key != settings.anthropic_api_key:
                    if anthropic_api_key:
                        st.success(
                            "Anthropic credentials updated. Reinitializing client..."
                        )
                        st.session_state["anthropic_api_key"] = anthropic_api_key
                        reset_llm_client_state()
                    else:
                        if "anthropic_api_key" in st.session_state:
                            del st.session_state["anthropic_api_key"]
                        st.warning(
                            "API key cleared. Please enter a valid API key to use Claude."
                        )

        # Update client if provider or model changes
        if (
            selected_provider != st.session_state["llm_provider"]
            or model_name != st.session_state["model_name"]
        ):
            st.session_state["llm_provider"] = selected_provider
            st.session_state["model_name"] = model_name
            # Update environment variables for runtime changes
            os.environ["LLM_PROVIDER"] = selected_provider
            if selected_provider == "ollama":
                os.environ["DEFAULT_MODEL"] = model_name
            elif selected_provider == "anthropic":
                os.environ["ANTHROPIC_MODEL"] = model_name
            else:
                os.environ["OPENAI_MODEL"] = model_name
            # Force reinitialization of client
            st.info("Provider or model changed. Reinitializing client...")
            st.session_state["model_available"] = False
            st.rerun()

        # Graph Options - Move to collapsible expander
        with st.expander("📊 Graph Options", expanded=False):
            # Switch for visualization type
            st.subheader("Visualization Settings")
            view_options = ["Semantic Graph", "Co-occurrence Network"]
            selected_view = st.radio("Analysis View", view_options)

            # Map selection to internal state
            st.session_state["current_view"] = (
                "semantic" if selected_view == "Semantic Graph" else "cooccurrence"
            )

            # Co-occurrence settings (only shown when that view is selected)
            if st.session_state["current_view"] == "cooccurrence":
                st.subheader("Co-occurrence Settings")

                # Window size for co-occurrence
                window_size = st.slider(
                    "Window Size",
                    min_value=1,
                    max_value=5,
                    value=settings.default_window_size,
                    help="Number of words to consider for co-occurrence (larger = more connections)",
                )
                st.session_state["window_size"] = window_size

                # Minimum frequency for words
                min_freq = st.slider(
                    "Minimum Word Frequency",
                    min_value=1,
                    max_value=5,
                    value=settings.default_min_frequency,
                    help="Minimum times a word must appear to be included",
                )
                st.session_state["min_freq"] = min_freq

                # POS tag selection
                pos_options = [
                    ("Nouns", "NOUN"),
                    ("Verbs", "VERB"),
                    ("Adjectives", "ADJ"),
                    ("Adverbs", "ADV"),
                    ("Proper Nouns", "PROPN"),
                ]
                pos_tag_options = [tag for _, tag in pos_options]
                selected_pos = st.multiselect(
                    "Part of Speech Filter",
                    options=pos_tag_options,
                    default=settings.default_pos_filter_list,
                    format_func=lambda tag: format_pos_option(tag, pos_options),
                    help="Filter words by part of speech",
                )
                st.session_state["selected_pos"] = selected_pos

        # Graph History Section - Move to collapsible expander
        with st.expander("📈 Graph History", expanded=False):
            # Show recent graphs
            try:
                history = st.session_state.graph_storage.get_graph_history(limit=10)

                if history:
                    for graph in history:
                        # Create a compact display for each graph
                        with st.expander(
                            f"📈 {graph['source_text'][:40]}...", expanded=False
                        ):
                            st.write(
                                f"**Languages:** {', '.join(graph['target_languages'])}"
                            )
                            st.write(
                                f"**Nodes:** {graph['node_count']}, **Edges:** {graph['edge_count']}"
                            )
                            st.write(f"**Created:** {graph['created_at'][:16]}")

                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🔄 Load", key=f"load_{graph['id']}"):
                                    # Load and display historical graph
                                    loaded_graph = (
                                        st.session_state.graph_storage.get_graph(
                                            graph["id"]
                                        )
                                    )
                                    if loaded_graph:
                                        st.session_state["current_graph_data"] = {
                                            "nodes": loaded_graph["nodes"],
                                            "edges": loaded_graph["edges"],
                                        }
                                        st.session_state["current_graph_id"] = graph[
                                            "id"
                                        ]
                                        st.rerun()

                            with col2:
                                if st.button("🗑️ Delete", key=f"delete_{graph['id']}"):
                                    if st.session_state.graph_storage.delete_graph(
                                        graph["id"]
                                    ):
                                        st.success("Graph deleted!")
                                        st.rerun()
                else:
                    st.info(
                        "No graphs saved yet. Generate your first graph to see it here!"
                    )
            except Exception as e:
                _report_error("loading graph history", e)

            # Add search functionality
            st.subheader("🔍 Search Graphs")
            search_query = st.text_input(
                "Search by text content",
                placeholder="Enter text to search...",
                key="graph_search",
            )

            if search_query:
                try:
                    search_results = (
                        st.session_state.graph_storage.search_graphs_by_text(
                            search_query, limit=5
                        )
                    )
                    if search_results:
                        st.write(f"Found {len(search_results)} matching graphs:")
                        for result in search_results:
                            st.write(f"• {result['source_text'][:50]}...")
                    else:
                        st.info("No matching graphs found.")
                except Exception as e:
                    _report_error("searching graphs", e)

            # Show storage statistics
            st.subheader("📊 Storage Info")
            try:
                stats = st.session_state.graph_storage.get_graph_statistics()
                st.write(f"**Total Graphs:** {stats['total_graphs']}")
                st.write(f"**Total Nodes:** {stats['total_nodes']}")
                st.write(f"**Storage Size:** {stats['storage_size_mb']} MB")
            except Exception as e:
                _report_error("loading storage stats", e)

            if st.button("🗑️ Clear All Data", key="clear_all_graphs"):
                st.session_state["confirm_clear_all_graphs"] = True

            if st.session_state.get("confirm_clear_all_graphs"):
                st.warning("Delete all saved graphs? This cannot be undone.")
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    if st.button("Yes, delete", key="confirm_clear_all_graphs_yes"):
                        st.session_state["confirm_clear_all_graphs"] = False
                        try:
                            if st.session_state.graph_storage.clear_all_data():
                                st.success("All data cleared!")
                                st.rerun()
                        except Exception as e:
                            _report_error("clearing data", e)
                with cancel_col:
                    if st.button("Cancel", key="confirm_clear_all_graphs_cancel"):
                        st.session_state["confirm_clear_all_graphs"] = False
                        st.rerun()

        # Debug toggle - Move to collapsible expander
        with st.expander("🐛 Debug", expanded=False):
            st.session_state["show_debug"] = st.checkbox(
                "Show Debug Logs",
                value=st.session_state["show_debug"],
                help="Show detailed logs of translation processing",
            )

    return source_lang, target_langs


def _render_debug_and_help_sections() -> None:
    """Render the collapsible debug-log panel and the dismissible how-to guide."""
    # Show debug logs if enabled
    if st.session_state["show_debug"]:
        with st.expander("Debug Logs", expanded=True):
            logs = get_recent_logs(50)
            if not logs:
                st.info("No logs yet. Perform actions to see logs here.")
            else:
                log_output = "\n".join(reversed(logs))
                st.code(log_output)

            if st.button("Clear Logs"):
                clear_logs()
                st.rerun()

    # Display a helpful guide if no translation has been made yet (collapsible and dismissible)
    if not st.session_state["help_dismissed"] and not st.session_state["chat_history"]:
        with st.expander("ℹ️ How to use the Translation Helper", expanded=False):
            st.markdown(
                f"""
            1. Select source language in the sidebar
            2. Select one or more target languages in the sidebar
            3. Type your text in the input box
            4. Click "Translate" to see the translations
            5. Explore the word relationships in either the semantic graph or co-occurrence network views

            **Example**: Try translating "Do you know my country?" from {get_language_name("en")} to both {get_language_name("es")} and {get_language_name("ca")}
            """
            )
            if st.button("Dismiss", key="dismiss_help"):
                st.session_state["help_dismissed"] = True
                st.rerun()


def _render_translation_panel(
    source_lang: str, target_langs: list[str]
) -> tuple[str, bool]:
    """Render the translation input/output panel and chat-history sidebar.

    Returns the entered source text and whether the translate button was pressed.
    """
    # Translation input/output section - side by side (50/50)
    st.subheader("💬 Translation")

    # Create main content area with collapsible right sidebar
    main_col, chat_sidebar_col = st.columns([2, 1])

    with main_col:
        input_col, output_col = st.columns([1, 1])

        with input_col:
            st.markdown("**Input**")
            # Translation input
            source_text = st.text_area(
                f"Enter text in {get_language_name(source_lang)}:",
                height=200,
                placeholder=f"Type your text in {get_language_name(source_lang)} here...",
                disabled=not st.session_state["model_available"],
                label_visibility="collapsed",
            )

            # The buttons are stacked rather than placed in their own columns:
            # this block already sits inside main_col > input_col, and Streamlit
            # allows only one level of column nesting. Both buttons are
            # full-width, so stacking reads the same.
            if len(target_langs) == 1:
                button_text = f"🔄 Translate to {get_language_name(target_langs[0])}"
            else:
                button_text = f"🔄 Translate to {len(target_langs)} languages"

            translate_button = st.button(
                button_text,
                use_container_width=True,
                disabled=not st.session_state["model_available"] or not source_text,
            )

            clear_button = st.button("🗑️ Clear History", use_container_width=True)
            if clear_button:
                st.session_state["chat_history"] = []
                st.session_state["graph_data"] = None
                st.session_state["cooccurrence_graphs"] = {}
                st.success("History cleared!")
                st.rerun()

        with output_col:
            # Google Translate-style translation output
            st.markdown("**Translation**")

            # Get the most recent translation results from chat history
            if st.session_state["chat_history"]:
                # Find the most recent assistant messages (translations)
                recent_translations = []
                for i in range(len(st.session_state["chat_history"]) - 1, -1, -1):
                    message = st.session_state["chat_history"][i]
                    if message["role"] == "assistant":
                        recent_translations.insert(0, message)
                        # Stop when we find a user message (start of a new translation request)
                        if (
                            i > 0
                            and st.session_state["chat_history"][i - 1]["role"]
                            == "user"
                        ):
                            break

                # Display translations in Google Translate style
                if recent_translations:
                    # Parse all translations
                    parsed_translations = []
                    for message in recent_translations:
                        trans_text, lang_code, lang_display = parse_translation_message(
                            message
                        )
                        if trans_text:
                            parsed_translations.append(
                                {
                                    "text": trans_text,
                                    "lang_code": lang_code,
                                    "lang_display": lang_display,
                                    "message": message,
                                }
                            )

                    if parsed_translations:
                        # Display each translation cleanly
                        for idx, trans in enumerate(parsed_translations):
                            # Language label (subtle, small) - light gray for dark theme
                            if trans["lang_display"]:
                                st.markdown(
                                    f'<div style="font-size: 0.75em; color: #B0B0B0; margin-bottom: 4px; font-weight: 500;">{html.escape(trans["lang_display"])}</div>',
                                    unsafe_allow_html=True,
                                )

                            # Translation text (large, prominent, Google Translate style) - light text for dark theme
                            escaped_text = html.escape(trans["text"])
                            st.markdown(
                                f"""
                            <div style="font-size: 1.5em; line-height: 1.6; color: #FAFAFA;
                                        padding: 12px 0; margin-bottom: 16px;
                                        border-bottom: 1px solid #4CC9F0; word-wrap: break-word;">
                                {escaped_text}
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            # TTS button (subtle, inline) - only show translation text, not language label
                            if (
                                st.session_state.get("model_available", False)
                                and trans["lang_code"]
                            ):
                                tts_key = f"tts_output_{hash(trans['text'] + trans['lang_code'])}"
                                # Streamlit allows only one level of column nesting, and
                                # this already sits two levels deep (main_col > output_col).
                                # use_container_width=False already keeps the button
                                # compact, so no column wrapper is needed for that.
                                if st.button(
                                    "🔊",
                                    key=tts_key,
                                    help=f"Play audio in {trans['lang_display'] or trans['lang_code']}",
                                    use_container_width=False,
                                ):
                                    text_to_speech(
                                        trans["text"], trans["lang_code"], tts_key
                                    )
                else:
                    # Empty state - light text for dark theme
                    st.markdown(
                        """
                    <div style="text-align: center; padding: 60px 20px; color: #B0B0B0;">
                        <div style="font-size: 3em; margin-bottom: 10px;">🌐</div>
                        <div style="font-size: 1.1em;">Your translations will appear here</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            else:
                # Empty state - light text for dark theme
                st.markdown(
                    """
                <div style="text-align: center; padding: 60px 20px; color: #B0B0B0;">
                    <div style="font-size: 3em; margin-bottom: 10px;">🌐</div>
                    <div style="font-size: 1.1em;">Your translations will appear here</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    # Right sidebar for chat history (collapsible via expander)
    with chat_sidebar_col:
        with st.expander("💬 Chat History", expanded=False):
            # Create a scrollable chat container
            chat_container = st.container(height=600)

            with chat_container:
                # Show existing messages or a placeholder
                if not st.session_state["chat_history"]:
                    st.markdown(
                        "<p style='color: #666; text-align: center; padding: 20px; font-size: 0.9em;'>Your translation history will appear here</p>",
                        unsafe_allow_html=True,
                    )
                else:
                    for i, message in enumerate(st.session_state["chat_history"]):
                        # For LLM responses (translations), use the target language for TTS
                        message_target_lang = None
                        if message["role"] == "assistant":
                            # Check if the message has a target_lang attribute
                            if "target_lang" in message:
                                message_target_lang = message["target_lang"]
                            # Fallback to analyzing the previous message
                            elif i > 0:
                                # Get the previous message to find the request details
                                prev_msg = st.session_state["chat_history"][i - 1]
                                if (
                                    prev_msg["role"] == "user"
                                    and "Translate" in prev_msg["content"]
                                ):
                                    # This is a translation response, extract the target language
                                    prev_content = prev_msg["content"]

                                    # Use match to check for target language in previous message
                                    pattern = "to (.*?):"
                                    match_result = re.search(pattern, prev_content)
                                    if match_result:
                                        target_text = match_result.group(1).strip()

                                        message_target_lang = None
                                        for (
                                            lang_code,
                                            lang_info,
                                        ) in LANGUAGE_MAP.items():
                                            if lang_info["name"] in target_text:
                                                message_target_lang = lang_code
                                                break

                        render_chat_message(
                            message["content"],
                            message["role"],
                            message_target_lang,
                            source_lang,
                        )

    return source_text, translate_button


def _filter_edges(edges: list, min_strength: float, selected_kinds) -> list:
    """Keep only edges at/above min_strength and whose relation is selected."""
    return [
        edge
        for edge in edges
        if edge.get("strength", 1.0) >= min_strength
        and edge.get("relation", "related") in selected_kinds
    ]


# Friendly labels for the relation-kind filter; anything not listed here falls
# back to a title-cased version of its raw relation string.
_RELATION_KIND_LABELS = {
    "translation": "🔤 Translation",
    "cognate": "🌍 Cognate",
    "cross_sentence": "📝 Same sentence",
}


def _render_graph_visualization_tabs() -> None:
    """Render the Semantic Graph / Co-occurrence Network tabs, including the
    word-selection and word-analysis UI nested inside the semantic graph tab."""
    # Graph visualization section - moved to tabs
    if st.session_state["graph_data"] or st.session_state.get("cooccurrence_graphs"):
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📊 Semantic Graph", "📈 Co-occurrence Network"])

        with tab1:
            # Show the visualization based on the selected view
            if st.session_state["graph_data"]:
                # Add header for the graph
                st.subheader("📊 Semantic Network Analysis")

                # Interactive Word Analysis Section
                st.markdown("### 🔍 Interactive Word Analysis")
                st.markdown(
                    "**Click any word to analyze it automatically** — its "
                    "categories (Synonyms, Etymology, Examples, ...) appear "
                    "as new nodes attached to it. Click a category to reveal "
                    "or hide its results, and click a result that's itself a "
                    "word to keep exploring from there."
                )

                # Display controls for the graph
                available_langs = list(st.session_state["graph_data"].keys())
                merge_graphs = True  # Default to merged view
                # Cross-language scores usually land well under 0.5, even for
                # correct direct translations - a 0.5 default hid nearly everything.
                min_strength = 0.0  # Default: show every relationship that was found

                # Peek at every relation kind that could actually appear, so the
                # kind filter's options match reality instead of a hardcoded
                # guess. Merging only adds edges (never removes any), so the
                # merged graph's kinds are a superset of any single-language
                # view's - cheap and safe to compute even when merging is off.
                if len(available_langs) > 1:
                    preview_edges = merge_language_graphs(
                        st.session_state["graph_data"]
                    )["edges"]
                else:
                    preview_edges = next(iter(st.session_state["graph_data"].values()))[
                        "edges"
                    ]
                available_kinds = sorted(
                    {edge.get("relation", "related") for edge in preview_edges}
                )
                selected_kinds = available_kinds

                with st.expander("Graph Options", expanded=False):
                    # Create columns for options
                    opt_col1, opt_col2 = st.columns([1, 1])

                    with opt_col1:
                        # Option to merge all graphs into one comprehensive view
                        merge_graphs = st.checkbox(
                            "Merge all language graphs",
                            value=True,
                            help="Show connections between different languages",
                        )

                    with opt_col2:
                        # Filter for minimum relationship strength
                        min_strength = st.slider(
                            "Minimum relationship strength",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.1,
                            help=(
                                "Hide relationships weaker than this. Cross-language "
                                "word-pair scores are often well under 0.5, even for "
                                "correct direct translations - start at 0 to see "
                                "everything, then raise it to declutter."
                            ),
                        )

                    selected_kinds = st.multiselect(
                        "Relationship types to show",
                        options=available_kinds,
                        default=available_kinds,
                        format_func=lambda k: _RELATION_KIND_LABELS.get(
                            k, k.replace("_", " ").title()
                        ),
                        help=(
                            "Each relationship type comes from a different signal - "
                            "Translation is LLM-reported, Cognate is a look-alike "
                            "heuristic. Deselect a type to hide it everywhere below."
                        ),
                    )

                # Display the graph based on selection

                if merge_graphs and len(available_langs) > 1:
                    # Create a merged graph with cross-language connections
                    merged_graph = merge_language_graphs(st.session_state["graph_data"])
                    merged_graph["edges"] = _filter_edges(
                        merged_graph["edges"], min_strength, selected_kinds
                    )

                    st.markdown(
                        f"**Combined graph showing relationships between {', '.join(available_langs)}**"
                    )
                    visualize_translation_graph(merged_graph)
                elif available_langs:
                    # Let user choose which language graph to show
                    selected_lang = st.selectbox(
                        "Select language graph",
                        options=available_langs,
                        format_func=get_language_display,
                    )

                    graph_data = st.session_state["graph_data"][selected_lang]
                    graph_data = {
                        "nodes": graph_data["nodes"],
                        "edges": _filter_edges(
                            graph_data["edges"], min_strength, selected_kinds
                        ),
                        "metadata": graph_data.get("metadata", {}),
                    }

                    # Display the selected graph
                    st.markdown(
                        f"**Semantic network for {get_language_display(selected_lang)}**"
                    )
                    visualize_translation_graph(graph_data)

            # Add a legend explaining the graph
            with st.expander("📊 Graph Legend", expanded=False):
                legend_col1, legend_col2, legend_col3 = st.columns(3)

                with legend_col1:
                    st.markdown("#### Language Colors")
                    st.markdown("🔵 **Blue** - English words")
                    st.markdown("🟡 **Yellow** - Spanish words")
                    st.markdown("🔴 **Red** - Catalan words")
                    st.markdown("💠 **Lighter shades** - Related words")

                with legend_col2:
                    st.markdown("#### Edge Types")
                    st.markdown("⚪ **White** - Translation (LLM-reported)")
                    st.markdown("🔶 **Gold dashed** - Cognate (looks alike)")
                    st.markdown("🟢 **Green** - Synonyms")
                    st.markdown("🔴 **Red** - Antonyms")
                    st.markdown("🟠 **Orange** - Hypernyms (broader terms)")
                    st.markdown("🟡 **Yellow** - Hyponyms (specific terms)")
                    st.markdown("🔵 **Cyan** - Contextual relation")
                    st.markdown("🟣 **Purple dashed** - Cross-sentence relation")

                with legend_col3:
                    st.markdown("#### Word Types")
                    st.markdown("🟠 **Orange border** - Noun")
                    st.markdown("🟢 **Green border** - Verb")
                    st.markdown("🔵 **Blue border** - Adjective")
                    st.markdown("🟡 **Yellow border** - Adverb")
                    st.markdown("🔴 **Red border** - Pronoun")
                    st.markdown("💗 **Pink border** - Preposition")
                    st.markdown("🟣 **Purple border** - Conjunction")
                    st.markdown("🔍 **Larger size** - Primary translation words")
                    st.markdown("🔎 **Smaller size** - Related words")

            # Word Analysis Section
            st.markdown("---")
            st.markdown("### 🔍 Word Analysis")

            # Build list of words from graph data for selection
            if st.session_state.get("graph_data"):
                # Collect all words from all language graphs
                word_options = []
                for lang, data in st.session_state["graph_data"].items():
                    for node in data.get("nodes", []):
                        word_label = node.get("label", "")
                        word_lang = node.get("language", lang)
                        word_pos = node.get("pos", "")
                        if word_label:
                            # Create display string and store data
                            display = f"{word_label} ({word_lang}) - {word_pos}"
                            word_options.append(
                                {
                                    "display": display,
                                    "word": word_label,
                                    "language": word_lang,
                                    "pos": word_pos,
                                }
                            )

                # Remove duplicates based on word+language
                seen = set()
                unique_options = []
                for opt in word_options:
                    key = (opt["word"], opt["language"])
                    if key not in seen:
                        seen.add(key)
                        unique_options.append(opt)

                if unique_options:
                    # Create selectbox for word selection
                    st.info("💡 **Select a word from the graph to analyze it:**")

                    display_options = ["-- Select a word --"] + [
                        opt["display"] for opt in unique_options
                    ]
                    selected_display = st.selectbox(
                        "Choose a word to analyze",
                        options=display_options,
                        key="word_analysis_selectbox",
                    )

                    if selected_display != "-- Select a word --":
                        # Find the selected word data
                        selected_word_data = next(
                            (
                                opt
                                for opt in unique_options
                                if opt["display"] == selected_display
                            ),
                            None,
                        )

                        if selected_word_data:
                            word = selected_word_data["word"]
                            language = selected_word_data["language"]
                            pos = selected_word_data["pos"]

                            st.success(
                                f"**Selected Word**: {word} ({language}) - {pos}"
                            )

                            # Analyze button
                            if st.button("🔍 Analyze Selected Word", type="primary"):
                                if st.session_state.get("model_available", False):
                                    with st.spinner(f"Analyzing '{word}' using LLM..."):
                                        # Reuse the session's cached client rather than
                                        # rebuilding one (which re-runs Ollama's
                                        # availability check) on every button press.
                                        client = get_llm_client()

                                        if client is None:
                                            st.error(
                                                "⚠️ Could not create an LLM client. Check the provider settings."
                                            )
                                        else:
                                            logger.info(
                                                f"Client status: {client.get_model_status()}"
                                            )

                                            analysis_data = run_async(
                                                analyze_selected_word(
                                                    word, language, client
                                                )
                                            )
                                            logger.info(
                                                f"Analysis result keys: {list(analysis_data.keys()) if analysis_data else 'None'}"
                                            )

                                            # Store the analysis for display
                                            st.session_state[
                                                "current_word_analysis"
                                            ] = analysis_data
                                            st.session_state["current_word"] = word
                                            st.session_state[
                                                "current_word_lang"
                                            ] = language
                                            st.rerun()
                                else:
                                    st.error(
                                        "⚠️ LLM model not available. Please check the model status above."
                                    )

                            # Display analysis if available
                            analysis_data = st.session_state.get(
                                "current_word_analysis"
                            )
                            if (
                                analysis_data
                                and st.session_state.get("current_word") == word
                            ):
                                if "error" not in analysis_data:
                                    display_word_analysis(word, language, analysis_data)
                                else:
                                    st.error(
                                        f"Analysis failed: {analysis_data['error']}"
                                    )
                else:
                    st.info("No words available in the graph for analysis.")
            else:
                st.info(
                    "No graph data available yet. Translate some text to generate graphs."
                )

        with tab2:
            # Show co-occurrence networks
            if st.session_state.get("cooccurrence_graphs"):
                # Add header for the co-occurrence network
                st.subheader("📊 Word Co-occurrence Network")

                available_langs = list(st.session_state["cooccurrence_graphs"].keys())

                # Let user choose which language graph to show
                if available_langs:
                    selected_lang = st.selectbox(
                        "Select language",
                        options=available_langs,
                        format_func=get_language_display,
                    )

                # Show information about this analysis
                st.markdown(
                    f"""
                Showing word co-occurrence network for **{get_language_display(selected_lang)}**

                * Nodes represent individual words
                * Larger nodes appear more frequently
                * Edges show words that appear close together in the text
                * Thicker edges indicate words that co-occur more frequently
                """
                )

                # Display the co-occurrence network
                graph = st.session_state["cooccurrence_graphs"][selected_lang]
                visualize_cooccurrence_network(graph, selected_lang)

                # Show network stats
                import networkx as nx

                st.subheader("Network Statistics")

                stat_cols = st.columns(3)
                with stat_cols[0]:
                    st.metric("Nodes (Words)", len(graph.nodes()))
                with stat_cols[1]:
                    st.metric("Edges (Co-occurrences)", len(graph.edges()))
                with stat_cols[2]:
                    if len(graph.nodes()) > 0:
                        density = nx.density(graph)
                        st.metric("Network Density", f"{density:.4f}")

                # Show most central words
                if len(graph.nodes()) > 0:
                    st.subheader("Most Important Words")

                    # Calculate centrality measures
                    degree_cent, betweenness_cent = compute_centrality(graph)

                    # Get top words by degree centrality
                    top_degree = sorted(
                        degree_cent.items(), key=get_centrality_sort_key
                    )[:10]
                    top_betweenness = sorted(
                        betweenness_cent.items(), key=get_centrality_sort_key
                    )[:10]

                    cent_cols = st.columns(2)
                    with cent_cols[0]:
                        st.markdown("**Most Connected Words**")
                        for word, score in top_degree:
                            st.markdown(f"• **{word}** ({score:.3f})")

                    with cent_cols[1]:
                        st.markdown("**Bridge Words**")
                        for word, score in top_betweenness:
                            st.markdown(f"• **{word}** ({score:.3f})")
            else:
                st.info(
                    "No co-occurrence data available yet. Translate some text to generate graphs."
                )


def _handle_translate_button(
    translate_button: bool, source_text: str, source_lang: str, target_langs: list[str]
) -> None:
    """Run the translation + graph-generation pipeline when the translate button
    was pressed, and store the results (or errors) into session state."""
    # Handle translation
    if translate_button and source_text and st.session_state["model_available"]:
        # Add user input to chat history
        target_lang_names = ", ".join(
            [get_language_name(lang) for lang in target_langs]
        )

        st.session_state["chat_history"].append(
            {
                "role": "user",
                "content": f"Translate from {get_language_name(source_lang)} to {target_lang_names}:\n\n{source_text}",
            }
        )

        # Perform the translations
        with st.spinner(f"Translating to {target_lang_names}..."):
            try:
                # Reuse the session's cached client instead of rebuilding it here.
                client = get_llm_client()
                if client is None:
                    st.error(
                        "❌ Could not create an LLM client. Please check the provider settings."
                    )
                    return

                # Store overall translation results
                all_graph_data = {}
                cooccurrence_graphs = {}

                successful_translations = {}
                translation_errors = {}

                # Translate to every target language concurrently. The clients are
                # genuinely async, so N languages cost roughly one round-trip rather
                # than N. Only the network calls are parallel - the spaCy work below
                # is CPU-bound and stays sequential.
                async def translate_all():
                    return await asyncio.gather(
                        *(
                            translate_text(client, source_text, source_lang, lang)
                            for lang in target_langs
                        ),
                        return_exceptions=True,
                    )

                results = run_async(translate_all())

                for target_lang, result in zip(target_langs, results):
                    if isinstance(result, Exception):
                        logger.error(f"Translation raised for {target_lang}: {result}")
                        translation_errors[target_lang] = str(result)
                        continue

                    if result.error or "Error code:" in result.text:
                        message = result.error or result.text
                        error_message = handle_translation_error(
                            message, source_lang, target_lang
                        )
                        translation_errors[target_lang] = error_message
                        logger.warning(
                            f"Translation failed for {target_lang}: {message}"
                        )
                        continue

                    successful_translations[target_lang] = result

                # Spanish and Catalan are close enough that a model occasionally
                # returns them the wrong way round. Checked once, after all
                # translations are in.
                if "es" in successful_translations and "ca" in successful_translations:
                    spanish_markers = [
                        "es",
                        "está",
                        "estás",
                        "la",
                        "el",
                        "los",
                        "las",
                        "y",
                        "eres",
                        "tienes",
                    ]
                    catalan_markers = [
                        "és",
                        "està",
                        "estàs",
                        "la",
                        "el",
                        "els",
                        "les",
                        "i",
                        "ets",
                        "tens",
                    ]

                    def count_markers(markers, text):
                        return sum(
                            1 for marker in markers if f" {marker} " in f" {text} "
                        )

                    es_text = successful_translations["es"].text
                    ca_text = successful_translations["ca"].text

                    if count_markers(catalan_markers, es_text) > count_markers(
                        spanish_markers, es_text
                    ) and count_markers(spanish_markers, ca_text) > count_markers(
                        catalan_markers, ca_text
                    ):
                        logger.warning(
                            "Detected possible language mismatch. Swapping Spanish and Catalan translations."
                        )
                        # Swap the whole result, not just the text - the alignment
                        # data was computed alongside whichever text it came with,
                        # so it's mislabeled the same way and needs to move with it.
                        successful_translations["es"], successful_translations["ca"] = (
                            successful_translations["ca"],
                            successful_translations["es"],
                        )

                # Read the co-occurrence settings once rather than per language.
                window_size = st.session_state.get("window_size", 2)
                min_freq = st.session_state.get("min_freq", 1)
                selected_pos = st.session_state.get(
                    "selected_pos", ["NOUN", "VERB", "ADJ"]
                )

                for target_lang, result in successful_translations.items():
                    # Add each translation as a separate message
                    translation_content = (
                        f"{get_language_display(target_lang)}: {result.text.strip()}"
                    )
                    st.session_state["chat_history"].append(
                        {
                            "role": "assistant",
                            "content": translation_content,
                            "target_lang": target_lang,  # Store target language for TTS
                        }
                    )

                    # Generate the graph data for this language (spaCy, no network I/O)
                    all_graph_data[target_lang] = run_async(
                        analyze_translation(
                            source_text,
                            [result.text],
                            [target_lang],
                            [result.alignment],
                        )
                    )

                    # Source text co-occurrence - only needs building once
                    if source_lang not in cooccurrence_graphs:
                        logger.info(
                            f"Building co-occurrence network for {source_lang} with {len(source_text.split())} words"
                        )
                        source_cooccurrence = build_word_cooccurrence_network(
                            source_text,
                            source_lang,
                            window_size=window_size,
                            min_freq=min_freq,
                            include_pos=selected_pos,
                        )
                        if len(source_cooccurrence.nodes()) > 0:
                            logger.info(
                                f"Built source co-occurrence network with {len(source_cooccurrence.nodes())} nodes"
                            )
                            cooccurrence_graphs[source_lang] = source_cooccurrence
                        else:
                            logger.warning(
                                f"Empty co-occurrence network for {source_lang}"
                            )

                    # Target text co-occurrence
                    logger.info(
                        f"Building co-occurrence network for {target_lang} with {len(result.text.split())} words"
                    )
                    target_cooccurrence = build_word_cooccurrence_network(
                        result.text,
                        target_lang,
                        window_size=window_size,
                        min_freq=min_freq,
                        include_pos=selected_pos,
                    )

                    if len(target_cooccurrence.nodes()) > 0:
                        logger.info(
                            f"Built target co-occurrence network with {len(target_cooccurrence.nodes())} nodes"
                        )
                        cooccurrence_graphs[target_lang] = target_cooccurrence
                    else:
                        logger.warning(f"Empty co-occurrence network for {target_lang}")

                # Display any translation errors separately
                if translation_errors:
                    st.error("⚠️ Some translations failed:")
                    for target_lang, error_msg in translation_errors.items():
                        display_translation_error(error_msg, target_lang)

                # Only proceed with successful translations
                if not successful_translations:
                    st.error(
                        "❌ All translations failed. Please check your API key and model selection."
                    )
                    return

                # Store all graph data
                st.session_state["graph_data"] = all_graph_data

                # Store co-occurrence graphs
                st.session_state["cooccurrence_graphs"] = cooccurrence_graphs

                # Store graphs in persistent storage
                if all_graph_data and len(all_graph_data) > 0:
                    for target_lang, graph_data in all_graph_data.items():
                        if graph_data and len(graph_data.get("nodes", [])) > 0:
                            try:
                                graph_id = st.session_state.graph_storage.store_graph(
                                    source_text=source_text,
                                    target_languages=[target_lang],
                                    nodes=graph_data["nodes"],
                                    edges=graph_data["edges"],
                                    user_session=st.session_state.get(
                                        "user_id", "anonymous"
                                    ),
                                    model_used=st.session_state.get(
                                        "llm_provider", "unknown"
                                    ),
                                    translation_text=successful_translations[
                                        target_lang
                                    ].text
                                    if target_lang in successful_translations
                                    else "",
                                )
                                logger.info(
                                    f"Stored graph {graph_id} for {target_lang}"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error storing graph for {target_lang}: {e}"
                                )

                st.success(
                    f"✅ Translation complete! Generated graphs for {len(successful_translations)} languages."
                )

            except Exception as e:
                logger.error(f"Translation error: {str(e)}")
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": f"Error: {str(e)}"}
                )

            # Refresh the UI
            st.rerun()


def main():
    # Initialize session state for help page if not exists
    if "show_help_page" not in st.session_state:
        st.session_state["show_help_page"] = False

    # Check if we should show the help page
    if st.session_state["show_help_page"]:
        show_language_graphs_help()
        return  # Exit the main function early

    # Create a header with visual distinction for dark theme
    st.title("Language Graph - Translation Helper")
    st.markdown(
        """
    Translate text between languages and visualize word relationships in an interactive graph.
    """
    )

    # Initialize session state FIRST
    default_session_state = {
        "llm_provider": settings.llm_provider.value,
        "model_name": settings.current_model,
        "graph_data": None,
        "cooccurrence_graphs": {},
        "chat_history": [],
        "show_debug": False,
        "audio_cache": {},
        "current_view": "semantic",
        "openai_organization": settings.openai_organization,
        "current_word_analysis": None,
        "current_word": None,
        "current_word_lang": None,
        "help_dismissed": False,
        "graph_selection": None,
        "graph_word_analyses": {},
        "graph_expanded_categories": set(),
        "graph_node_positions": {},
    }
    for key, value in default_session_state.items():
        st.session_state.setdefault(key, value)

    # Initialize graph storage
    if "graph_storage" not in st.session_state:
        st.session_state.graph_storage = get_graph_storage()

    # Now get LLM provider and model from properly initialized session state
    llm_provider = st.session_state["llm_provider"]
    model_name = st.session_state["model_name"]

    # Use the cached LLM client, rebuilding it only when the provider, the model or
    # the credentials changed. get_llm_client() already implements exactly this, so
    # the caching lives in one place rather than being repeated at each call site.
    if st.session_state.pop("needs_client_reinit", False):
        st.session_state["llm_client"] = None

    previous = st.session_state.get("llm_client")
    client = get_llm_client()

    if client is not previous:
        # Force the status line to re-check against the new client.
        for key in ("model_status_displayed_once", "last_model_available"):
            st.session_state.pop(key, None)
        logger.info(f"Created new LLM client for {llm_provider}:{model_name}")

    if client is None:
        st.error(
            "⚠️ Could not initialise an LLM client. Check the provider settings in the sidebar."
        )
        st.session_state["model_available"] = False
        return

    # Display model status and check if it's available (with caching)
    model_available = display_model_status(client)

    # Update model availability in session state
    st.session_state["model_available"] = model_available

    source_lang, target_langs = _render_sidebar()

    _render_debug_and_help_sections()

    source_text, translate_button = _render_translation_panel(source_lang, target_langs)

    _render_graph_visualization_tabs()

    _handle_translate_button(translate_button, source_text, source_lang, target_langs)


if __name__ == "__main__":
    main()
