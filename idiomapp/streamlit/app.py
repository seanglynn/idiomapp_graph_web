import os
import re
import json
import logging
import asyncio
import tempfile
import time
import html
from typing import Dict, List, Any, Optional

# Third-party imports
import streamlit as st 
import networkx as nx
from pyvis.network import Network
import spacy
import textacy
from langdetect import detect, LangDetectException
from streamlit.logger import get_logger
from gtts import gTTS
import base64

# Internal imports
from idiomapp.utils.llm_utils import LLMClient, get_available_models, is_ollama_running
from idiomapp.utils.logging_utils import get_logger, get_recent_logs, clear_logs
from idiomapp.config import settings
from idiomapp.utils.nlp_utils import (
    analyze_parts_of_speech,
    split_into_sentences,
    calculate_word_similarity,
    build_word_cooccurrence_network,
    visualize_cooccurrence_network,
    detect_language,
    get_network_stats,
    load_spacy_model,
    get_language_color,
)
from idiomapp.utils.audio_utils import (
    generate_audio,
    clean_text_for_tts,
    extract_translation_content
)

# Set up logging
logger = get_logger("streamlit_app")

# Language mapping dictionary for consistent reference
LANGUAGE_MAP = {
    "en": {"name": "English", "flag": "🇬🇧", "tts_code": "en"},
    "es": {"name": "Spanish", "flag": "🇪🇸", "tts_code": "es"},
    "ca": {"name": "Catalan", "flag": "🏴󠁥󠁳󠁣󠁴󠁿", "tts_code": "es", "tts_note": "(via Spanish TTS)"}
} # TODO: Add more TTS;  Add language detection

# TTS language mapping
TTS_LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'ca': 'Catalan (via Spanish TTS)'
}

# Set up page configuration - use Streamlit's native theming
st.set_page_config(
    page_title="Idiomapp",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add minimal custom styling that works with dark theme
st.markdown("""
<style>
    /* Improve chat message readability for dark theme */
    .chat-message-user {
        padding: 15px !important;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #4361EE;
        white-space: pre-wrap;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .chat-message-ai {
        padding: 15px !important;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #4CC9F0;
        white-space: pre-wrap;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
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
</style>
""", unsafe_allow_html=True)

def render_chat_message(message, role, target_lang=None):
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
                unsafe_allow_html=True
            )
                
        case "assistant" | "ai":
            css_class = "chat-message-ai"
            prefix = "AI"
            
            # Check if this is a translation message (contains language name + flag)
            is_translation = False
            for lang_code, lang_info in LANGUAGE_MAP.items():
                if f"{lang_info['name']} {lang_info['flag']}:" in message:
                    is_translation = True
                    break
                
            # For AI responses, create container with message and TTS button
            formatted_content = process_message_content(message)
            
            # Create a unique key for this message using content hash
            # This ensures unique keys even if the app reruns
            message_hash = hash(message)
            message_key = f"tts_{message_hash}"
            
            # Display message 
            if is_translation:
                st.markdown(
                    f"""<div class='{css_class}'>
                    {formatted_content}
                    </div>""", 
                    unsafe_allow_html=True
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
                            
                            if len(segments) > 1:
                                # The content is after the pattern, might need to clean up
                                # Get segments that follow the pattern
                                for i in range(1, len(segments)):
                                    content = segments[i].strip()
                                    
                                    # If this is not the last segment, need to extract up to the next language
                                    if i < len(segments) - 1:
                                        # Find the next language marker
                                        next_lang_marker = None
                                        for check_lang, check_info in LANGUAGE_MAP.items():
                                            check_pattern = f"\n\n{check_info['name']} {check_info['flag']}:"
                                            if check_pattern in content:
                                                next_lang_marker = content.find(check_pattern)
                                                break
                                        
                                        if next_lang_marker is not None:
                                            content = content[:next_lang_marker].strip()
                                    
                                    # Store the translation content for this language
                                    translation_segments[lang_code] = {
                                        "content": content,
                                        "lang_name": lang_info['name'],
                                        "flag": lang_info['flag']
                                    }
                    
                    # Generate audio players for each language segment
                    if translation_segments:
                        for lang_code, segment in translation_segments.items():
                            try:
                                # Log attempt to generate audio
                                logger.info(f"Generating audio for {lang_code} segment")
                                
                                # Prepare the translation text with the language header
                                translation_text = f"{segment['lang_name']} {segment['flag']}: {segment['content']}"
                                
                                # Generate the audio HTML for this segment
                                audio_html = generate_audio(translation_text, lang_code)
                                
                                # Show the audio player with a clear label
                                st.markdown(f"<p style='margin: 5px 0; color: #CCCCCC; font-size: 12px;'>Audio for {segment['lang_name']} {segment['flag']}</p>", unsafe_allow_html=True)
                                st.markdown(audio_html, unsafe_allow_html=True)
                                logger.info(f"Audio player displayed for {lang_code}")
                            except Exception as e:
                                logger.error(f"Error generating audio for {lang_code}: {str(e)}")
                                st.error(f"Audio error: {str(e)}")
                else:
                    st.warning("Audio unavailable - AI model not ready")
            else:
                st.markdown(
                    f"""<div class='{css_class}'>
                    <strong>{prefix}:</strong> {formatted_content}
                    </div>""", 
                    unsafe_allow_html=True
                )
            
                # Show audio for non-translation messages
                model_available = st.session_state.get("model_available", False)
                if model_available and role == "assistant" and target_lang:
                    try:
                        # Generate audio for this message
                        audio_html = generate_audio(message, target_lang)
                        st.markdown(audio_html, unsafe_allow_html=True)
                    except Exception as e:
                        logger.error(f"Error generating audio: {str(e)}")
                        st.error(f"Audio error: {str(e)}")
                    
        case _:
            # Default handling for unknown roles
            st.markdown(f"**Message ({role}):** {html.escape(message)}", unsafe_allow_html=True)

def process_message_content(message):
    """Process message content to handle code blocks and HTML escaping"""
    content = []
    lines = message.split('\n')
    
    # Simple code block detection
    in_code_block = False
    for line in lines:
        # Use match statement to handle different line content types
        match line.strip():
            case code_start if code_start.startswith('```'):
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
            logger.info(f"Language not specified, detecting language")
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
    # Get status information from the client
    status = client.get_model_status()
    provider = status.get("provider", "unknown")
    model_name = status.get("model_name", "unknown")
    is_available = status.get("available", False)
    
    # Create a container for the status message
    status_container = st.empty()
    
    # Display status message based on availability
    if is_available:
        status_container.success(f"✅ {provider.title()} model '{model_name}' is available")
        return True
    else:
        # Different message based on provider
        if provider == "ollama":
            host = status.get("host", "unknown")
            status_container.error(
                f"⚠️ Ollama model '{model_name}' is not available. "
                f"Host: {host}"
            )
        elif provider == "openai":
            status_container.error(
                f"⚠️ OpenAI model '{model_name}' is not available. "
                f"API key {'is not set' if not status.get('api_key_set') else 'may be invalid'}"
            )
        else:
            status_container.error(f"⚠️ LLM provider '{provider}' is not available.")
            
        return False

async def translate_text(client, source_text, source_lang, target_lang):
    """
    Translate text using the Ollama model.
    
    Args:
        client: The Ollama client
        source_text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        Translated text
    """
    logger.info(f"Translating from {source_lang} to {target_lang}: {source_text}")
    
    # Special prompt for Catalan to ensure proper Catalan and not Spanish
    if target_lang == "ca":
        catalan_note = """
        IMPORTANT: Translate to proper Catalan, NOT Spanish. Catalan uses different words and grammar than Spanish.
        Key differences:
        - Spanish "el" → Catalan "el", but Spanish "la" → Catalan "la"
        - Spanish "¿Cómo estás?" → Catalan "Com estàs?"
        - Spanish "gracias" → Catalan "gràcies"
        - Catalan often uses apostrophes: l'home, d'aigua
        - Catalan has è and ò accents that don't exist in Spanish
        """
        prompt = f"""
        Translate the following text 
            from {LANGUAGE_MAP[source_lang]['name']} 
            to {LANGUAGE_MAP[target_lang]['name']} (Catalan): "{source_text}"
        
        {catalan_note}
        
        Translation ({LANGUAGE_MAP[target_lang]['name']} {LANGUAGE_MAP[target_lang]['flag']} LANG_TAG_{target_lang}): 
        """
        system_prompt = "You are a Catalan language expert fluent in both Catalan and Spanish, aware of their differences. Provide ONLY the translation without any explanation or clarification."
    else:
        prompt = f"""
        Translate the following text 
            from {LANGUAGE_MAP[source_lang]['name']} 
            to {LANGUAGE_MAP[target_lang]['name']}: "{source_text}"
        Translation ({LANGUAGE_MAP[target_lang]['name']} {LANGUAGE_MAP[target_lang]['flag']} LANG_TAG_{target_lang}): 
        """
        system_prompt = "You are a translation expert. Provide ONLY the translation without any explanation or clarification."
    
    try:
        translation = await client.generate_text(prompt, system_prompt=system_prompt)
        
        # Clean up the translation (remove quotes if they exist)
        translation = translation.strip('"\'').strip()
        
        # Check for explanatory text before the actual translation
        # Common patterns where the model explains instead of just translating
        explanation_patterns = [
            # Look for quotes that might contain the actual translation
            r'["""\'\'\'](.*?)[\"""\'\'\']',
            # Look for phrases that indicate the model is about to give the translation
            r'(?:here is|here\'s|the translation is|translated as)[:\s]+(.+)',
            # Look for the target language name followed by a colon/translation
            rf'{LANGUAGE_MAP[target_lang]["name"]}[:\s]+(.+)',
            # Look for lines that start with the actual translation (after removing language tag)
            r'^\s*([^"\'\n].+)$'
        ]
        
        # Try to extract just the translation part using the patterns
        extracted_translation = None
        for pattern in explanation_patterns:
            matches = re.search(pattern, translation, re.IGNORECASE | re.DOTALL)
            if matches:
                extracted_translation = matches.group(1).strip()
                # If we found something promising, use it
                if len(extracted_translation) > 0 and not extracted_translation.startswith("http"):
                    translation = extracted_translation
                    break
        
        # Verify the correct language tag is present and remove it
        expected_tag = f"LANG_TAG_{target_lang}"
        if expected_tag in translation:
            translation = translation.replace(expected_tag, "").strip()
        else:
            logger.warning(f"Language tag {expected_tag} not found in translation response")
        
        # Clean up any remaining quotation marks
        translation = translation.strip('"\'').strip()
        
        # If the translation contains multiple paragraphs or explanation, try to extract just the translation
        if "\n\n" in translation:
            # Try to find the actual translation (usually the shortest paragraph or quoted text)
            paragraphs = [p.strip() for p in translation.split("\n\n") if p.strip()]
            if paragraphs:
                # Use the shortest non-empty paragraph that's not just a language marker
                paragraphs = [p for p in paragraphs if len(p) > 3 and not p.startswith("LANG_TAG")]
                if paragraphs:
                    translation = min(paragraphs, key=len)
        
        logger.info(f"Translation result: {translation}")
        return translation
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return f"Translation error: {str(e)}"

async def analyze_translation(source_text, target_texts, target_langs):
    """
    Analyze translation and generate graph with related words.
    
    Args:
        source_text: Source text to translate
        target_texts: List of translations
        target_langs: List of target languages
        
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
            "translations": target_texts
        }
    }
    
    # Detect source language if not English
    detected_source_lang = detect_language(source_text)
    graph_data["metadata"]["source_lang"] = detected_source_lang
    
    # Split texts into sentences
    source_sentences = split_into_sentences(source_text)
    target_sentences_by_lang = {}
    
    for lang, text in zip(target_langs, target_texts):
        target_sentences_by_lang[lang] = split_into_sentences(text)
    
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
                    sentence_group
                )
    
    # Add cross-sentence relationships if multiple sentences
    if len(source_sentences) > 1:
        add_cross_sentence_relationships(graph_data)
    
    # Add cross-language relationships if multiple target languages
    if len(target_langs) > 1:
        add_cross_language_relationships(graph_data, target_langs)
    
    return graph_data

def process_sentence_pair(source_sentence, target_sentence, source_lang, target_lang, 
                           graph_data, added_nodes, word_relations_cache, sentence_group=""):
    """Process a pair of sentences in different languages and add them to the graph"""
    
    logger.info(f"Processing sentence pair: {source_lang} to {target_lang}")
    
    try:
        # Analyze parts of speech for source and target sentence
        source_pos = analyze_parts_of_speech(source_sentence, source_lang)
        target_pos = analyze_parts_of_speech(target_sentence, target_lang)
        
        # Add source words as nodes
        for word_data in source_pos:
            # Handle string input case (from fallback tokenization)
            if isinstance(word_data, str):
                word = word_data
                pos = "unknown"
                details = ""
            else:
                # Normal dictionary case
                word = word_data["word"]
                pos = word_data["pos"]
                details = word_data.get("details", "")
            
            # Create unique ID for node
            node_id = f"{word}_{source_lang}{sentence_group}"
            
            # Skip if already added
            if node_id in added_nodes:
                continue
                
            # Add node to graph
            graph_data["nodes"].append({
                "id": node_id,
                "label": word,
                "language": source_lang,
                "pos": pos,
                "details": details,
                "node_type": "primary",
                "group": f"{source_lang}{sentence_group}",
                "sentence_group": sentence_group
            })
            added_nodes.add(node_id)
        
        # Add target words as nodes
        for word_data in target_pos:
            # Handle string input case (from fallback tokenization)
            if isinstance(word_data, str):
                word = word_data
                pos = "unknown"
                details = ""
            else:
                # Normal dictionary case
                word = word_data["word"]
                pos = word_data["pos"]
                details = word_data.get("details", "")
            
            # Create unique ID for node
            node_id = f"{word}_{target_lang}{sentence_group}"
            
            # Skip if already added
            if node_id in added_nodes:
                continue
                
            # Add node to graph
            graph_data["nodes"].append({
                "id": node_id,
                "label": word,
                "language": target_lang,
                "pos": pos,
                "details": details,
                "node_type": "primary",
                "group": f"{target_lang}{sentence_group}",
                "sentence_group": sentence_group
            })
            added_nodes.add(node_id)
        
        # Add edges between source and target words based on alignment
        for source_word_data in source_pos:
            # Handle string input case (from fallback tokenization)
            if isinstance(source_word_data, str):
                source_word = source_word_data
                source_pos_val = "unknown"
            else:
                # Normal dictionary case
                source_word = source_word_data["word"]
                source_pos_val = source_word_data["pos"]
            
            source_id = f"{source_word}_{source_lang}{sentence_group}"
            
            # For each target word, establish a direct translation edge if appropriate
            for target_word_data in target_pos:
                # Handle string input case (from fallback tokenization)
                if isinstance(target_word_data, str):
                    target_word = target_word_data
                    target_pos_val = "unknown"
                else:
                    # Normal dictionary case
                    target_word = target_word_data["word"]
                    target_pos_val = target_word_data["pos"]
                
                target_id = f"{target_word}_{target_lang}{sentence_group}"
                
                try:
                    # Use enhanced word similarity analysis
                    similarity_info = calculate_word_similarity(
                        source_word, target_word, source_lang, target_lang)
                    
                    # Basic sanity check for similarity_info structure
                    if not isinstance(similarity_info, dict):
                        logger.error(f"Invalid similarity_info type: {type(similarity_info)} for {source_word}/{target_word}")
                        # Create a default similarity info dictionary
                        similarity_info = {
                            "score": 0.0,
                            "relationship_type": "unknown",
                            "description": "Unable to determine relationship",
                            "linguistic_features": {}
                        }
                        
                    # Safely extract data
                    similarity_score = similarity_info.get("score", 0)
                    relationship_type = similarity_info.get("relationship_type", "unknown")
                    relationship_description = similarity_info.get("description", "Related words")
                    
                    # Only add edges for words that seem related above a threshold
                    if similarity_score > 0.3:
                        # Create a detailed label based on the relationship type
                        if relationship_type == "direct_translation":
                            edge_label = "direct translation"
                        elif relationship_type == "cognate":
                            edge_label = "cognate"
                        elif relationship_type == "semantic_equivalent":
                            edge_label = "equivalent"
                        else:
                            edge_label = relationship_type.replace("_", " ")
                        
                        # Create detailed tooltip with linguistic information
                        linguistic_features = similarity_info.get("linguistic_features", {})
                        pos_match = linguistic_features.get("pos_match", False)
                        is_cognate = linguistic_features.get("is_cognate", False)
                        
                        # Build tooltip with rich information
                        tooltip_parts = [
                            f"{relationship_description}",
                            f"Source: {source_word} ({source_pos_val})" if source_pos_val else f"Source: {source_word}",
                            f"Target: {target_word} ({target_pos_val})" if target_pos_val else f"Target: {target_word}"
                        ]
                        
                        if pos_match:
                            tooltip_parts.append("Same part of speech ✓")
                        
                        if is_cognate:
                            tooltip_parts.append("Historical cognate words ✓")
                        
                        # Add edit distance if available
                        edit_distance = linguistic_features.get("edit_distance")
                        if edit_distance:
                            tooltip_parts.append(f"String similarity: {edit_distance:.2f}")
                        
                        tooltip = "; ".join(tooltip_parts)
                        
                        # Add translation edge
                        graph_data["edges"].append({
                            "from": source_id,
                            "to": target_id,
                            "relation": relationship_type,
                            "strength": similarity_score,
                            "label": edge_label,
                            "description": relationship_description,
                            "title": tooltip  # This will be used for the edge tooltip
                        })
                except Exception as e:
                    logger.error(f"Error processing word pair {source_word}/{target_word}: {type(e).__name__}: {str(e)}")
                    continue
        
        # Process related words for source and target sentences
        try:
            process_related_words(source_pos, source_lang, target_lang, graph_data, 
                                added_nodes, word_relations_cache, sentence_group)
        except Exception as e:
            logger.error(f"Error processing source related words: {type(e).__name__}: {str(e)}")
            
        try:
            process_related_words(target_pos, target_lang, source_lang, graph_data, 
                                added_nodes, word_relations_cache, sentence_group, is_target=True)
        except Exception as e:
            logger.error(f"Error processing target related words: {type(e).__name__}: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in process_sentence_pair: {type(e).__name__}: {str(e)}")
        # Don't re-raise - allow processing to continue with other sentences

def add_cross_sentence_relationships(graph_data):
    """Add relationships between words across different sentences"""
    try:
        # Group nodes by sentence
        sentence_groups = {}
        for node in graph_data["nodes"]:
            group = node.get("sentence_group", "")
            if group not in sentence_groups:
                sentence_groups[group] = []
            sentence_groups[group].append(node)
        
        # Create connections between related words in different sentences
        processed_pairs = set()
        
        for group1, nodes1 in sentence_groups.items():
            for group2, nodes2 in sentence_groups.items():
                # Skip same group or already processed pairs
                if group1 == group2 or (group1, group2) in processed_pairs:
                    continue
                    
                processed_pairs.add((group1, group2))
                processed_pairs.add((group2, group1))
                
                # Find words with same part of speech to connect
                for node1 in nodes1:
                    # Skip non-primary nodes and nodes with unknown pos
                    if node1.get("node_type", "") != "primary":
                        continue
                        
                    pos1 = node1.get("pos", "unknown")
                    if pos1 == "unknown":
                        continue
                        
                    # Find matching POS in the other sentence
                    for node2 in nodes2:
                        # Skip non-primary nodes and nodes with different languages
                        if node2.get("node_type", "") != "primary" or node2.get("language", "") != node1.get("language", ""):
                            continue
                            
                        pos2 = node2.get("pos", "unknown")
                        if pos2 == pos1:
                            try:
                                # Use the enhanced similarity function for words in the same language
                                similarity_info = calculate_word_similarity(
                                    node1["label"], 
                                    node2["label"], 
                                    node1.get("language", "en"), 
                                    node2.get("language", "en")
                                )
                                
                                # Extract similarity score and information
                                if not isinstance(similarity_info, dict):
                                    logger.warning(f"Invalid similarity_info format for cross-sentence comparison")
                                    continue
                                
                                similarity_score = similarity_info.get("score", 0)
                                relationship_type = similarity_info.get("relationship_type", "cross_sentence")
                                description = similarity_info.get("description", f"Related {pos1} words across sentences")
                                
                                # Connect only if there's some similarity or same POS for key types
                                if similarity_score >= 0.3 or pos1 in ["noun", "verb", "adjective"]:
                                    # Create a tooltip with linguistic information
                                    tooltip = f"{description}; Same part of speech: {pos1}; Similarity: {similarity_score:.2f}"
                                    
                                    graph_data["edges"].append({
                                        "from": node1["id"],
                                        "to": node2["id"],
                                        "relation": "cross_sentence",
                                        "strength": max(0.4, similarity_score),
                                        "label": f"related {pos1}",
                                        "description": description,
                                        "title": tooltip,
                                        "color": "#AA44BB",  # Purple for cross-sentence
                                        "dashes": True
                                    })
                            except Exception as e:
                                logger.error(f"Error in cross-sentence processing: {str(e)}")
                                continue
    except Exception as e:
        logger.error(f"Error in add_cross_sentence_relationships: {str(e)}")
        # Don't re-raise, allow processing to continue

async def get_word_translation_map(client, source_words, target_words, source_lang, target_lang):
    """Use LLM to map source words to their translations in target words"""
    if not source_words or not target_words:
        return {}
        
    # Limit to reasonable numbers to avoid overly complex prompts
    max_words = 20
    source_sample = source_words[:max_words]
    target_sample = target_words[:max_words]
    
    # Get language names
    source_lang_name = LANGUAGE_MAP.get(source_lang, {}).get("name", source_lang)
    target_lang_name = LANGUAGE_MAP.get(target_lang, {}).get("name", target_lang)
    
    # Create the prompt
    prompt = f"""Map each {source_lang_name} word to its corresponding {target_lang_name} translation.

{source_lang_name} words: {', '.join(source_sample)}
{target_lang_name} words: {', '.join(target_sample)}

Provide a JSON mapping like this:
{{
  "source_word1": "target_word1",
  "source_word2": "target_word2",
  ...
}}

Only include mappings where you're confident about the translation. It's okay to leave out words.
"""
    
    try:
        # Call the API to get the mapping - use generate_text instead of generate
        response_text = await client.generate_text(
            prompt=prompt,
            system_prompt="You are a translation assistant that provides JSON mappings between words."
        )
        
        # Extract JSON from response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}")
        
        if json_start != -1 and json_end != -1:
            json_text = response_text[json_start:json_end+1]
            
            try:
                # Parse the JSON data
                mapping = json.loads(json_text)
                logger.info(f"Successfully parsed word translation map with {len(mapping)} mappings")
                return mapping
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse word translation map: {str(e)}")
        
        return {}
        
    except Exception as e:
        logger.error(f"Error getting word translation map: {str(e)}")
        return {}

# Add this helper function right before the visualize_translation_graph function
def sanitize_tooltip_text(text):
    """
    Sanitize text for tooltips by removing HTML tags and converting special characters.
    
    Args:
        text: The text to sanitize
        
    Returns:
        Text with HTML tags removed and special characters escaped
    """
    # Remove HTML tags but keep the content between them
    text = re.sub(r'<[^>]*>', '', text)
    
    # Replace < and > with their HTML entities to prevent any remaining tags from being interpreted
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    
    return text

def visualize_translation_graph(graph_data):
    """
    Visualize translation and related words using PyVis.
    
    Args:
        graph_data: Dictionary with nodes and edges
    """
    logger.info(f"Visualizing translation graph")
    
    # Validate graph data
    if not graph_data or not isinstance(graph_data, dict):
        logger.warning("Invalid or empty graph data provided")
        st.warning("No graph data available to visualize.")
        return
    
    # Check if this is an error result
    if graph_data.get("metadata", {}).get("error"):
        logger.warning("Graph data contains error, skipping visualization")
        st.warning("Unable to generate graph due to translation errors.")
        return
    
    # Check if we have any nodes to display
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    
    if not nodes:
        logger.warning("No nodes in graph data")
        st.info("No words to display in the graph. Try translating more text.")
        return
    
    # Filter out any error-related nodes
    valid_nodes = []
    error_keywords = ["translation", "failed", "error", "try", "again"]
    
    for node in nodes:
        node_label = node.get("label", "").lower()
        if not any(keyword in node_label for keyword in error_keywords):
            valid_nodes.append(node)
        else:
            logger.warning(f"Filtering out error-related node: {node.get('label', '')}")
    
    if not valid_nodes:
        logger.warning("All nodes filtered out as error-related")
        st.warning("Unable to generate meaningful graph from the translation results.")
        return
    
    # Update graph data with filtered nodes
    graph_data["nodes"] = valid_nodes
    
    # Filter edges that reference filtered-out nodes
    valid_node_ids = {node["id"] for node in valid_nodes}
    valid_edges = []
    
    for edge in edges:
        if edge.get("from") in valid_node_ids and edge.get("to") in valid_node_ids:
            valid_edges.append(edge)
        else:
            logger.debug(f"Filtering out edge with invalid node references: {edge.get('from')} -> {edge.get('to')}")
    
    graph_data["edges"] = valid_edges
    
    # Create a network with dark mode friendly colors
    net = Network(height="600px", width="100%", bgcolor="#0E1117", font_color="#FAFAFA")
    
    # Set options with dark theme colors and improved physics for force-directed layout
    net.barnes_hut()
    net.set_options("""
    {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "color": {
          "border": "#4361EE",
          "background": "#4CC9F0"
        },
        "font": {
          "size": 16,
          "face": "Arial",
          "color": "#FAFAFA"
        },
        "shadow": true
      },
      "edges": {
        "color": {
          "color": "#AAAAAA",
          "highlight": "#F72585",
          "hover": "#F72585"
        },
        "smooth": {
          "enabled": true,
          "type": "dynamic"
        },
        "shadow": false,
        "width": 2
      },
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -100,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08,
          "damping": 0.4
        },
        "solver": "forceAtlas2Based",
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 100
        }
      },
      "interaction": {
        "dragNodes": true,
        "hideEdgesOnDrag": false,
        "hideNodesOnDrag": false,
        "hover": true,
        "navigationButtons": true,
        "multiselect": true
      }
    }
    """)
    
    # Group colors for different languages and relations
    group_colors = {
        "en": "#4361EE",    # Blue for English
        "es": "#FFD700",    # Yellow for Spanish (was #F72585 Pink)
        "ca": "#FF3B30",    # Red for Catalan (was #7209B7 Purple)
        "en-related": "#90E0EF",  # Light blue for English related
        "es-related": "#FFF1A3",  # Light yellow for Spanish related (was #FF9EC4 Light pink)
        "ca-related": "#FF8C7C"   # Light red for Catalan related (was #C77DFF Light purple)
    }
    
    # Add sentence-specific color variations
    for i in range(1, 10):  # Support up to 10 sentences
        suffix = f"-s{i}"
        # Create slight variations for each sentence group
        group_colors[f"en{suffix}"] = adjust_color(group_colors["en"], i * 10)
        group_colors[f"es{suffix}"] = adjust_color(group_colors["es"], i * 10)
        group_colors[f"ca{suffix}"] = adjust_color(group_colors["ca"], i * 10)
        group_colors[f"en-related{suffix}"] = adjust_color(group_colors["en-related"], i * 10)
        group_colors[f"es-related{suffix}"] = adjust_color(group_colors["es-related"], i * 10) 
        group_colors[f"ca-related{suffix}"] = adjust_color(group_colors["ca-related"], i * 10)
    
    # Language name mapping for node tooltips
        language_names = {
        "en": "English",
        "es": "Spanish",
        "ca": "Catalan"
    }
    
    # POS color accents to highlight parts of speech
    pos_border_colors = {
        "noun": "#FF9500",      # Orange for nouns
        "verb": "#4CD964",      # Green for verbs
        "adjective": "#5AC8FA", # Blue for adjectives
        "adverb": "#FFCC00",    # Yellow for adverbs
        "pronoun": "#FF3B30",   # Red for pronouns
        "preposition": "#FF2D55", # Pink for prepositions
        "conjunction": "#5856D6", # Purple for conjunctions
        "interjection": "#FF9500", # Orange for interjections
        "determiner": "#C7C7CC",  # Gray for determiners
        "unknown": "#4361EE"    # Default blue for unknown
    }
    
    # Set edge label visibility (default to hiding for cleaner visualization)
    show_edge_labels = False
    
    # Add nodes
    for node in graph_data["nodes"]:
        # Determine node language and group
        node_lang = node.get("language", "unknown")
        group = node.get("group", "default")
        
        # Get color based on group
        color = group_colors.get(group, "#4CC9F0")  # Default color if group not found
        
        # Determine node size based on type
        if node["node_type"] == "primary":
            size = 30  # Larger size for primary translation words
        else:
            size = 20  # Smaller size for related words
        
        # Get proper language name for tooltip
        lang_name = language_names.get(node_lang, node_lang.upper())
        
        # Get part of speech and customize border color
        pos = node.get("pos", "unknown")
        details = sanitize_tooltip_text(node.get("details", ""))
        border_color = pos_border_colors.get(pos.lower(), "#4361EE")
        
        # Create enriched tooltip with POS and details
        tooltip = f"{node['label']} ({lang_name}); "
        if pos and pos != "unknown":
            tooltip += f"Part of speech: {pos};"
            if details:
                tooltip += f"Details: {details};"
                
        # Determine if this is part of a multi-sentence translation
        sentence_group = node.get("sentence_group", "")
        if sentence_group and sentence_group.startswith("-s"):
            tooltip += f"Sentence: {sentence_group[2:]};"
        
        # Add the node
        net.add_node(
            node["id"], 
            label=node["label"], 
            title=tooltip,
            color={"background": color, "border": border_color},
            size=size,
            group=group,
            physics=True
        )
    
    # Add edges with relation types
    for edge in graph_data["edges"]:
        # Get relation info
        relation = edge.get("relation", "related")
        strength = edge.get("strength", 0.5)
        
        # Customize edge appearance based on relation type
        if relation == "direct_translation" or relation == "translation":
            # Direct translation edges are thicker and white
            width = 3 * strength
            color = "#FFFFFF"  # White for translation edges
            arrow = False  # No arrow for translations (bidirectional)
            smooth = {"enabled": True, "type": "curvedCW"}
        elif relation == "cognate":
            # Cognate edges are gold colored
            width = 2.5 * strength
            color = "#FFD700"  # Gold for cognates
            arrow = False
            smooth = {"enabled": True, "type": "curvedCW"}
        elif relation == "semantic_equivalent":
            # Semantic equivalents are teal
            width = 2.5 * strength
            color = "#00B8D4"  # Teal for semantic equivalents
            arrow = False
            smooth = {"enabled": True, "type": "curvedCW"}
        elif relation == "cross_sentence":
            # Cross-sentence connections are purple and dashed
            width = 1.5 * strength
            color = edge.get("color", "#AA44BB")  # Purple for cross-sentence
            arrow = True
            smooth = {"enabled": True, "type": "curvedCCW"}
            dashes = True
        elif "semantic_similarity" in relation:
            # Cross-language similarities are orange and dashed
            width = 2 * strength
            color = edge.get("color", "#FFAA00")  # Orange for semantic similarities
            arrow = True
            smooth = {"enabled": True, "type": "curvedCW"}
            dashes = True
        else:
            # Other relations vary by type
            width = 1 + strength
            # Color based on relation type
            relation_colors = {
                "synonym": "#00FF00",      # Green for synonyms
                "antonym": "#FF0000",      # Red for antonyms
                "hypernym": "#FFA500",     # Orange for hypernyms
                "hyponym": "#FFFF00",      # Yellow for hyponyms
                "contextual": "#00FFFF",   # Cyan for contextual
                "common_prefix": "#A0A0FF", # Light blue for common prefix
                "common_suffix": "#FFA0FF", # Light purple for common suffix
                "char_similarity": "#A0FFA0", # Light green for character similarity
                "related_term": "#A0FFFF",   # Light cyan for related terms
                "weak_relation": "#AAAAAA"   # Gray for weak relations
            }
            color = relation_colors.get(relation, "#AAAAAA")
            arrow = True
            smooth = {"enabled": True, "type": "continuous"}
            dashes = False
        
        # Get edge label and tooltip
        label = sanitize_tooltip_text(edge.get("label", relation.replace("_", " ")))
        
        # Use our enhanced tooltip if available, otherwise fall back to basic info
        if "title" in edge:
            title = sanitize_tooltip_text(edge["title"])
        elif "description" in edge:
            title = sanitize_tooltip_text(edge["description"])
        else:
            # Create edge title (tooltip) from basic information
            title = f"{label} ({strength:.2f})"
        
        # Add the edge with appropriate styling
        edge_options = {
            "title": title,
            "label": label if show_edge_labels else "",
            "width": width,
            "color": {"color": color, "highlight": "#F72585"},
            "smooth": smooth,
            "arrows": "to" if arrow else "",
            "dashes": edge.get("dashes", dashes if 'dashes' in locals() else False),
            "physics": True
        }
        
        if relation in ["cross_sentence", "semantic_similarity"] or "dashes" in edge:
            edge_options["dashes"] = True
        
        # Add the edge
        net.add_edge(
            edge["from"], 
            edge["to"], 
            **edge_options
        )
    
    # Create a temporary HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        path = tmpfile.name
        net.save_graph(path)
    
    # Display the graph
    with open(path, 'r', encoding='utf-8') as f:
        html_string = f.read()
    
    # Clean up the temp file
    os.unlink(path)
    
    # Display the network
    st.components.v1.html(html_string, height=600)

def adjust_color(hex_color, amount):
    """Adjust a hex color by lightening or darkening it"""
    # Convert hex to RGB
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Adjust each channel
    adjusted_rgb = []
    for channel in rgb:
        adjusted = channel + amount
        adjusted = max(0, min(255, adjusted))  # Clamp to valid range
        adjusted_rgb.append(adjusted)
    
    # Convert back to hex
    return '#%02x%02x%02x' % tuple(adjusted_rgb)

def add_cross_language_relationships(graph_data, target_langs):
    """Add relationships between words in different languages"""
    logger.info(f"Adding cross-language relationships for {len(target_langs)} languages")
    
    try:
        # Group nodes by language and POS
        nodes_by_lang_pos = {}
        
        # Initialize for each language
        for lang in target_langs:
            nodes_by_lang_pos[lang] = {}
        
        # Process all nodes
        for node in graph_data["nodes"]:
            # Skip related words
            node_type = node.get("node_type", "")
            if node_type != "primary":
                continue
                
            # Get language and POS
            lang = node.get("language", "")
            pos = node.get("pos", "unknown")
            
            # Skip nodes with unspecified language
            if lang not in nodes_by_lang_pos:
                continue
                
            # Skip nodes with unknown POS
            if pos == "unknown":
                continue
                
            # Add node to the appropriate group
            if pos not in nodes_by_lang_pos[lang]:
                nodes_by_lang_pos[lang][pos] = []
                
            nodes_by_lang_pos[lang][pos].append(node)
        
        # Create connections between related words in different languages
        processed_pairs = set()
        
        for lang1 in target_langs:
            for lang2 in target_langs:
                if lang1 == lang2 or (lang1, lang2) in processed_pairs:
                    continue
                    
                processed_pairs.add((lang1, lang2))
                
                # For each part of speech, find potential matches
                for pos in set(nodes_by_lang_pos[lang1].keys()) & set(nodes_by_lang_pos[lang2].keys()):
                    for node1 in nodes_by_lang_pos[lang1][pos]:
                        for node2 in nodes_by_lang_pos[lang2][pos]:
                            try:
                                # If the nodes are in same sentence group, good candidate for connection
                                same_sentence = node1.get("sentence_group", "") == node2.get("sentence_group", "")
                                
                                # Use enhanced similarity calculation
                                similarity_info = calculate_word_similarity(
                                    node1["label"], node2["label"], lang1, lang2)
                                
                                # Get similarity score and relationship data
                                if not isinstance(similarity_info, dict):
                                    logger.warning(f"Invalid similarity_info format for cross-language comparison")
                                    continue
                                
                                similarity_score = similarity_info.get("score", 0)
                                relationship_type = similarity_info.get("relationship_type", "cross_language")
                                description = similarity_info.get("description", "Related words across languages")
                                
                                # Connect nodes if semantically similar
                                min_threshold = 0.2 if same_sentence else 0.4
                                
                                if similarity_score >= min_threshold:
                                    # Create a tooltip with translation information
                                    tooltip = f"{description}; {node1['label']} ({lang1}) ↔ {node2['label']} ({lang2})"
                                    
                                    if same_sentence:
                                        tooltip += "; Same sentence ✓"
                                        
                                    # Add cross-language edge
                                    graph_data["edges"].append({
                                        "from": node1["id"],
                                        "to": node2["id"],
                                        "relation": "cross_language",
                                        "strength": similarity_score,
                                        "label": relationship_type.replace("_", " "),
                                        "description": description,
                                        "title": tooltip,
                                        "color": "#4CC9F0",  # Blue for cross-language
                                        "width": 2,
                                        "dashes": True
                                    })
                            except Exception as e:
                                logger.error(f"Error processing cross-language pair: {str(e)}")
                                continue
    except Exception as e:
        logger.error(f"Error in add_cross_language_relationships: {str(e)}")
        # Don't re-raise - allow processing to continue

def process_related_words(words_data, source_lang, target_lang, graph_data, 
                     added_nodes, word_relations_cache, sentence_group="", is_target=False):
    """
    Process related words for a list of words and add them to the graph.
    
    Args:
        words_data: List of word data dictionaries with parts of speech
        source_lang: Source language code
        target_lang: Target language code
        graph_data: The graph data structure to update
        added_nodes: Set of already added node IDs to avoid duplicates
        word_relations_cache: Cache of word relations to avoid duplicates
        sentence_group: Optional sentence group identifier
        is_target: Whether these are target language words
    """
    # Skip if no words to process
    if not words_data:
        return
    
    # Get language for these words (either source or target lang)
    lang = target_lang if is_target else source_lang
    
    # For now, we'll use a simple predefined set of related words for common categories
    # In a real implementation, this would be replaced with a call to a language model
    
    # Process each word
    for word_data in words_data:
        # Handle string input case (from fallback tokenization)
        if isinstance(word_data, str):
            word = word_data
            pos = "unknown"
        else:
            # Normal dictionary case
            word = word_data["word"]
            pos = word_data["pos"]
        
        # Skip words that don't have a clear POS
        if pos == "unknown":
            continue
            
        # Create a simple cache key
        cache_key = f"{word}_{lang}_{pos}"
        
        # Skip if we've already processed this word
        if cache_key in word_relations_cache:
            related_words = word_relations_cache[cache_key]
        else:
            # Generate related words (in a real implementation, this would call a language model)
            related_words = generate_simple_related_words(word, pos, lang)
            word_relations_cache[cache_key] = related_words
        
        # Skip if no related words found
        if not related_words:
            continue
            
        # Add related word nodes
        word_id = f"{word}_{lang}{sentence_group}"
        
        for related_word, relation_type in related_words:
            # Create a unique ID for this related word
            related_id = f"{related_word}_{lang}-related{sentence_group}"
            
            # Skip if already added
            if related_id in added_nodes:
                continue
                
            # Add node for related word
            graph_data["nodes"].append({
                "id": related_id,
                "label": related_word,
                "language": lang,
                "pos": pos,  # Assume same POS as original word
                "details": relation_type,
                "node_type": "related",
                "group": f"{lang}-related{sentence_group}",
                "sentence_group": sentence_group
            })
            added_nodes.add(related_id)
            
            # Add edge from original word to related word
            # Customize strength based on relation type
            relation_strengths = {
                "synonym": 0.9,
                "antonym": 0.7,
                "hypernym": 0.6,
                "hyponym": 0.6,
                "contextual": 0.5
            }
            
            strength = relation_strengths.get(relation_type, 0.5)
            
            graph_data["edges"].append({
                "from": word_id,
                "to": related_id,
                "relation": relation_type,
                "strength": strength,
                "label": relation_type
            })

# Improve
def generate_simple_related_words(word, pos, language):
    """Generate some simple related words for common words in various languages"""
    # This is a very simplified approach for demonstration
    # In a real implementation, this would call a language model API
    
    # Some common word relationships in English
    if language == "en":
        if word == "good":
            return [
                ("excellent", "synonym"),
                ("bad", "antonym"),
                ("quality", "hypernym"),
                ("great", "synonym"),
                ("rating", "contextual")
            ]
        elif word == "happy":
            return [
                ("joyful", "synonym"),
                ("sad", "antonym"),
                ("emotion", "hypernym"),
                ("ecstatic", "hyponym"),
                ("birthday", "contextual")
            ]
    
    # Some common word relationships in Spanish
    elif language == "es":
        if word == "bueno":
            return [
                ("excelente", "synonym"),
                ("malo", "antonym"),
                ("calidad", "hypernym"),
                ("genial", "synonym"),
                ("valoración", "contextual")
            ]
        elif word == "feliz":
            return [
                ("alegre", "synonym"),
                ("triste", "antonym"),
                ("emoción", "hypernym"),
                ("extático", "hyponym"),
                ("cumpleaños", "contextual")
            ]
    
    # Some common word relationships in Catalan
    elif language == "ca":
        if word == "bo":
            return [
                ("excel·lent", "synonym"),
                ("dolent", "antonym"),
                ("qualitat", "hypernym"),
                ("genial", "synonym"),
                ("valoració", "contextual")
            ]
        elif word == "feliç":
            return [
                ("content", "synonym"),
                ("trist", "antonym"),
                ("emoció", "hypernym"),
                ("extàtic", "hyponym"),
                ("aniversari", "contextual")
            ]
    
    # Default: return empty list if no predefined relations
    return []

def merge_language_graphs(graph_data_dict):
    """Merge multiple language graphs into a single graph with cross-language connections"""
    import copy
    
    if not graph_data_dict or len(graph_data_dict) == 0:
        return None
    
    # Create a new graph combining all nodes and edges
    merged_graph = {
        "nodes": [],
        "edges": [],
        "metadata": {
            "source_lang": next(iter(graph_data_dict.values()))["metadata"]["source_lang"],
            "target_langs": [],
            "source_text": next(iter(graph_data_dict.values()))["metadata"]["source_text"],
            "translations": {}
        }
    }
    
    # Track all nodes we've added to avoid duplicates
    added_nodes = set()
    
    # Add nodes and edges from each language graph
    for lang, graph in graph_data_dict.items():
        # Update metadata
        merged_graph["metadata"]["target_langs"].append(lang)
        if "translations" in graph["metadata"]:
            merged_graph["metadata"]["translations"][lang] = graph["metadata"]["translations"]
        
        # Add nodes
        for node in graph["nodes"]:
            if node["id"] not in added_nodes:
                merged_graph["nodes"].append(copy.deepcopy(node))
                added_nodes.add(node["id"])
        
        # Add edges
        for edge in graph["edges"]:
            merged_graph["edges"].append(copy.deepcopy(edge))
    
    # Now add cross-language relationships
    target_langs = merged_graph["metadata"]["target_langs"]
    if len(target_langs) > 1:
        add_cross_language_relationships(merged_graph, target_langs)
    
    logger.info(f"Merged {len(graph_data_dict)} language graphs into a single graph")
    return merged_graph

def visualize_cooccurrence_network(graph, lang_code=None):
    """
    Visualize a word co-occurrence network using Pyvis.
    
    Args:
        graph: networkx.Graph with word co-occurrence network
        lang_code: Language code for color coding (optional)
    
    Returns:
        Pyvis network visualization HTML
    """
    try:
        # Create a network with dark mode friendly colors
        net = Network(height="600px", width="100%", bgcolor="#0E1117", font_color="#FAFAFA")
        
        # Set options for visualization
        net.barnes_hut()
        net.set_options("""
        {
          "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "color": {
              "border": "#4361EE",
              "background": "#4CC9F0"
            },
            "font": {
              "size": 16,
              "face": "Arial",
              "color": "#FAFAFA"
            },
            "shadow": true
          },
          "edges": {
            "color": {
              "color": "#AAAAAA",
              "highlight": "#F72585",
              "hover": "#F72585"
            },
            "smooth": {
              "enabled": true,
              "type": "dynamic"
            },
            "shadow": false,
            "width": 2
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -5000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04
            },
            "stabilization": {
              "iterations": 1000
            }
          },
          "interaction": {
            "dragNodes": true,
            "hideEdgesOnDrag": false,
            "hideNodesOnDrag": false,
            "hover": true
          }
        }
        """)
        
        # TODO: Dynamic - Language-specific colors - now using shared function from nlp_utils
        lang_colors = {
            None: "#4CC9F0"   # Default light blue if no language specified
        }
        
        # Check if graph has nodes
        if len(graph.nodes()) == 0:
            logger.warning("Co-occurrence graph is empty - no nodes to display")
            return "<div class='alert alert-warning'>No co-occurrence data available for this text. Try a longer text or adjust co-occurrence settings.</div>"
        
        # Convert node IDs to strings to ensure compatibility with Pyvis
        nodes_to_add = []
        for node in graph.nodes():
            # Make sure node is a string
            node_id = str(node)
            
            # Get node weight (degree in the graph)
            size = 20 + (graph.degree(node) * 3)
            
            # Create tooltip with node information
            tooltip = f"Word: {sanitize_tooltip_text(str(node))}; Co-occurrences: {graph.degree(node)}"
            
            # Get color for the language from our shared function in nlp_utils
            node_color = get_language_color(lang_code) if lang_code else "#4CC9F0"
            
            # Add to our list of nodes to add
            nodes_to_add.append((node_id, {
                "label": node_id,
                "title": tooltip,
                "color": {"background": node_color, "border": "#4361EE"},
                "size": size
            }))
        
        # Add all nodes
        for node_id, node_data in nodes_to_add:
            net.add_node(
                node_id,
                label=node_data["label"],
                title=node_data["title"],
                color=node_data["color"],
                size=node_data["size"]
            )
        
        # Add edges with weights - ensure string conversion for source/target
        for source, target, data in graph.edges(data=True):
            # Convert source and target to strings
            source_id = str(source)
            target_id = str(target)
            
            # Get edge weight (count or other measure)
            weight = data.get('weight', 1)
            width = 1 + (weight / 2)  # Scale width based on weight
            
            # Create edge tooltip (safely)
            edge_tooltip = f"Co-occurrence: {weight}"
            
            # Add the edge
            net.add_edge(
                source_id,
                target_id,
                title=edge_tooltip,
                width=width,
                color="#FFFFFF" if weight > 2 else "#AAAAAA"
            )
        
        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
            path = tmpfile.name
            net.save_graph(path)
        
        # Read the HTML
        with open(path, 'r', encoding='utf-8') as f:
            html_string = f.read()
        
        # Clean up the temp file
        os.unlink(path)
        
        return html_string
        
    except Exception as e:
        logger.error(f"Error visualizing co-occurrence network: {str(e)}")
        return f"<div class='alert alert-danger'>Error creating visualization: {str(e)}</div>"

def show_language_graphs_help():
    """
    Display the language graphs help page using markdown files from the docs/ folder.
    """
    st.title("Understanding Language Graphs")
    
    # Path to docs directory
    docs_path = "docs/"
    
    # Find all markdown files recursively
    import os
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
                file_names = [os.path.basename(f).replace('.md', '') for f in md_files]
                selected_index = st.selectbox(
                    "Select documentation:",
                    range(len(file_names)),
                    format_func=lambda i: file_names[i]
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

def display_nlp_legend():
    """Display a legend explaining NLP terminology and color coding used in the graph."""
    
    # Use the same colors as in the visualization
    pos_border_colors = {
        "noun": "#FF5733",  # Reddish-orange for nouns
        "verb": "#33A1FF",  # Blue for verbs
        "adjective": "#33FF57",  # Green for adjectives
        "adverb": "#FF33A1",  # Pink for adverbs
        "pronoun": "#A133FF",  # Purple for pronouns
        "determiner": "#FFD133",  # Yellow for determiners
        "preposition": "#33FFD1",  # Cyan for prepositions
        "conjunction": "#D133FF",  # Magenta for conjunctions
        "unknown": "#4361EE"  # Default blue
    }
    
    # Create expandable section for the legend
    with st.expander("📖 **NLP Graph Legend - Understanding the Visualization**", expanded=False):
        st.markdown("""
        ### Node Colors and Types
        
        Nodes in the graph represent words, and their colors indicate the language and part of speech.
        The border color of a node indicates its part of speech (POS):
        """)
        
        # Part of speech explanations with colored borders to match the graph
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            - <span style="border: 2px solid {pos_border_colors['noun']}; padding: 2px 5px; border-radius: 4px;">NOUN/PROPN</span>: Nouns (person, place, thing) and proper nouns (names, locations)
            - <span style="border: 2px solid {pos_border_colors['verb']}; padding: 2px 5px; border-radius: 4px;">VERB</span>: Action words or states of being
            - <span style="border: 2px solid {pos_border_colors['adjective']}; padding: 2px 5px; border-radius: 4px;">ADJ</span>: Words that describe nouns
            - <span style="border: 2px solid {pos_border_colors['adverb']}; padding: 2px 5px; border-radius: 4px;">ADV</span>: Words that modify verbs, adjectives, or other adverbs
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            - <span style="border: 2px solid {pos_border_colors['pronoun']}; padding: 2px 5px; border-radius: 4px;">PRON</span>: Words that substitute for nouns (I, you, he, she)
            - <span style="border: 2px solid {pos_border_colors['determiner']}; padding: 2px 5px; border-radius: 4px;">DET</span>: Articles and determiners (the, a, this, that)
            - <span style="border: 2px solid {pos_border_colors['preposition']}; padding: 2px 5px; border-radius: 4px;">ADP</span>: Prepositions (in, on, at, by)
            - <span style="border: 2px solid {pos_border_colors['conjunction']}; padding: 2px 5px; border-radius: 4px;">CONJ</span>: Words that connect phrases or clauses
            """, unsafe_allow_html=True)
            
        st.markdown("""
        ### Dependency Relations
        
        The tooltip shows dependency relations between words:
        
        - **ROOT**: The main word in a sentence, usually the main verb
        - **nsubj**: Nominal subject - the subject of a clause
        - **dobj/obj**: Direct object - the object directly affected by the verb
        - **amod**: Adjectival modifier - an adjective that modifies a noun
        - **det**: Determiner - an article or determiner that modifies a noun
        
        ### Entity Types
        
        Some words are recognized as named entities, which are specific real-world objects:
        
        - **PERSON**: Names of people
        - **LOC/GPE**: Locations or geopolitical entities (countries, cities)
        - **ORG**: Organizations, companies, institutions
        - **MISC**: Miscellaneous entities, including nationalities, languages
        - **DATE/TIME**: Calendar and time references
        
        ### Edge Types and Strength
        
        - Edges show relationships between words
        - Thicker edges indicate stronger relationships
        - Dashed edges indicate cross-language or cross-sentence connections
        """)

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
    st.markdown("""
    Translate text between languages and visualize word relationships in an interactive graph.
    """)
    
    # Get LLM provider and model from settings
    llm_provider = settings.llm_provider.value
    model_name = settings.current_model
    
    # Use the factory method to create the appropriate client
    client = LLMClient.create(provider=llm_provider, model_name=model_name)
    
    # Display model status and check if it's available
    model_available = display_model_status(client)
    
    # Initialize session state
    if "translations" not in st.session_state:
        st.session_state["translations"] = {}
    if "graph_data" not in st.session_state:
        st.session_state["graph_data"] = None
    if "cooccurrence_graphs" not in st.session_state:
        st.session_state["cooccurrence_graphs"] = {}
    if "llm_provider" not in st.session_state:
        st.session_state["llm_provider"] = llm_provider
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = model_name
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "show_debug" not in st.session_state:
        st.session_state["show_debug"] = False
    if "audio_cache" not in st.session_state:
        st.session_state["audio_cache"] = {}
    if "model_available" not in st.session_state:
        st.session_state["model_available"] = model_available
    if "current_view" not in st.session_state:
        st.session_state["current_view"] = "semantic"
    else:
        st.session_state["model_available"] = model_available
    
    # Add a sidebar with translation settings
    with st.sidebar:
        st.header("Translation Settings")
        
        # Help button at the top of the sidebar with improved styling
        st.markdown("""
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
        """, unsafe_allow_html=True)
        
        # Hidden button that will be triggered by the custom HTML button
        if st.button("📚 Understanding Language Graphs", key="doc_button_hidden", help="Learn about language graphs and how to use them", use_container_width=True):
            st.session_state["show_help_page"] = True
            st.rerun()
        
        # Add a visual separator
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # LLM Provider selection
        st.subheader("LLM Provider")
        provider_options = ["ollama", "openai"]
        selected_provider = st.selectbox(
            "AI Provider",
            provider_options,
            index=provider_options.index(st.session_state["llm_provider"]) if st.session_state["llm_provider"] in provider_options else 0,
            format_func=lambda x: x.title(),  # Capitalize for display
            help="Select the AI provider to use for translation"
        )
        
        # Show provider-specific options
        if selected_provider == "ollama":
            available_models = get_available_models() if st.session_state["llm_provider"] == "ollama" else ["llama3.2:latest"]
            model_name = st.selectbox(
                f"Ollama Model {'' if st.session_state['model_available'] and st.session_state['llm_provider'] == 'ollama' else '⚠️'}",
                available_models,
                index=available_models.index(st.session_state["model_name"]) if st.session_state["model_name"] in available_models else 0,
                help="Select the Ollama model to use for translation",
                disabled=not (st.session_state["model_available"] and st.session_state["llm_provider"] == "ollama")
            )
        elif selected_provider == "openai":
            openai_models = settings.openai_models_list
            model_name = st.selectbox(
                f"OpenAI Model {'' if st.session_state['model_available'] and st.session_state['llm_provider'] == 'openai' else '⚠️'}",
                openai_models,
                index=openai_models.index(st.session_state["model_name"]) if st.session_state["model_name"] in openai_models else 0,
                help="Select the OpenAI model to use for translation",
                disabled=not (st.session_state["model_available"] and st.session_state["llm_provider"] == "openai")
            )
            
            # Show API key input if OpenAI is selected
            openai_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=settings.openai_api_key,
                help="Enter your OpenAI API key to use ChatGPT"
            )
            
            # Update session state and environment if API key changes
            if openai_api_key != settings.openai_api_key:
                # Update the global config settings temporarily 
                # Note: This is a runtime update, the config object itself doesn't persist changes
                os.environ["OPENAI_API_KEY"] = openai_api_key
                if openai_api_key:
                    st.success("API key updated. Reinitializing client...")
                    # Force reinitialization of client with new API key
                    st.session_state["model_available"] = False
                    st.rerun()
        
        # Update client if provider or model changes
        if selected_provider != st.session_state["llm_provider"] or model_name != st.session_state["model_name"]:
            st.session_state["llm_provider"] = selected_provider
            st.session_state["model_name"] = model_name
            # Update environment variables for runtime changes
            os.environ["LLM_PROVIDER"] = selected_provider
            if selected_provider == "ollama":
                os.environ["DEFAULT_MODEL"] = model_name
            else:
                os.environ["OPENAI_MODEL"] = model_name
            # Force reinitialization of client
            st.info("Provider or model changed. Reinitializing client...")
            st.session_state["model_available"] = False
            st.rerun()
        
        # Language selection
        st.header("Language Settings")
        source_lang = st.selectbox(
            "Source Language",
            settings.supported_languages_list,
            index=settings.supported_languages_list.index(settings.default_source_language) if settings.default_source_language in settings.supported_languages_list else 0,
            format_func=lambda x: f"{LANGUAGE_MAP[x]['name']} {LANGUAGE_MAP[x]['flag']}",
            help="Select the source language"
        )
        
        # Multiple target languages selection
        target_langs = st.multiselect(
            "Target Languages",
            settings.supported_languages_list,
            default=settings.default_target_languages_list,
            format_func=lambda x: f"{LANGUAGE_MAP[x]['name']} {LANGUAGE_MAP[x]['flag']}",
            help="Select one or more target languages"
        )
        
        # Ensure at least one target language is selected
        if not target_langs:
            st.warning("Please select at least one target language")
            target_langs = settings.default_target_languages_list[:1]  # Use first default target language
        
        # Model selection with status indication
            available_models = get_available_models()
            model_name = st.selectbox(
            f"Translation Model {'' if st.session_state['model_available'] else '⚠️'}",
                available_models,
                index=available_models.index(st.session_state["model_name"]) if st.session_state["model_name"] in available_models else 0,
            help="Select the AI model to use for translation",
            disabled=not st.session_state["model_available"]
        )
        
        if not st.session_state["model_available"]:
            st.error("⚠️ Selected model is not available. AI features are disabled.")
            st.info("Check the model status above and make sure it's properly installed.")
        else:
            st.success("✅ AI model is ready to use")
            
            st.session_state["model_name"] = model_name
            
        # Switch for visualization type
        st.header("Visualization Settings")
        view_options = ["Semantic Graph", "Co-occurrence Network"]
        selected_view = st.radio("Analysis View", view_options)
        
        # Map selection to internal state
        st.session_state["current_view"] = "semantic" if selected_view == "Semantic Graph" else "cooccurrence"
        
        # Co-occurrence settings (only shown when that view is selected)
        if st.session_state["current_view"] == "cooccurrence":
            st.subheader("Co-occurrence Settings")
            
            # Window size for co-occurrence
            window_size = st.slider(
                "Window Size", 
                min_value=1, 
                max_value=5, 
                value=settings.default_window_size,
                help="Number of words to consider for co-occurrence (larger = more connections)"
            )
            st.session_state["window_size"] = window_size
            
            # Minimum frequency for words
            min_freq = st.slider(
                "Minimum Word Frequency", 
                min_value=1, 
                max_value=5, 
                value=settings.default_min_frequency,
                help="Minimum times a word must appear to be included"
            )
            st.session_state["min_freq"] = min_freq
            
            # POS tag selection
            pos_options = [
                ("Nouns", "NOUN"), 
                ("Verbs", "VERB"), 
                ("Adjectives", "ADJ"), 
                ("Adverbs", "ADV"),
                ("Proper Nouns", "PROPN")
            ]
            selected_pos = st.multiselect(
                "Part of Speech Filter",
                options=[tag for _, tag in pos_options],
                default=settings.default_pos_filter_list,
                format_func=lambda x: next((name for name, tag in pos_options if tag == x), x),
                help="Filter words by part of speech"
            )
            st.session_state["selected_pos"] = selected_pos
        
        # Debug toggle
            st.session_state["show_debug"] = st.checkbox(
                "Show Debug Logs", 
                value=st.session_state["show_debug"],
            help="Show detailed logs of translation processing"
            )
    
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
    
    # Create a two-column layout for the main content area
    main_col, chat_col = st.columns([3, 2])
    
    with main_col:
        # Show the visualization based on the selected view
        if st.session_state["current_view"] == "semantic" and st.session_state["graph_data"]:
            # Add header for the graph
            st.subheader("📊 Semantic Network Analysis")
            
            # Display controls for the graph
            with st.expander("Graph Options", expanded=False):
                # Add option to choose which languages to display
                available_langs = list(st.session_state["graph_data"].keys())
                
                # Create columns for options
                opt_col1, opt_col2 = st.columns([1, 1])
                
                with opt_col1:
                    # Option to merge all graphs into one comprehensive view
                    merge_graphs = st.checkbox("Merge all language graphs", value=True, 
                                             help="Show connections between different languages")
                
                with opt_col2:
                    # Filter for minimum relationship strength
                    min_strength = st.slider("Minimum relationship strength", 
                                           min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                                           help="Only show strong relationships above this threshold")
            
            # Display the graph based on selection
            if merge_graphs and len(available_langs) > 1:
                # Create a merged graph with cross-language connections
                merged_graph = merge_language_graphs(st.session_state["graph_data"])
                
                # Filter edges by strength
                if min_strength > 0:
                    filtered_edges = [edge for edge in merged_graph["edges"] 
                                     if edge.get("strength", 1.0) >= min_strength]
                    merged_graph["edges"] = filtered_edges
                
                st.markdown(f"**Combined graph showing relationships between {', '.join(available_langs)}**")
                visualize_translation_graph(merged_graph)
            elif available_langs:
                # Let user choose which language graph to show
                selected_lang = st.selectbox("Select language graph", 
                                           options=available_langs,
                                           format_func=lambda x: f"{LANGUAGE_MAP[x]['name']} {LANGUAGE_MAP[x]['flag']}")
                
                # Filter edges by strength if needed
                graph_data = st.session_state["graph_data"][selected_lang]
                if min_strength > 0:
                    filtered_edges = [edge for edge in graph_data["edges"] 
                                     if edge.get("strength", 1.0) >= min_strength]
                    # Create a copy of the graph with filtered edges
                    filtered_graph = {
                        "nodes": graph_data["nodes"],
                        "edges": filtered_edges,
                        "metadata": graph_data.get("metadata", {})
                    }
                    graph_data = filtered_graph
                
                # Display the selected graph
                st.markdown(f"**Semantic network for {LANGUAGE_MAP[selected_lang]['name']} {LANGUAGE_MAP[selected_lang]['flag']}**")
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
                    st.markdown("⚪ **White** - Direct translation")
                    st.markdown("🔶 **Gold** - Cognates (similar words)")
                    st.markdown("🔹 **Teal** - Semantic equivalents")
                    st.markdown("🟢 **Green** - Synonyms")
                    st.markdown("🔴 **Red** - Antonyms")
                    st.markdown("🟠 **Orange** - Hypernyms (broader terms)")
                    st.markdown("🟡 **Yellow** - Hyponyms (specific terms)")
                    st.markdown("🔵 **Cyan** - Contextual relation")
                    st.markdown("🟣 **Purple dashed** - Cross-sentence relation")
                    st.markdown("🟠 **Orange dashed** - Cross-language similarity")
                    st.markdown("🔷 **Light blue** - Common prefix")
                    st.markdown("🔮 **Light purple** - Common suffix")
                    
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
        
        # Show co-occurrence networks if that view is selected 
        elif st.session_state["current_view"] == "cooccurrence" and "cooccurrence_graphs" in st.session_state and st.session_state["cooccurrence_graphs"]:
            # Add header for the co-occurrence network
            st.subheader("📊 Word Co-occurrence Network")
            
            available_langs = list(st.session_state["cooccurrence_graphs"].keys())
            
            # Let user choose which language graph to show
            if available_langs:
                selected_lang = st.selectbox(
                    "Select language", 
                    options=available_langs,
                    format_func=lambda x: f"{LANGUAGE_MAP[x]['name']} {LANGUAGE_MAP[x]['flag']}"
                )
                
                # Show information about this analysis
                st.markdown(f"""
                Showing word co-occurrence network for **{LANGUAGE_MAP[selected_lang]['name']} {LANGUAGE_MAP[selected_lang]['flag']}**
                
                * Nodes represent individual words
                * Larger nodes appear more frequently
                * Edges show words that appear close together in the text
                * Thicker edges indicate words that co-occur more frequently
                """)
                
                # Display the co-occurrence network
                graph = st.session_state["cooccurrence_graphs"][selected_lang]
                html_string = visualize_cooccurrence_network(graph, selected_lang)
                st.components.v1.html(html_string, height=610)
                
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
                    degree_cent = nx.degree_centrality(graph)
                    betweenness_cent = nx.betweenness_centrality(graph)
                    
                    # Get top words by degree centrality
                    top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
                    top_betweenness = sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    cent_cols = st.columns(2)
                    with cent_cols[0]:
                        st.markdown("**Most Connected Words**")
                        for word, score in top_degree:
                            st.markdown(f"• **{word}** ({score:.3f})")
                    
                    with cent_cols[1]:
                        st.markdown("**Bridge Words**")
                        for word, score in top_betweenness:
                            st.markdown(f"• **{word}** ({score:.3f})")
                
        # Show placeholder if no data available
        elif not st.session_state["graph_data"] and not st.session_state.get("cooccurrence_graphs"):
            st.info("No graph data available yet. Translate some text to generate graphs.")
            
        # Display a helpful guide if no translation has been made yet
        if not st.session_state["chat_history"]:
            st.info(f"""
            ### How to use the Translation Helper
            1. Select source language in the sidebar
            2. Select one or more target languages in the sidebar
            3. Type your text in the input box
            4. Click "Translate" to see the translations
            5. Explore the word relationships in either the semantic graph or co-occurrence network views
            
            **Example**: Try translating "Do you know my country?" from {LANGUAGE_MAP["en"]["name"]} to both {LANGUAGE_MAP["es"]["name"]} and {LANGUAGE_MAP["ca"]["name"]}
            """)
        
        # Translation input section below the graph
        st.subheader("💬 Translation Input")
        
        # Translation input
        source_text = st.text_area(
            f"Enter text in {LANGUAGE_MAP[source_lang]['name']}:",
            height=100,
            placeholder=f"Type your text in {LANGUAGE_MAP[source_lang]['name']} here...",
            disabled=not st.session_state["model_available"]
        )
        
        btn_col1, btn_col2 = st.columns([1, 1])
        
        with btn_col1:
            # Create button text based on number of target languages
            if len(target_langs) == 1:
                button_text = f"🔄 Translate to {LANGUAGE_MAP[target_langs[0]]['name']}"
            else:
                button_text = f"🔄 Translate to {len(target_langs)} languages"
                
            translate_button = st.button(
                button_text, 
                use_container_width=True,
                disabled=not st.session_state["model_available"] or not source_text
            )
        
        with btn_col2:
            clear_button = st.button("🗑️ Clear History", use_container_width=True)
            if clear_button:
                st.session_state["chat_history"] = []
                st.session_state["translations"] = {}
                st.session_state["graph_data"] = None
                st.session_state["cooccurrence_graphs"] = {}
                st.success("History cleared!")
                st.rerun()
    
    # Chat history on the right
    with chat_col:
        st.subheader("💬 Translation Chat")
        
        # Create custom CSS for styling the chat
        st.markdown("""
        <style>
        .chat-message-user, .chat-message-ai {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
        }
        .chat-message-user {
            background-color: rgba(67, 97, 238, 0.1);
            border-left: 3px solid #4361EE;
        }
        .chat-message-ai {
            background-color: rgba(76, 201, 240, 0.1);
            border-left: 3px solid #4CC9F0;
        }
        .stChatContainer {
            height: 500px;
            overflow-y: auto;
            border: 1px solid #4361EE;
            border-radius: 10px;
            padding: 15px;
            background-color: #1E1E1E;
        }
        audio::-webkit-media-controls-panel {
            background-color: #333333;
        }
        audio::-webkit-media-controls-play-button {
            background-color: #4361EE;
            border-radius: 50%;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create a scrollable chat container with fixed height
        chat_container = st.container(height=500)
        
        with chat_container:
            # Add instructions at the top
            st.info("Scroll to view translation history.")
            
            # Show existing messages or a placeholder
            if not st.session_state["chat_history"]:
                st.markdown("<p style='color: #666; text-align: center; padding: 20px;'>Your translation history will appear here</p>", unsafe_allow_html=True)
            else:
                for i, message in enumerate(st.session_state["chat_history"]):
                    # For AI responses (translations), use the target language for TTS
                    message_target_lang = None
                    if message["role"] == "assistant":
                        # Check if the message has a target_lang attribute
                        if "target_lang" in message:
                            message_target_lang = message["target_lang"]
                        # Fallback to analyzing the previous message
                        elif i > 0:
                            # Get the previous message to find the request details
                            prev_msg = st.session_state["chat_history"][i-1]
                            if prev_msg["role"] == "user" and "Translate" in prev_msg["content"]:
                                # This is a translation response, extract the target language
                                prev_content = prev_msg["content"]
                                
                                # Use match to check for target language in previous message
                                pattern = "to (.*?):"
                                match_result = re.search(pattern, prev_content)
                                if match_result:
                                    target_text = match_result.group(1).strip()
                                    
                                    match target_text:
                                        case text if "Spanish" in text:
                                            message_target_lang = "es"
                                        case text if "English" in text:
                                            message_target_lang = "en"
                                        case text if "Catalan" in text:
                                            message_target_lang = "ca"
                                        case _:
                                            message_target_lang = None
                            
                    render_chat_message(message["content"], message["role"], message_target_lang)

    # Handle translation
    if translate_button and source_text and st.session_state["model_available"]:
        # Add user input to chat history
        target_lang_names = ", ".join([
            LANGUAGE_MAP[lang]['name'] for lang in target_langs
        ])
        
        st.session_state["chat_history"].append({
            "role": "user", 
            "content": f"Translate from {LANGUAGE_MAP[source_lang]['name']} to {target_lang_names}:\n\n{source_text}"
        })
        
        # Perform the translations
        with st.spinner(f"Translating to {target_lang_names}..."):
            # Run the async function in a synchronous context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Create Ollama client with selected model
                client = LLMClient.create(provider=st.session_state["llm_provider"], model_name=st.session_state["model_name"])
                
                # Store overall translation results
                all_translations = {}
                all_graph_data = {}
                cooccurrence_graphs = {}
                
                # Process each target language
                for target_lang in target_langs:
                    # Get the translation for this language
                    translation = loop.run_until_complete(
                        translate_text(client, source_text, source_lang, target_lang)
                    )
                    
                    # Store the translation
                    all_translations[target_lang] = translation
                    
                    # Verify if Spanish and Catalan translations might be swapped
                    if "es" in all_translations and "ca" in all_translations and len(all_translations) >= 2:
                        # Check for Spanish markers in Catalan translation
                        spanish_markers = ["es", "está", "estás", "la", "el", "los", "las", "y", "eres", "tienes"]
                        catalan_markers = ["és", "està", "estàs", "la", "el", "els", "les", "i", "ets", "tens"]
                        
                        # Count occurrences of Spanish vs Catalan markers
                        spanish_count_in_es = sum(1 for marker in spanish_markers if f" {marker} " in f" {all_translations['es']} ")
                        catalan_count_in_es = sum(1 for marker in catalan_markers if f" {marker} " in f" {all_translations['es']} ")
                        spanish_count_in_ca = sum(1 for marker in spanish_markers if f" {marker} " in f" {all_translations['ca']} ")
                        catalan_count_in_ca = sum(1 for marker in catalan_markers if f" {marker} " in f" {all_translations['ca']} ")
                        
                        # If Spanish translation looks more like Catalan and vice versa, swap them
                        if catalan_count_in_es > spanish_count_in_es and spanish_count_in_ca > catalan_count_in_ca:
                            logger.warning("Detected possible language mismatch. Swapping Spanish and Catalan translations.")
                            all_translations["es"], all_translations["ca"] = all_translations["ca"], all_translations["es"]
                    
                    # Add each translation as a separate message
                    translation_content = f"{LANGUAGE_MAP[target_lang]['name']} {LANGUAGE_MAP[target_lang]['flag']}: {translation.strip()}"
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": translation_content,
                        "target_lang": target_lang  # Store target language for TTS
                    })
                    
                    # Generate the graph data for this language
                    graph_data = loop.run_until_complete(
                        analyze_translation(source_text, [translation], [target_lang])
                    )
                    
                    # Store the graph data
                    all_graph_data[target_lang] = graph_data
                    
                    # Generate co-occurrence network for source and target texts
                    # Source text co-occurrence 
                    if source_lang not in cooccurrence_graphs:
                        window_size = st.session_state.get("window_size", 2)
                        min_freq = st.session_state.get("min_freq", 1)
                        selected_pos = st.session_state.get("selected_pos", ["NOUN", "VERB", "ADJ"])
                        
                        logger.info(f"Building co-occurrence network for {source_lang} with {len(source_text.split())} words")
                        source_cooccurrence = build_word_cooccurrence_network(
                            source_text, 
                            source_lang, 
                            window_size=window_size,
                            min_freq=min_freq,
                            include_pos=selected_pos
                        )
                        if len(source_cooccurrence.nodes()) > 0:
                            logger.info(f"Built source co-occurrence network with {len(source_cooccurrence.nodes())} nodes")
                            cooccurrence_graphs[source_lang] = source_cooccurrence
                        else:
                            logger.warning(f"Empty co-occurrence network for {source_lang}")
                    
                    # Target text co-occurrence
                    window_size = st.session_state.get("window_size", 2)
                    min_freq = st.session_state.get("min_freq", 1)
                    selected_pos = st.session_state.get("selected_pos", ["NOUN", "VERB", "ADJ"])
                    
                    logger.info(f"Building co-occurrence network for {target_lang} with {len(translation.split())} words")
                    target_cooccurrence = build_word_cooccurrence_network(
                        translation, 
                        target_lang, 
                        window_size=window_size,
                        min_freq=min_freq,
                        include_pos=selected_pos
                    )
                    
                    if len(target_cooccurrence.nodes()) > 0:
                        logger.info(f"Built target co-occurrence network with {len(target_cooccurrence.nodes())} nodes")
                        cooccurrence_graphs[target_lang] = target_cooccurrence
                    else:
                        logger.warning(f"Empty co-occurrence network for {target_lang}")
                
                # Store all translations in session state
                st.session_state["translations"][source_text] = {
                    "source_lang": source_lang,
                    "target_langs": target_langs,
                    "translations": all_translations
                }
                
                # Store all graph data
                st.session_state["graph_data"] = all_graph_data
                
                # Store co-occurrence graphs
                st.session_state["cooccurrence_graphs"] = cooccurrence_graphs
                
            except Exception as e:
                logger.error(f"Translation error: {str(e)}")
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": f"Error: {str(e)}"
                })
            finally:
                loop.close()
                
            # Refresh the UI
            st.rerun()

if __name__ == "__main__":
    main() 