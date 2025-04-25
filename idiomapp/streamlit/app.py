import os
import re
import json
import logging
import asyncio
import tempfile
import time
import random
import html
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO

# Third-party imports
import streamlit as st 
import numpy as np
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import spacy
import textacy
from langdetect import detect, LangDetectException

# Internal imports
from idiomapp.utils.ollama_utils import OllamaClient, get_available_models
from idiomapp.utils.logging_utils import setup_logging, get_recent_logs, clear_logs
from idiomapp.utils.nlp_utils import (
    analyze_parts_of_speech,
    split_into_sentences,
    calculate_similarity,
    calculate_word_similarity,
    build_word_cooccurrence_network,
    visualize_cooccurrence_network,
    detect_language,
    get_network_stats
)
from idiomapp.utils.audio_utils import (
    generate_audio,
    clean_text_for_tts,
    extract_translation_content
)
from gtts import gTTS
import base64

# Set up logging
logger = setup_logging("streamlit_app")

# Language mapping dictionary for consistent reference
LANGUAGE_MAP = {
    "en": {"name": "English", "flag": "🇬🇧", "tts_code": "en"},
    "es": {"name": "Spanish", "flag": "🇪🇸", "tts_code": "es"},
    "ca": {"name": "Catalan", "flag": "🏴󠁥󠁳󠁣󠁴󠁿", "tts_code": "es", "tts_note": "(via Spanish TTS)"}
} # TODO: Add more TTS;  Add language detection

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

def create_graph(graph_type, num_nodes, randomize_edges=False):
    """Create different types of graphs based on user selection"""
    
    logger.info(f"Creating {graph_type} graph with {num_nodes} nodes")
    
    # Use match statement to create graph based on type
    match graph_type:
        case "Path":
            G = nx.path_graph(num_nodes)
        case "Cycle":
            G = nx.cycle_graph(num_nodes)
        case "Star":
            G = nx.star_graph(num_nodes - 1)
        case "Complete":
            G = nx.complete_graph(num_nodes)
        case "Barabasi-Albert":
            # For Barabasi-Albert, m must be at least 1 and less than n
            m = min(3, num_nodes - 1)
            if m > 0:
                G = nx.barabasi_albert_graph(num_nodes, m)
            else:
                G = nx.path_graph(num_nodes)  # Fallback
        case _:
            G = nx.path_graph(num_nodes)  # Default
    
    # Add some random edges if requested
    if randomize_edges and num_nodes > 2:
        num_random_edges = random.randint(1, 3)
        for _ in range(num_random_edges):
            a, b = random.sample(list(G.nodes()), 2)
            if not G.has_edge(a, b):
                G.add_edge(a, b)
        logger.info(f"Added {num_random_edges} random edges to graph")
    
    return G

def visualize_graph_pyvis(G, central_node=None):
    """Create an interactive visualization of the graph using Pyvis"""
    
    logger.info(f"Visualizing graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Create a network with dark mode friendly colors
    net = Network(height="600px", width="100%", bgcolor="#0E1117", font_color="#FAFAFA")
    
    # Set options with dark theme colors
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
        "smooth": false,
        "shadow": false,
        "width": 3
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "springConstant": 0.04,
          "springLength": 95
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
    
    # Add nodes and edges to the network
    for node in G.nodes():
        # Special color for central node with high contrast for better focus
        if central_node is not None and node == central_node:
            net.add_node(node, label=f"Node {node}", title=f"Node {node}", color="#F72585", size=30)
        else:
            net.add_node(node, label=f"Node {node}", title=f"Node {node}", size=25)
    
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])
    
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
    st.components.v1.html(html_string, height=610)

def node_analytics(G):
    """Display analytics about the graph"""
    
    logger.info("Calculating graph analytics")
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.subheader("📊 Graph Analytics")
    
    # Use clearer layout with more structure for easier focus
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Key Metrics")
        st.metric("Number of Nodes", len(G.nodes()))
        st.metric("Number of Edges", len(G.edges()))
        
        if len(G.nodes()) > 0:
            # Calculate degree centrality
            degree_centrality = nx.degree_centrality(G)
            most_central_node = max(degree_centrality, key=degree_centrality.get)
            st.metric("Most Central Node", f"Node {most_central_node}")
    
    with col2:
        st.markdown("##### Structure Analysis")
        # Check if the graph is connected
        is_connected = nx.is_connected(G) if len(G.nodes()) > 0 else False
        st.metric("Connected", "Yes" if is_connected else "No")
        
        # Average clustering
        try:
            avg_clustering = nx.average_clustering(G)
            st.metric("Average Clustering", f"{avg_clustering:.4f}")
        except:
            st.metric("Average Clustering", "N/A")
    
    return most_central_node if len(G.nodes()) > 0 else None

def get_graph_description(G):
    """Generate a text description of the graph for AI analysis"""
    nodes = len(G.nodes())
    edges = len(G.edges())
    density = nx.density(G)
    
    description = f"""
    Graph with {nodes} nodes and {edges} edges.
    Graph density: {density:.4f}
    """
    
    # Add more characteristics if the graph has nodes
    if nodes > 0:
        # Degree information
        degrees = [d for n, d in G.degree()]
        avg_degree = sum(degrees) / len(degrees)
        max_degree = max(degrees)
        min_degree = min(degrees)
        
        # Centrality information
        degree_centrality = nx.degree_centrality(G)
        most_central = max(degree_centrality, key=degree_centrality.get)
        
        description += f"""
        Average degree: {avg_degree:.2f}
        Maximum degree: {max_degree}
        Minimum degree: {min_degree}
        Most central node: {most_central}
        Is connected: {nx.is_connected(G)}
        """
        
        # Add clustering information if applicable
        try:
            avg_clustering = nx.average_clustering(G)
            description += f"Average clustering coefficient: {avg_clustering:.4f}\n"
        except:
            pass
    
    logger.info(f"Generated graph description with {len(description)} characters")
    return description

async def get_ai_analysis(G, model_name):
    """Get AI analysis of the graph using Ollama"""
    
    logger.info(f"Starting analysis with model: {model_name}")
    
    # Create Ollama client with selected model
    client = OllamaClient(model_name=model_name)
    
    # Get graph description
    description = get_graph_description(G)
    logger.info(f"Generated graph description with {len(description)} characters")
    
    # Get AI analysis
    try:
        logger.info(f"Sending analysis request to Ollama...")
        analysis = await client.analyze_graph(description)
        logger.info(f"Received analysis response from Ollama")
        
        logger.info(f"Sending improvement suggestions request to Ollama...")
        suggestions = await client.suggest_graph_improvements(description)
        logger.info(f"Received improvement suggestions from Ollama")
        
        return {
            "analysis": analysis,
            "suggestions": suggestions
        }
    except Exception as e:
        logger.error(f"Error during AI analysis: {str(e)}")
        return {
            "analysis": {"analysis": f"Error analyzing graph: {str(e)}", "summary": "Error"},
            "suggestions": ["Could not generate suggestions"]
        }

async def chat_with_ai(model_name, message, chat_history):
    """Chat with AI about graphs"""
    # Create Ollama client with selected model
    client = OllamaClient(model_name=model_name)
    
    # Prepare context from chat history
    context = "You are a helpful assistant specializing in graph theory and network analysis. "
    context += "You're discussing a graph visualization with the user. "
    
    logger.info(f"Preparing chat with model: {model_name}")
    logger.info(f"Message length: {len(message)} characters")
    
    # Prepare system prompt with brief chat history for context
    if len(chat_history) > 0:
        context += "Here's a summary of your conversation so far: "
        for entry in chat_history[-3:]:  # Include last 3 exchanges for context
            if entry["role"] == "user":
                context += f"User asked: {entry['content']}. "
            else:
                context += f"You responded about: {entry['content'][:50]}... "
        logger.info(f"Added chat context from {len(chat_history[-3:])} previous messages")
    
    # Generate response
    try:
        logger.info("Sending message to Ollama...")
        response = await client.generate_text(message, system_prompt=context)
        logger.info(f"Received response with {len(response)} characters")
        return response
    except Exception as e:
        logger.error(f"Error in chat with AI: {str(e)}")
        return f"Error: {str(e)}"

def render_chat_message(message, role, target_lang=None):
    """
    Render a chat message with proper HTML escaping and styling.
    
    Args:
        message (str): The message content to render
        role (str): The role of the message sender ('user' or 'assistant')
        target_lang (str, optional): Language code for TTS
    
    Returns:
        None: Renders the message directly using st.markdown
    """
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
    """Display the model download status and progress"""
    status = client.get_model_status()
    
    match status["status"]:
        case "downloading":
            # Create a progress bar for downloading
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Check progress periodically
            progress = status["download_progress"]
            progress_bar.progress(int(progress)/100)
            status_text.info(f"Downloading model {status['model_name']}... {progress:.1f}%")
            
            # Add a small wait to allow UI to update
            import time
            time.sleep(0.1)
            
            # If download just started (progress < 5%), provide additional info
            if progress < 5:
                st.warning("""
                **Model download in progress.**
                
                This may take several minutes depending on the model size and your internet connection.
                You can continue using the app, but AI features will not work until the download completes.
                """)
                
            return False
            
        case "not_found":
            st.error(f"""
            **Model {status['model_name']} not available.**
            
            Error: {status['error']}
            
            Please start the model download manually with:
            ```
            docker exec -it idiomapp-ollama /bin/bash
            ollama pull {status['model_name']}
            ```
            """)
            return False
            
        case "unknown":
            st.warning(f"""
            **Model status unknown.**
            
            Could not determine if model {status['model_name']} is available.
            AI features may not work correctly.
            
            Error: {status['error']}
            """)
            return False
            
        case "available":
            # Only show a small success message that auto-dismisses
            pass
            
        case _:
            # Unrecognized status, treat as unavailable
            st.warning(f"Unknown model status: {status['status']}")
            return False
    
    return True

async def translate_text(client, source_text, source_lang, target_lang):
    """
    Translate text using the Ollama model.
    
    Args:
        client: The Ollama client
        source_text: Text to translate
        source_lang: Source language
        target_lang: Target language
        
    Returns:
        str: Translated text
    """
    logger.info(f"Translating from {source_lang} to {target_lang}: {source_text}")
    
    prompt = f"""
    Translate the following {source_lang} text to {target_lang}:
    
    "{source_text}"
    
    Please provide only the translation without any additional explanation or notes.
    """
    
    try:
        translation = await client.generate_text(prompt, 
            system_prompt="You are a helpful translation assistant. Your task is to translate text accurately while preserving meaning and context.")
        
        # Clean up the translation (remove quotes if they exist)
        translation = translation.strip('"\'').strip()
        
        logger.info(f"Translation result: {translation}")
        return translation
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return f"Translation error: {str(e)}"

async def generate_related_words(client, word, language):
    """
    Generate related words for a given word in a specified language.
    
    Args:
        client: The Ollama client
        word: The word to find relations for
        language: The language of the word
        
    Returns:
        list: List of related words with relationship types
    """
    logger.info(f"Generating related words for '{word}' in {language}")
    
    prompt = f"""
    For the {language} word "{word}", provide 5 related words and their relationship types.
    Format the response as follows:
    
    word1:relationship_type
    word2:relationship_type
    word3:relationship_type
    word4:relationship_type
    word5:relationship_type
    
    Relationship types should be one of: synonym, antonym, hypernym, hyponym, or contextual.
    Example: for "happy" in English, response might be:
    joyful:synonym
    sad:antonym
    emotion:hypernym
    ecstatic:hyponym
    birthday:contextual
    
    Provide only the list without any additional explanation.
    """
    
    try:
        response = await client.generate_text(prompt, 
            system_prompt="You are a linguistic assistant specialized in word relationships.")
        
        # Parse the response into a list of word:relation pairs
        related_words = []
        for line in response.strip().split('\n'):
            if ':' in line:
                parts = line.strip().split(':', 1)
                if len(parts) == 2:
                    word, relation = parts
                    related_words.append((word.strip(), relation.strip()))
        
        logger.info(f"Found {len(related_words)} related words for '{word}'")
        return related_words
    except Exception as e:
        logger.error(f"Error generating related words: {str(e)}")
        return []

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
    
    # Analyze parts of speech for source and target sentence
    source_pos = analyze_parts_of_speech(source_sentence, source_lang)
    target_pos = analyze_parts_of_speech(target_sentence, target_lang)
    
    # Add source words as nodes
    for word_data in source_pos:
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
        source_word = source_word_data["word"]
        source_id = f"{source_word}_{source_lang}{sentence_group}"
        
        # For each target word, establish a direct translation edge if appropriate
        for target_word_data in target_pos:
            target_word = target_word_data["word"]
            target_id = f"{target_word}_{target_lang}{sentence_group}"
            
            # Simple heuristic for direct translations - could be enhanced with alignment models
            # For now, we'll use a simple similarity/string distance metric
            translation_strength = calculate_word_similarity(
                source_word, target_word, source_lang, target_lang)
            
            # Only add edges for words that seem related
            if translation_strength > 0.3:
                # Add translation edge
                graph_data["edges"].append({
                    "from": source_id,
                    "to": target_id,
                    "relation": "translation",
                    "strength": translation_strength,
                    "label": "translation"
                })
    
    # Process related words for source and target sentences
    process_related_words(source_pos, source_lang, target_lang, graph_data, 
                         added_nodes, word_relations_cache, sentence_group)
    
    process_related_words(target_pos, target_lang, source_lang, graph_data, 
                         added_nodes, word_relations_cache, sentence_group, is_target=True)

def add_cross_sentence_relationships(graph_data):
    """Add relationships between words across different sentences"""
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
                if node1["node_type"] != "primary":
                    continue
                    
                pos1 = node1.get("pos", "unknown")
                if pos1 == "unknown":
                    continue
                    
                # Find matching POS in the other sentence
                for node2 in nodes2:
                    if node2["node_type"] != "primary" or node2["language"] != node1["language"]:
                        continue
                        
                    pos2 = node2.get("pos", "unknown")
                    if pos2 == pos1:
                        # Calculate similarity to determine if they should be connected
                        similarity = calculate_similarity(node1["label"], node2["label"])
                        
                        # Connect only if there's some similarity or same POS for key types
                        if similarity >= 0.3 or pos1 in ["noun", "verb", "adjective"]:
                            graph_data["edges"].append({
                                "from": node1["id"],
                                "to": node2["id"],
                                "relation": "cross_sentence",
                                "strength": max(0.4, similarity),
                                "label": f"related {pos1}",
                                "color": "#AA44BB",  # Purple for cross-sentence
                                "dashes": True
                            })

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
        # Call the API to get the mapping
        response = await client.generate(
            prompt=prompt,
            max_tokens=1000,
            stop_sequences=None
        )
        
        response_text = response.get("response", "")
        
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
        "es": "#F72585",    # Pink for Spanish
        "ca": "#7209B7",    # Purple for Catalan
        "en-related": "#90E0EF",  # Light blue for English related
        "es-related": "#FF9EC4",  # Light pink for Spanish related
        "ca-related": "#C77DFF"   # Light purple for Catalan related
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
        tooltip = f"{node['label']} ({lang_name})"
        if pos and pos != "unknown":
            tooltip += f"<br>Part of speech: <b>{pos}</b>"
            if details:
                tooltip += f"<br>Details: {details}"
                
        # Determine if this is part of a multi-sentence translation
        sentence_group = node.get("sentence_group", "")
        if sentence_group and sentence_group.startswith("-s"):
            tooltip += f"<br>Sentence: {sentence_group[2:]}"
        
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
        if relation == "translation":
            # Translation edges are thicker and white
            width = 3 * strength
            color = "#FFFFFF"  # White for translation edges
            arrow = False  # No arrow for translations (bidirectional)
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
                "contextual": "#00FFFF"    # Cyan for contextual
            }
            color = relation_colors.get(relation, "#AAAAAA")
            arrow = True
            smooth = {"enabled": True, "type": "continuous"}
            dashes = False
        
        # Create edge label based on relation and strength
        label = sanitize_tooltip_text(edge.get("label", relation))
        
        # Create edge title (tooltip)
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
    """
    Add semantic relationships between words in different target languages.
    
    Args:
        graph_data: The graph data structure
        target_langs: List of target languages
    """
    if len(target_langs) < 2:
        return
        
    # Create mappings of nodes by language and part of speech
    nodes_by_lang_pos = {}
    
    for lang in target_langs:
        nodes_by_lang_pos[lang] = {}
        
    # Group nodes by language and POS
    for node in graph_data["nodes"]:
        # Only consider target language nodes
        if node["language"] not in target_langs:
            continue
            
        # Only consider primary nodes (not related words)
        if node["node_type"] != "primary":
            continue
            
        pos = node.get("pos", "unknown")
        if pos == "unknown":
            continue
            
        lang = node["language"]
        if pos not in nodes_by_lang_pos[lang]:
            nodes_by_lang_pos[lang][pos] = []
            
        nodes_by_lang_pos[lang][pos].append(node)
    
    # For each language pair, connect nodes with the same part of speech
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
                        # If the nodes are in same sentence group, good candidate for connection
                        same_sentence = node1.get("sentence_group") == node2.get("sentence_group")
                        
                        # Calculate similarity
                        similarity = calculate_word_similarity(
                            node1["label"], node2["label"], lang1, lang2)
                        
                        # Connect nodes if semantically similar
                        min_threshold = 0.2 if same_sentence else 0.4
                        if similarity > min_threshold:
                            graph_data["edges"].append({
                                "from": node1["id"],
                                "to": node2["id"],
                                "relation": f"semantic_similarity_{lang1}_{lang2}",
                                "strength": similarity,
                                "label": f"semantic_{pos}",
                                "color": "#FFAA00",  # Orange for semantic cross-language
                                "dashes": True
                            })

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
        
        # Language-specific colors
        lang_colors = {
            "en": "#4361EE",  # Blue for English
            "es": "#F72585",  # Pink for Spanish
            "ca": "#7209B7",  # Purple for Catalan
            None: "#4CC9F0"   # Default light blue
        }
        
        # Add nodes
        for node in graph.nodes():
            # Get node weight (degree in the graph)
            size = 20 + (graph.degree(node) * 3)
            
            # Create tooltip with node information
            tooltip = f"Word: {sanitize_tooltip_text(str(node))}<br>Co-occurrences: {graph.degree(node)}"
            
            # Add the node with appropriate styling
            net.add_node(
                node,
                label=node,
                title=tooltip,
                color={"background": lang_colors.get(lang_code, "#4CC9F0")},
                size=size
            )
        
        # Add edges with weights
        for source, target, data in graph.edges(data=True):
            # Get edge weight (count or other measure)
            weight = data.get('weight', 1)
            width = 1 + (weight / 2)  # Scale width based on weight
            
            # Create edge tooltip (already using appropriate HTML)
            edge_tooltip = f"Co-occurrence: {weight}"
            
            # Add the edge
            net.add_edge(
                source,
                target,
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
        return f"<div>Error creating visualization: {str(e)}</div>"

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
    
    # Initialize the Ollama client
    model_name = os.environ.get("DEFAULT_MODEL", "llama3.2:latest")
    client = OllamaClient(model_name)
    
    # Display model status and check if it's available
    model_available = display_model_status(client)
    
    # Initialize session state
    if "translations" not in st.session_state:
        st.session_state["translations"] = {}
    if "graph_data" not in st.session_state:
        st.session_state["graph_data"] = None
    if "cooccurrence_graphs" not in st.session_state:
        st.session_state["cooccurrence_graphs"] = {}
    if "ollama_model" not in st.session_state:
        st.session_state["ollama_model"] = model_name
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
        
        # Create a custom styled button that stands out
        # doc_button_html = """
        # <div class="doc-button" onclick="document.getElementById('doc_button_hidden').click()">
        #     📚 Understanding Language Graphs
        # </div>
        # """
        # st.markdown(doc_button_html, unsafe_allow_html=True)
        
        # Hidden button that will be triggered by the custom HTML button
        if st.button("📚 Understanding Language Graphs", key="doc_button_hidden", help="Learn about language graphs and how to use them", use_container_width=True):
            st.session_state["show_help_page"] = True
            st.rerun()
        
        # Add a visual separator
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Language selection
        source_lang = st.selectbox(
            "Source Language",
            ["en", "es", "ca"],
            format_func=lambda x: f"{LANGUAGE_MAP[x]['name']} {LANGUAGE_MAP[x]['flag']}",
            help="Select the source language"
        )
        
        # Multiple target languages selection
        target_langs = st.multiselect(
            "Target Languages",
            ["es", "en", "ca"],
            default=["es", "ca"],  # Default to Spanish and Catalan
            format_func=lambda x: f"{LANGUAGE_MAP[x]['name']} {LANGUAGE_MAP[x]['flag']}",
            help="Select one or more target languages"
        )
        
        # Ensure at least one target language is selected
        if not target_langs:
            st.warning("Please select at least one target language")
            target_langs = ["es"]  # Default to Spanish if none selected
            
        # Model selection with status indication
            available_models = get_available_models()
            model_name = st.selectbox(
            f"Translation Model {'' if st.session_state['model_available'] else '⚠️'}",
                available_models,
                index=available_models.index(st.session_state["ollama_model"]) if st.session_state["ollama_model"] in available_models else 0,
            help="Select the AI model to use for translation",
            disabled=not st.session_state["model_available"]
        )
        
        if not st.session_state["model_available"]:
            st.error("⚠️ Selected model is not available. AI features are disabled.")
            st.info("Check the model status above and make sure it's properly installed.")
        else:
            st.success("✅ AI model is ready to use")
            
            st.session_state["ollama_model"] = model_name
            
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
                value=2,
                help="Number of words to consider for co-occurrence (larger = more connections)"
            )
            st.session_state["window_size"] = window_size
            
            # Minimum frequency for words
            min_freq = st.slider(
                "Minimum Word Frequency", 
                min_value=1, 
                max_value=5, 
                value=1,
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
                default=["NOUN", "VERB", "ADJ"],
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
                    st.markdown("🔴 **Pink** - Spanish words")
                    st.markdown("🟣 **Purple** - Catalan words")
                    st.markdown("💠 **Lighter shades** - Related words")
                    
                with legend_col2:
                    st.markdown("#### Edge Types")
                    st.markdown("⚪ **White** - Translation")
                    st.markdown("🟢 **Green** - Synonym")
                    st.markdown("🔴 **Red** - Antonym")
                    st.markdown("🟠 **Orange** - Hypernym (broader term)")
                    st.markdown("🟡 **Yellow** - Hyponym (more specific term)")
                    st.markdown("🔵 **Cyan** - Contextual relation")
                    st.markdown("🟠 **Orange dashed** - Cross-language similarity")
                    
                with legend_col3:
                    st.markdown("#### Word Types")
                    st.markdown("🟠 **Orange border** - Noun")
                    st.markdown("🟢 **Green border** - Verb")
                    st.markdown("🔵 **Blue border** - Adjective")
                    st.markdown("🟡 **Yellow border** - Adverb")
                    st.markdown("🔴 **Red border** - Pronoun")
                    st.markdown("💗 **Pink border** - Preposition")
                    st.markdown("🟣 **Purple border** - Conjunction")
                
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
                client = OllamaClient(model_name=st.session_state["ollama_model"])
                
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
                    
                    # Add each translation as a separate message
                    translation_content = f"{LANGUAGE_MAP[target_lang]['name']} {LANGUAGE_MAP[target_lang]['flag']}: {translation}"
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
                        
                        source_cooccurrence = build_word_cooccurrence_network(
                            source_text, 
                            source_lang, 
                            window_size=window_size,
                            min_freq=min_freq,
                            include_pos=selected_pos
                        )
                        cooccurrence_graphs[source_lang] = source_cooccurrence
                    
                    # Target text co-occurrence
                    window_size = st.session_state.get("window_size", 2)
                    min_freq = st.session_state.get("min_freq", 1)
                    selected_pos = st.session_state.get("selected_pos", ["NOUN", "VERB", "ADJ"])
                    
                    target_cooccurrence = build_word_cooccurrence_network(
                        translation, 
                        target_lang, 
                        window_size=window_size,
                        min_freq=min_freq,
                        include_pos=selected_pos
                    )
                    cooccurrence_graphs[target_lang] = target_cooccurrence
                
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