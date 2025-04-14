import streamlit as st
import networkx as nx
import random
import matplotlib.pyplot as plt
import tempfile
from pyvis.network import Network
import os
import asyncio
import html
import base64
import re
from datetime import datetime
from io import BytesIO
from gtts import gTTS
from langdetect import detect, LangDetectException
from idiomapp.utils.ollama_utils import OllamaClient, get_available_models
from idiomapp.utils.logging_utils import setup_logging, get_recent_logs, clear_logs

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
    'ca': 'Catalan (via Spanish TTS)',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'nl': 'Dutch',
    'ja': 'Japanese',
    'zh-cn': 'Chinese',
    'ru': 'Russian'
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
                        pattern = f"{lang_info['name']} {lang_info['flag']}: "
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

def detect_language(text, specified_lang=None):
    """
    Detect language of text using langdetect library or use specified language.
    
    Args:
        text (str): The text to analyze
        specified_lang (str, optional): Explicitly specified language code
        
    Returns:
        str: Language code ('en', 'es', or 'ca')
    """
    # If language is explicitly specified, use that
    if specified_lang in ['en', 'es', 'ca']:
        logger.info(f"Using explicitly specified language: {specified_lang}")
        return specified_lang
        
    try:
        # Default to English if text is too short
        if len(text) < 20:
            logger.info("Text too short for reliable detection, defaulting to English")
            return 'en'
        
        # Use langdetect to identify the language
        detected_lang = detect(text)
        
        # Log the detected language
        logger.info(f"Language detected: {detected_lang}")
        
        # Map language codes to our supported languages using match
        match detected_lang:
            case 'es':
                logger.info("Detected Spanish, muy bien!")
                return 'es'
            case 'ca':
                logger.info("Detected Catalan, molt bé!")
                return 'ca'
            case _:
                # Default to English for any other language
                logger.info(f"Detected {detected_lang}, using English")
                return 'en'
            
    except LangDetectException as e:
        logger.warning(f"Language detection failed: {str(e)}. Defaulting to English.")
        return 'en'  # Default to English in case of errors

def text_to_speech(text, message_key=None, specified_lang=None):
    """
    Convert text to speech using Google Text-to-Speech and return an embedded audio player
    
    Args:
        text (str): The text to convert to speech
        message_key (str, optional): Key for caching the audio
        specified_lang (str, optional): Explicitly specify language code
        
    Returns:
        str: HTML with an embedded audio player
    """
    try:
        # Check if we have this audio cached already with the same language
        cache_key = f"{message_key}_{specified_lang}"
        if message_key and cache_key in st.session_state["audio_cache"]:
            logger.info(f"Using cached audio for message {message_key}")
            return st.session_state["audio_cache"][cache_key]
        
        logger.info(f"Converting text to speech (length: {len(text)} characters)")
        
        # Use specified language or detect it
        lang = specified_lang if specified_lang else detect_language(text)
        
        # Clean the text for TTS by removing markdown formatting
        clean_text = text
        
        # If this is a translation response, extract only the translated part
        if lang in LANGUAGE_MAP and f"{LANGUAGE_MAP[lang]['name']} {LANGUAGE_MAP[lang]['flag']}:" in clean_text:
            # Extract only the translation part after the language identifier
            prefix = f"{LANGUAGE_MAP[lang]['name']} {LANGUAGE_MAP[lang]['flag']}:"
            clean_text = clean_text.split(prefix, 1)[1].strip() if prefix in clean_text else clean_text
        
        # Remove markdown formatting like **bold** or *italic*
        clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text)
        clean_text = re.sub(r'\*(.*?)\*', r'\1', clean_text)
        
        # Get TTS language code from language map
        if lang in LANGUAGE_MAP:
            tts_lang = LANGUAGE_MAP[lang]['tts_code']
            tts_note = LANGUAGE_MAP[lang].get('tts_note', '')
            lang_name = f"{LANGUAGE_MAP[lang]['name']} {tts_note}".strip()
        else:
            tts_lang = lang
            lang_name = TTS_LANGUAGE_NAMES.get(lang, lang.upper())
            
        logger.info(f"Using TTS language: {tts_lang} for {lang_name}")
        
        # Truncate very long text to avoid errors
        max_length = 1000
        if len(clean_text) > max_length:
            logger.warning(f"Text too long ({len(clean_text)} chars), truncating to {max_length} chars")
            clean_text = clean_text[:max_length] + "... (text truncated for audio)"
        
        # Create a BytesIO object to store the audio
        audio_bytes = BytesIO()
        
        # Generate audio from text using gTTS with detected language
        tts = gTTS(text=clean_text, lang=tts_lang, slow=False)
        tts.write_to_fp(audio_bytes)
        
        # Reset the pointer to the start of the buffer
        audio_bytes.seek(0)
        
        # Encode the audio data as base64
        audio_base64 = base64.b64encode(audio_bytes.read()).decode()
        
        # Create HTML for the embedded audio player with dark theme styling
        audio_html = f'''
        <div style="margin-top: 10px; padding: 10px; background-color: #1E1E1E; border-radius: 8px; border: 1px solid #4361EE;">
            <p style="margin-bottom: 5px; color: #CCCCCC; font-size: 12px;">Audio: {lang_name}</p>
            <audio controls style="width: 100%; height: 40px; background-color: #333333; border-radius: 4px;">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        </div>
        '''
        
        logger.info(f"Text-to-speech conversion successful")
        
        # Cache the audio if we have a key
        if message_key:
            st.session_state["audio_cache"][cache_key] = audio_html
            
        return audio_html
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {str(e)}")
        return f"<div style='color: #FF5C5C; padding: 10px;'>TTS Error: {str(e)}</div>"

def generate_audio(text, lang_code):
    """Generate audio HTML for a given text and language code"""
    try:
        logger.info(f"Generating audio for language {lang_code}")
        
        # Clean the text for TTS by removing markdown formatting and extracting translation
        clean_text = text
        
        # If this is a translation response, extract only the translated part
        if lang_code in LANGUAGE_MAP and f"{LANGUAGE_MAP[lang_code]['name']} {LANGUAGE_MAP[lang_code]['flag']}:" in clean_text:
            # Extract only the translation part after the language identifier
            prefix = f"{LANGUAGE_MAP[lang_code]['name']} {LANGUAGE_MAP[lang_code]['flag']}:"
            clean_text = clean_text.split(prefix, 1)[1].strip() if prefix in clean_text else clean_text
        
        # Remove markdown formatting like **bold** or *italic*
        clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text)
        clean_text = re.sub(r'\*(.*?)\*', r'\1', clean_text)
        
        # Get TTS language code from language map
        if lang_code in LANGUAGE_MAP:
            tts_lang = LANGUAGE_MAP[lang_code]['tts_code']
            tts_note = LANGUAGE_MAP[lang_code].get('tts_note', '')
            lang_name = f"{LANGUAGE_MAP[lang_code]['name']} {tts_note}".strip()
        else:
            tts_lang = lang_code
            lang_name = TTS_LANGUAGE_NAMES.get(lang_code, lang_code.upper())
            
        logger.info(f"Using TTS language: {tts_lang} for {lang_name}")
        
        # Truncate very long text to avoid errors
        max_length = 1000
        if len(clean_text) > max_length:
            logger.warning(f"Text too long ({len(clean_text)} chars), truncating to {max_length} chars")
            clean_text = clean_text[:max_length] + "... (text truncated for audio)"
        
        # Create a BytesIO object to store the audio
        audio_bytes = BytesIO()
        
        # Generate audio from text using gTTS with detected language
        tts = gTTS(text=clean_text, lang=tts_lang, slow=False)
        tts.write_to_fp(audio_bytes)
        
        # Reset the pointer to the start of the buffer
        audio_bytes.seek(0)
        
        # Encode the audio data as base64
        audio_base64 = base64.b64encode(audio_bytes.read()).decode()
        
        # Generate a more simple audio HTML with smaller footprint
        audio_html = f'''
        <div style="margin-top: 10px; padding: 8px; background-color: #1E1E1E; border-radius: 5px; border: 1px solid #4361EE;">
            <audio controls style="width: 100%; height: 30px;">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        </div>
        '''
        
        logger.info(f"Audio HTML generated successfully")
        return audio_html
        
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return f"<div style='color: red; padding: 5px; font-size: 12px;'>Audio error: {str(e)}</div>"

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

async def analyze_translation(client, source_text, translation, source_lang, target_lang):
    """
    Create graph data for words in the translation.
    
    Args:
        client: The Ollama client
        source_text: Original text
        translation: Translated text
        source_lang: Source language
        target_lang: Target language
        
    Returns:
        dict: Graph data structure for visualization
    """
    logger.info(f"Analyzing translation for graph visualization")
    
    # Split the texts into words (simple split by space as a basic approach)
    source_words = [word.strip('.,!?:;()[]{}""\'').lower() for word in source_text.split() if word.strip('.,!?:;()[]{}""\'')]
    target_words = [word.strip('.,!?:;()[]{}""\'').lower() for word in translation.split() if word.strip('.,!?:;()[]{}""\'')]
    
    # Initialize graph data
    graph_data = {
        "nodes": [],
        "edges": []
    }
    
    # Add source words to graph
    for word in source_words:
        if word:  # Skip empty strings
            graph_data["nodes"].append({
                "id": f"{source_lang}:{word}",
                "label": word,
                "group": source_lang,
                "language": source_lang
            })
    
    # Add target words to graph
    for word in target_words:
        if word:  # Skip empty strings
            graph_data["nodes"].append({
                "id": f"{target_lang}:{word}",
                "label": word,
                "group": target_lang,
                "language": target_lang
            })
    
    # Add edges between source and target words (assuming they align roughly)
    min_len = min(len(source_words), len(target_words))
    for i in range(min_len):
        if source_words[i] and target_words[i]:
            graph_data["edges"].append({
                "from": f"{source_lang}:{source_words[i]}",
                "to": f"{target_lang}:{target_words[i]}",
                "relation": "translation"
            })
    
    # Initialize word relation cache if it doesn't exist in session state
    if "word_relations_cache" not in st.session_state:
        st.session_state["word_relations_cache"] = {}
    
    # Check which words need related words (not in cache)
    words_to_process = []
    for word in set(target_words):  # Use set to avoid duplicates
        if not word:
            continue
        
        cache_key = f"{target_lang}:{word}"
        if cache_key not in st.session_state["word_relations_cache"]:
            words_to_process.append(word)
    
    # If we have words that need processing, batch them together
    if words_to_process:
        # Limit to max 5 words to avoid overly long prompts
        batch_size = 5
        batches = [words_to_process[i:i+batch_size] for i in range(0, len(words_to_process), batch_size)]
        
        for batch in batches:
            # Process the batch
            batch_related_words = await generate_related_words_batch(client, batch, target_lang)
            
            # Store in cache
            for word, related in batch_related_words.items():
                cache_key = f"{target_lang}:{word}"
                st.session_state["word_relations_cache"][cache_key] = related
    
    # Now add all related words from the cache
    for word in target_words:
        if not word:
            continue
            
        cache_key = f"{target_lang}:{word}"
        if cache_key in st.session_state["word_relations_cache"]:
            related_words = st.session_state["word_relations_cache"][cache_key]
            
            # Add related words and connections to the graph
            for related_word, relation_type in related_words:
                node_id = f"{target_lang}:{related_word}"
                
                # Check if node already exists
                if not any(node["id"] == node_id for node in graph_data["nodes"]):
                    graph_data["nodes"].append({
                        "id": node_id,
                        "label": related_word,
                        "group": f"{target_lang}-related",
                        "language": target_lang
                    })
                
                # Add edge for relationship
                graph_data["edges"].append({
                    "from": f"{target_lang}:{word}",
                    "to": node_id,
                    "relation": relation_type
                })
    
    logger.info(f"Generated graph with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
    return graph_data

async def generate_related_words_batch(client, words, language):
    """
    Generate related words for multiple words at once.
    
    Args:
        client: The Ollama client
        words: List of words to find relations for
        language: The language of the words
        
    Returns:
        dict: Dictionary mapping words to their related words
    """
    if not words:
        return {}
        
    logger.info(f"Generating related words for {len(words)} words in {language} as batch")
    
    # Build a prompt that asks for related words for each word
    words_list = ", ".join([f'"{word}"' for word in words])
    
    prompt = f"""
    For each of these {language} words: {words_list}, provide 3 related words with their relationship types.
    
    Format your response exactly like this:
    
    WORD: first_word
    related1:relationship_type
    related2:relationship_type
    related3:relationship_type
    
    WORD: second_word
    related1:relationship_type
    related2:relationship_type
    related3:relationship_type
    
    ...and so on for each word.
    
    Relationship types must be one of: synonym, antonym, hypernym, hyponym, or contextual.
    Be accurate and consistent with the format.
    """
    
    try:
        response = await client.generate_text(prompt, 
            system_prompt="You are a linguistic assistant specialized in word relationships. Respond only with the requested format.")
        
        # Parse the response to extract related words for each word
        results = {}
        current_word = None
        current_relations = []
        
        for line in response.strip().split('\n'):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check if this is a word header
            if line.startswith("WORD:"):
                # If we were processing a word, save its relations
                if current_word is not None and current_relations:
                    results[current_word] = current_relations
                    
                # Start a new word
                current_word = line.replace("WORD:", "").strip()
                current_relations = []
            
            # Process relation if we're inside a word section
            elif current_word is not None and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    related_word, relation = parts
                    current_relations.append((related_word.strip(), relation.strip()))
        
        # Add the last word's relations
        if current_word is not None and current_relations:
            results[current_word] = current_relations
            
        logger.info(f"Successfully processed batch of {len(results)} words")
        return results
        
    except Exception as e:
        logger.error(f"Error generating related words in batch: {str(e)}")
        return {word: [] for word in words}  # Return empty results on error

def visualize_translation_graph(graph_data):
    """
    Visualize translation and related words using PyVis.
    
    Args:
        graph_data: Dictionary with nodes and edges
    """
    logger.info(f"Visualizing translation graph")
    
    # Create a network with dark mode friendly colors - taller height for new layout
    net = Network(height="500px", width="100%", bgcolor="#0E1117", font_color="#FAFAFA")
    
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
        "width": 2
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
    
    # Group colors for different languages and relations
    group_colors = {
        "en": "#4361EE",    # Blue for English
        "es": "#F72585",    # Pink for Spanish - VERIFIED
        "ca": "#7209B7",    # Purple for Catalan - CHANGED
        "fr": "#7209B7",    # Purple for French
        "de": "#4CC9F0",    # Light blue for German
        "it": "#3A0CA3",    # Dark purple for Italian
        "pt": "#4895EF",    # Blue-purple for Portuguese
        "ru": "#560BAD",    # Medium purple for Russian
        "zh": "#F77F00",    # Orange for Chinese
        "ja": "#FCBF49",    # Yellow for Japanese
        "ar": "#D62828",    # Red for Arabic
        "en-related": "#90E0EF",  # Light blue for English related
        "es-related": "#FF9EC4",  # Light pink for Spanish related - VERIFIED
        "ca-related": "#C77DFF",  # Light purple for Catalan related - CHANGED
        "fr-related": "#C8B6FF",  # Light purple for French related
        "de-related": "#BDE0FE",  # Very light blue for German related
        "it-related": "#CDB4DB",  # Light purple for Italian related
        "pt-related": "#A2D2FF",  # Light blue for Portuguese related
        "ru-related": "#BDB2FF",  # Light purple for Russian related
        "zh-related": "#FFD6A5",  # Light orange for Chinese related
        "ja-related": "#FFFCBF",  # Light yellow for Japanese related
        "ar-related": "#FFADAD"   # Light red for Arabic related
    }
    
    # Language name mapping for node tooltips
    language_names = {
        "en": "English",
        "es": "Spanish",
        "ca": "Catalan",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ar": "Arabic"
    }
    
    # Add nodes
    for node in graph_data["nodes"]:
        # Determine node language and group
        node_lang = node.get("language", "unknown")
        group = node.get("group", "default")
        
        # Make sure the group matches the language (to fix the Spanish/Catalan issue)
        if "-related" in group:
            # For related nodes, preserve the -related suffix but ensure correct language prefix
            base_lang = group.split("-")[0]
            if base_lang != node_lang:
                group = f"{node_lang}-related"
        else:
            # For primary nodes, group should match language exactly
            if group != node_lang:
                group = node_lang
                
        # Get color based on corrected group
        color = group_colors.get(group, "#4CC9F0")  # Default color if group not found
        
        # Handle different node types
        if "-related" in group:
            size = 20  # Smaller size for related words
        else:
            size = 30  # Larger size for translation words
        
        # Get proper language name for tooltip
        lang_name = language_names.get(node_lang, node_lang.upper())
        
        net.add_node(
            node["id"], 
            label=node["label"], 
            title=f"{node['label']} ({lang_name})",
            color=color,
            size=size,
            group=group  # Use corrected group
        )
    
    # Add edges with relation types
    for edge in graph_data["edges"]:
        if edge.get("relation") == "translation":
            # Translation edges are thicker
            width = 3
            color = "#FFFFFF"  # White for translation edges
        else:
            # Other relations are thinner
            width = 1
            # Color based on relation type
            relation_colors = {
                "synonym": "#00FF00",      # Green for synonyms
                "antonym": "#FF0000",      # Red for antonyms
                "hypernym": "#FFA500",     # Orange for hypernyms
                "hyponym": "#FFFF00",      # Yellow for hyponyms
                "contextual": "#00FFFF"    # Cyan for contextual
            }
            color = relation_colors.get(edge.get("relation", ""), "#AAAAAA")
        
        # Add the edge with relation as title
        net.add_edge(
            edge["from"], 
            edge["to"], 
            title=edge.get("relation", "related"),
            color=color,
            width=width
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
    
    # Display the network - use taller height for the new layout
    st.components.v1.html(html_string, height=500)

def main():
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
    else:
        st.session_state["model_available"] = model_available
    
    # Add a sidebar with translation settings
    with st.sidebar:
        st.header("Translation Settings")
        
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
            default=["es"],  # Default to Spanish
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
        # Graph section first - above the chat input
        if st.session_state["graph_data"]:
            st.subheader("🔄 Word Relationships Graphs")
            
            # Add an info box explaining the graph
            st.markdown("""
            <div style="padding: 10px; border-radius: 5px; margin-bottom: 15px; background-color: rgba(67, 97, 238, 0.1); border-left: 4px solid #4361EE;">
            <p>These graphs show word relationships between languages. Interactive features:</p>
            <ul>
              <li><strong>Drag nodes</strong> to rearrange the visualization</li>
              <li><strong>Hover over words</strong> to see details</li>
              <li><strong>Zoom in/out</strong> with mouse wheel or trackpad</li>
              <li><strong>Different colors</strong> represent different relationship types</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Create tabs for each language
            if len(st.session_state["graph_data"]) > 1:
                # Use tabs when we have multiple languages
                lang_tabs = st.tabs([
                    f"{LANGUAGE_MAP[lang]['name']} {LANGUAGE_MAP[lang]['flag']}" 
                    for lang in st.session_state["graph_data"].keys()
                ])
                
                # Display each language graph in its own tab
                for i, (lang, graph_data) in enumerate(st.session_state["graph_data"].items()):
                    with lang_tabs[i]:
                        st.caption(f"Word relationships for {LANGUAGE_MAP[lang]['name']} translation")
                        visualize_translation_graph(graph_data)
            else:
                # Single language case - no tabs needed
                lang, graph_data = next(iter(st.session_state["graph_data"].items()))
                st.caption(f"Word relationships for {LANGUAGE_MAP[lang]['name']} translation")
                visualize_translation_graph(graph_data)
            
            # Add a legend explaining the graph
            with st.expander("📊 Graph Legend", expanded=False):
                legend_col1, legend_col2 = st.columns(2)
                
                with legend_col1:
                    st.markdown("#### Node Colors")
                    st.markdown("🔵 **Blue** - English words")
                    st.markdown("🔴 **Pink** - Spanish words")
                    st.markdown("🟣 **Purple** - Catalan words")
                    st.markdown("💠 **Lighter shades** - Related words")
                    
                with legend_col2:
                    st.markdown("#### Edge Colors")
                    st.markdown("⚪ **White** - Translation")
                    st.markdown("🟢 **Green** - Synonym")
                    st.markdown("🔴 **Red** - Antonym")
                    st.markdown("🟠 **Orange** - Hypernym (broader term)")
                    st.markdown("🟡 **Yellow** - Hyponym (more specific term)")
                    st.markdown("🔵 **Cyan** - Contextual relation")
        
        # Display a helpful guide if no translation has been made yet
        if not st.session_state["chat_history"]:
            st.info(f"""
            ### How to use the Translation Helper
            1. Select source language in the sidebar
            2. Select one or more target languages in the sidebar
            3. Type your text in the input box
            4. Click "Translate" to see the translations
            5. Explore the word relationship graphs that appear
            
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
                        analyze_translation(client, source_text, translation, source_lang, target_lang)
                    )
                    
                    # Store the graph data
                    all_graph_data[target_lang] = graph_data
                
                # Store all translations in session state
                st.session_state["translations"][source_text] = {
                    "source_lang": source_lang,
                    "target_langs": target_langs,
                    "translations": all_translations
                }
                
                # Store all graph data
                st.session_state["graph_data"] = all_graph_data
                
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