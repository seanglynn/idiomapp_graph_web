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
from idiomapp.ollama_utils import OllamaClient, get_available_models
from idiomapp.logging_utils import setup_logging, get_recent_logs, clear_logs

# Set up logging
logger = setup_logging("graph_app")

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
    }
    /* Audio player styling for dark theme */
    audio {
        width: 100%;
        border-radius: 8px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

def create_graph(graph_type, num_nodes, randomize_edges=False):
    """Create different types of graphs based on user selection"""
    
    logger.info(f"Creating {graph_type} graph with {num_nodes} nodes")
    
    if graph_type == "Path":
        G = nx.path_graph(num_nodes)
    elif graph_type == "Cycle":
        G = nx.cycle_graph(num_nodes)
    elif graph_type == "Star":
        G = nx.star_graph(num_nodes - 1)
    elif graph_type == "Complete":
        G = nx.complete_graph(num_nodes)
    elif graph_type == "Barabasi-Albert":
        # For Barabasi-Albert, m must be at least 1 and less than n
        m = min(3, num_nodes - 1)
        if m > 0:
            G = nx.barabasi_albert_graph(num_nodes, m)
        else:
            G = nx.path_graph(num_nodes)  # Fallback
    else:
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

def render_chat_message(message, role):
    """
    Render a chat message with proper HTML escaping and styling.
    
    Args:
        message (str): The message content to render
        role (str): The role of the message sender ('user' or 'assistant')
    
    Returns:
        None: Renders the message directly using st.markdown
    """
    # Determine message style based on role
    if role == "user":
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
    else:
        css_class = "chat-message-ai"
        prefix = "AI"
        
        # For AI responses, create container with message and TTS button
        formatted_content = process_message_content(message)
        
        # Create a unique key for this message
        message_index = len(st.session_state.get("chat_history", []))
        message_key = f"tts_{message_index}"
        audio_key = f"audio_{message_key}"
        
        # Initialize the audio state if it doesn't exist
        if audio_key not in st.session_state:
            st.session_state[audio_key] = False
        
        # Create columns for the message and listen button
        msg_col, btn_col = st.columns([9, 1])
        
        # Display message in the first column
        with msg_col:
            st.markdown(
                f"""<div class='{css_class}'>
                <strong>{prefix}:</strong> {formatted_content}
                </div>""", 
                unsafe_allow_html=True
            )
        
        # Display listen button in the second column
        with btn_col:
            if st.button("🔊", key=message_key, help="Listen to this response"):
                st.session_state[audio_key] = True
        
        # Show audio player below if the button was clicked
        if st.session_state[audio_key]:
            with st.spinner("Generating audio..."):
                audio_html = text_to_speech(message, message_key)
                st.markdown(audio_html, unsafe_allow_html=True)

def process_message_content(message):
    """Process message content to handle code blocks and HTML escaping"""
    content = []
    lines = message.split('\n')
    
    # Simple code block detection
    in_code_block = False
    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            content.append(f"<pre>{line}</pre>" if in_code_block else "</pre>")
        elif in_code_block:
            # Don't escape inside code blocks
            content.append(line)
        else:
            # Escape HTML outside code blocks
            content.append(html.escape(line))
    
    # Join lines with line breaks
    return "<br>".join(content)

def detect_language(text):
    """
    Detect language of text using langdetect library.
    Focused on recognizing English and Spanish.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        str: Language code ('en' or 'es')
    """
    try:
        # Default to English if text is too short
        if len(text) < 20:
            logger.info("Text too short for reliable detection, defaulting to English")
            return 'en'
        
        # Use langdetect to identify the language
        detected_lang = detect(text)
        
        # Log the detected language
        logger.info(f"Language detected: {detected_lang}")
        
        # For this application, only handling English and Spanish
        # Map other languages to the closest match
        if detected_lang == 'es':
            logger.info(f"Detected {detected_lang}, muy bien!")
            return 'es'  # Spanish
        elif detected_lang == 'ca':
            # Catalan, Portuguese, Galician are similar to Spanish
            logger.info(f"Detected {detected_lang}, molt bé!")
            return 'es'
        else:
            # Default to English for any other language
            return 'en'
            
    except LangDetectException as e:
        logger.warning(f"Language detection failed: {str(e)}. Defaulting to English.")
        return 'en'  # Default to English in case of errors

def text_to_speech(text, message_key=None):
    """
    Convert text to speech using Google Text-to-Speech and return audio player HTML
    
    Args:
        text (str): The text to convert to speech
        message_key (str, optional): Key for caching the audio
        
    Returns:
        str: HTML for an audio player with the speech
    """
    try:
        # Check if we have this audio cached already
        if message_key and message_key in st.session_state["audio_cache"]:
            logger.info(f"Using cached audio for message {message_key}")
            return st.session_state["audio_cache"][message_key]
        
        logger.info(f"Converting text to speech (length: {len(text)} characters)")
        
        # Detect language (English or Spanish)
        lang = detect_language(text)
        
        # Truncate very long text to avoid errors (gTTS has limits)
        max_length = 3000
        if len(text) > max_length:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {max_length} chars")
            text = text[:max_length] + "... (text truncated for audio)"
        
        # Create a BytesIO object to store the audio
        audio_bytes = BytesIO()
        
        # Generate audio from text using gTTS with detected language
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(audio_bytes)
        
        # Reset the pointer to the start of the buffer
        audio_bytes.seek(0)
        
        # Encode the audio data as base64
        audio_base64 = base64.b64encode(audio_bytes.read()).decode()
        
        # Get user-friendly language name
        language_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'ja': 'Japanese',
            'zh-cn': 'Chinese',
            'ru': 'Russian'
        }
        lang_name = language_names.get(lang, lang.upper())
        
        # Create HTML for an audio player with dark theme styling
        audio_html = f"""
        <div style="margin: 10px 0; padding: 10px; background-color: #262730; border-radius: 8px; border: 1px solid #4361EE;">
            <div style="font-size: 14px; margin-bottom: 5px; color: #FAFAFA;">🔊 Audio version ({lang_name}):</div>
            <audio controls style="width: 100%; height: 40px;">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        </div>
        """
        
        logger.info(f"Text-to-speech conversion successful using {lang_name}")
        
        # Cache the audio if we have a key
        if message_key:
            st.session_state["audio_cache"][message_key] = audio_html
            
        return audio_html
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {str(e)}")
        return f"<div style='color: #FF5C5C; padding: 10px; background-color: #3A1C1C; border-radius: 8px; border: 1px solid #FF5C5C;'>TTS Error: {str(e)}</div>"

def main():
    # Create a cleaner header with visual distinction for dark theme
    st.markdown("<h1 style='text-align: center; color: #4CC9F0;'>IdiomApp Graph Explorer</h1>", unsafe_allow_html=True)
    
    # Initialize session state
    if "graph" not in st.session_state:
        st.session_state["graph"] = create_graph("Path", 8)
    if "ai_analysis" not in st.session_state:
        st.session_state["ai_analysis"] = None
    if "ollama_model" not in st.session_state:
        st.session_state["ollama_model"] = "llama3.2:latest"
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "view" not in st.session_state:
        st.session_state["view"] = "visualization"  # or "chat"
    if "show_debug" not in st.session_state:
        st.session_state["show_debug"] = False
    if "audio_cache" not in st.session_state:
        st.session_state["audio_cache"] = {}
    
    # Add a sidebar with clear structure for ADHD-friendly navigation
    with st.sidebar:
        # Create tab-based layout to group related controls
        tab1, tab2, tab3 = st.tabs(["🔄 Graph", "🤖 AI", "⚙️ Settings"])
        
        with tab1:
            # Simple, focused graph type selection with visual cues
            graph_type = st.selectbox(
                "Graph Type",
                ["Path", "Cycle", "Star", "Complete", "Barabasi-Albert"],
                help="Select the type of graph structure to visualize"
            )
            
            # Slider with clear bounds and feedback
            num_nodes = st.slider(
                "Number of Nodes", 
                min_value=3, 
                max_value=20, 
                value=8,
                help="Adjust the number of nodes in the graph"
            )
            
            # Checkboxes with clear purpose
            col1, col2 = st.columns(2)
            with col1:
                randomize_edges = st.checkbox(
                    "Random Edges",
                    value=False,
                    help="Add random connections between nodes"
                )
            with col2:
                highlight_central = st.checkbox(
                    "Highlight Central",
                    value=True,
                    help="Highlight the most central node"
                )
            
            # Clearly labeled button with action feedback
            if st.button("🔄 Generate New Graph", use_container_width=True):
                with st.spinner("Creating new graph..."):
                    st.session_state["graph"] = create_graph(graph_type, num_nodes, randomize_edges)
                    # Reset AI analysis when generating a new graph
                    st.session_state["ai_analysis"] = None
                    st.success("New graph created!")
        
        with tab2:
            # Get available models or use defaults
            available_models = get_available_models()
            
            # Model selection with clear indication of what it does
            model_name = st.selectbox(
                "Ollama Model",
                available_models,
                index=available_models.index(st.session_state["ollama_model"]) if st.session_state["ollama_model"] in available_models else 0,
                help="Select the AI model to use for analysis"
            )
            st.session_state["ollama_model"] = model_name
            
            # Clear button for AI analysis
            if st.button("🔍 Analyze Graph", use_container_width=True):
                with st.spinner("AI is analyzing your graph..."):
                    # Run the async function in a synchronous context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        logger.info("Creating event loop for async analysis")
                        st.session_state["ai_analysis"] = loop.run_until_complete(
                            get_ai_analysis(st.session_state["graph"], model_name)
                        )
                        logger.info("Analysis completed successfully")
                        st.success("Analysis complete!")
                    except Exception as e:
                        logger.error(f"Error in analysis: {str(e)}")
                        st.error(f"Analysis failed: {str(e)}")
                    finally:
                        loop.close()
            
            # View toggle with clear visual distinction
            view_options = ["📊 Graph View", "💬 Chat"]
            
            # Radio buttons with clear visual grouping
            view_selection = st.radio(
                "Switch View",
                view_options,
                index=0 if st.session_state["view"] == "visualization" else 1,
                help="Toggle between graph view and chat interface"
            )
            st.session_state["view"] = "visualization" if view_selection == "📊 Graph View" else "chat"
        
        with tab3:
            # Debug toggle with clear purpose
            st.session_state["show_debug"] = st.checkbox(
                "Show Debug Logs", 
                value=st.session_state["show_debug"],
                help="Show detailed logs of AI processing"
            )
    
    # Show debug logs if enabled
    if st.session_state["show_debug"]:
        with st.expander("Debug Logs", expanded=True):
            # Get the recent logs
            logs = get_recent_logs(50)
            
            # Message for when no logs are available
            if not logs:
                st.info("No logs yet. Perform actions to see logs here.")
            else:
                # Display logs in a code block with the newest first
                log_output = "\n".join(reversed(logs))
                st.code(log_output)
            
            # Add a button to clear the logs
            if st.button("Clear Logs"):
                clear_logs()
                st.rerun()
    
    # Main area - determine which view to show
    if st.session_state["view"] == "visualization":
        # Two columns layout for main content in visualization view
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display graph analytics
            most_central_node = node_analytics(st.session_state["graph"])
            
            # Only highlight if requested
            central_node = most_central_node if highlight_central else None
            
            # Display the graph
            visualize_graph_pyvis(st.session_state["graph"], central_node=central_node)
        
        with col2:
            # Display AI Analysis if available
            if st.session_state["ai_analysis"]:
                st.subheader("🤖 AI Analysis")
                
                with st.expander("📋 Graph Analysis", expanded=True):
                    st.markdown(st.session_state["ai_analysis"]["analysis"]["analysis"])
                
                with st.expander("💡 Improvement Suggestions", expanded=True):
                    for suggestion in st.session_state["ai_analysis"]["suggestions"]:
                        st.markdown(f"✨ {suggestion}")
            else:
                st.subheader("About Graph Visualization")
                
                st.markdown("""
                This interactive graph visualization allows you to explore different graph structures.
                
                **Interaction:**
                - Drag nodes to reposition
                - Zoom with mouse wheel
                - Pan by dragging background
                
                **Use the AI Analysis tab** in the sidebar to get AI insights about your graph.
                """)
    else:
        # Chat interface view - simpler design
        st.subheader("💬 Chat with AI about Graphs")
        
        # Chat info and debug toggle in the same row
        col1, col2 = st.columns([3, 1])
        with col1:
            # Brief instruction
            st.info("Ask questions about graph theory, network analysis, or the current graph visualization.")
        with col2:
            # Add debug toggle directly in the chat interface
            chat_debug = st.checkbox("Show Chat Logs", value=False)
        
        # Display current graph properties
        with st.expander("📊 Current Graph Properties", expanded=False):
            st.code(get_graph_description(st.session_state["graph"]))
        
        # Show debug logs if chat debug is enabled
        if chat_debug:
            with st.expander("Chat Debug Logs", expanded=True):
                logs = get_recent_logs(20, filter_text="ollama")
                
                if not logs:
                    st.info("No Ollama logs found. Send a message to see the logs.")
                else:
                    # Display logs in a code block with the newest first
                    log_output = "\n".join(reversed(logs))
                    st.code(log_output)
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            # Style the chat messages for better readability
            for message in st.session_state["chat_history"]:
                render_chat_message(message["content"], message["role"])
        
        # Input for new message
        user_message = st.text_area("Your message:", height=100, placeholder="Type your question about graphs here...")
        
        # Action buttons in columns for better layout
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("💬 Send Message", use_container_width=True):
                if user_message:
                    # Add user message to history
                    st.session_state["chat_history"].append({"role": "user", "content": user_message})
                    
                    # Get AI response
                    with st.spinner("AI is thinking..."):
                        # Run the async function in a synchronous context
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            logger.info("Creating chat event loop")
                            ai_response = loop.run_until_complete(
                                chat_with_ai(
                                    st.session_state["ollama_model"],
                                    user_message,
                                    st.session_state["chat_history"]
                                )
                            )
                            logger.info("Chat completion successful")
                        except Exception as e:
                            logger.error(f"Error in chat: {str(e)}")
                            ai_response = f"Error processing request: {str(e)}"
                        finally:
                            loop.close()
                    
                    # Add AI response to history
                    st.session_state["chat_history"].append({"role": "assistant", "content": ai_response})
                    
                    # Clear input box by rerunning
                    st.rerun()
        
        with col2:
            # Add button to clear chat history
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state["chat_history"] = []
                st.success("Chat history cleared!")
                st.rerun()

if __name__ == "__main__":
    main() 