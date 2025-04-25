"""
Natural Language Processing utilities.
Uses textacy and spaCy for advanced NLP capabilities.
"""

import os
import re
import logging
import tempfile
from typing import List, Dict, Any, Optional, Union, Tuple

# NLP libraries
import spacy
import textacy
from textacy.extract.keyterms import textrank
from textacy.representations.network import build_cooccurrence_network
import networkx as nx
from pyvis.network import Network
from langdetect import detect, LangDetectException

# Setup logging
logger = logging.getLogger(__name__)

# Language model mapping
LANG_MODELS = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "ca": "ca_core_news_sm"
}

def load_spacy_model(language: str) -> spacy.language.Language:
    """
    Load the appropriate spaCy language model, downloading it if necessary.
    
    Args:
        language: ISO language code (en, es, ca)
        
    Returns:
        Loaded spaCy language model
    """
    # Get the appropriate model name
    model_name = LANG_MODELS.get(language, "en_core_web_sm")
    
    try:
        # Try to load the model
        nlp = spacy.load(model_name)
        return nlp
    except OSError:
        # If model is not found, download it
        logger.info(f"Downloading language model: {model_name}")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
        return nlp

def detect_language(text: str, specified_lang: Optional[str] = None) -> str:
    """
    Detect the language of a text or use the specified language.
    
    Args:
        text: The text to analyze
        specified_lang: Optional language code to use instead of detection
        
    Returns:
        ISO language code (en, es, ca, etc.)
    """
    if not text or len(text.strip()) < 3:
        return "en"  # Default to English for very short texts
    
    # Use specified language if provided
    if specified_lang in LANG_MODELS:
        return specified_lang
    
    try:
        # Use langdetect to detect the language
        detected = detect(text)
        
        # Map some language codes to our supported models
        lang_mapping = {
            "en": "en",
            "es": "es",
            "ca": "ca",
            "pt": "es",  # Use Spanish model for Portuguese
            "it": "es",  # Use Spanish model for Italian
            "fr": "es"   # Use Spanish model for French
        }
        
        return lang_mapping.get(detected, "en")
    except LangDetectException:
        logger.warning(f"Could not detect language for text: {text[:50]}...")
        return "en"  # Default to English

def analyze_parts_of_speech(sentence: str, language: str) -> List[Dict[str, Any]]:
    """
    Analyze parts of speech for words in a sentence using textacy and spaCy.
    
    Args:
        sentence: The sentence to analyze
        language: The language of the sentence
        
    Returns:
        List of words with their parts of speech information
    """
    logger.info(f"Analyzing parts of speech for: {sentence}")
    
    try:
        # Load the appropriate model
        nlp = load_spacy_model(language)
        
        # Process the text
        doc = nlp(sentence)
        
        # Map spaCy POS tags to simpler categories
        pos_mapping = {
            "NOUN": "noun",
            "PROPN": "noun",
            "VERB": "verb",
            "AUX": "verb",
            "ADJ": "adjective",
            "ADV": "adverb",
            "PRON": "pronoun",
            "DET": "determiner",
            "ADP": "preposition",
            "CCONJ": "conjunction",
            "SCONJ": "conjunction",
            "INTJ": "interjection",
            "NUM": "number",
            "SYM": "symbol",
            "PART": "particle"
        }
        
        # Extract word data
        result = []
        for token in doc:
            # Skip punctuation and whitespace
            if token.is_punct or token.is_space:
                continue
            
            # Get part of speech or default to unknown
            pos = pos_mapping.get(token.pos_, "unknown")
            
            # Get additional details
            details = f"{token.tag_}"
            if token.lemma_ != token.text:
                details += f" (lemma: {token.lemma_})"
            
            # Get dependency information
            if token.dep_ != "ROOT":
                details += f", {token.dep_} of '{token.head.text}'"
            else:
                details += ", ROOT"
            
            # Add named entity information if available
            if token.ent_type_:
                details += f", entity: {token.ent_type_}"
            
            # Create word data entry
            result.append({
                "word": token.text.lower(),
                "pos": pos,
                "details": details,
                "lemma": token.lemma_,
                "dep": token.dep_,
                "is_entity": bool(token.ent_type_),
                "entity_type": token.ent_type_ if token.ent_type_ else None
            })
        
        # Use textacy to extract additional information
        try:
            # Get key terms if available
            doc_terms = textrank(doc, normalize="lemma")
            for term, _ in doc_terms:
                # Find all tokens that are part of this key term
                for token_data in result:
                    if token_data["lemma"] in term:
                        token_data["is_keyterm"] = True
                        token_data["details"] += ", keyterm"
        except Exception as term_error:
            logger.warning(f"Error extracting key terms: {term_error}")
            
        logger.info(f"Found {len(result)} words with parts of speech")
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing parts of speech: {str(e)}")
        # Fallback to a simple tokenization
        return [{"word": word.lower(), "pos": "unknown", "details": "", "lemma": word.lower()} 
                for word in sentence.split() if word not in '.,;!?"\'()[]{}']

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using spaCy's sentence segmentation.
    
    Args:
        text: The text to segment into sentences
        
    Returns:
        List of sentences
    """
    try:
        # Detect language of text
        lang = detect_language(text)
        # Load appropriate language model
        nlp = load_spacy_model(lang)
        # Process text with spaCy
        doc = nlp(text)
        # Extract sentences
        sentences = [sent.text.strip() for sent in doc.sents]
        logger.info(f"Split text into {len(sentences)} sentences")
        return sentences
    except Exception as e:
        logger.error(f"Error splitting text into sentences: {str(e)}")
        # Fallback to a simple regex-based approach
        logger.info("Using fallback sentence splitting")
        # Simple regex to split on sentence-ending punctuation
        simple_sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in simple_sentences if s.strip()]

def calculate_similarity(word1: str, word2: str) -> float:
    """
    Calculate string similarity between two words in the same language.
    
    Args:
        word1: First word
        word2: Second word
        
    Returns:
        Similarity score between 0 and 1
    """
    # Convert to lower case
    word1 = word1.lower()
    word2 = word2.lower()
    
    # Exact match
    if word1 == word2:
        return 1.0
    
    # Empty strings
    if not word1 or not word2:
        return 0.0
    
    # Length difference
    length_diff = abs(len(word1) - len(word2)) / max(len(word1), len(word2))
    length_similarity = 1 - length_diff
    
    # Character overlap
    common_chars = set(word1) & set(word2)
    if not common_chars:
        return 0.0
    
    char_similarity = len(common_chars) / (len(set(word1)) + len(set(word2)) - len(common_chars))
    
    # Simple edit distance (very basic implementation)
    # This could be improved with a proper Levenshtein distance
    distance = 0
    min_len = min(len(word1), len(word2))
    for i in range(min_len):
        if word1[i] != word2[i]:
            distance += 1
    
    distance += abs(len(word1) - len(word2))
    max_distance = max(len(word1), len(word2))
    edit_similarity = 1 - (distance / max_distance if max_distance > 0 else 0)
    
    # Combine the different measures
    similarity = (length_similarity * 0.3) + (char_similarity * 0.3) + (edit_similarity * 0.4)
    
    return similarity

def calculate_word_similarity(word1: str, word2: str, lang1: str, lang2: str) -> float:
    """
    Calculate similarity between two words in different languages.
    
    Args:
        word1: First word
        word2: Second word
        lang1: Language of the first word
        lang2: Language of the second word
        
    Returns:
        Similarity score between 0 and 1
    """
    # For same language, use direct string comparison
    if lang1 == lang2:
        return calculate_similarity(word1, word2)
    
    # Convert to lower case for comparison
    word1 = word1.lower()
    word2 = word2.lower()
    
    # Common prefixes or character patterns between languages
    if len(word1) > 2 and len(word2) > 2:
        # Check for common prefix (first 3 characters)
        if word1[:3] == word2[:3]:
            return 0.7
        
        # Check for common suffix (last 3 characters)
        if word1[-3:] == word2[-3:]:
            return 0.6
    
    # Count common characters
    common_chars = set(word1) & set(word2)
    if not common_chars:
        return 0.0
    
    # Calculate Jaccard similarity
    similarity = len(common_chars) / (len(set(word1) | set(word2)))
    
    # Adjust based on language pairs (some language pairs share more vocabulary)
    language_pair_boost = {
        ("en", "es"): 0.1,  # English-Spanish
        ("es", "en"): 0.1,
        ("en", "ca"): 0.05, # English-Catalan
        ("ca", "en"): 0.05,
        ("es", "ca"): 0.2,  # Spanish-Catalan (more similar)
        ("ca", "es"): 0.2
    }
    
    # Apply language pair specific boost
    similarity += language_pair_boost.get((lang1, lang2), 0)
    
    # Cap at 1.0
    return min(1.0, similarity)

def build_word_cooccurrence_network(text: str, language: str, window_size: int = 2, 
                                    min_freq: int = 1, include_pos: List[str] = None) -> nx.Graph:
    """
    Build a word co-occurrence network from text using textacy.
    
    Args:
        text: Text to analyze
        language: Language of text
        window_size: Size of sliding window for co-occurrence
        min_freq: Minimum frequency required for words to be included
        include_pos: List of POS tags to include (None = all)
        
    Returns:
        networkx.Graph with word co-occurrence network
    """
    try:
        # Load the appropriate language model
        nlp = load_spacy_model(language)
        
        # Create a textacy Doc
        doc = textacy.make_spacy_doc(text, lang=language) 
        
        # Define word filters
        pos_tags = include_pos if include_pos else ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]
        
        # Add count attribute to tokens
        # First get the frequency of each token
        token_counts = {}
        for token in doc:
            token_text = token.text.lower()
            if token_text not in token_counts:
                token_counts[token_text] = 0
            token_counts[token_text] += 1
        
        # Then add the count to each token as a custom attribute
        if not spacy.tokens.Token.has_extension("counts"):
            spacy.tokens.Token.set_extension("counts", default=0)
        
        for token in doc:
            token._.counts = token_counts.get(token.text.lower(), 0)
        
        # Filter terms: include only content words with specified POS tags
        def term_filter(term):
            return (
                term.pos_ in pos_tags and 
                not term.is_stop and 
                not term.is_punct and 
                term._.counts >= min_freq
            )
        
        # Build co-occurrence network with textacy
        graph = build_cooccurrence_network(
            doc,
            window_size=window_size,
            edge_weighting="count",
            term_filter=term_filter
        )
        
        logger.info(f"Built co-occurrence network with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph
    
    except Exception as e:
        logger.error(f"Error building co-occurrence network: {str(e)}")
        # Return an empty graph on failure
        return nx.Graph()

def visualize_cooccurrence_network(graph: nx.Graph, lang_code: Optional[str] = None) -> str:
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
            tooltip = f"Word: {node}<br>Co-occurrences: {graph.degree(node)}"
            
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
            
            # Add the edge
            net.add_edge(
                source,
                target,
                title=f"Co-occurrence: {weight}",
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

def get_network_stats(graph: nx.Graph) -> Dict[str, Any]:
    """
    Calculate various network statistics for a graph.
    
    Args:
        graph: networkx.Graph to analyze
        
    Returns:
        Dictionary of network statistics
    """
    if len(graph.nodes()) == 0:
        return {
            "node_count": 0,
            "edge_count": 0,
            "density": 0,
            "avg_degree": 0
        }
    
    try:
        stats = {
            "node_count": len(graph.nodes()),
            "edge_count": len(graph.edges()),
            "density": nx.density(graph),
            "avg_degree": sum(dict(graph.degree()).values()) / len(graph.nodes()),
            "connected_components": nx.number_connected_components(graph),
        }
        
        # Calculate centrality measures if there are enough nodes
        if len(graph.nodes()) > 1:
            # Degree centrality
            degree_cent = nx.degree_centrality(graph)
            # Get top nodes by degree centrality
            top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
            stats["top_degree_nodes"] = top_degree
            
            # Betweenness centrality (if graph is large enough)
            if len(graph.nodes()) > 2 and nx.is_connected(graph):
                betweenness_cent = nx.betweenness_centrality(graph)
                top_betweenness = sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:10]
                stats["top_betweenness_nodes"] = top_betweenness
        
        return stats
    
    except Exception as e:
        logger.error(f"Error calculating network stats: {str(e)}")
        return {
            "node_count": len(graph.nodes()),
            "edge_count": len(graph.edges()),
            "error": str(e)
        } 