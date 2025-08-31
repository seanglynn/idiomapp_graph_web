"""
Natural Language Processing utilities.
Uses textacy and spaCy for advanced NLP capabilities.
"""

import os
import re
import logging
import tempfile
from typing import List, Dict, Any, Optional

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

# Color scheme for languages (shared between functions)
LANGUAGE_COLORS = {
    "en": "#4361EE",  # Blue for English
    "es": "#FFD700",  # Yellow for Spanish
    "ca": "#FF3B30",  # Red for Catalan
    "en-related": "#90E0EF",  # Light blue for English related
    "es-related": "#FFF1A3",  # Light yellow for Spanish related
    "ca-related": "#FF8C7C"   # Light red for Catalan related
}

def get_language_color(lang_code: str, is_related: bool = False) -> str:
    """
    Get the standard color for a language.
    
    Args:
        lang_code: The language code (en, es, ca)
        is_related: Whether this is a related word (lighter color)
        
    Returns:
        Hex color code for the language
    """
    if is_related:
        key = f"{lang_code}-related"
        return LANGUAGE_COLORS.get(key, "#4CC9F0")  # Default to light blue
    else:
        return LANGUAGE_COLORS.get(lang_code, "#4CC9F0")  # Default to blue

# Model cache to prevent redundant loading
_MODEL_CACHE = {}

def load_spacy_model(language: str) -> spacy.language.Language:
    """
    Load the appropriate spaCy language model, downloading it if necessary.
    Uses caching to prevent redundant loading and downloading.
    
    Args:
        language: ISO language code (en, es, ca)
        
    Returns:
        Loaded spaCy language model
    """
    global _MODEL_CACHE
    
    # Get the appropriate model name
    model_name = LANG_MODELS.get(language, "en_core_web_sm")
    
    # Check if model is already in cache
    if language in _MODEL_CACHE:
        logger.info(f"Using cached model for {language}")
        return _MODEL_CACHE[language]
    
    try:
        # Try to load the model
        nlp = spacy.load(model_name)
        logger.info(f"Successfully loaded language model: {model_name}")
        _MODEL_CACHE[language] = nlp
        return nlp
    except OSError:
        # If model is not found, download it
        logger.info(f"Model {model_name} not found locally. Attempting to download...")
        try:
            # Use the Python API for more reliable download
            import subprocess
            import sys
            
            # Try downloading with subprocess first (more reliable in container environments)
            try:
                logger.info(f"Downloading {model_name} using subprocess")
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
                logger.info(f"Successfully downloaded {model_name} using subprocess")
            except subprocess.SubprocessError as sub_err:
                # If subprocess fails, try the normal spaCy CLI
                logger.warning(f"Subprocess download failed: {sub_err}. Trying spaCy API.")
                spacy.cli.download(model_name)
                logger.info(f"Successfully downloaded {model_name} using spaCy API")
            
            # Try loading again after download
            try:
                nlp = spacy.load(model_name)
                logger.info(f"Successfully loaded {model_name} after download")
                _MODEL_CACHE[language] = nlp
                return nlp
            except Exception as load_error:
                logger.error(f"Error loading model after download: {str(load_error)}")
                # Last resort - create a blank model
                logger.warning(f"Creating blank model for {language}")
                nlp = spacy.blank(language)
                _MODEL_CACHE[language] = nlp
                return nlp
                
        except Exception as download_error:
            # If downloading failed, create a blank model
            logger.error(f"Error downloading {model_name}: {str(download_error)}. Creating blank model.")
            nlp = spacy.blank(language)
            logger.warning(f"Created blank model for {language} as fallback")
            _MODEL_CACHE[language] = nlp
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
        # Load the appropriate model (uses cached model if available)
        nlp = load_spacy_model(language)
        
        # Check if this is a blank model with limited capabilities
        is_blank_model = len(nlp.pipeline) == 0
        
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
            
            # For blank models, the details will be limited
            if is_blank_model:
                details = "No detailed analysis available (using blank model)"
                result.append({
                    "word": token.text.lower(),
                    "pos": "unknown",
                    "details": details,
                    "lemma": token.text.lower(),
                    "dep": "unknown",
                    "is_entity": False,
                    "entity_type": None
                })
                continue
            
            # Get additional details for full models
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
        
        # Use textacy to extract additional information if not a blank model
        if not is_blank_model:
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

def calculate_word_similarity(word1: str, word2: str, lang1: str, lang2: str) -> dict:
    """
    Calculate similarity between two words in different languages using NLP techniques.
    Provides rich information about the relationship between words.
    
    Args:
        word1: First word
        word2: Second word
        lang1: Language of the first word
        lang2: Language of the second word
        
    Returns:
        Dictionary containing similarity score and relationship information
    """
    try:
        # Create a cache key for this word pair
        cache_key = f"{word1}_{lang1}_{word2}_{lang2}"
        
        # Check if we have a cached result
        if hasattr(calculate_word_similarity, "cache") and cache_key in calculate_word_similarity.cache:
            return calculate_word_similarity.cache[cache_key]
        
        # Initialize cache if doesn't exist yet
        if not hasattr(calculate_word_similarity, "cache"):
            calculate_word_similarity.cache = {}
        
        # For same language, use vector similarity
        if lang1 == lang2:
            return _calculate_same_language_similarity(word1, word2, lang1)
        
        # Load spaCy models for both languages
        try:
            nlp1 = load_spacy_model(lang1)
            nlp2 = load_spacy_model(lang2)
            
            # Process the words to get tokens
            doc1 = nlp1(word1.lower())
            doc2 = nlp2(word2.lower())
            
            # Get the tokens (use first token if multiple)
            token1 = doc1[0] if len(doc1) > 0 else None
            token2 = doc2[0] if len(doc2) > 0 else None
            
            if token1 is None or token2 is None:
                # Fallback to string similarity if tokens not available
                return _calculate_fallback_similarity(word1, word2, lang1, lang2)
            
            # Extract linguistic features
            lemma1 = token1.lemma_
            lemma2 = token2.lemma_
            pos1 = token1.pos_
            pos2 = token2.pos_
            
            # Calculate string similarity measures
            # Normalized edit distance (Levenshtein)
            from textdistance import levenshtein
            edit_distance = levenshtein.normalized_similarity(word1.lower(), word2.lower())
            
            # Calculate character overlap (Jaccard similarity)
            common_chars = set(word1.lower()) & set(word2.lower())
            char_overlap = len(common_chars) / (len(set(word1.lower()) | set(word2.lower())))
            
            # Check for cognates (words that look similar across languages)
            is_cognate = False
            cognate_confidence = 0.0
            
            # Cognate detection rules based on common patterns between language pairs
            if len(word1) > 3 and len(word2) > 3:
                # Common prefix
                prefix_match = word1.lower()[:3] == word2.lower()[:3]
                # Common suffix
                suffix_match = word1.lower()[-3:] == word2.lower()[-3:]
                
                if prefix_match and suffix_match:
                    is_cognate = True
                    cognate_confidence = 0.9
                elif prefix_match:
                    is_cognate = True
                    cognate_confidence = 0.7
                elif suffix_match:
                    is_cognate = True
                    cognate_confidence = 0.6
                elif edit_distance > 0.7:  # High string similarity
                    is_cognate = True
                    cognate_confidence = 0.8
            
            # Apply language pair specific boosting
            language_pair_boost = {
                ("en", "es"): 0.1,  # English-Spanish
                ("es", "en"): 0.1,
                ("en", "ca"): 0.05, # English-Catalan
                ("ca", "en"): 0.05,
                ("es", "ca"): 0.2,  # Spanish-Catalan (very similar)
                ("ca", "es"): 0.2
            }
            
            # Calculate the base similarity score
            pos_match = pos1 == pos2
            
            # Apply adjustments based on linguistic features
            base_similarity = (
                (edit_distance * 0.4) +
                (char_overlap * 0.3) +
                (0.1 if pos_match else 0) +
                language_pair_boost.get((lang1, lang2), 0)
            )
            
            # Determine relationship type
            if is_cognate:
                relationship_type = "cognate"
                confidence = cognate_confidence
                description = f"Cognate words that share common etymology ({cognate_confidence:.1f})"
            elif edit_distance > 0.8:
                relationship_type = "direct_translation"
                confidence = edit_distance
                description = f"Direct translation equivalent ({edit_distance:.2f})"
            elif pos_match and base_similarity > 0.4:
                relationship_type = "semantic_equivalent"
                confidence = base_similarity
                description = f"Semantic equivalent ({base_similarity:.2f})"
            elif base_similarity > 0.3:
                relationship_type = "related_term"
                confidence = base_similarity
                description = f"Related term ({base_similarity:.2f})"
            else:
                relationship_type = "weak_relation"
                confidence = base_similarity
                description = f"Weak relation ({base_similarity:.2f})"
            
            # Structure detailed similarity information
            similarity_info = {
                "score": min(1.0, base_similarity),  # Cap at 1.0
                "relationship_type": relationship_type,
                "confidence": confidence,
                "description": description,
                "linguistic_features": {
                    "pos_match": pos_match,
                    "pos1": pos1,
                    "pos2": pos2,
                    "lemma1": lemma1,
                    "lemma2": lemma2,
                    "edit_distance": edit_distance,
                    "char_overlap": char_overlap,
                    "is_cognate": is_cognate
                }
            }
            
            # Cache the result
            calculate_word_similarity.cache[cache_key] = similarity_info
            return similarity_info
            
        except Exception as e:
            logger.error(f"Error in advanced word similarity: {str(e)}")
            # Fall back to the basic similarity if there's an error
            return _calculate_fallback_similarity(word1, word2, lang1, lang2)
    except Exception as e:
        # If any unexpected error occurs, return a default dictionary
        logger.error(f"Critical error in calculate_word_similarity for {word1}/{word2}: {str(e)}")
        return {
            "score": 0.0,
            "relationship_type": "error",
            "confidence": 0.0,
            "description": f"Error calculating similarity: {str(e)}",
            "linguistic_features": {}
        }

def _calculate_same_language_similarity(word1: str, word2: str, language: str) -> dict:
    """Calculate similarity for words in the same language"""
    try:
        # Load spaCy model
        nlp = load_spacy_model(language)
        
        # Get vector similarity if available
        doc1 = nlp(word1.lower())
        doc2 = nlp(word2.lower())
        
        # Check if words are identical
        if word1.lower() == word2.lower():
            return {
                "score": 1.0,
                "relationship_type": "identical",
                "confidence": 1.0,
                "description": "Identical words",
                "linguistic_features": {
                    "pos_match": True,
                    "identical": True
                }
            }
        
        # Get tokens
        token1 = doc1[0] if len(doc1) > 0 else None
        token2 = doc2[0] if len(doc2) > 0 else None
        
        if token1 is None or token2 is None:
            # Fallback if tokens not available
            similarity = calculate_similarity(word1, word2)
            return {
                "score": similarity,
                "relationship_type": "string_similar",
                "confidence": similarity,
                "description": f"String similarity ({similarity:.2f})",
                "linguistic_features": {}
            }
        
        # Check if lemmas are the same (same base word)
        same_lemma = token1.lemma_ == token2.lemma_
        pos_match = token1.pos_ == token2.pos_
        
        # Try vector similarity if available
        vector_similarity = 0.0
        if token1.has_vector and token2.has_vector:
            vector_similarity = token1.similarity(token2)
        
        # Calculate string similarity as fallback
        string_similarity = calculate_similarity(word1, word2)
        
        # Determine base score based on available metrics
        if vector_similarity > 0:
            base_score = vector_similarity * 0.6 + string_similarity * 0.4
        else:
            base_score = string_similarity
            
        # Boost for same lemma or POS
        if same_lemma:
            base_score = min(1.0, base_score + 0.3)
            relationship_type = "morphological_variant"
            description = f"Morphological variants of same word ({base_score:.2f})"
        elif pos_match and base_score > 0.5:
            relationship_type = "same_pos_semantic"
            description = f"Same part of speech with semantic similarity ({base_score:.2f})"
        elif base_score > 0.7:
            relationship_type = "highly_similar"
            description = f"Highly similar words ({base_score:.2f})"
        else:
            relationship_type = "somewhat_related"
            description = f"Somewhat related words ({base_score:.2f})"
        
        return {
            "score": base_score,
            "relationship_type": relationship_type,
            "confidence": base_score,
            "description": description,
            "linguistic_features": {
                "pos_match": pos_match,
                "pos1": token1.pos_,
                "pos2": token2.pos_,
                "same_lemma": same_lemma,
                "lemma1": token1.lemma_,
                "lemma2": token2.lemma_,
                "vector_similarity": vector_similarity if token1.has_vector and token2.has_vector else None
            }
        }
    except Exception as e:
        logger.error(f"Error in same-language similarity: {str(e)}")
        # Simple fallback
        similarity = calculate_similarity(word1, word2)
        return {
            "score": similarity,
            "relationship_type": "string_similar",
            "confidence": similarity,
            "description": f"String similarity ({similarity:.2f})",
            "linguistic_features": {}
        }

def _calculate_fallback_similarity(word1: str, word2: str, lang1: str, lang2: str) -> dict:
    """Fallback similarity calculation when NLP methods fail"""
    # Convert to lower case for comparison
    word1 = word1.lower()
    word2 = word2.lower()
    
    # Common prefixes or character patterns between languages
    if len(word1) > 2 and len(word2) > 2:
        # Check for common prefix (first 3 characters)
        if word1[:3] == word2[:3]:
            return {
                "score": 0.7,
                "relationship_type": "common_prefix",
                "confidence": 0.7,
                "description": "Common prefix suggests possible relation",
                "linguistic_features": {
                    "common_prefix": word1[:3]
                }
            }
        
        # Check for common suffix (last 3 characters)
        if word1[-3:] == word2[-3:]:
            return {
                "score": 0.6,
                "relationship_type": "common_suffix",
                "confidence": 0.6,
                "description": "Common suffix suggests possible relation",
                "linguistic_features": {
                    "common_suffix": word1[-3:]
                }
            }
    
    # Count common characters
    common_chars = set(word1) & set(word2)
    if not common_chars:
        return {
            "score": 0.0,
            "relationship_type": "unrelated",
            "confidence": 0.9,
            "description": "No apparent relation",
            "linguistic_features": {}
        }
    
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
    similarity = min(1.0, similarity)
    
    return {
        "score": similarity,
        "relationship_type": "char_similarity",
        "confidence": similarity,
        "description": f"Character similarity ({similarity:.2f})",
        "linguistic_features": {
            "common_chars": list(common_chars),
            "char_overlap": similarity
        }
    }

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
        
        # Check if this is a blank model (limited functionality)
        is_blank_model = len(nlp.pipeline) == 0
        
        if is_blank_model:
            logger.warning(f"Using blank model for {language}. Creating simplified co-occurrence network.")
            return _build_simple_cooccurrence_network(text, window_size, min_freq)
        
        # Create a textacy Doc
        try:
            doc = textacy.make_spacy_doc(text, lang=nlp)
        except ValueError as e:
            logger.warning(f"Error creating textacy Doc: {e}. Using simplified approach.")
            return _build_simple_cooccurrence_network(text, window_size, min_freq)
        
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
            try:
                spacy.tokens.Token.set_extension("counts", default=0)
            except ValueError:
                # Extension already exists, ignore
                pass
        
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
        try:
            graph = build_cooccurrence_network(
                doc,
                window_size=window_size,
                edge_weighting="count",
                term_filter=term_filter
            )
            
            # If graph is empty, fall back to the simple approach
            if len(graph.nodes()) == 0:
                logger.warning(f"Empty graph from textacy. Using simplified approach for {language}.")
                return _build_simple_cooccurrence_network(text, window_size, min_freq)
                
            logger.info(f"Built co-occurrence network with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            return graph
            
        except Exception as e:
            logger.warning(f"Error in textacy network building: {e}. Using simplified approach.")
            return _build_simple_cooccurrence_network(text, window_size, min_freq)
    
    except Exception as e:
        logger.error(f"{e.__class__.__name__} building co-occurrence network: {str(e)}")
        # Try the simple approach as a last resort
        try:
            return _build_simple_cooccurrence_network(text, window_size, min_freq)
        except Exception:
            # If everything fails, return an empty graph
            return nx.Graph()

def _build_simple_cooccurrence_network(text: str, window_size: int = 2, min_freq: int = 1) -> nx.Graph:
    """
    Build a simple co-occurrence network without requiring advanced NLP.
    This is a fallback when language models or textacy are not available.
    
    Args:
        text: Text to process
        window_size: Size of sliding window
        min_freq: Minimum word frequency to include
        
    Returns:
        networkx.Graph with word co-occurrence network
    """
    # Create a new graph
    G = nx.Graph()
    
    # Tokenize text (simple approach)
    # Remove punctuation and convert to lowercase
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = clean_text.split()
    
    # Skip if no words
    if not words:
        return G
        
    # Count word frequencies
    word_counts = {}
    for word in words:
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1
    
    # Filter words by frequency
    filtered_words = []
    for i, word in enumerate(words):
        if word_counts[word] >= min_freq:
            filtered_words.append((i, word))
    
    # Add nodes to the graph
    for _, word in filtered_words:
        if not G.has_node(word):
            G.add_node(word)
    
    # Add edges based on co-occurrence in the sliding window
    for i, (pos1, word1) in enumerate(filtered_words):
        # Look ahead within window_size
        for j in range(i + 1, len(filtered_words)):
            pos2, word2 = filtered_words[j]
            # Check if within window and not the same word
            if pos2 - pos1 <= window_size and word1 != word2:
                # Add or update edge
                if G.has_edge(word1, word2):
                    G[word1][word2]["weight"] += 1
                else:
                    G.add_edge(word1, word2, weight=1)
    
    logger.info(f"Built simple co-occurrence network with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G

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
            tooltip = f"Word: {node}; Co-occurrences: {graph.degree(node)}"
            
            # Get color for the language from our shared function
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
            
            # Add the edge
            net.add_edge(
                source_id,
                target_id,
                title=f"Co-occurrence: {weight}",
                width=width,
                color="#FFFFFF" if weight > 2 else "#AAAAAA"
            )
        
        # TODO AVOID: Create a temporary HTML file
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