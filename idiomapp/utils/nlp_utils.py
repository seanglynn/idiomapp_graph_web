"""
Natural Language Processing utilities.
Uses textacy and spaCy for advanced NLP capabilities.
"""
import re
import asyncio
import logging
from typing import List, Dict, Any, Optional

# NLP libraries
import spacy
import textacy
from textacy.extract.keyterms import textrank
from textacy.representations.network import build_cooccurrence_network
import networkx as nx
from langdetect import detect, LangDetectException

# Color scheme imported from central config
from pydantic import ValidationError

from idiomapp.config import GROUP_COLORS as LANGUAGE_COLORS, LANG_MODELS
from idiomapp.utils.schemas import WordAnalysis, WORD_ANALYSIS_GROUPS, prompt_example
from idiomapp.utils.analysis_cache import get_word_analysis_cache

# Setup logging
logger = logging.getLogger(__name__)


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
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", model_name]
                )
                logger.info(f"Successfully downloaded {model_name} using subprocess")
            except subprocess.SubprocessError as sub_err:
                # If subprocess fails, try the normal spaCy CLI
                logger.warning(
                    f"Subprocess download failed: {sub_err}. Trying spaCy API."
                )
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
                # Try alternative models as fallback
                nlp = _try_alternative_models(language)
                if nlp:
                    _MODEL_CACHE[language] = nlp
                    return nlp
                # Last resort - create a blank model
                logger.warning(f"Creating blank model for {language}")
                nlp = spacy.blank(language)
                _MODEL_CACHE[language] = nlp
                return nlp

        except Exception as download_error:
            # If downloading failed, create a blank model
            logger.error(
                f"Error downloading {model_name}: {str(download_error)}. Creating blank model."
            )
            # Use the language code directly for blank models (spaCy handles this correctly)
            nlp = spacy.blank(language)
            logger.warning(f"Created blank model for {language} as fallback")
            _MODEL_CACHE[language] = nlp
            return nlp


def _try_alternative_models(language: str) -> Optional[spacy.language.Language]:
    """
    Try to load alternative SpaCy models for a language.

    Args:
        language: ISO language code

    Returns:
        Loaded SpaCy model or None if all alternatives fail
    """
    # Alternative model mappings for better fallback
    # Based on https://spacy.io/models - using web-trained models for better general text handling
    alternative_models = {
        "es": ["es_core_web_sm", "es_core_web_md", "es_core_web_lg", "es_core_news_sm"],
        "ca": ["ca_core_web_sm", "ca_core_web_md", "ca_core_web_lg", "ca_core_news_sm"],
        "en": [
            "en_core_web_md",
            "en_core_web_lg",
            "en_core_web_trf",
            "en_core_news_sm",
        ],
    }

    alternatives = alternative_models.get(language, [])

    for alt_model in alternatives:
        try:
            logger.info(f"Trying alternative model: {alt_model}")
            nlp = spacy.load(alt_model)
            logger.info(f"Successfully loaded alternative model: {alt_model}")
            return nlp
        except OSError:
            logger.warning(f"Alternative model {alt_model} not available")
            continue

    # If no alternatives work, try downloading a smaller model
    try:
        fallback_model = f"{language}_core_web_sm"
        logger.info(f"Attempting to download fallback model: {fallback_model}")
        spacy.cli.download(fallback_model)
        nlp = spacy.load(fallback_model)
        logger.info(f"Successfully loaded fallback model: {fallback_model}")
        return nlp
    except Exception as e:
        logger.error(f"Failed to load fallback model {fallback_model}: {e}")
        return None


def clear_model_cache():
    """Clear the SpaCy model cache to force reloading of models."""
    _MODEL_CACHE.clear()
    logger.info("SpaCy model cache cleared")


def get_model_status():
    """Get the status of loaded SpaCy models."""
    status = {}
    for lang, model in _MODEL_CACHE.items():
        if hasattr(model, "vocab") and len(model.vocab) > 1000:
            status[lang] = "full_model"
        else:
            status[lang] = "blank_model"
    return status


def ensure_models_available():
    """
    Ensure that all required SpaCy models are available.
    Downloads missing models and provides status information.
    """
    import subprocess
    import sys

    models_to_check = list(LANG_MODELS.values())
    missing_models = []
    available_models = []

    for model in models_to_check:
        try:
            # Try to load the model
            spacy.load(model)
            available_models.append(model)
            logger.info(f"Model {model} is available")
        except OSError:
            missing_models.append(model)
            logger.warning(f"Model {model} is missing")

    # Download missing models
    for model in missing_models:
        try:
            logger.info(f"Downloading missing model: {model}")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            logger.info(f"Successfully downloaded {model}")
            available_models.append(model)
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to download {model}: {e}")

    return {
        "available": available_models,
        "missing": missing_models,
        "total_required": len(models_to_check),
        "total_available": len(available_models),
    }


def get_recommended_model_size(language: str, use_case: str = "general") -> str:
    """
    Get recommended model size based on language and use case.

    Args:
        language: Language code
        use_case: Use case ("general", "production", "research")

    Returns:
        Recommended model name
    """
    # Based on https://spacy.io/models recommendations
    recommendations = {
        "general": {
            "en": "en_core_web_sm",  # Fast, efficient for most use cases
            "es": "es_core_web_sm",  # Good balance of speed and accuracy
            "ca": "ca_core_web_sm",  # Catalan web model
        },
        "production": {
            "en": "en_core_web_md",  # Better accuracy for production
            "es": "es_core_web_md",  # Improved accuracy for Spanish
            "ca": "ca_core_web_md",  # Better Catalan accuracy
        },
        "research": {
            "en": "en_core_web_lg",  # Best accuracy for research
            "es": "es_core_web_lg",  # Highest Spanish accuracy
            "ca": "ca_core_web_lg",  # Best Catalan accuracy
        },
    }

    return recommendations.get(use_case, recommendations["general"]).get(
        language, f"{language}_core_web_sm"
    )


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

        # Check if detected language is supported, otherwise default to English
        if detected in LANG_MODELS:
            return detected
        else:
            logger.info(
                f"Detected language '{detected}' not supported, defaulting to English"
            )
            return "en"
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
            "PART": "particle",
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
                result.append(
                    {
                        "word": token.text.lower(),
                        "pos": "unknown",
                        "details": details,
                        "lemma": token.text.lower(),
                        "dep": "unknown",
                        "is_entity": False,
                        "entity_type": None,
                    }
                )
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
            result.append(
                {
                    "word": token.text.lower(),
                    "pos": pos,
                    "details": details,
                    "lemma": token.lemma_,
                    "dep": token.dep_,
                    "is_entity": bool(token.ent_type_),
                    "entity_type": token.ent_type_ if token.ent_type_ else None,
                }
            )

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
        return [
            {
                "word": word.lower(),
                "pos": "unknown",
                "details": "",
                "lemma": word.lower(),
            }
            for word in sentence.split()
            if word not in ".,;!?\"'()[]{}"
        ]


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
        simple_sentences = re.split(r"(?<=[.!?])\s+", text)
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

    char_similarity = len(common_chars) / (
        len(set(word1)) + len(set(word2)) - len(common_chars)
    )

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
    similarity = (
        (length_similarity * 0.3) + (char_similarity * 0.3) + (edit_similarity * 0.4)
    )

    return similarity


def detect_cognate(word1: str, word2: str) -> Optional[float]:
    """
    Check whether two words look like cognates - words that share a common
    etymology and so look alike across languages (e.g. "nation"/"nación").

    Reliable only for what it actually measures - visual/etymological
    similarity - not for "this is the correct translation of that word": a
    lexically different but correct translation (e.g. "moon"/"luna") won't be
    caught here, and that's by design; see `process_sentence_pair`'s
    `alignment_pairs` for the LLM-provided signal that does catch that case.

    Returns:
        A confidence in [0.6, 0.9] if the words look like cognates, else None.
    """
    if len(word1) <= 3 or len(word2) <= 3:
        return None

    from textdistance import levenshtein

    w1, w2 = word1.lower(), word2.lower()
    prefix_match = w1[:3] == w2[:3]
    suffix_match = w1[-3:] == w2[-3:]

    if prefix_match and suffix_match:
        return 0.9
    if prefix_match:
        return 0.7
    if suffix_match:
        return 0.6
    if levenshtein.normalized_similarity(w1, w2) > 0.7:
        return 0.8
    return None


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
                "linguistic_features": {"pos_match": True, "identical": True},
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
                "linguistic_features": {},
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
            description = (
                f"Same part of speech with semantic similarity ({base_score:.2f})"
            )
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
                "vector_similarity": vector_similarity
                if token1.has_vector and token2.has_vector
                else None,
            },
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
            "linguistic_features": {},
        }


def _word_pos_details(word_data: dict) -> tuple:
    """
    Extract (word, pos, details) from a POS-tagged word dict.

    `analyze_parts_of_speech()` always returns dicts - including its own
    exception-fallback path, which still builds `{"word": ..., "pos": "unknown",
    ...}` rather than a bare string - so no other input shape reaches this.
    """
    return word_data["word"], word_data["pos"], word_data.get("details", "")


def _add_word_nodes(word_pos_list, lang, sentence_group, graph_data, added_nodes):
    """Add a graph node for each word in a POS-tagged word list, skipping duplicates."""
    for word_data in word_pos_list:
        word, pos, details = _word_pos_details(word_data)

        node_id = f"{word}_{lang}{sentence_group}"
        if node_id in added_nodes:
            continue

        graph_data["nodes"].append(
            {
                "id": node_id,
                "label": word,
                "language": lang,
                "pos": pos,
                "details": details,
                "node_type": "primary",
                "group": f"{lang}{sentence_group}",
                "sentence_group": sentence_group,
            }
        )
        added_nodes.add(node_id)


def process_sentence_pair(
    source_sentence,
    target_sentence,
    source_lang,
    target_lang,
    graph_data,
    added_nodes,
    word_relations_cache,
    sentence_group="",
    alignment_pairs=frozenset(),
):
    """
    Process a pair of sentences in different languages and add them to the graph.

    alignment_pairs: a set of (source_word, target_word) tuples, lowercased,
    that the translation LLM reported as corresponding to each other - see
    `idiomapp.streamlit.app.analyze_translation`. Used to draw `translation`
    edges; independently, `detect_cognate` draws `cognate` edges wherever two
    words look alike, regardless of alignment - a pair can honestly be both,
    or either, so both checks always run.
    """

    logger.info(f"Processing sentence pair: {source_lang} to {target_lang}")

    try:
        # Analyze parts of speech for source and target sentence
        source_pos = analyze_parts_of_speech(source_sentence, source_lang)
        target_pos = analyze_parts_of_speech(target_sentence, target_lang)

        _add_word_nodes(
            source_pos, source_lang, sentence_group, graph_data, added_nodes
        )
        _add_word_nodes(
            target_pos, target_lang, sentence_group, graph_data, added_nodes
        )

        for source_word_data in source_pos:
            source_word, source_pos_val, _ = _word_pos_details(source_word_data)
            source_id = f"{source_word}_{source_lang}{sentence_group}"

            for target_word_data in target_pos:
                target_word, target_pos_val, _ = _word_pos_details(target_word_data)
                target_id = f"{target_word}_{target_lang}{sentence_group}"

                try:
                    if (source_word.lower(), target_word.lower()) in alignment_pairs:
                        graph_data["edges"].append(
                            {
                                "from": source_id,
                                "to": target_id,
                                "relation": "translation",
                                "strength": 1.0,
                                "label": "translation",
                                "description": "Reported as a translation pair",
                                "title": f"{source_word} ({source_pos_val or '?'}) → {target_word} ({target_pos_val or '?'})",
                            }
                        )

                    cognate_confidence = detect_cognate(source_word, target_word)
                    if cognate_confidence is not None:
                        graph_data["edges"].append(
                            {
                                "from": source_id,
                                "to": target_id,
                                "relation": "cognate",
                                "strength": cognate_confidence,
                                "label": "cognate",
                                "description": f"Cognate words that share common etymology ({cognate_confidence:.1f})",
                                "title": f"{source_word} ({source_pos_val or '?'}) ↔ {target_word} ({target_pos_val or '?'})",
                                "dashes": True,
                            }
                        )
                except Exception as e:
                    logger.error(
                        f"Error processing word pair {source_word}/{target_word}: {type(e).__name__}: {str(e)}"
                    )
                    continue

        # Process related words for source and target sentences
        try:
            process_related_words(
                source_pos,
                source_lang,
                target_lang,
                graph_data,
                added_nodes,
                word_relations_cache,
                sentence_group,
            )
        except Exception as e:
            logger.error(
                f"Error processing source related words: {type(e).__name__}: {str(e)}"
            )

        try:
            process_related_words(
                target_pos,
                target_lang,
                source_lang,
                graph_data,
                added_nodes,
                word_relations_cache,
                sentence_group,
                is_target=True,
            )
        except Exception as e:
            logger.error(
                f"Error processing target related words: {type(e).__name__}: {str(e)}"
            )

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
                        if node2.get("node_type", "") != "primary" or node2.get(
                            "language", ""
                        ) != node1.get("language", ""):
                            continue

                        pos2 = node2.get("pos", "unknown")
                        if pos2 == pos1:
                            try:
                                # Both nodes are already guaranteed same-language by
                                # the skip condition above - go straight to the
                                # same-language (real word-vector-based) path.
                                similarity_info = _calculate_same_language_similarity(
                                    node1["label"],
                                    node2["label"],
                                    node1.get("language", "en"),
                                )

                                # Extract similarity score and information
                                similarity_score = similarity_info.get("score", 0)
                                similarity_info.get(
                                    "relationship_type", "cross_sentence"
                                )
                                description = similarity_info.get(
                                    "description",
                                    f"Related {pos1} words across sentences",
                                )

                                # Connect only if there's some similarity or same POS for key types
                                if similarity_score >= 0.3 or pos1 in [
                                    "noun",
                                    "verb",
                                    "adjective",
                                ]:
                                    # Create a tooltip with linguistic information
                                    tooltip = f"{description}; Same part of speech: {pos1}; Similarity: {similarity_score:.2f}"

                                    graph_data["edges"].append(
                                        {
                                            "from": node1["id"],
                                            "to": node2["id"],
                                            "relation": "cross_sentence",
                                            "strength": max(0.4, similarity_score),
                                            "label": f"related {pos1}",
                                            "description": description,
                                            "title": tooltip,
                                            "color": "#AA44BB",  # Purple for cross-sentence
                                            "dashes": True,
                                        }
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error in cross-sentence processing: {str(e)}"
                                )
                                continue
    except Exception as e:
        logger.error(f"Error in add_cross_sentence_relationships: {str(e)}")
        # Don't re-raise, allow processing to continue


def add_cross_language_relationships(graph_data, target_langs):
    """Add relationships between words in different languages"""
    logger.info(
        f"Adding cross-language relationships for {len(target_langs)} languages"
    )

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
                for pos in set(nodes_by_lang_pos[lang1].keys()) & set(
                    nodes_by_lang_pos[lang2].keys()
                ):
                    for node1 in nodes_by_lang_pos[lang1][pos]:
                        for node2 in nodes_by_lang_pos[lang2][pos]:
                            try:
                                # No alignment data reaches this function - it only
                                # ever compares target-vs-target nodes (e.g. es<->ca),
                                # and this app never translates target-to-target - so
                                # cognate detection is the only signal available here.
                                confidence = detect_cognate(
                                    node1["label"], node2["label"]
                                )

                                if confidence is not None:
                                    same_sentence = node1.get(
                                        "sentence_group", ""
                                    ) == node2.get("sentence_group", "")

                                    tooltip = (
                                        f"Cognate words that share common etymology "
                                        f"({confidence:.1f}); {node1['label']} ({lang1}) "
                                        f"↔ {node2['label']} ({lang2})"
                                    )
                                    if same_sentence:
                                        tooltip += "; Same sentence ✓"

                                    graph_data["edges"].append(
                                        {
                                            "from": node1["id"],
                                            "to": node2["id"],
                                            "relation": "cognate",
                                            "strength": confidence,
                                            "label": "cognate",
                                            "description": "Cognate words that share common etymology",
                                            "title": tooltip,
                                            "dashes": True,
                                        }
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error processing cross-language pair: {str(e)}"
                                )
                                continue
    except Exception as e:
        logger.error(f"Error in add_cross_language_relationships: {str(e)}")
        # Don't re-raise - allow processing to continue


# Edge strength by relation type, for related-word edges in process_related_words.
_RELATION_STRENGTHS = {
    "synonym": 0.9,
    "antonym": 0.7,
    "hypernym": 0.6,
    "hyponym": 0.6,
    "contextual": 0.5,
}


def process_related_words(
    words_data,
    source_lang,
    target_lang,
    graph_data,
    added_nodes,
    word_relations_cache,
    sentence_group="",
    is_target=False,
):
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
        word, pos, _ = _word_pos_details(word_data)

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
            graph_data["nodes"].append(
                {
                    "id": related_id,
                    "label": related_word,
                    "language": lang,
                    "pos": pos,  # Assume same POS as original word
                    "details": relation_type,
                    "node_type": "related",
                    "group": f"{lang}-related{sentence_group}",
                    "sentence_group": sentence_group,
                }
            )
            added_nodes.add(related_id)

            # Add edge from original word to related word, strength by relation type
            strength = _RELATION_STRENGTHS.get(relation_type, 0.5)

            graph_data["edges"].append(
                {
                    "from": word_id,
                    "to": related_id,
                    "relation": relation_type,
                    "strength": strength,
                    "label": relation_type,
                }
            )


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
                ("rating", "contextual"),
            ]
        elif word == "happy":
            return [
                ("joyful", "synonym"),
                ("sad", "antonym"),
                ("emotion", "hypernym"),
                ("ecstatic", "hyponym"),
                ("birthday", "contextual"),
            ]

    # Some common word relationships in Spanish
    elif language == "es":
        if word == "bueno":
            return [
                ("excelente", "synonym"),
                ("malo", "antonym"),
                ("calidad", "hypernym"),
                ("genial", "synonym"),
                ("valoración", "contextual"),
            ]
        elif word == "feliz":
            return [
                ("alegre", "synonym"),
                ("triste", "antonym"),
                ("emoción", "hypernym"),
                ("extático", "hyponym"),
                ("cumpleaños", "contextual"),
            ]

    # Some common word relationships in Catalan
    elif language == "ca":
        if word == "bo":
            return [
                ("excel·lent", "synonym"),
                ("dolent", "antonym"),
                ("qualitat", "hypernym"),
                ("genial", "synonym"),
                ("valoració", "contextual"),
            ]
        elif word == "feliç":
            return [
                ("content", "synonym"),
                ("trist", "antonym"),
                ("emoció", "hypernym"),
                ("extàtic", "hyponym"),
                ("aniversari", "contextual"),
            ]

    # Default: return empty list if no predefined relations
    return []


def merge_language_graphs(graph_data_dict):
    """Merge multiple language graphs into a single graph with cross-language connections"""
    if not graph_data_dict or len(graph_data_dict) == 0:
        return None

    # Create a new graph combining all nodes and edges
    merged_graph = {
        "nodes": [],
        "edges": [],
        "metadata": {
            "source_lang": next(iter(graph_data_dict.values()))["metadata"][
                "source_lang"
            ],
            "target_langs": [],
            "source_text": next(iter(graph_data_dict.values()))["metadata"][
                "source_text"
            ],
            "translations": {},
        },
    }

    # Track all nodes we've added to avoid duplicates
    added_nodes = set()

    # Add nodes and edges from each language graph
    for lang, graph in graph_data_dict.items():
        # Update metadata
        merged_graph["metadata"]["target_langs"].append(lang)
        if "translations" in graph["metadata"]:
            merged_graph["metadata"]["translations"][lang] = graph["metadata"][
                "translations"
            ]

        # Add nodes. A shallow copy is enough - every node/edge dict here is flat
        # (strings/floats/bools, no nested mutable containers) and nothing after
        # the merge mutates the copies in place.
        for node in graph["nodes"]:
            if node["id"] not in added_nodes:
                merged_graph["nodes"].append(dict(node))
                added_nodes.add(node["id"])

        # Add edges
        for edge in graph["edges"]:
            merged_graph["edges"].append(dict(edge))

    # Now add cross-language relationships
    target_langs = merged_graph["metadata"]["target_langs"]
    if len(target_langs) > 1:
        add_cross_language_relationships(merged_graph, target_langs)

    logger.info(f"Merged {len(graph_data_dict)} language graphs into a single graph")
    return merged_graph


def build_word_cooccurrence_network(
    text: str,
    language: str,
    window_size: int = 2,
    min_freq: int = 1,
    include_pos: List[str] = None,
) -> nx.Graph:
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
            logger.warning(
                f"Using blank model for {language}. Creating simplified co-occurrence network."
            )
            return _build_simple_cooccurrence_network(text, window_size, min_freq)

        # Create a textacy Doc
        try:
            doc = textacy.make_spacy_doc(text, lang=nlp)
        except ValueError as e:
            logger.warning(
                f"Error creating textacy Doc: {e}. Using simplified approach."
            )
            return _build_simple_cooccurrence_network(text, window_size, min_freq)

        # Define word filters
        pos_tags = (
            include_pos if include_pos else ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]
        )

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
                term.pos_ in pos_tags
                and not term.is_stop
                and not term.is_punct
                and term._.counts >= min_freq
            )

        # Build co-occurrence network with textacy
        try:
            # textacy 0.13 expects Sequence[str], not a spaCy Doc
            terms = [token.text.lower() for token in doc if term_filter(token)]

            if not terms:
                logger.warning(f"No terms passed POS/frequency filter for {language}")
                return _build_simple_cooccurrence_network(text, window_size, min_freq)

            graph = build_cooccurrence_network(
                terms,
                window_size=window_size,
                edge_weighting="count",
            )

            # If graph is empty, fall back to the simple approach
            if len(graph.nodes()) == 0:
                logger.warning(
                    f"Empty graph from textacy. Using simplified approach for {language}."
                )
                return _build_simple_cooccurrence_network(text, window_size, min_freq)

            logger.info(
                f"Built co-occurrence network with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
            )
            return graph

        except Exception as e:
            logger.warning(
                f"Error in textacy network building: {e}. Using simplified approach."
            )
            return _build_simple_cooccurrence_network(text, window_size, min_freq)

    except Exception as e:
        logger.error(f"{e.__class__.__name__} building co-occurrence network: {str(e)}")
        # Try the simple approach as a last resort
        try:
            return _build_simple_cooccurrence_network(text, window_size, min_freq)
        except Exception as fallback_error:
            logger.error(f"Fallback approach also failed: {fallback_error}")
            # If everything fails, return an empty graph
            return nx.Graph()


def _build_simple_cooccurrence_network(
    text: str, window_size: int = 2, min_freq: int = 1
) -> nx.Graph:
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
    clean_text = re.sub(r"[^\w\s]", "", text.lower())
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

    logger.info(
        f"Built simple co-occurrence network with {len(G.nodes)} nodes and {len(G.edges)} edges"
    )
    return G


def get_network_stats(graph: nx.Graph) -> Dict[str, Any]:
    """
    Calculate various network statistics for a graph.

    Args:
        graph: networkx.Graph to analyze

    Returns:
        Dictionary of network statistics
    """
    if len(graph.nodes()) == 0:
        return {"node_count": 0, "edge_count": 0, "density": 0, "avg_degree": 0}

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
            top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[
                :10
            ]
            stats["top_degree_nodes"] = top_degree

            # Betweenness centrality (if graph is large enough)
            if len(graph.nodes()) > 2 and nx.is_connected(graph):
                betweenness_cent = nx.betweenness_centrality(graph)
                top_betweenness = sorted(
                    betweenness_cent.items(), key=lambda x: x[1], reverse=True
                )[:10]
                stats["top_betweenness_nodes"] = top_betweenness

        return stats

    except Exception as e:
        logger.error(f"Error calculating network stats: {str(e)}")
        return {
            "node_count": len(graph.nodes()),
            "edge_count": len(graph.edges()),
            "error": str(e),
        }


async def analyze_word_linguistics(
    word: str, language: str, client=None
) -> Dict[str, Any]:
    """
    Analyze a word's linguistic properties using LLM for rich language learning information.

    The LLM half of this is cached to disk (see `analysis_cache.py`), keyed on the
    word, language, and the provider/model that produced it - a repeat request for
    the same combination is served from the cache instead of calling the LLM again.
    A cache hit skips spaCy too, since the cached dict already has everything a
    fresh run would produce.

    Args:
        word: The word to analyze
        language: Code (en, es, ca)
        client: Optional LLM client for enhanced analysis

    Returns:
        Dictionary with comprehensive linguistic information
    """
    cache = get_word_analysis_cache()
    provider = model_name = None
    if client is not None:
        status = client.get_model_status()
        provider, model_name = status.get("provider"), status.get("model_name")
        cached = cache.get(word, language, provider, model_name)
        if cached is not None:
            logger.info(
                f"Using cached word analysis for '{word}' ({language}, {provider}:{model_name})"
            )
            return cached

    try:
        # Load spaCy model for basic analysis
        nlp = load_spacy_model(language)
        doc = nlp(word)

        if len(doc) == 0:
            return {"error": "Could not process word"}

        token = doc[0]

        # Check if we're using a blank model (limited linguistic knowledge)
        is_blank_model = not hasattr(nlp, "vocab") or len(nlp.vocab) < 1000

        # Basic linguistic analysis from spaCy
        analysis = {
            "word": word,
            "language": language,
            "pos": token.pos_,
            "lemma": token.lemma_,
            "tag": token.tag_,
            "dep": token.dep_,
            "is_alpha": token.is_alpha,
            "is_stop": token.is_stop,
            "is_punct": token.is_punct,
            "is_space": token.is_space,
            "shape": token.shape_,
            "is_title": token.is_title,
            "is_lower": token.is_lower,
            "is_upper": token.is_upper,
            "is_digit": token.is_digit,
            "like_num": token.like_num,
            "like_url": token.like_url,
            "like_email": token.like_email,
            "is_entity": bool(token.ent_type_),
            "entity_type": token.ent_type_ if token.ent_type_ else None,
            "has_vector": token.has_vector,
            "vector_norm": float(token.vector_norm) if token.has_vector else None,
            "is_oov": token.is_oov,
            "is_sent_start": token.is_sent_start,
            "is_sent_end": token.is_sent_end,
            "is_quote": token.is_quote,
            "is_bracket": token.is_bracket,
            "is_currency": token.is_currency,
            "is_left_punct": token.is_left_punct,
            "is_right_punct": token.is_right_punct,
        }

        # If using a blank model, try to improve POS detection with basic rules
        if is_blank_model:
            improved_pos = _improve_pos_detection(word, language)
            if improved_pos:
                analysis["pos"] = improved_pos
                analysis["pos_confidence"] = "low (using basic rules)"
                logger.info(
                    f"Improved POS detection for '{word}' from blank model: {improved_pos}"
                )

        # Enhanced analysis using LLM if available
        if client:
            try:
                enhanced_info = await _get_llm_word_analysis(
                    word, language, analysis["pos"], client
                )
                analysis.update(enhanced_info)
                if "llm_error" not in enhanced_info:
                    cache.set(word, language, provider, model_name, analysis)
            except Exception as e:
                logger.warning(f"LLM analysis failed for {word}: {e}")

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing word {word}: {str(e)}")
        return {"error": f"Analysis failed: {str(e)}"}


def _improve_pos_detection(word: str, language: str) -> Optional[str]:
    """
    Improve part-of-speech detection for blank SpaCy models using basic linguistic rules.

    Args:
        word: The word to analyze
        language: Language code

    Returns:
        Improved POS tag or None if no improvement possible
    """
    word_lower = word.lower()

    if language == "es":  # Spanish
        # Spanish verb endings
        if word_lower.endswith(("ar", "er", "ir")):
            return "VERB"  # Infinitive
        elif word_lower.endswith(
            (
                "o",
                "as",
                "a",
                "amos",
                "áis",
                "an",
                "es",
                "e",
                "emos",
                "éis",
                "en",
                "is",
                "es",
                "e",
                "imos",
                "ís",
                "en",
            )
        ):
            return "VERB"  # Conjugated forms

        # Spanish noun endings
        elif word_lower.endswith(("o", "e", "r", "l", "n")):
            return "NOUN"  # Likely masculine
        elif word_lower.endswith(("a", "ión", "dad", "tad", "tud", "ez", "eza")):
            return "NOUN"  # Likely feminine

        # Spanish adjective endings
        elif word_lower.endswith(("o", "a", "os", "as")):
            return "ADJ"

        # Spanish adverb endings
        elif word_lower.endswith(("mente")):
            return "ADV"

    elif language == "ca":  # Catalan
        # Catalan verb endings
        if word_lower.endswith(("ar", "er", "re")):
            return "VERB"  # Infinitive
        elif word_lower.endswith(("o", "es", "a", "em", "eu", "en")):
            return "VERB"  # Conjugated forms

        # Catalan noun endings
        elif word_lower.endswith(("o", "e", "r", "l", "n")):
            return "NOUN"  # Likely masculine
        elif word_lower.endswith(("a", "ció", "tat", "tut", "esa")):
            return "NOUN"  # Likely feminine

        # Catalan adjective endings
        elif word_lower.endswith(("o", "a", "os", "es")):
            return "ADJ"

    elif language == "en":  # English
        # English verb endings
        if word_lower.endswith(("ing", "ed", "s")):
            return "VERB"

        # English noun endings
        elif word_lower.endswith(
            ("tion", "sion", "ness", "ment", "ity", "ance", "ence")
        ):
            return "NOUN"

        # English adjective endings
        elif word_lower.endswith(("al", "ful", "ous", "ive", "able", "ible")):
            return "ADJ"

        # English adverb endings
        elif word_lower.endswith(("ly")):
            return "ADV"

    return None


def _word_analysis_prompt(lang_name: str, word: str, pos: str, example: str) -> str:
    """Build the per-group word-analysis prompt around a schema-derived example."""
    return f"""Analyze the {lang_name} word "{word}" (part of speech: {pos}).

Return a JSON object with these fields (include all that apply):

{example}

Respond ONLY with the JSON object, no other text. Make sure the JSON is valid."""


async def _get_llm_word_analysis(
    word: str, language: str, pos: str, client
) -> Dict[str, Any]:
    """
    Get enhanced linguistic analysis using LLM for language learning insights.

    Fires one call per WordAnalysis group (meaning/usage/grammar/pronunciation/
    learner_notes) concurrently, rather than one call for the whole object -
    Claude's structured-output endpoint rejects the combined schema outright (its
    compiled grammar is too large once every group sits under one call), while
    each group's schema alone is comfortably within the limit. See the
    ``WordAnalysis`` docstring in ``schemas.py`` for the full story. Each prompt's
    illustrative JSON is generated from the group's own schema via
    ``prompt_example()``, so the fields a provider is asked for can never drift
    from the fields the response is actually validated against.

    A group that errors does not fail the whole analysis - the other groups'
    results are still merged and returned, matching this module's existing
    tolerance for partial/sparse LLM output.

    Args:
        word: Word to analyze
        language: Language code
        pos: Part of speech
        client: LLM client

    Returns:
        Dictionary with enhanced analysis
    """
    lang_names = {"en": "English", "es": "Spanish", "ca": "Catalan"}
    lang_name = lang_names.get(language, language)

    system_prompt = (
        f"You are a {lang_name} linguistics expert. Respond only with valid JSON. "
        f"No markdown, no explanation, just the JSON object."
    )

    logger.debug(
        f"Calling LLM for word analysis: {word} ({language}), client={type(client).__name__}"
    )

    group_names = list(WORD_ANALYSIS_GROUPS)
    try:
        raw_results = await asyncio.gather(
            *(
                client.generate_json(
                    _word_analysis_prompt(lang_name, word, pos, prompt_example(schema)),
                    system_prompt=system_prompt,
                    schema=schema,
                )
                for schema in WORD_ANALYSIS_GROUPS.values()
            ),
            return_exceptions=True,
        )
    except Exception as e:
        logger.error(f"LLM analysis failed for {word}: {e}", exc_info=True)
        return {"llm_error": str(e)}

    merged: Dict[str, Any] = {}
    failures = []
    for group_name, result in zip(group_names, raw_results):
        if isinstance(result, BaseException):
            failures.append(f"{group_name}: {result}")
        elif "error" in result:
            failures.append(f"{group_name}: {result['error']}")
        else:
            merged[group_name] = result

    if not merged:
        logger.warning(f"LLM analysis failed for every group for {word}: {failures}")
        return {"llm_error": "; ".join(failures) or "Empty response from LLM"}
    if failures:
        logger.warning(f"Partial word analysis for {word}: {failures}")

    try:
        analysis = WordAnalysis.model_validate(merged)
    except ValidationError as e:
        logger.warning(f"Word analysis failed validation for {word}: {e}")
        return {"llm_error": "LLM returned data in an unexpected shape"}

    display_data = analysis.to_display_dict()
    logger.debug(f"Parsed word analysis for {word} ({len(display_data)} fields)")
    return display_data
