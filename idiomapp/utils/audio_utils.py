"""
Audio utilities.
Includes text-to-speech and audio processing functions.
"""

import io
import base64
import logging
from typing import Optional
from gtts import gTTS

from idiomapp.config import TTS_LANG_CODES, LANGUAGE_MARKERS
from idiomapp.utils.nlp_utils import detect_language

# Set up logging
logger = logging.getLogger(__name__)


def text_to_speech(text: str, lang_code: str) -> str:
    """
    Convert text to speech and return base64 encoded audio data.
    
    Args:
        text: Text to convert to speech
        lang_code: Language code for TTS
        
    Returns:
        Base64 encoded audio data as string
    """
    try:
        # Generate the audio
        tts = gTTS(text=text, lang=lang_code, slow=False)
        
        # Convert to base64
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_data = base64.b64encode(mp3_fp.read()).decode('utf-8')
        
        logger.info(f"Audio generated successfully for {lang_code}")
        return audio_data
        
    except Exception as e:
        logger.error(f"Error generating TTS: {str(e)}")
        return ""


def clean_text_for_tts(text: str) -> str:
    """
    Clean text for text-to-speech generation.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text suitable for TTS
    """
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = " ".join(text.split())
    
    # Remove any HTML tags that might have slipped through
    import re
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove any language markers that might be in the text
    for marker in LANGUAGE_MARKERS.values():
        text = text.replace(marker, "")
    
    # Clean up any remaining artifacts
    text = text.strip()
    
    return text


def extract_translation_text(
    full_message: str, 
    source_language: str, 
    target_language: str
) -> str:
    """
    Extract clean translation text from a multi-language message.
    
    Args:
        full_message: Complete message containing translations
        source_language: Source language code (e.g., 'ca')
        target_language: Target language code (e.g., 'en')
        
    Returns:
        Clean text for the target language, or empty string if not found
    """
    logger.info(f"Extracting {target_language} translation from {source_language} source")
    logger.info(f"Full message length: {len(full_message)} characters")
    
    # Get the language marker for the target language
    if target_language not in LANGUAGE_MARKERS:
        logger.warning(f"Unknown target language: {target_language}")
        return ""
    
    target_marker = LANGUAGE_MARKERS[target_language]
    logger.info(f"Looking for marker: '{target_marker}'")
    
    # Check if the marker exists in the message
    if target_marker not in full_message:
        logger.warning(f"Target language marker '{target_marker}' not found in message")
        return ""
    
    # Split by the target language marker
    parts = full_message.split(target_marker)
    if len(parts) < 2:
        logger.warning(f"Message split into {len(parts)} parts, expected at least 2")
        return ""
    
    # Get the content after the marker
    content = parts[1].strip()
    logger.info(f"Initial content after marker: '{content[:100]}...'")
    
    # Find the next language marker to trim the content
    next_marker_pos = -1
    for lang_code, marker in LANGUAGE_MARKERS.items():
        if lang_code != target_language and marker in content:
            pos = content.find(marker)
            if pos != -1 and (next_marker_pos == -1 or pos < next_marker_pos):
                next_marker_pos = pos
    
    # Trim to the next marker if found
    if next_marker_pos != -1:
        content = content[:next_marker_pos].strip()
        logger.info(f"Content trimmed to next marker: '{content[:100]}...'")
    
    # Clean up the extracted content
    content = clean_text_for_tts(content)
    
    logger.info(f"Final extracted text: '{content[:100]}...' (length: {len(content)})")
    return content


def generate_audio(text: str, source_language: str = "unknown", target_language: str = None) -> str:
    """
    Generate audio for text in the target language.
    
    This function is backwards-compatible and can handle both old and new calling patterns:
    
    Legacy (old): generate_audio(text, lang_code) - where lang_code is the target language
    New: generate_audio(text, source_language, target_language)
    
    Args:
        text: Clean text to convert to speech
        source_language: Source language code (for validation, optional)
        target_language: Target language code for TTS (optional for backwards compatibility)
        
    Returns:
        HTML for embedded audio player
    """
    # Handle backwards compatibility: if target_language is None, try to detect it
    if target_language is None:
        if source_language != "unknown":
            # New calling pattern: source_language is actually the target language
            target_language = source_language
            source_language = "unknown"
        else:
            # Legacy calling pattern: try to detect language
            detected_lang = detect_language(text)
            target_language = detected_lang
            logger.info(f"Detected language: {detected_lang}")
    
    logger.info(f"Generating audio: {source_language} -> {target_language}")
    logger.info(f"Text length: {len(text)} characters")
    
    # Validate inputs
    if not text or not text.strip():
        logger.error("No text provided for audio generation")
        return f"<span style='color:orange'>⚠️ No text available for audio generation</span>"
    
    if target_language not in TTS_LANG_CODES:
        logger.error(f"Unsupported target language for TTS: {target_language}")
        return f"<span style='color:red'>❌ Audio not supported for {target_language}</span>"
    
    # Get TTS language code
    tts_lang = TTS_LANG_CODES[target_language]
    logger.info(f"Using TTS language: {tts_lang}")
    
    try:
        # Generate the audio file
        audio_file = text_to_speech(text, tts_lang)
        
        if audio_file:
            # Create HTML audio player
            audio_html = f"""
            <audio controls style="width: 100%; margin: 5px 0;">
                <source src="data:audio/mp3;base64,{audio_file}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            """
            logger.info(f"Audio generated successfully for {target_language}")
            return audio_html
        else:
            logger.error("Audio file generation failed")
            return f"<span style='color:red'>❌ Audio generation failed</span>"
            
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return f"<span style='color:red'>❌ Audio error: {str(e)}</span>"


def process_translation_audio(
    full_message: str, 
    source_language: str, 
    target_language: str
) -> str:
    """
    Main function that extracts translation text and generates audio.
    
    Args:
        full_message: Complete message containing translations
        source_language: Source language code
        target_language: Target language code
        
    Returns:
        HTML for embedded audio player
    """
    logger.info(f"Processing translation audio: {source_language} -> {target_language}")
    
    try:
        # Step 1: Extract the translation text
        translation_text = extract_translation_text(full_message, source_language, target_language)
        
        if not translation_text:
            logger.warning(f"No translation text extracted for {target_language}")
            return f"<span style='color:orange'>⚠️ No translation text found for {target_language}</span>"
        
        # Step 2: Generate audio from the extracted text
        audio_html = generate_audio(translation_text, source_language, target_language)
        
        return audio_html
        
    except Exception as e:
        logger.error(f"Error processing translation audio: {str(e)}")
        return f"<span style='color:red'>❌ Audio processing error: {str(e)}</span>" 