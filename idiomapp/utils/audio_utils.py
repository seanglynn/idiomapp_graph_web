"""
Audio utilities.
Includes text-to-speech and audio processing functions.
"""

import logging
import re
import base64
from typing import Dict, Optional, Any

# Text-to-speech
from gtts import gTTS

# Setup logging
logger = logging.getLogger(__name__)

# Language code mapping for gTTS
TTS_LANG_CODES = {
    "en": "en",
    "es": "es",
    "ca": "es"  # Use Spanish for Catalan (gTTS limitation)
}

def clean_text_for_tts(text: str) -> str:
    """
    Clean text for text-to-speech processing.
    Removes markdown formatting, URLs, emojis, etc.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text suitable for TTS
    """
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    text = re.sub(r'__(.*?)__', r'\1', text)      # Underline
    text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Fix common issues
    text = text.replace('&', 'and')
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_translation_content(text: str, language_mark: str) -> str:
    """
    Extract translation content from a message that contains multiple translations.
    
    Args:
        text: Message text with translations in multiple languages
        language_mark: Language marker to extract (e.g., "English 🇬🇧:")
        
    Returns:
        Extracted translation text for the specified language
    """
    # Split by language marker
    if language_mark in text:
        parts = text.split(language_mark)
        if len(parts) > 1:
            # Get the content after the marker
            content = parts[1].strip()
            
            # Find the end of this language section (next language marker)
            next_lang_pos = -1
            language_markers = ["English 🇬🇧:", "Spanish 🇪🇸:", "Catalan 🇪🇸:"]
            for marker in language_markers:
                if marker != language_mark and marker in content:
                    pos = content.find(marker)
                    if pos != -1 and (next_lang_pos == -1 or pos < next_lang_pos):
                        next_lang_pos = pos
            
            # Extract just this language's content
            if next_lang_pos != -1:
                content = content[:next_lang_pos].strip()
            
            return content
    
    # If no specific translation found, return the original text
    return text

def generate_audio(text: str, lang_code: Optional[str] = None) -> str:
    """
    Generate an audio link for a given text and language code.
    
    Args:
        text: Text to convert to speech
        lang_code: Language code (en, es, ca)
        
    Returns:
        HTML for a direct audio link
    """
    logger.info(f"Generating audio for text ({len(text)} chars)")
    
    try:
        # Clean the text for TTS
        original_text = text
        text = clean_text_for_tts(text)
        
        # Check if this is a multi-language translation message
        language_markers = {
            "en": "English 🇬🇧:",
            "es": "Spanish 🇪🇸:",
            "ca": "Catalan 🇪🇸:"
        }
        
        if lang_code in language_markers and any(marker in original_text for marker in language_markers.values()):
            # Extract just the translation for this language
            language_mark = language_markers[lang_code]
            text = extract_translation_content(original_text, language_mark)
        
        # Use the specified language or default to English
        tts_lang = TTS_LANG_CODES.get(lang_code, "en")
        
        # Generate the audio
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        
        # Convert to base64
        import io
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_data = base64.b64encode(mp3_fp.read()).decode('utf-8')
        
        # Create a direct link that opens in a new tab
        data_url = f"data:audio/mp3;base64,{audio_data}"
        
        # Get language name for the button
        lang_names = {
            "en": "English",
            "es": "Spanish",
            "ca": "Catalan"
        }
        lang_name = lang_names.get(lang_code, "")
        
        # Create a link styled as a button
        audio_link = f"""
        <a href="{data_url}" target="_blank" style="
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: rgba(67, 97, 238, 0.1);
            border: 1px solid #4361EE;
            color: #4361EE;
            text-decoration: none;
            margin: 5px;
            font-size: 16px;
            transition: all 0.3s ease;
            " title="Play audio ({lang_name})">🔊</a>
        """
        
        logger.info("Audio link generated successfully")
        return audio_link
    
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return f"<span style='color:red'>Audio error: {str(e)}</span>" 