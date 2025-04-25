# Idiom App Utility Modules

This directory contains utility modules that provide various functionalities for the Idiom App application.

## Module Structure

### `nlp_utils.py`

Natural Language Processing utilities based on [textacy](https://textacy.readthedocs.io/) and [spaCy](https://spacy.io/):

- `analyze_parts_of_speech`: Analyze parts of speech in text using spaCy
- `split_into_sentences`: Split text into sentences using textacy
- `calculate_similarity`: Calculate string similarity between words
- `calculate_word_similarity`: Calculate similarity between words in different languages
- `build_word_cooccurrence_network`: Build co-occurrence networks using textacy
- `visualize_cooccurrence_network`: Visualize networks using pyvis
- `detect_language`: Detect the language of a text
- `get_network_stats`: Calculate various network statistics

### `audio_utils.py`

Audio processing utilities including text-to-speech functionality:

- `generate_audio`: Generate audio links for text in different languages
- `clean_text_for_tts`: Clean text for text-to-speech processing
- `extract_translation_content`: Extract translation content from messages

### `ollama_utils.py`

Utilities for interfacing with Ollama LLM:

- `OllamaClient`: Client for Ollama API
- `get_available_models`: Get available Ollama models

### `logging_utils.py`

Utilities for logging and debugging:

- `setup_logging`: Configure logging for the application
- `get_recent_logs`: Get recent log messages
- `clear_logs`: Clear log history

## Usage

Import the modules as needed:

```python
from idiomapp.utils.nlp_utils import analyze_parts_of_speech, split_into_sentences
from idiomapp.utils.audio_utils import generate_audio
from idiomapp.utils.ollama_utils import OllamaClient
from idiomapp.utils.logging_utils import setup_logging
```

## Dependencies

- textacy
- spaCy
- networkx
- pyvis
- gTTS (Google Text-to-Speech)
- langdetect 