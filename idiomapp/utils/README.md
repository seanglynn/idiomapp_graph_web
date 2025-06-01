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

### `llm_utils.py`

Utilities for interfacing with LLM providers (Ollama, OpenAI):

- `LLMClient`: Abstract base class for LLM providers
- `OllamaClient`: Client for Ollama API
- `OpenAIClient`: Client for OpenAI/ChatGPT API
- `get_available_models`: Get available Ollama models

### `logging_utils.py`

Logging utilities for the application:

- `get_logger`: Get or create a cached logger for the specified module (recommended)
- `setup_logging`: Legacy function that wraps get_logger for backward compatibility
- `get_recent_logs`: Get recent log messages for UI display
- `clear_logs`: Clear the recent logs buffer

The logging system uses LRU cache to efficiently manage logger instances and avoid repeated initialization.

### Usage

```python
from idiomapp.utils.logging_utils import get_logger

# Get a cached logger instance
logger = get_logger("module_name")
logger.info("This is a test message")
```

## Usage

Import the modules as needed:

```python
from idiomapp.utils.nlp_utils import analyze_parts_of_speech, split_into_sentences
from idiomapp.utils.audio_utils import generate_audio
from idiomapp.utils.llm_utils import LLMClient
from idiomapp.utils.logging_utils import get_logger
```

## Dependencies

- textacy
- spaCy
- networkx
- pyvis
- gTTS (Google Text-to-Speech)
- langdetect
- openai
- ollama
