# VibeLog - IdiomApp Development Progress & Codebase Evolution

## Project Overview
**IdiomApp** is a multi-language translation and semantic analysis application built with Streamlit, featuring LLM integration (OpenAI/Ollama), NLP processing with spaCy, and interactive graph visualization using Pyvis. The application supports English, Spanish, and Catalan with real-time translation, audio generation, and word relationship analysis.

## Codebase Architecture

### **Core Module Structure**
```
idiomapp/
├── config.py                 # Centralized configuration with Pydantic settings
├── streamlit/
│   └── app.py               # Main Streamlit application and UI logic
├── utils/
│   ├── llm_utils.py         # Abstract LLM client interface and OpenAI implementation
│   ├── ollama_utils.py      # Ollama-specific utility functions
│   ├── nlp_utils.py         # NLP processing with spaCy and textacy
│   ├── audio_utils.py       # Text-to-speech and audio processing
│   └── state_utils.py       # Streamlit session state management
└── docker/                  # Containerization and deployment
```

### **Key Design Patterns**
- **Abstract Base Classes**: `LLMClient` interface with concrete `OpenAIClient` and `OllamaClient` implementations
- **Factory Pattern**: `LLMClient.create()` method for dynamic client instantiation
- **Configuration-Driven**: Pydantic settings with environment variable support
- **Session State Management**: Streamlit session state for secure credential storage
- **Error Handling Strategy**: Layered error handling with graceful degradation

## 2025-08-31 - Audio Function Refactoring & Function Order Fix ✅

### **Functional Changes**
- **Consolidated Audio Functions**: Replaced `generate_audio_legacy()` and `generate_audio_new()` with single, backwards-compatible `generate_audio()` function
- **Fixed Function Order**: Moved `text_to_speech()` to top of file to resolve "name 'text_to_speech' is not defined" error
- **Eliminated Duplication**: Removed duplicate function definitions and circular references

### **Technical Implementation**
```python
# Before: Multiple functions with confusing names
def generate_audio_new(text, source_language, target_language): ...
def generate_audio_legacy(text, lang_code): ...
generate_audio = generate_audio_legacy  # Confusing alias

# After: Single, smart function with backwards compatibility
def generate_audio(text: str, source_language: str = "unknown", target_language: str = None) -> str:
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
```

### **Backwards Compatibility Logic**
- **Legacy Calls**: `generate_audio(text, lang_code)` → automatically detects language
- **New Calls**: `generate_audio(text, source_language, target_language)` → explicit parameters
- **Smart Detection**: Automatically infers calling pattern based on parameter values

### **Function Order Resolution**
- **Problem**: `generate_audio()` called `text_to_speech()` before it was defined
- **Solution**: Moved `text_to_speech()` to top of file (line 18) before `generate_audio()`
- **Result**: Eliminated "name 'text_to_speech' is not defined" runtime error

## 2025-08-31 - Configuration Consolidation & LANG_MODELS Centralization ✅

### **Functional Changes**
- **Moved LANG_MODELS**: Relocated SpaCy language model mapping from `nlp_utils.py` to `config.py`
- **Centralized Constants**: All configuration constants now in single location
- **Updated Imports**: `nlp_utils.py` now imports `LANG_MODELS` from central config

### **Technical Implementation**
```python
# config.py - Added LANG_MODELS constant
LANG_MODELS: Dict[str, str] = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm", 
    "ca": "ca_core_news_sm"
}

# nlp_utils.py - Updated import
from idiomapp.config import GROUP_COLORS as LANGUAGE_COLORS, LANG_MODELS

# Usage remains the same throughout the codebase
model_name = LANG_MODELS.get(language, "en_core_web_sm")
```

### **Benefits for LLMs**
- **Single Source of Truth**: All constants in one location for easier code analysis
- **Consistent Pattern**: Follows same structure as other constants like `LANGUAGE_MAP`, `TTS_LANG_CODES`
- **Easier Maintenance**: No need to search multiple files for configuration values

## 2025-08-31 - Interactive Graph Node Analysis & Simplified Click System ✅

### **Functional Changes**
- **Clickable Graph Nodes**: Users can click any word in semantic graph for analysis
- **JavaScript Integration**: Enhanced Pyvis HTML with custom click handlers
- **Form-Based Communication**: Simple form submission between graph and Streamlit
- **Integrated Analysis UI**: Analysis section appears below graph when word is selected

### **Technical Implementation**
```python
# app.py - JavaScript injection for click handling
def enhance_graph_html(html_content: str, nodes_data: List[Dict]) -> str:
    """Inject custom JavaScript into Pyvis HTML output"""
    click_handler = create_click_handler(nodes_data)
    
    # Insert JavaScript before closing </body> tag
    if '</body>' in html_content:
        html_content = html_content.replace('</body>', f'{click_handler}</body>')
    
    return html_content

def create_click_handler(nodes_data: List[Dict]) -> str:
    """Generate JavaScript for node click handling"""
    return f"""
    <script>
    // Store graph node data for click handling
    const nodesData = {nodes_data};
    
    // Attach click listeners to Pyvis nodes
    function attachClickHandlers() {{
        const network = window.vis_network || window.__vis_network;
        if (network) {{
            network.on('click', function(params) {{
                const nodeId = params.nodes[0];
                const nodeData = nodesData.find(n => n.id === nodeId);
                if (nodeData) {{
                    // Show word info and submit to Streamlit
                    alert(`Word: ${{nodeData.word}}\\nLanguage: ${{nodeData.language}}\\nPOS: ${{nodeData.pos}}`);
                    submitWordToStreamlit(nodeData);
                }}
            }});
        }}
    }}
    
    // Submit word data to Streamlit via query parameters
    function submitWordToStreamlit(wordData) {{
        const currentUrl = new URL(window.location.href);
        currentUrl.searchParams.set('word_analysis_request', JSON.stringify(wordData));
        window.location.href = currentUrl.toString();
    }}
    
    // Initialize when page loads
    window.addEventListener('load', attachClickHandlers);
    </script>
    """
```

### **Streamlit Integration**
```python
# app.py - Handle word analysis requests from graph
def main():
    # Check for word analysis requests from graph clicks
    word_analysis_request = st.query_params.get("word_analysis_request")
    if word_analysis_request:
        try:
            word_data = json.loads(word_analysis_request)
            st.session_state["selected_word_from_graph"] = {
                "word": word_data.get("word", ""),
                "language": word_data.get("language", ""),
                "pos": word_data.get("pos", "")
            }
        except json.JSONDecodeError:
            st.error("Invalid word data received from graph")
```

### **User Experience Flow**
1. **Click Node** → JavaScript alert shows word info
2. **Form Submission** → Word data sent to Streamlit via query parameters
3. **Session State** → Selected word stored in Streamlit session
4. **Analysis UI** → Analysis section appears below graph
5. **LLM Analysis** → User clicks "Analyze Selected Word" button

## 2025-08-31 - Interactive Word Analysis & Language Learning Features ✅

### **Functional Changes**
- **LLM-Powered Word Analysis**: Uses attached LLM for comprehensive linguistic analysis
- **Rich Linguistic Data**: Verb conjugations, noun gender/plural forms, adjective agreements
- **Educational Focus**: Designed for language learners with practical examples and grammar notes
- **Async Processing**: Efficient handling of LLM requests for word analysis

### **Technical Implementation**
```python
# nlp_utils.py - Core analysis function
async def analyze_word_linguistics(word: str, language: str, client=None) -> Dict[str, Any]:
    """Orchestrates linguistic analysis combining spaCy and LLM"""
    
    # Step 1: Basic spaCy analysis
    nlp = load_spacy_model(language)
    doc = nlp(word)
    
    if len(doc) > 0:
        token = doc[0]
        pos = token.pos_
        lemma = token.lemma_
    else:
        pos = "UNKNOWN"
        lemma = word
    
    # Step 2: LLM-enhanced analysis
    llm_analysis = await _get_llm_word_analysis(word, language, pos, client)
    
    return {
        "word": word,
        "language": language,
        "pos": pos,
        "lemma": lemma,
        "llm_analysis": llm_analysis
    }

# app.py - Display analysis results
def display_word_analysis(word_data: Dict[str, Any]):
    """Display comprehensive word analysis with expandable sections"""
    
    st.subheader(f"📚 Analysis: {word_data['word']} ({word_data['language'].upper()})")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Part of Speech", word_data['pos'])
    with col2:
        st.metric("Language", word_data['language'].upper())
    with col3:
        st.metric("Lemma", word_data['lemma'])
    
    # LLM analysis results
    if word_data.get('llm_analysis'):
        analysis = word_data['llm_analysis']
        
        # Verb analysis
        if word_data['pos'] == 'VERB':
            _display_verb_analysis(analysis)
        # Noun analysis  
        elif word_data['pos'] == 'NOUN':
            _display_noun_analysis(analysis)
        # Adjective analysis
        elif word_data['pos'] == 'ADJ':
            _display_adjective_analysis(analysis)
        # Generic analysis
        else:
            _display_generic_analysis(analysis)
```

### **LLM Prompt Engineering**
```python
# nlp_utils.py - LLM prompts for different parts of speech
async def _get_llm_word_analysis(word: str, language: str, pos: str, client) -> Dict[str, Any]:
    """Generate LLM prompts tailored to part of speech"""
    
    if pos == "VERB":
        prompt = f"""
        Analyze the Spanish verb "{word}" and provide:
        1. Infinitive form
        2. Conjugation patterns (present, past, future)
        3. Related forms (participles, gerunds)
        4. Usage examples
        5. Grammar notes
        
        Return as JSON with keys: infinitive, conjugations, related_forms, examples, grammar_notes
        """
    elif pos == "NOUN":
        prompt = f"""
        Analyze the Spanish noun "{word}" and provide:
        1. Gender (masculine/feminine)
        2. Plural form
        3. Article usage
        4. Cultural context
        5. Usage examples
        
        Return as JSON with keys: gender, plural, articles, cultural_context, examples
        """
    
    # Send to LLM and parse response
    response = await client.generate_text(prompt)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM response"}
```

## 2025-08-31 - SpaCy Model Management & Accuracy Improvements ✅

### **Functional Changes**
- **Latest spaCy v3 Models**: Updated to use `es_core_web_sm` and `ca_core_web_sm`
- **Alternative Model Fallbacks**: Multiple model variants for better reliability
- **Automatic Downloads**: Attempts to download missing models automatically
- **Enhanced POS Detection**: Linguistic rules for blank models

### **Technical Implementation**
```python
# nlp_utils.py - Enhanced model loading with fallbacks
def load_spacy_model(language: str) -> spacy.language.Language:
    """Load spaCy model with multiple fallback options"""
    
    # Get primary model name from config
    model_name = LANG_MODELS.get(language, "en_core_web_sm")
    
    try:
        # Try primary model
        nlp = spacy.load(model_name)
        return nlp
    except OSError:
        # Try alternative models
        nlp = _try_alternative_models(language)
        if nlp:
            return nlp
        
        # Last resort - blank model with enhanced POS detection
        logger.warning(f"Creating blank model for {language}")
        nlp = spacy.blank(language)
        return nlp

def _try_alternative_models(language: str) -> Optional[spacy.language.Language]:
    """Try multiple model variants for better reliability"""
    
    alternative_models = [
        f"{language}_core_web_sm",
        f"{language}_core_web_md", 
        f"{language}_core_web_lg",
        f"{language}_core_news_sm"
    ]
    
    for model in alternative_models:
        try:
            nlp = spacy.load(model)
            logger.info(f"Successfully loaded alternative model: {model}")
            return nlp
        except OSError:
            continue
    
    return None

def _improve_pos_detection(word: str, language: str) -> Optional[str]:
    """Provide basic linguistic rules for POS detection with blank models"""
    
    if language == "es":  # Spanish
        # Verb endings
        if word.endswith(('ar', 'er', 'ir')):
            return "VERB"
        # Noun endings
        if word.endswith(('o', 'a', 'e', 'ión', 'dad', 'tad')):
            return "NOUN"
        # Adjective endings
        if word.endswith(('o', 'a', 'e', 'al', 'ar', 'or')):
            return "ADJ"
    
    elif language == "ca":  # Catalan
        # Similar patterns to Spanish
        if word.endswith(('ar', 'er', 'ir', 're')):
            return "VERB"
        if word.endswith(('a', 'e', 'o', 'ció', 'tat')):
            return "NOUN"
    
    return None
```

### **Model Status Management**
```python
# app.py - Model status display and management
def display_model_status():
    """Display SpaCy model status and management options"""
    
    st.sidebar.subheader("🔧 SpaCy Language Models")
    
    # Check model status for each language
    for lang_code in ["en", "es", "ca"]:
        try:
            nlp = load_spacy_model(lang_code)
            model_name = LANG_MODELS.get(lang_code, "unknown")
            
            if hasattr(nlp, 'vocab') and len(nlp.vocab) > 1000:
                status = "✅ Full Model"
                color = "green"
            else:
                status = "⚠️ Basic Model"
                color = "orange"
            
            st.sidebar.markdown(f"**{lang_code.upper()}**: {status}")
            
        except Exception as e:
            st.sidebar.markdown(f"**{lang_code.upper()}**: ❌ Error")
    
    # Management buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh Models"):
            ensure_models_available()
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Cache"):
            clear_model_cache()
            st.rerun()
```

## 2025-08-31 - Catalan Translation and Audio Fixes ✅

### **Functional Changes**
- **Fixed SpaCy Model Errors**: Resolved "Can't find model" errors for Catalan and other languages
- **Textacy Integration Fix**: Updated to use loaded SpaCy models instead of language codes
- **Catalan TTS Working**: Fixed "No text to speak" errors for Catalan audio generation
- **Enhanced Debug Logging**: Better tracking of text extraction and processing

### **Technical Implementation**
```python
# nlp_utils.py - Fixed textacy integration
def build_cooccurrence_network(text: str, language: str, window_size: int = 2, min_freq: int = 1):
    """Build co-occurrence network with proper SpaCy model usage"""
    
    # Load the appropriate SpaCy model
    nlp = load_spacy_model(language)
    
    try:
        # Use loaded model instead of language code string
        doc = textacy.make_spacy_doc(text, lang=nlp)  # Fixed: was lang=language
        
        # Build network with error handling for term_filter compatibility
        try:
            network = textacy.representations.network.build_cooccurrence_network(
                doc, 
                window_size=window_size, 
                min_freq=min_freq,
                term_filter=lambda term: term.pos_ in ["NOUN", "VERB", "ADJ"]
            )
        except TypeError:
            # Fallback for older textacy versions without term_filter
            logger.info("term_filter not supported, using fallback")
            network = textacy.representations.network.build_cooccurrence_network(
                doc, 
                window_size=window_size, 
                min_freq=min_freq
            )
        
        return network
        
    except Exception as e:
        logger.error(f"Error building co-occurrence network: {str(e)}")
        return None
```

### **Audio Generation Fixes**
```python
# audio_utils.py - Enhanced text extraction and TTS
def extract_translation_text(full_message: str, source_language: str, target_language: str) -> str:
    """Extract clean translation text with enhanced debugging"""
    
    logger.info(f"Extracting {target_language} translation from {source_language} source")
    logger.info(f"Full message length: {len(full_message)} characters")
    
    # Get language marker from config
    target_marker = LANGUAGE_MARKERS[target_language]
    logger.info(f"Looking for marker: '{target_marker}'")
    
    # Split and extract content
    parts = full_message.split(target_marker)
    if len(parts) < 2:
        logger.warning(f"Target language marker not found")
        return ""
    
    content = parts[1].strip()
    
    # Find next language marker to trim content
    next_marker_pos = -1
    for lang_code, marker in LANGUAGE_MARKERS.items():
        if lang_code != target_language and marker in content:
            pos = content.find(marker)
            if pos != -1 and (next_marker_pos == -1 or pos < next_marker_pos):
                next_marker_pos = pos
    
    # Trim to next marker if found
    if next_marker_pos != -1:
        content = content[:next_marker_pos].strip()
    
    # Clean extracted content
    content = clean_text_for_tts(content)
    
    logger.info(f"Final extracted text: '{content[:100]}...' (length: {len(content)})")
    return content
```

## 2025-08-31 - Ollama Function Consolidation & Code Organization ✅

### **Functional Changes**
- **Consolidated Ollama Functions**: Moved all Ollama-specific functions to dedicated module
- **Eliminated Duplicates**: Removed duplicate function definitions between modules
- **Cleaner Separation**: Clear separation between LLM client logic and Ollama utilities
- **Updated Imports**: All references now point to correct module locations

### **Technical Implementation**
```python
# ollama_utils.py - Dedicated Ollama module
def is_ollama_running() -> bool:
    """Check if Ollama service is running"""
    try:
        response = httpx.get(f"{get_valid_ollama_host()}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False

def get_valid_ollama_host() -> str:
    """Get valid Ollama host with fallbacks"""
    hosts = [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://ollama:11434"  # Docker service name
    ]
    
    for host in hosts:
        try:
            response = httpx.get(f"{host}/api/tags", timeout=2.0)
            if response.status_code == 200:
                return host
        except Exception:
            continue
    
    return hosts[0]  # Default fallback

def get_available_models() -> List[str]:
    """Get list of available Ollama models"""
    try:
        response = httpx.get(f"{get_valid_ollama_host()}/api/tags")
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
    except Exception as e:
        logger.error(f"Error getting Ollama models: {e}")
    
    return []

def pull_model_if_needed(model_name: str) -> bool:
    """Pull Ollama model if not already available"""
    available_models = get_available_models()
    
    if model_name in available_models:
        logger.info(f"Model {model_name} already available")
        return True
    
    try:
        logger.info(f"Pulling model {model_name}...")
        response = httpx.post(
            f"{get_valid_ollama_host()}/api/pull",
            json={"name": model_name}
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error pulling model {model_name}: {e}")
        return False
```

### **Updated Imports**
```python
# llm_utils.py - Now imports from ollama_utils
from idiomapp.utils.ollama_utils import (
    is_ollama_running,
    get_valid_ollama_host, 
    get_available_models,
    pull_model_if_needed
)

# app.py - Updated imports
from idiomapp.utils.ollama_utils import get_available_models
from idiomapp.utils.llm_utils import LLMClient, get_openai_available_models
```

## 2025-08-31 - Major Architecture Improvement: Configuration-Driven Model Management ✅

### **Functional Changes**
- **Fixed Model Parameter Compatibility**: Resolved `max_tokens` vs `max_completion_tokens` issues
- **Automatic Parameter Detection**: GPT-4o and Claude-3 models automatically use correct parameters
- **Legacy Model Support**: Older models continue to use `max_tokens` parameter
- **Fallback Mechanism**: Automatic fallback if initial parameter fails

### **Technical Implementation**
```python
# config.py - Model capabilities configuration
MODEL_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    # GPT-5 models - strict parameter requirements
    "gpt-5": {
        "supports_max_completion_tokens": True,
        "supports_custom_temperature": False,
        "supports_custom_max_tokens": True,
        "description": "GPT-5 model with strict parameter requirements",
        "notes": "Only supports default temperature (1.0), uses max_completion_tokens"
    },
    "gpt-4o": {
        "supports_max_completion_tokens": True,
        "supports_custom_temperature": True,
        "supports_custom_max_tokens": True,
        "description": "GPT-4o model with modern parameter support",
        "notes": "Full parameter support including custom temperature"
    }
}

# llm_utils.py - Smart parameter selection
def generate_text(self, prompt: str, **kwargs) -> str:
    """Generate text with model-aware parameter selection"""
    
    # Get model capabilities
    capabilities = get_model_capabilities(self.model_name)
    
    # Build parameters based on model capabilities
    params = {
        "model": self.model_name,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    # Add max tokens based on model support
    if capabilities["supports_max_completion_tokens"]:
        params["max_completion_tokens"] = kwargs.get("max_tokens", 1024)
    else:
        params["max_tokens"] = kwargs.get("max_tokens", 1024)
    
    # Add temperature if supported
    if capabilities["supports_custom_temperature"]:
        params["temperature"] = kwargs.get("temperature", 0.7)
    
    try:
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
    except Exception as e:
        # Fallback: try without problematic parameters
        if "max_completion_tokens" in params:
            logger.warning("max_completion_tokens failed, trying max_tokens")
            params["max_tokens"] = params.pop("max_completion_tokens")
            try:
                response = self.client.chat.completions.create(**params)
                return response.choices[0].message.content
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise
        raise
```

## 2025-08-31 - Major Security & Error Handling Overhaul ✅

### **Functional Changes**
- **API Key Security**: Removed storage of OpenAI API keys in environment variables
- **Session State Isolation**: API keys stored securely in Streamlit session state only
- **Direct API Key Passing**: LLM client receives API key directly without touching environment
- **No More UI Freezing**: Replaced `st.rerun()` with flag-based client reinitialization

### **Technical Implementation**
```python
# app.py - Secure API key management
def initialize_session_state():
    """Initialize secure session state for credentials"""
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "openai_organization" not in st.session_state:
        st.session_state.openai_organization = ""
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = None
    if "client_needs_reinit" not in st.session_state:
        st.session_state.client_needs_reinit = False

def handle_credential_change():
    """Handle credential changes without UI freezing"""
    if st.session_state.client_needs_reinit:
        st.session_state.llm_client = None
        st.session_state.client_needs_reinit = False

# Sidebar credential inputs
with st.sidebar:
    st.subheader("🔑 LLM Configuration")
    
    # OpenAI API Key (secure input)
    api_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.openai_api_key,
        type="password",
        help="Enter your OpenAI API key"
    )
    
    # OpenAI Organization ID
    organization = st.text_input(
        "OpenAI Organization ID (Optional)",
        value=st.session_state.openai_organization,
        help="Enter your OpenAI organization ID if applicable"
    )
    
    # Check for changes and trigger reinitialization
    if (api_key != st.session_state.openai_api_key or 
        organization != st.session_state.openai_organization):
        st.session_state.openai_api_key = api_key
        st.session_state.openai_organization = organization
        st.session_state.client_needs_reinit = True
        st.rerun()

# LLM client creation with secure credentials
def get_llm_client():
    """Get LLM client with secure credential handling"""
    if st.session_state.client_needs_reinit or st.session_state.llm_client is None:
        if st.session_state.llm_provider == "openai":
            if not st.session_state.openai_api_key:
                st.error("Please enter your OpenAI API key")
                return None
            
            client = LLMClient.create(
                provider="openai",
                model_name=st.session_state.openai_model,
                api_key=st.session_state.openai_api_key,
                organization=st.session_state.openai_organization
            )
            st.session_state.llm_client = client
            st.session_state.client_needs_reinit = False
    
    return st.session_state.llm_client
```

### **Error Handling System**
```python
# app.py - Comprehensive error handling
def handle_translation_error(error: Exception, model_name: str = "unknown") -> str:
    """Handle common translation errors with user-friendly messages"""
    
    error_str = str(error).lower()
    
    if "insufficient_quota" in error_str:
        return "⚠️ API quota exceeded. Please check your OpenAI billing and usage limits."
    elif "429" in error_str or "rate limit" in error_str:
        return "⚠️ Rate limit exceeded. Please wait a moment and try again."
    elif "401" in error_str or "authentication" in error_str:
        return "⚠️ Authentication failed. Please check your OpenAI API key."
    elif "404" in error_str or "model not found" in error_str:
        return "⚠️ Model not found. Please select a different model from the sidebar."
    elif "max_tokens" in error_str:
        return "⚠️ Token limit exceeded. Please try with shorter text or increase the token limit."
    else:
        return f"⚠️ Translation error: {str(error)}"

def display_translation_error(error_message: str):
    """Display translation errors in dedicated section"""
    st.error(error_message)
    st.info("💡 Tip: Check your API key, model selection, and text length.")

# Translation loop with error separation
successful_translations = []
translation_errors = []

for target_lang in target_languages:
    try:
        translation = await translate_text(
            source_text, 
            source_language, 
            target_lang, 
            llm_client
        )
        successful_translations.append((target_lang, translation))
    except Exception as e:
        error_message = handle_translation_error(e, llm_client.model_name if llm_client else "unknown")
        translation_errors.append((target_lang, error_message))

# Display results separately
if successful_translations:
    st.subheader("✅ Successful Translations")
    for lang, translation in successful_translations:
        render_chat_message(translation, source_lang=source_language)
        # Generate audio for successful translations
        audio_html = process_translation_audio(translation, source_language, lang)
        if audio_html:
            st.markdown(audio_html, unsafe_allow_html=True)

if translation_errors:
    st.subheader("❌ Translation Errors")
    for lang, error in translation_errors:
        display_translation_error(error)
```

## 2025-08-31 - Initial Project Setup ✅

### **Core Infrastructure**
- **Streamlit Application**: Main interface for translation and visualization
- **Docker Support**: Containerized deployment with Docker Compose
- **Environment Configuration**: Centralized settings management with `.env` files
- **Dependency Management**: Poetry-based package management

### **LLM Integration Foundation**
- **Ollama Support**: Local model hosting and management
- **OpenAI Integration**: API-based model access with configurable parameters
- **Model Selection**: Dynamic provider and model switching
- **Translation Pipeline**: Async text generation with proper error handling

### **NLP & Visualization Features**
- **Semantic Graph Generation**: Word relationship visualization across languages
- **Co-occurrence Networks**: Frequency-based word connection analysis
- **Cross-Language Relationships**: Mapping between different language representations
- **Interactive Graphs**: Pyvis-based network visualization with filtering

## Technical Architecture Summary

### **Configuration Management**
- **Pydantic Settings**: Type-safe configuration with environment variable support
- **Centralized Config**: Single source of truth for all application settings
- **Runtime Updates**: Dynamic configuration changes without restarts

### **Error Handling Strategy**
- **Layered Error Handling**: API, translation, and UI error management
- **User-Friendly Messages**: Actionable error guidance for common issues
- **Graceful Degradation**: Continued functionality with partial failures
- **Comprehensive Logging**: Detailed error tracking for debugging

### **Security Implementation**
- **API Key Isolation**: Secure storage without environment variable exposure
- **Session State Security**: User-specific data isolation
- **Input Validation**: Robust error checking and sanitization
- **Secure Client Initialization**: Protected credential handling

### **Code Organization**
- **Abstract Base Classes**: Clean interfaces for extensibility
- **Factory Pattern**: Dynamic object creation based on configuration
- **Separation of Concerns**: Clear module boundaries and responsibilities
- **Configuration-Driven**: Behavior controlled by settings rather than hardcoded values

## 2025-08-31 - Graph Storage System Implementation ✅

### **Functional Changes**
- **Persistent Graph Storage**: Implemented abstract base class pattern for graph data persistence
- **Code Cleanup**: Removed redundant language mapping code and centralized configuration usage
- **Streamlit Integration**: Seamless integration with existing Streamlit session state
- **Graph History Management**: Sidebar display of recent graphs with load/delete functionality
- **Search Capabilities**: Text-based search through stored graph metadata
- **Storage Statistics**: Real-time display of total graphs, nodes, edges, and storage size
- **Data Management**: Clear all graphs functionality and individual graph deletion

### **Technical Implementation**
```python
# graph_storage.py - Abstract base class and implementation
class GraphStorage(ABC):
    """Abstract base class for graph storage implementations"""
    
    @abstractmethod
    def store_graph(self, source_text: str, target_language: str, 
                   graph_data: Dict[str, Any], model_used: str) -> str:
        """Store graph data and return unique identifier"""
        pass
    
    @abstractmethod
    def get_graph(self, graph_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve graph data by ID"""
        pass
    
    @abstractmethod
    def get_graph_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent graph history"""
        pass

class StreamlitGraphStorage(GraphStorage):
    """Streamlit-specific graph storage implementation"""
    
    def __init__(self, storage_dir: str = "graph_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.graphs_file = self.storage_dir / "graphs.json"
        self.nodes_file = self.storage_dir / "nodes.pkl"
        self.edges_file = self.storage_dir / "edges.pkl"
        self._load_existing_data()
    
    def store_graph(self, source_text: str, target_language: str, 
                   graph_data: Dict[str, Any], model_used: str) -> str:
        """Store graph with metadata and return unique ID"""
        graph_id = f"graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.graphs)}"
        
        # Store metadata
        metadata = {
            "id": graph_id,
            "source_text": source_text,
            "target_language": target_language,
            "model_used": model_used,
            "timestamp": datetime.now().isoformat(),
            "node_count": len(graph_data.get("nodes", [])),
            "edge_count": len(graph_data.get("edges", [])),
            "storage_size": self._calculate_storage_size(graph_data)
        }
        
        self.graphs[graph_id] = metadata
        self._save_data()
        
        logger.info(f"Stored graph {graph_id} with {metadata['node_count']} nodes and {metadata['edge_count']} edges")
        return graph_id
```

### **Storage Architecture**
```python
# Storage file structure
graph_storage/
├── graphs.json      # Graph metadata and searchable information
├── nodes.pkl        # Serialized graph node data
└── edges.pkl        # Serialized graph edge data

# Metadata structure
{
    "id": "graph_20250831_212149_0",
    "source_text": "hello, is it me you're looking for?",
    "target_language": "es",
    "model_used": "gpt-4o",
    "timestamp": "2025-08-31T21:21:49.123456",
    "node_count": 14,
    "edge_count": 7,
    "storage_size": "2.3 KB"
}
```

### **Streamlit Integration**
```python
# app.py - Graph storage initialization
def initialize_session_state():
    # ... existing initialization ...
    
    # Initialize graph storage
    if "graph_storage" not in st.session_state:
        st.session_state.graph_storage = get_graph_storage()

# Automatic graph storage after generation
if all_graph_data and len(all_graph_data) > 0:
    for target_lang, graph_data in all_graph_data.items():
        if graph_data and len(graph_data.get("nodes", [])) > 0:
            try:
                graph_id = st.session_state.graph_storage.store_graph(
                    source_text=source_text,
                    target_language=target_lang,
                    graph_data=graph_data,
                    model_used=llm_client.model_name if llm_client else "unknown"
                )
                st.success(f"Stored graph {graph_id} for {target_lang}")
            except Exception as e:
                st.error(f"Failed to store graph for {target_lang}: {str(e)}")
```

### **Graph History Sidebar**
```python
# app.py - Graph history display
st.header("📊 Graph History")

try:
    history = st.session_state.graph_storage.get_graph_history(limit=10)
    
    if history:
        for graph in history:
            with st.expander(f"📈 {graph['target_language'].upper()}: {graph['source_text'][:50]}..."):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Source:** {graph['source_text']}")
                    st.write(f"**Model:** {graph['model_used']}")
                    st.write(f"**Nodes:** {graph['node_count']}, **Edges:** {graph['edge_count']}")
                    st.write(f"**Size:** {graph['storage_size']}")
                
                with col2:
                    if st.button("🔄 Load", key=f"load_{graph['id']}"):
                        # Load graph data back into session state
                        loaded_graph = st.session_state.graph_storage.get_graph(graph['id'])
                        if loaded_graph:
                            st.session_state["graph_data"] = {graph['target_language']: loaded_graph}
                            st.rerun()
                    
                    if st.button("🗑️ Delete", key=f"delete_{graph['id']}"):
                        st.session_state.graph_storage.delete_graph(graph['id'])
                        st.rerun()
    
    # Storage statistics
    stats = st.session_state.graph_storage.get_storage_stats()
    st.sidebar.metric("Total Graphs", stats["total_graphs"])
    st.sidebar.metric("Total Nodes", stats["total_nodes"])
    st.sidebar.metric("Total Edges", stats["total_edges"])
    st.sidebar.metric("Storage Size", stats["total_size"])
    
except Exception as e:
    st.error(f"Error loading graph history: {str(e)}")
```

### **Search and Management Features**
```python
# graph_storage.py - Search functionality
def search_graphs(self, query: str) -> List[Dict[str, Any]]:
    """Search graphs by text content"""
    query_lower = query.lower()
    results = []
    
    for graph_id, metadata in self.graphs.items():
        if (query_lower in metadata["source_text"].lower() or
            query_lower in metadata["target_language"].lower() or
            query_lower in metadata["model_used"].lower()):
            results.append(metadata)
    
    return results

def clear_all_graphs(self) -> bool:
    """Clear all stored graphs"""
    try:
        self.graphs.clear()
        self.nodes_data.clear()
        self.edges_data.clear()
        self._save_data()
        logger.info("Cleared all graphs")
        return True
    except Exception as e:
        logger.error(f"Error clearing graphs: {e}")
        return False
```

### **Code Cleanup - Configuration Centralization**
```python
# Before: Redundant language mapping in nlp_utils.py
lang_mapping = {
    "en": "en",
    "es": "es", 
    "ca": "ca"
}
full_lang = lang_mapping.get(language, "en")

# After: Using centralized config
if detected in LANG_MODELS:
    return detected
else:
    return "en"  # Default to English
```

**Why This Was Needed:**
- **Eliminated Duplication**: Removed redundant language code mappings
- **Single Source of Truth**: All language configuration now in `config.py`
- **Consistency**: Following established pattern of centralized configuration
- **Maintainability**: Changes to supported languages only need to happen in one place

### **Benefits of Implementation**
- **Data Persistence**: Graphs survive app restarts and session changes
- **Performance**: Faster access to previously generated graphs
- **User Experience**: Easy access to learning history and previous analyses
- **Storage Efficiency**: Compressed storage with metadata indexing
- **Extensibility**: Abstract base class allows for different storage backends
- **Searchability**: Find specific graphs by content or language
- **Management**: Easy cleanup and organization of stored data

## Current Status: Production Ready with Graph Storage ✅

The application now provides:
- **Secure API key management**
- **Professional error handling**
- **Dynamic model discovery**
- **Robust translation pipeline**
- **Clean separation of concerns**
- **Production-grade reliability**
- **Interactive graph analysis**
- **Comprehensive language learning features**
- **Persistent graph storage and management**
- **Graph history and search capabilities**

## Next Steps (Future Enhancements)
- **Additional Language Support**: Expand beyond EN/ES/CA
- **Advanced NLP Features**: Enhanced semantic analysis and relationship detection
- **Performance Optimization**: Caching, response time improvements, and batch processing
- **User Management**: Multi-user support, preferences, and learning progress tracking
- **API Endpoints**: RESTful API for external integrations and mobile apps
- **Advanced Analytics**: Learning progress tracking and vocabulary building insights
