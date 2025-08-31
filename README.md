# IdiomApp
Visualizing linguistic connections through interactive graphs and networks.

**¿Qué es esto?** 
- Establecer relaciones entre palabras y frases en varios idiomas.

**¿Por qué?**
- Quiero aprender idioma**s** más rápido.

**¿Cómo?**
- Con LLMs, NLP, semantic graph & co-occurrence networks obvio!


## Project Structure

- `idiomapp/streamlit/`: Main Streamlit application
- `idiomapp/utils/`: Utility modules for LLM integration, NLP, TTS, state management, and logging
  - `state_utils.py`: Centralized state management with caching and session state utilities
  - `llm_utils.py`: LLM client abstractions for Ollama and OpenAI
  - `nlp_utils.py`: NLP processing and graph generation utilities
  - `audio_utils.py`: Text-to-speech and audio processing
  - `logging_utils.py`: Centralized logging configuration
- `archive/`: Archived code (previous FastAPI application)

## Quick Start

### Local Development

```bash
# Install dependencies
make install

# Run the Streamlit app with auto-refresh
make run-graph-dev
```

### Docker (Recommended)

```bash
# Start application containers
make docker-start

# Stop when finished
make docker-down
```

Access the application at: http://localhost:8503

## Docker Commands

- `make docker-start`: Start containers (interactive mode with logs)
- `make docker-down`: Stop containers
- `make docker-shell`: Access Streamlit container shell
- `make ollama-shell`: Access Ollama container shell

## Docker Setup

This application can be run in Docker containers for easy deployment. Docker-specific files are organized in the `docker/` directory:

- `docker/Dockerfile`: Container definition for the Streamlit application
- `docker/docker-entrypoint.sh`: Entry point script that handles configuration and startup

### Running with Docker

```bash
# Start in interactive mode (recommended for development)
make docker-start

# Or run in background
make docker-start-detached
```

Access the application at: http://localhost:8503

## Configuration

Create a `.env` file with the following settings:

```
# LLM Provider Configuration
# Choose between "ollama" or "openai"
LLM_PROVIDER=ollama

# Ollama configuration (if using Ollama)
OLLAMA_HOST=http://localhost:11434  # Use http://ollama:11434 for Docker
DEFAULT_MODEL=llama3.2:latest

# OpenAI configuration (if using OpenAI)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Logging
LOG_LEVEL=INFO

# Streamlit configuration
STREAMLIT_SERVER_PORT=8503
STREAMLIT_SERVER_HEADLESS=false    # Use true for Docker/production
STREAMLIT_SERVER_ENABLECORS=false
STREAMLIT_SERVER_ADDRESS=localhost  # Use 0.0.0.0 for Docker
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider to use (`ollama` or `openai`) | `ollama` |
| `OLLAMA_HOST` | URL of the Ollama service (when using Ollama) | `http://localhost:11434` |
| `DEFAULT_MODEL` | Default Ollama model to use | `llama3.2:latest` |
| `OPENAI_API_KEY` | OpenAI API key (when using OpenAI) | empty |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-3.5-turbo` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `STREAMLIT_SERVER_PORT` | Port for Streamlit server | `8503` |
| `STREAMLIT_SERVER_HEADLESS` | Run in headless mode | `false` |
| `STREAMLIT_SERVER_ENABLECORS` | Enable CORS | `false` |
| `STREAMLIT_SERVER_ADDRESS` | Bind server to address | `0.0.0.0` in Docker, `localhost` for local dev |

## LLM Integration

This application supports two different LLM providers:

### Ollama (Default)

Uses the local Ollama service with models like `llama3.2:latest`. Requires the Ollama service to be running and accessible.

### OpenAI

Uses OpenAI's API with models like `gpt-3.5-turbo`, `gpt-4`, etc. Requires an OpenAI API key.

You can switch between providers in the UI or by setting the `LLM_PROVIDER` environment variable.

## Security Note

When running locally, the app is configured to only be accessible via `localhost`.

When running in Docker, the server binds to `0.0.0.0` (all interfaces) to make it accessible from your host machine at http://localhost:8503. The Docker container is isolated, but the port is mapped to your localhost.

## Troubleshooting

If you encounter Docker issues:
1. Verify Docker and Docker Compose are installed and running
2. Check Ollama accessibility from the Streamlit container
3. Use `make docker-shell` or `make ollama-shell` for debugging 
