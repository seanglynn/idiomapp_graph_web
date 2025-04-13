# IdiomApp

WIP

## Project Structure

- `idiomapp/streamlit/`: Main Streamlit application
- `idiomapp/utils/`: Utility modules for Ollama integration and logging
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

## Configuration

Create a `.env` file with:

```
# Ollama configuration
OLLAMA_HOST=http://ollama:11434
DEFAULT_MODEL=llama3.2:latest

# Logging
LOG_LEVEL=INFO
```

## Troubleshooting

If you encounter Docker issues:
1. Verify Docker and Docker Compose are installed and running
2. Check Ollama accessibility from the Streamlit container
3. Use `make docker-shell` or `make ollama-shell` for debugging 
