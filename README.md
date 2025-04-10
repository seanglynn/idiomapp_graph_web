# IdiomApp Graph Explorer

Interactive graph visualization applications with AI-powered analysis.

## Features

- **Interactive Graph Visualization**: View and manipulate graph structures with intuitive controls
- **Streamlit Interface**: User-friendly interface for graph exploration
- **AI-Powered Analysis**: Get insights about your graphs using Ollama LLM integration
- **Text-to-Speech**: Listen to AI explanations with automatic language detection
- **FastAPI Backend**: Alternative lightweight API for graph data integration

## Project Structure

```
idiomapp/
├── api/              # FastAPI application
│   └── app.py        # FastAPI app entry point
├── streamlit/        # Streamlit application
│   └── app.py        # Streamlit app entry point
└── utils/            # Shared utilities
    ├── logging_utils.py  # Logging configuration
    └── ollama_utils.py   # Ollama LLM integration
```

## Installation

1. Make sure you have Python 3.12+ installed
2. Clone this repository
3. Install dependencies:

```bash
make install
```

## Running the Applications

### Streamlit Graph Explorer

The main application with full features and AI integration:

```bash
make run-graph
```

For development with auto-refresh:

```bash
make run-graph-dev
```

Your browser will automatically open to the Streamlit interface (typically http://localhost:8503).

### FastAPI Backend

A lightweight API for graph data:

```bash
make run-fastapi
```

Then open your browser and navigate to http://localhost:8001

## Environment Configuration

The application uses a `.env` file for configuration:

```
# Logging configuration
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Ollama configuration
OLLAMA_HOST=http://127.0.0.1:11434
DEFAULT_MODEL=llama3.2:latest
```

## Working with the Poetry Environment

To activate the Poetry virtual environment for interactive use:

```bash
make init
```

## Development

To install the project in development mode:

```bash
make install-dev
```

## API Endpoints

- `/` - Main page with interactive graph visualization
- `/api/graph-data` - JSON API endpoint that returns graph data

## Customization

You can modify the graph generation logic in `idiomapp/main.py` by editing the `generate_graph_data()` function. 