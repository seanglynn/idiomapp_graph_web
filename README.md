# IdiomApp Visualization

A Python web application that visualizes graph data using FastAPI and D3.js.

## Features

- Interactive graph visualization with D3.js
- Auto-populated graph data from Python backend
- Ability to generate new random graphs on demand
- Zoom and pan functionality

## Installation

1. Make sure you have Python 3.12+ installed
2. Clone this repository
3. Install dependencies:

```bash
make install
```

## Running the Application

Start the FastAPI development server:

```bash
make run-fastapi
```

Then open your browser and navigate to http://localhost:8001

## Running Streamlit Applications

Streamlit provides an alternative way to visualize your graph data with interactive components:

1. Run the Streamlit application:

```bash
make run-graph
```

2. Your browser will automatically open to the Streamlit app (typically at http://localhost:8501)

3. For development with auto-refresh:

```bash
make run-graph-dev
```

This will automatically refresh the application whenever you save changes to your code.

## Working with the Poetry Environment

To activate the Poetry virtual environment for interactive use:

```bash
make init
```

This starts a new shell with the project's environment activated, allowing you to run Python commands without prefixing them with `poetry run`.

## API Endpoints

- `/` - Main page with interactive graph visualization
- `/api/graph-data` - JSON API endpoint that returns graph data

## Customization

You can modify the graph generation logic in `idiomapp/main.py` by editing the `generate_graph_data()` function. 