# Idiom App Visualization

A Python web application that visualizes graph data using FastAPI and D3.js.

## Features

- Interactive graph visualization with D3.js
- Auto-populated graph data from Python backend
- Ability to generate new random graphs on demand
- Zoom and pan functionality

## Installation

1. Make sure you have Python 3.12+ installed
2. Clone this repository
3. Install dependencies with Poetry:

```bash
poetry install
```

## Running the Application

Start the development server:

```bash
poetry run python run.py
```

Then open your browser and navigate to http://localhost:8001

## API Endpoints

- `/` - Main page with interactive graph visualization
- `/api/graph-data` - JSON API endpoint that returns graph data

## Customization

You can modify the graph generation logic in `idiomapp/main.py` by editing the `generate_graph_data()` function. 