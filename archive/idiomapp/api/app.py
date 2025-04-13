"""
FastAPI application for the IdiomApp graph data API.
"""
import networkx as nx
import random
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from pathlib import Path

from idiomapp.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging("api_app")

# Create FastAPI app
app = FastAPI(title="IdiomApp API", description="API for graph data visualization")

# Setup paths for templates and static files
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page with graph visualization."""
    logger.info("Rendering index page")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/graph-data")
async def get_graph_data():
    """Return JSON data for graph visualization."""
    logger.info("Generating graph data")
    return generate_graph_data()


def generate_graph_data(num_nodes=10):
    """Generate random graph data for visualization."""
    # Create a random graph
    G = nx.gnp_random_graph(num_nodes, 0.3)
    
    # Format the data for D3.js
    nodes = [{"id": i, "name": f"Node {i}"} for i in G.nodes()]
    links = [{"source": u, "target": v} for u, v in G.edges()]
    
    logger.info(f"Generated graph with {len(nodes)} nodes and {len(links)} links")
    return {"nodes": nodes, "links": links}


def start():
    """Entry point for running the FastAPI application."""
    uvicorn.run("idiomapp.api.app:app", host="127.0.0.1", port=8001, reload=True)


if __name__ == "__main__":
    start() 