from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import networkx as nx
from pathlib import Path

app = FastAPI()

# Mount static directory
app.mount(
    "/static", 
    StaticFiles(directory=Path(__file__).parent / "static"), 
    name="static"
)

# Set up templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Function to generate simple graph data
def generate_graph_data():
    # Create a very simple graph
    G = nx.path_graph(5)  # Simple linear graph with 5 nodes (0-4)
    
    # Add just one cross connection
    G.add_edge(1, 3)
    
    # Get node positions using a simple layout
    pos = nx.spring_layout(G, seed=42)
    
    # Get edges
    edges = []
    for source, target in G.edges():
        edges.append({
            "source": source,
            "target": target
        })
    
    # Get nodes
    nodes = []
    for node in G.nodes():
        position = pos[node]
        nodes.append({
            "id": node,
            "x": float(position[0]),
            "y": float(position[1])
        })
    
    return {"nodes": nodes, "edges": edges}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    graph_data = generate_graph_data()
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "graph_data": graph_data}
    )

@app.get("/api/graph-data")
async def get_graph_data():
    return generate_graph_data()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("graph_web.main:app", host="0.0.0.0", port=8001, reload=True) 