"""
Graph storage utilities for IdiomApp using abstract base class pattern.
Provides both immediate access and persistent storage for graph data.
"""

import json
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

from idiomapp.utils.logging_utils import get_logger

# Set up logging using the cached logger
logger = get_logger("graph_storage")


class GraphStorage(ABC):
    """
    Abstract base class for graph storage implementations.
    Defines the interface that all storage backends must implement.
    """
    
    @abstractmethod
    def store_graph(self, 
                   source_text: str, 
                   target_languages: List[str], 
                   nodes: List[Dict], 
                   edges: List[Dict], 
                   user_session: str = None, 
                   model_used: str = None,
                   translation_text: str = None) -> str:
        """
        Store a complete graph with metadata, nodes, and edges.
        
        Args:
            source_text: Original text that was translated
            target_languages: List of target languages used
            nodes: List of graph nodes
            edges: List of graph edges
            user_session: User session identifier
            model_used: LLM model used for generation
            translation_text: Full translation text
            
        Returns:
            str: Generated graph ID
        """
        pass
    
    @abstractmethod
    def get_graph(self, graph_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a complete graph by ID.
        
        Args:
            graph_id: ID of the graph to retrieve
            
        Returns:
            Dict containing graph metadata, nodes, and edges, or None if not found
        """
        pass
    
    @abstractmethod
    def get_graph_history(self, user_session: str = None, limit: int = 20) -> List[Dict]:
        """
        Get recent graph generation history.
        
        Args:
            user_session: Filter by user session (optional)
            limit: Maximum number of graphs to return
            
        Returns:
            List of graph metadata sorted by creation date
        """
        pass
    
    @abstractmethod
    def delete_graph(self, graph_id: str) -> bool:
        """
        Delete a graph and all its data.
        
        Args:
            graph_id: ID of the graph to delete
            
        Returns:
            bool: True if deletion was successful
        """
        pass
    
    @abstractmethod
    def search_graphs_by_text(self, search_text: str, limit: int = 10) -> List[Dict]:
        """
        Search graphs by source text content.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of matching graph metadata
        """
        pass
    
    @abstractmethod
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics about stored graphs.
        
        Returns:
            Dict containing statistics
        """
        pass
    
    @abstractmethod
    def clear_all_data(self) -> bool:
        """
        Clear all stored graph data.
        
        Returns:
            bool: True if clearing was successful
        """
        pass


class StreamlitGraphStorage(GraphStorage):
    """
    Streamlit-specific graph storage implementation using file persistence.
    Combines fast access with persistent storage for graph results.
    """
    
    def __init__(self, storage_dir: str = "./graph_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # File paths for different storage types
        self.graphs_file = self.storage_dir / "graphs.json"
        self.nodes_file = self.storage_dir / "nodes.pkl"
        self.edges_file = self.storage_dir / "edges.pkl"
        
        # Initialize storage files if they don't exist
        self._init_storage_files()
    
    def _init_storage_files(self):
        """Initialize storage files with empty structures if they don't exist"""
        if not self.graphs_file.exists():
            self._save_graphs_metadata({})
        
        if not self.nodes_file.exists():
            self._save_nodes_data({})
        
        if not self.edges_file.exists():
            self._save_edges_data({})
    
    def _save_graphs_metadata(self, data: Dict):
        """Save graph metadata to JSON file"""
        try:
            with open(self.graphs_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Error saving graphs metadata: {e}")
    
    def _load_graphs_metadata(self) -> Dict:
        """Load graph metadata from JSON file"""
        try:
            with open(self.graphs_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading graphs metadata: {e}")
            return {}
    
    def _save_nodes_data(self, data: Dict):
        """Save nodes data using pickle for complex objects"""
        try:
            with open(self.nodes_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving nodes data: {e}")
    
    def _load_nodes_data(self) -> Dict:
        """Load nodes data using pickle"""
        try:
            with open(self.nodes_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading nodes data: {e}")
            return {}
    
    def _save_edges_data(self, data: Dict):
        """Save edges data using pickle for complex objects"""
        try:
            with open(self.edges_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving edges data: {e}")
    
    def _load_edges_data(self) -> Dict:
        """Load edges data using pickle"""
        try:
            with open(self.edges_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading edges data: {e}")
            return {}
    
    def store_graph(self, 
                   source_text: str, 
                   target_languages: List[str], 
                   nodes: List[Dict], 
                   edges: List[Dict], 
                   user_session: str = None, 
                   model_used: str = None,
                   translation_text: str = None) -> str:
        """
        Store a complete graph with metadata, nodes, and edges.
        
        Args:
            source_text: Original text that was translated
            target_languages: List of target languages used
            nodes: List of graph nodes
            edges: List of graph edges
            user_session: User session identifier
            model_used: LLM model used for generation
            translation_text: Full translation text
            
        Returns:
            str: Generated graph ID
        """
        # Generate unique graph ID
        graph_id = f"graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self._load_graphs_metadata())}"
        
        # Prepare graph metadata
        graph_metadata = {
            "id": graph_id,
            "source_text": source_text,
            "target_languages": target_languages,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "created_at": datetime.now().isoformat(),
            "user_session": user_session or "anonymous",
            "model_used": model_used or "unknown",
            "translation_text": translation_text or ""
        }
        
        # Load existing data
        graphs_metadata = self._load_graphs_metadata()
        nodes_data = self._load_nodes_data()
        edges_data = self._load_edges_data()
        
        # Add new graph data
        graphs_metadata[graph_id] = graph_metadata
        nodes_data[graph_id] = nodes
        edges_data[graph_id] = edges
        
        # Save all data
        self._save_graphs_metadata(graphs_metadata)
        self._save_nodes_data(nodes_data)
        self._save_edges_data(edges_data)
        
        logger.info(f"Stored graph {graph_id} with {len(nodes)} nodes and {len(edges)} edges")
        return graph_id
    
    def get_graph(self, graph_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a complete graph by ID.
        
        Args:
            graph_id: ID of the graph to retrieve
            
        Returns:
            Dict containing graph metadata, nodes, and edges, or None if not found
        """
        graphs_metadata = self._load_graphs_metadata()
        nodes_data = self._load_nodes_data()
        edges_data = self._load_edges_data()
        
        if graph_id not in graphs_metadata:
            return None
        
        return {
            "metadata": graphs_metadata[graph_id],
            "nodes": nodes_data.get(graph_id, []),
            "edges": edges_data.get(graph_id, [])
        }
    
    def get_graph_history(self, user_session: str = None, limit: int = 20) -> List[Dict]:
        """
        Get recent graph generation history.
        
        Args:
            user_session: Filter by user session (optional)
            limit: Maximum number of graphs to return
            
        Returns:
            List of graph metadata sorted by creation date
        """
        graphs_metadata = self._load_graphs_metadata()
        
        # Convert to list and sort by creation date
        graphs_list = list(graphs_metadata.values())
        graphs_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Filter by user session if specified
        if user_session:
            graphs_list = [g for g in graphs_list if g.get('user_session') == user_session]
        
        return graphs_list[:limit]
    
    def delete_graph(self, graph_id: str) -> bool:
        """
        Delete a graph and all its data.
        
        Args:
            graph_id: ID of the graph to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            # Load existing data
            graphs_metadata = self._load_graphs_metadata()
            nodes_data = self._load_nodes_data()
            edges_data = self._load_edges_data()
            
            # Remove graph data
            if graph_id in graphs_metadata:
                del graphs_metadata[graph_id]
            if graph_id in nodes_data:
                del nodes_data[graph_id]
            if graph_id in edges_data:
                del edges_data[graph_id]
            
            # Save updated data
            self._save_graphs_metadata(graphs_metadata)
            self._save_nodes_data(nodes_data)
            self._save_edges_data(edges_data)
            
            logger.info(f"Deleted graph {graph_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting graph {graph_id}: {e}")
            return False
    
    def search_graphs_by_text(self, search_text: str, limit: int = 10) -> List[Dict]:
        """
        Search graphs by source text content.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of matching graph metadata
        """
        graphs_metadata = self._load_graphs_metadata()
        search_text_lower = search_text.lower()
        
        matches = []
        for graph in graphs_metadata.values():
            source_text = graph.get('source_text', '').lower()
            translation_text = graph.get('translation_text', '').lower()
            
            if (search_text_lower in source_text or 
                search_text_lower in translation_text):
                matches.append(graph)
        
        # Sort by creation date and limit results
        matches.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return matches[:limit]
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics about stored graphs.
        
        Returns:
            Dict containing statistics
        """
        graphs_metadata = self._load_graphs_metadata()
        nodes_data = self._load_nodes_data()
        edges_data = self._load_edges_data()
        
        total_graphs = len(graphs_metadata)
        total_nodes = sum(len(nodes_data.get(gid, [])) for gid in graphs_metadata)
        total_edges = sum(len(edges_data.get(gid, [])) for gid in graphs_metadata)
        
        # Language distribution
        language_counts = {}
        for graph in graphs_metadata.values():
            for lang in graph.get('target_languages', []):
                language_counts[lang] = language_counts.get(lang, 0) + 1
        
        return {
            "total_graphs": total_graphs,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "language_distribution": language_counts,
            "storage_size_mb": self._get_storage_size()
        }
    
    def _get_storage_size(self) -> float:
        """Calculate storage size in MB"""
        try:
            total_size = 0
            for file_path in [self.graphs_file, self.nodes_file, self.edges_file]:
                if file_path.exists():
                    total_size += file_path.stat().st_size
            return round(total_size / (1024 * 1024), 2)
        except Exception:
            return 0.0
    
    def clear_all_data(self) -> bool:
        """
        Clear all stored graph data.
        
        Returns:
            bool: True if clearing was successful
        """
        try:
            self._save_graphs_metadata({})
            self._save_nodes_data({})
            self._save_edges_data({})
            logger.info("Cleared all graph data")
            return True
        except Exception as e:
            logger.error(f"Error clearing all data: {e}")
            return False


# Future storage implementations can be added here:

class InMemoryGraphStorage(GraphStorage):
    """
    In-memory graph storage implementation for testing and development.
    Data is lost when the application restarts.
    """
    
    def __init__(self):
        self._graphs_metadata = {}
        self._nodes_data = {}
        self._edges_data = {}
        self._counter = 0
    
    def store_graph(self, 
                   source_text: str, 
                   target_languages: List[str], 
                   nodes: List[Dict], 
                   edges: List[Dict], 
                   user_session: str = None, 
                   model_used: str = None,
                   translation_text: str = None) -> str:
        """Store graph in memory"""
        graph_id = f"mem_graph_{self._counter}"
        self._counter += 1
        
        graph_metadata = {
            "id": graph_id,
            "source_text": source_text,
            "target_languages": target_languages,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "created_at": datetime.now().isoformat(),
            "user_session": user_session or "anonymous",
            "model_used": model_used or "unknown",
            "translation_text": translation_text or ""
        }
        
        self._graphs_metadata[graph_id] = graph_metadata
        self._nodes_data[graph_id] = nodes
        self._edges_data[graph_id] = edges
        
        logger.info(f"Stored graph {graph_id} in memory with {len(nodes)} nodes and {len(edges)} edges")
        return graph_id
    
    def get_graph(self, graph_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve graph from memory"""
        if graph_id not in self._graphs_metadata:
            return None
        
        return {
            "metadata": self._graphs_metadata[graph_id],
            "nodes": self._nodes_data.get(graph_id, []),
            "edges": self._edges_data.get(graph_id, [])
        }
    
    def get_graph_history(self, user_session: str = None, limit: int = 20) -> List[Dict]:
        """Get graph history from memory"""
        graphs_list = list(self._graphs_metadata.values())
        graphs_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        if user_session:
            graphs_list = [g for g in graphs_list if g.get('user_session') == user_session]
        
        return graphs_list[:limit]
    
    def delete_graph(self, graph_id: str) -> bool:
        """Delete graph from memory"""
        if graph_id in self._graphs_metadata:
            del self._graphs_metadata[graph_id]
        if graph_id in self._nodes_data:
            del self._nodes_data[graph_id]
        if graph_id in self._edges_data:
            del self._edges_data[graph_id]
        return True
    
    def search_graphs_by_text(self, search_text: str, limit: int = 10) -> List[Dict]:
        """Search graphs in memory"""
        search_text_lower = search_text.lower()
        matches = []
        
        for graph in self._graphs_metadata.values():
            source_text = graph.get('source_text', '').lower()
            translation_text = graph.get('translation_text', '').lower()
            
            if (search_text_lower in source_text or 
                search_text_lower in translation_text):
                matches.append(graph)
        
        matches.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return matches[:limit]
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics from memory"""
        total_graphs = len(self._graphs_metadata)
        total_nodes = sum(len(self._nodes_data.get(gid, [])) for gid in self._graphs_metadata)
        total_edges = sum(len(self._edges_data.get(gid, [])) for gid in self._graphs_metadata)
        
        language_counts = {}
        for graph in self._graphs_metadata.values():
            for lang in graph.get('target_languages', []):
                language_counts[lang] = language_counts.get(lang, 0) + 1
        
        return {
            "total_graphs": total_graphs,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "language_distribution": language_counts,
            "storage_size_mb": 0.0  # In-memory storage has no file size
        }
    
    def clear_all_data(self) -> bool:
        """Clear all data from memory"""
        self._graphs_metadata.clear()
        self._nodes_data.clear()
        self._edges_data.clear()
        self._counter = 0
        return True


# Convenience functions for getting storage instances
def get_graph_storage() -> GraphStorage:
    """Get the default graph storage instance (StreamlitGraphStorage)."""
    return StreamlitGraphStorage()

def get_in_memory_storage() -> GraphStorage:
    """Get an in-memory storage instance for testing."""
    return InMemoryGraphStorage()

def get_storage_by_type(storage_type: str = "streamlit") -> GraphStorage:
    """
    Get storage instance by type.
    
    Args:
        storage_type: Type of storage ("streamlit", "memory", etc.)
        
    Returns:
        GraphStorage instance
    """
    if storage_type == "memory":
        return InMemoryGraphStorage()
    else:
        return StreamlitGraphStorage()
