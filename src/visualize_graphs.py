#!/usr/bin/env python
"""@Author:     sai.chen
@FileName:   visualize_graphs.py
@Date:       2025/09/23
@Description:
-----------------------------------------------------------
Visualization script for the RAG Research Agent graphs.

This script provides visualization capabilities for understanding the relationships
between the different graphs in the system.
-----------------------------------------------------------
"""

try:
    from IPython.display import Image, display

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    print("IPython not available - visualization will not work in notebook environment")

from src.graphs import index_graph, main_graph, researcher_graph
from src.log_util import logger


def draw_graph(graph, name: str = "Graph") -> None:
    """Visualize the graph in Jupyter Notebook (if available)

    Args:
        graph: The graph to visualize
        name: Name of the graph for display purposes
    """
    try:
        if not IPYTHON_AVAILABLE:
            logger.warning("IPython is not installed. Cannot visualize graph.")
            return

        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
        logger.info(f"Successfully displayed {name}")
    except Exception as e:
        logger.warning(f"Could not visualize {name}: {e}")


def visualize_all_graphs() -> None:
    """Visualize all graphs in the system to understand their structure and relationships."""
    print("Visualizing all graphs in the RAG Research Agent system...")
    print("\n1. Index Graph")
    print("--------------")
    draw_graph(index_graph, "Index Graph")

    print("\n2. Researcher Graph")
    draw_graph(researcher_graph, "Researcher Graph")

    print("\n3. Retrieval Graph")
    draw_graph(main_graph, "Retrieval Graph")


if __name__ == "__main__":
    visualize_all_graphs()
