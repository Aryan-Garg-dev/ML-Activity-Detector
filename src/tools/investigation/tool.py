"""InvestigationTool: Multi-hop graph analysis for entity connectivity.

Follows suspicious links between entities (e.g. sender to receiver) to 
uncover networks of suspicious activity (layering, rings).
"""

from typing import Any
from loguru import logger
from pydantic import BaseModel, Field
import networkx as nx

from core.config import AppConfig
from core.types import ToolName
from schemas.contracts import ToolResult
from storage.duckdb import DuckDBClient


class InvestigationInput(BaseModel):
    """Input schema for InvestigationTool."""
    start_account_id: str = Field(description="The starting account ID to investigate.")
    max_depth: int = Field(default=2, description="Maximum number of hops to explore.")


def execute_investigation(
    input: InvestigationInput,
    config: AppConfig,
    db_client: DuckDBClient,
    query_id: str,
    step_id: int,
) -> ToolResult:
    """Execute multi-hop graph analysis starting from an entity."""
    logger.info("Executing InvestigationTool for account [{account}] (depth: {depth})", account=input.start_account_id, depth=input.max_depth)

    acc_id = str(input.start_account_id)
    depth = input.max_depth

    # Use networkx to build a graph of transactions up to max_depth
    # We query duckdb for transactions where sender or receiver is in our current layer of nodes
    
    current_nodes = {acc_id}
    visited_nodes = set()
    edges = []
    
    for _ in range(depth):
        if not current_nodes:
            break
            
        nodes_list = list(current_nodes)
        placeholders = ", ".join(["?"] * len(nodes_list))
        
        # Query outgoing and incoming transactions
        query = f"""
            SELECT sender_account_id, receiver_account_id, tx_amount
            FROM transactions
            WHERE sender_account_id IN ({placeholders})
               OR receiver_account_id IN ({placeholders})
            LIMIT 500
        """
        params = nodes_list + nodes_list
        rows = db_client.query(query, params)
        
        next_nodes = set()
        for r in rows:
            sender = str(r[0])
            receiver = str(r[1])
            amount = float(r[2])
            edges.append({"source": sender, "target": receiver, "amount": amount})
            
            if sender not in visited_nodes:
                next_nodes.add(sender)
            if receiver not in visited_nodes:
                next_nodes.add(receiver)
                
        visited_nodes.update(current_nodes)
        current_nodes = next_nodes - visited_nodes

    # Build NetworkX graph to compute some metrics
    G = nx.DiGraph()
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e["amount"])
        
    metrics = {
        "nodes_explored": len(G.nodes),
        "edges_found": len(G.edges),
        "cycles": [cycle for cycle in nx.simple_cycles(G) if len(cycle) > 1] if len(G.nodes) < 200 else []
    }
    
    return ToolResult(
        tool_name=ToolName.INVESTIGATION,
        step_id=step_id,
        status="ok",
        data={
            "investigation_graph": {
                "nodes": list(G.nodes),
                "edges": edges
            },
            "metrics": metrics
        },
        rows_count=len(G.nodes)
    )
