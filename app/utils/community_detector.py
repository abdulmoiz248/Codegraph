import json
import logging
import importlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .pydantic_models import CommunityDetectionResult, LeidenCommunity

logger = logging.getLogger(__name__)


# Prefer the real Leiden stack, but keep import optional so the app still runs.
try:  # pragma: no cover - optional dependency path
    ig = importlib.import_module("igraph")
    leidenalg = importlib.import_module("leidenalg")
    _LEIDEN_AVAILABLE = True
except Exception:
    ig = None
    leidenalg = None
    _LEIDEN_AVAILABLE = False


DEFAULT_EDGE_WEIGHTS = {
    "FUNCTION_CALLS_FUNCTION": 3.0,
    "CLASS_INHERITS_FROM": 2.5,
    "CLASS_CONTAINS_METHOD": 1.8,
    "FILE_CONTAINS_FUNCTION": 0.8,
    "FILE_CONTAINS_CLASS": 0.8,
    "DEFAULT": 1.0,
}


def _edge_weight(rel_type: str) -> float:
    return DEFAULT_EDGE_WEIGHTS.get(rel_type, DEFAULT_EDGE_WEIGHTS["DEFAULT"])


def _eligible_node_ids(graph: Dict) -> List[str]:
    # Exclude FILE nodes from the community graph by default; communities are more useful on code entities.
    return [n["id"] for n in graph.get("nodes", []) if n.get("type") != "FILE"]


def _build_vertex_index(node_ids: List[str]) -> Dict[str, int]:
    return {node_id: idx for idx, node_id in enumerate(node_ids)}


def _build_igraph(graph: Dict, node_ids: List[str]):
    index = _build_vertex_index(node_ids)
    edges: List[Tuple[int, int]] = []
    weights: List[float] = []

    for edge in graph.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if source not in index or target not in index:
            continue

        # Leiden works best on an undirected graph for community discovery.
        if source == target:
            continue

        edges.append((index[source], index[target]))
        weights.append(_edge_weight(edge.get("rel_type", "DEFAULT")))

    g = ig.Graph(n=len(node_ids), edges=edges, directed=False)
    if weights:
        g.es["weight"] = weights
    g.vs["node_id"] = node_ids
    return g


def apply_leiden_communities(graph: Dict, resolution: float = 1.0, seed: int = 42) -> Dict:
    """
    Apply Leiden community detection to a graph payload.

    Args:
        graph (Dict): Graph with "nodes" and "edges"
        resolution (float): Leiden resolution parameter
        seed (int): RNG seed for deterministic results when supported

    Returns:
        Dict: Graph payload enriched with community assignments

    Raises:
        ImportError: If igraph/leidenalg are unavailable
    """
    if not _LEIDEN_AVAILABLE:
        raise ImportError(
            "Leiden requires 'python-igraph' and 'leidenalg'. Install with: "
            "uv pip install python-igraph leidenalg"
        )

    node_ids = _eligible_node_ids(graph)
    if not node_ids:
        logger.warning("No eligible nodes found for Leiden community detection")
        graph["communities"] = []
        graph["summary"] = {**graph.get("summary", {}), "community_count": 0}
        return graph

    g = _build_igraph(graph, node_ids)
    if g.vcount() == 0:
        logger.warning("Leiden graph has no vertices after filtering")
        graph["communities"] = []
        graph["summary"] = {**graph.get("summary", {}), "community_count": 0}
        return graph

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"] if g.ecount() else None,
        resolution_parameter=resolution,
        seed=seed,
    )

    membership = partition.membership
    community_map: Dict[int, List[str]] = defaultdict(list)
    node_lookup = {idx: node_id for idx, node_id in enumerate(node_ids)}

    for vertex_idx, community_id in enumerate(membership):
        community_map[community_id].append(node_lookup[vertex_idx])

    # Annotate nodes in-place.
    member_to_community = {member: community_id for community_id, members in community_map.items() for member in members}
    for node in graph.get("nodes", []):
        node_id = node.get("id")
        if node_id in member_to_community:
            node["community_id"] = member_to_community[node_id]

    communities = [
        LeidenCommunity(
            community_id=community_id,
            size=len(members),
            members=members,
        )
        for community_id, members in sorted(community_map.items(), key=lambda item: item[0])
    ]

    graph["communities"] = [community.model_dump() if hasattr(community, "model_dump") else community.dict() for community in communities]
    graph["summary"] = {
        **graph.get("summary", {}),
        "community_count": len(communities),
        "community_sizes": [c.size for c in communities],
        "community_algorithm": "leiden",
        "community_resolution": resolution,
    }

    return graph


def apply_leiden_communities_from_file(input_file: str, output_file: str = None, resolution: float = 1.0) -> str:
    """
    Load a graph JSON file, apply Leiden communities, and write the result back.

    Args:
        input_file (str): Path to input graph JSON
        output_file (str, optional): Output path. Defaults to input_file with `_leiden` suffix.
        resolution (float): Leiden resolution parameter

    Returns:
        str: Output file path
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Graph file not found: {input_path}")

    if output_file is None:
        output_file = str(input_path.with_name(f"{input_path.stem}_leiden{input_path.suffix}"))

    with open(input_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    graph = apply_leiden_communities(graph, resolution=resolution)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    logger.info(f"Leiden communities saved to {output_path}")
    return str(output_path)
