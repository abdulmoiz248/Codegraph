import logging
import json
from pathlib import Path
from typing import List, Dict
from .code_chunker import chunk_repository, CodeChunk
from .llm_enricher import GeminiEnricher
from .deduplicator import EntityDeduplicator, apply_deduplication_mapping
from .pydantic_models import EnrichedCodeChunk, DeduplicationGroup

logger = logging.getLogger(__name__)


class EnrichmentPipeline:
    """
    Complete pipeline for code enrichment:
    1. Chunk code by function/class boundaries
    2. Enrich with LLM semantic descriptions
    3. Deduplicate entities across files
    4. Output enriched graph
    """
    
    def __init__(self, sim_threshold: float = 0.85):
        """
        Initialize enrichment pipeline.
        
        Args:
            sim_threshold (float): Similarity threshold for deduplication
        """
        self.enricher = GeminiEnricher()
        self.deduplicator = EntityDeduplicator(similarity_threshold=sim_threshold)
        self.enriched_chunks: List[EnrichedCodeChunk] = []
        self.dedup_groups: List[DeduplicationGroup] = []
    
    def enrich_repository(self, repo_path: str) -> Dict:
        """
        Full enrichment pipeline for a repository.
        
        Args:
            repo_path (str): Path to repository
        
        Returns:
            Dict with enriched data and deduplication info
        """
        logger.info("=== Phase 2: LLM Enrichment Pipeline ===")
        
        # Step 1: Chunk the code
        logger.info("Step 1/4: Chunking code by function/class boundaries...")
        chunks_by_file = chunk_repository(repo_path)
        all_chunks = [chunk for chunks in chunks_by_file.values() for chunk in chunks]
        logger.info(f"  ✓ Found {len(all_chunks)} code chunks")
        
        # Step 2: Enrich chunks with LLM
        logger.info("Step 2/4: Enriching chunks with Gemini LLM...")
        self.enriched_chunks = self.enricher.enrich_batch(all_chunks)
        logger.info(f"  ✓ Enriched {len(self.enriched_chunks)} chunks")
        
        # Step 3: Deduplicate entities
        logger.info("Step 3/4: Deduplicating entities across files...")
        id_to_canonical, self.dedup_groups = self.deduplicator.deduplicate(self.enriched_chunks)
        logger.info(f"  ✓ Found {len(self.dedup_groups)} deduplication groups")
        
        # Step 4: Build enriched graph
        logger.info("Step 4/4: Building enriched graph...")
        enriched_graph = self._build_enriched_graph(id_to_canonical)
        logger.info(f"  ✓ Built enriched graph with {len(enriched_graph['nodes'])} nodes")
        
        return enriched_graph
    
    def _build_enriched_graph(self, id_to_canonical: Dict[str, str]) -> Dict:
        """
        Build the enriched graph output.
        
        Args:
            id_to_canonical (Dict[str, str]): ID deduplication mapping
        
        Returns:
            Dict representing the enriched graph
        """
        # Keep only canonical chunks
        canonical_chunks = apply_deduplication_mapping(self.enriched_chunks, id_to_canonical)
        
        # Build nodes from enriched chunks
        nodes = []
        for chunk in canonical_chunks:
            node = {
                "id": chunk.id,
                "type": chunk.description.type.upper(),
                "name": chunk.description.name,
                "filepath": chunk.filepath,
                "lineno": chunk.lineno,
                "summary": chunk.description.summary,
                "purpose": chunk.description.purpose,
                "complexity": chunk.description.complexity,
                "tags": chunk.description.tags
            }
            nodes.append(node)
        
        # Build edges from semantic relationships
        edges = []
        
        # Explicit semantic edges from hidden_relationships
        for chunk in canonical_chunks:
            for sem_edge in chunk.description.hidden_relationships:
                canonical_target = id_to_canonical.get(sem_edge.target, sem_edge.target)
                
                edge = {
                    "source": chunk.id,
                    "target": canonical_target,
                    "rel_type": sem_edge.rel_type,
                    "confidence": sem_edge.confidence,
                    "reason": sem_edge.reason
                }
                edges.append(edge)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "deduplication_groups": [g.dict() for g in self.dedup_groups],
            "summary": {
                "total_chunks": len(self.enriched_chunks),
                "canonical_chunks": len(canonical_chunks),
                "dedup_groups": len(self.dedup_groups),
                "total_edges": len(edges),
                "edge_types": self._count_edge_types(edges)
            }
        }
    
    def _count_edge_types(self, edges: List[Dict]) -> Dict[str, int]:
        """Count edges by type."""
        counts = {}
        for edge in edges:
            rel_type = edge.get("rel_type", "UNKNOWN")
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts
    
    def save_enriched_graph(self, graph: Dict, output_path: str) -> str:
        """
        Save enriched graph to JSON file.
        
        Args:
            graph (Dict): Enriched graph
            output_path (str): Output file path
        
        Returns:
            str: Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(graph, f, indent=2)
        
        logger.info(f"✓ Enriched graph saved to {output_path}")
        return str(output_path)


def run_enrichment_pipeline(repo_path: str, output_file: str) -> str:
    """
    Run complete enrichment pipeline.
    
    Args:
        repo_path (str): Path to repository
        output_file (str): Output file for enriched graph
    
    Returns:
        str: Path to output file
    
    Raises:
        ValueError: If repo or paths invalid
        RuntimeError: If enrichment fails
    """
    repo_path = Path(repo_path)
    if not repo_path.exists():
        raise ValueError(f"Repository path not found: {repo_path}")
    
    try:
        pipeline = EnrichmentPipeline()
        enriched_graph = pipeline.enrich_repository(str(repo_path))
        return pipeline.save_enriched_graph(enriched_graph, output_file)
    
    except Exception as e:
        logger.error(f"Enrichment pipeline failed: {e}")
        raise RuntimeError(f"Failed to run enrichment pipeline: {e}") from e
