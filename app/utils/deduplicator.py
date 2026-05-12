import logging
from typing import List, Dict, Set, Tuple
from difflib import SequenceMatcher
from .pydantic_models import EnrichedCodeChunk, DeduplicationGroup

logger = logging.getLogger(__name__)


class EntityDeduplicator:
    """Identify and merge duplicate/similar code entities across files."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize deduplicator.
        
        Args:
            similarity_threshold (float): Similarity score (0-1) to consider entities identical
        """
        self.similarity_threshold = similarity_threshold
        self.dedup_groups: Dict[str, DeduplicationGroup] = {}
    
    def deduplicate(self, enriched_chunks: List[EnrichedCodeChunk]) -> Tuple[
        Dict[str, str],  # id -> canonical_id mapping
        List[DeduplicationGroup]  # dedup groups
    ]:
        """
        Find duplicates and merge them.
        
        Args:
            enriched_chunks (List[EnrichedCodeChunk]): Chunks to deduplicate
        
        Returns:
            Tuple of:
                - Mapping from all IDs to canonical ID
                - List of deduplication groups
        """
        # Group by name first (fast pre-filter)
        by_name: Dict[str, List[EnrichedCodeChunk]] = {}
        for chunk in enriched_chunks:
            name = chunk.description.name
            if name not in by_name:
                by_name[name] = []
            by_name[name].append(chunk)
        
        id_to_canonical = {}
        groups = []
        
        # For each name group, find similar items
        for name, chunks in by_name.items():
            if len(chunks) == 1:
                id_to_canonical[chunks[0].id] = chunks[0].id
                continue
            
            # Find clusters of similar chunks
            clusters = self._cluster_similar(chunks)
            
            for cluster in clusters:
                if len(cluster) == 1:
                    id_to_canonical[cluster[0].id] = cluster[0].id
                else:
                    # Create dedup group
                    canonical = cluster[0]
                    canonical_id = canonical.id
                    aliases = [c.id for c in cluster[1:]]
                    
                    scores = [self._similarity(canonical, c) for c in cluster[1:]]
                    avg_similarity = sum(scores) / len(scores) if scores else 1.0
                    
                    group = DeduplicationGroup(
                        canonical_id=canonical_id,
                        aliases=aliases,
                        similarity_score=avg_similarity,
                        reason=f"Same {canonical.description.type} name '{name}' found in multiple files"
                    )
                    groups.append(group)
                    
                    # Map all to canonical
                    for chunk in cluster:
                        id_to_canonical[chunk.id] = canonical_id
                    
                    logger.info(
                        f"Deduplicated '{name}' across {len(cluster)} files: "
                        f"canonical={canonical_id}, similarity={avg_similarity:.2%}"
                    )
        
        self.dedup_groups = {g.canonical_id: g for g in groups}
        return id_to_canonical, groups
    
    def _cluster_similar(self, chunks: List[EnrichedCodeChunk]) -> List[List[EnrichedCodeChunk]]:
        """
        Cluster similar chunks together using agglomerative clustering.
        
        Args:
            chunks (List[EnrichedCodeChunk]): Chunks to cluster
        
        Returns:
            List of clusters (each cluster is a list of similar chunks)
        """
        if not chunks:
            return []
        
        # Start with each chunk as its own cluster
        clusters = [[chunk] for chunk in chunks]
        
        # Merge clusters that are similar enough
        while True:
            best_sim = 0
            best_i, best_j = -1, -1
            
            # Find most similar pair of clusters
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Use average similarity between all pairs
                    sims = [
                        self._similarity(ch_i, ch_j)
                        for ch_i in clusters[i]
                        for ch_j in clusters[j]
                    ]
                    avg_sim = sum(sims) / len(sims) if sims else 0
                    
                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_i, best_j = i, j
            
            # Stop if no good matches
            if best_sim < self.similarity_threshold:
                break
            
            # Merge the two best clusters
            clusters[best_i].extend(clusters[best_j])
            clusters.pop(best_j)
        
        return clusters
    
    def _similarity(self, chunk1: EnrichedCodeChunk, chunk2: EnrichedCodeChunk) -> float:
        """
        Calculate semantic similarity between two chunks.
        
        Uses multiple signals:
        - Source code similarity
        - Description similarity
        - Purpose/tags similarity
        
        Args:
            chunk1, chunk2 (EnrichedCodeChunk): Chunks to compare
        
        Returns:
            float: Similarity score 0-1
        """
        scores = []
        
        # Source code similarity
        code_sim = self._string_similarity(chunk1.source_code, chunk2.source_code)
        scores.append(code_sim * 0.4)  # Weight: 40%
        
        # Description similarity
        desc_sim = self._string_similarity(
            chunk1.description.summary,
            chunk2.description.summary
        )
        scores.append(desc_sim * 0.3)  # Weight: 30%
        
        # Purpose similarity
        purpose_sim = self._string_similarity(
            chunk1.description.purpose,
            chunk2.description.purpose
        )
        scores.append(purpose_sim * 0.2)  # Weight: 20%
        
        # Tag overlap
        tags1 = set(chunk1.description.tags)
        tags2 = set(chunk2.description.tags)
        if tags1 or tags2:
            tag_sim = len(tags1 & tags2) / len(tags1 | tags2) if (tags1 | tags2) else 1.0
            scores.append(tag_sim * 0.1)  # Weight: 10%
        
        return sum(scores)
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using sequence matcher."""
        if not s1 or not s2:
            return 1.0 if s1 == s2 else 0.0
        
        matcher = SequenceMatcher(None, s1.lower(), s2.lower())
        return matcher.ratio()


def apply_deduplication_mapping(
    enriched_chunks: List[EnrichedCodeChunk],
    id_to_canonical: Dict[str, str]
) -> List[EnrichedCodeChunk]:
    """
    Update chunk IDs based on deduplication mapping.
    
    Args:
        enriched_chunks (List[EnrichedCodeChunk]): Original chunks
        id_to_canonical (Dict[str, str]): ID mapping
    
    Returns:
        List[EnrichedCodeChunk]: Chunks with updated canonical IDs
    """
    result = []
    
    for chunk in enriched_chunks:
        canonical_id = id_to_canonical.get(chunk.id, chunk.id)
        
        # Only keep canonical versions (avoid duplicates in output)
        if chunk.id == canonical_id:
            result.append(chunk)
    
    return result
