from typing import Any, List
import importlib

try:  # pragma: no cover - prefer pydantic when available, but avoid hard dependency
    _pydantic = importlib.import_module("pydantic")
    BaseModel = _pydantic.BaseModel
    Field = _pydantic.Field
    ValidationError = _pydantic.ValidationError
except Exception:  # pragma: no cover - fallback for environments without pydantic
    class ValidationError(Exception):
        pass

    def Field(default=..., default_factory=None, **_kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    class BaseModel:
        def __init__(self, **data: Any):
            for name, class_value in self.__class__.__dict__.items():
                if name.startswith("_") or callable(class_value):
                    continue
                if name in data:
                    value = data[name]
                elif class_value is not ...:
                    value = class_value
                else:
                    raise TypeError(f"Missing required field: {name}")
                setattr(self, name, value)

        def dict(self) -> dict:
            return dict(self.__dict__)

        def model_dump(self) -> dict:
            return self.dict()


class SemanticEdge(BaseModel):
    """Represents a semantic relationship between code elements."""
    target: str = Field(..., description="Name of the target function/class")
    rel_type: str = Field(..., description="Type of relationship (e.g., USES, DEPENDS_ON, SHARES_STATE)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score 0-1")
    reason: str = Field(..., description="Why this relationship exists")


class CodeElementDescription(BaseModel):
    """LLM-generated semantic description of a code element."""
    name: str = Field(..., description="Name of the function/class")
    type: str = Field(..., description="Type: 'function' or 'class'")
    summary: str = Field(..., description="1-2 sentence summary of what it does")
    purpose: str = Field(..., description="The core purpose/responsibility")
    inputs: str = Field(default="", description="What it takes as input")
    outputs: str = Field(default="", description="What it produces/returns")
    side_effects: str = Field(default="", description="Any side effects (DB writes, API calls, etc.)")
    hidden_relationships: List[SemanticEdge] = Field(
        default_factory=list,
        description="Relationships not directly visible in code (shared state, shared config, etc.)"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization (e.g., 'authentication', 'data-validation', 'api-handler')"
    )
    complexity: str = Field(
        default="medium",
        description="Estimated complexity: 'simple', 'medium', 'complex'"
    )


def validate_code_element_description(data: dict) -> CodeElementDescription:
    """Build and validate a code element description from raw JSON data."""
    hidden_relationships = [
        edge if isinstance(edge, SemanticEdge) else SemanticEdge(**edge)
        for edge in data.get("hidden_relationships", [])
    ]
    data = dict(data)
    data["hidden_relationships"] = hidden_relationships
    return CodeElementDescription(**data)


class EnrichedCodeChunk(BaseModel):
    """A code chunk combined with LLM-generated enrichment."""
    id: str = Field(..., description="Unique identifier (filepath:name)")
    source_code: str = Field(..., description="Original source code")
    filepath: str = Field(..., description="File path")
    lineno: int = Field(..., description="Starting line number")
    description: CodeElementDescription = Field(..., description="LLM-generated description")


class EnrichmentBatch(BaseModel):
    """A batch of enriched code elements for processing."""
    chunks: List[EnrichedCodeChunk]
    total_tokens_used: int = 0
    total_cost: float = 0.0


class DeduplicationGroup(BaseModel):
    """Group of identical or very similar code elements across files."""
    canonical_id: str = Field(..., description="The primary ID for this group")
    aliases: List[str] = Field(
        default_factory=list,
        description="Alternative IDs that map to the same entity"
    )
    similarity_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Average similarity score")
    reason: str = Field(..., description="Why these were deduplicated")


class CommunityNode(BaseModel):
    """A node assigned to a Leiden community."""
    id: str = Field(..., description="Graph node identifier")
    community_id: int = Field(..., description="Leiden community identifier")


class LeidenCommunity(BaseModel):
    """A single Leiden community with member node ids."""
    community_id: int = Field(..., description="Community identifier")
    size: int = Field(..., description="Number of nodes in the community")
    members: List[str] = Field(default_factory=list, description="Node ids that belong to this community")


class CommunityDetectionResult(BaseModel):
    """Result payload for Leiden community detection."""
    nodes: List[dict] = Field(default_factory=list)
    edges: List[dict] = Field(default_factory=list)
    communities: List[LeidenCommunity] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)
