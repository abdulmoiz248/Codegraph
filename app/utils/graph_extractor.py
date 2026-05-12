import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Node:
    id: str
    type: str  # FILE, CLASS, FUNCTION
    name: str
    filepath: str = None
    lineno: int = None


@dataclass
class Edge:
    source: str
    target: str
    rel_type: str  # FILE_CONTAINS_FUNCTION, FUNCTION_CALLS_FUNCTION, CLASS_INHERITS_FROM


class RepositoryGraphExtractor:
    """Extract file structure and import graph from a Python repository."""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.file_to_functions: Dict[str, List[str]] = {}
        self.file_to_classes: Dict[str, List[str]] = {}
    
    def extract(self) -> Dict:
        """Extract the complete graph from the repository."""
        logger.info(f"Starting extraction from: {self.root_dir}")
        
        # Find all Python files
        py_files = list(self.root_dir.rglob("*.py"))
        logger.info(f"Found {len(py_files)} Python files")
        
        # Process each file
        for filepath in py_files:
            self._process_file(filepath)
        
        logger.info(f"Extracted {len(self.nodes)} nodes and {len(self.edges)} edges")
        return self._build_output()
    
    def _process_file(self, filepath: Path) -> None:
        """Process a single Python file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            relative_path = filepath.relative_to(self.root_dir)
            file_id = f"FILE:{relative_path}"
            
            # Create file node
            self.nodes[file_id] = Node(
                id=file_id,
                type="FILE",
                name=str(relative_path),
                filepath=str(filepath)
            )
            
            # Extract classes and functions
            self._extract_definitions(tree, file_id, filepath, relative_path)
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {filepath}: {e}")
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
    
    def _extract_definitions(self, tree: ast.AST, file_id: str, filepath: Path, relative_path: Path) -> None:
        """Extract classes and functions from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self._process_class(node, file_id, filepath, relative_path)
            elif isinstance(node, ast.FunctionDef):
                self._process_function(node, file_id, filepath, relative_path)
    
    def _process_class(self, node: ast.ClassDef, file_id: str, filepath: Path, relative_path: Path) -> None:
        """Process a class definition."""
        class_id = f"CLASS:{relative_path}:{node.name}"
        
        self.nodes[class_id] = Node(
            id=class_id,
            type="CLASS",
            name=node.name,
            filepath=str(filepath),
            lineno=node.lineno
        )
        
        # FILE contains CLASS
        self.edges.append(Edge(file_id, class_id, "FILE_CONTAINS_CLASS"))
        
        # Handle inheritance
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_class_id = f"CLASS:{relative_path}:{base.id}"
                self.edges.append(Edge(class_id, base_class_id, "CLASS_INHERITS_FROM"))
        
        # Extract methods within class
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self._process_function(item, file_id, filepath, relative_path, class_id)
    
    def _process_function(self, node: ast.FunctionDef, file_id: str, filepath: Path, 
                         relative_path: Path, parent_class_id: str = None) -> None:
        """Process a function definition."""
        if parent_class_id:
            func_id = f"{parent_class_id}:{node.name}"
            self.edges.append(Edge(parent_class_id, func_id, "CLASS_CONTAINS_METHOD"))
        else:
            func_id = f"FUNCTION:{relative_path}:{node.name}"
            self.edges.append(Edge(file_id, func_id, "FILE_CONTAINS_FUNCTION"))
        
        self.nodes[func_id] = Node(
            id=func_id,
            type="FUNCTION",
            name=node.name,
            filepath=str(filepath),
            lineno=node.lineno
        )
        
        # Extract function calls
        self._extract_function_calls(node, func_id, filepath, relative_path)
    
    def _extract_function_calls(self, tree: ast.AST, func_id: str, filepath: Path, relative_path: Path) -> None:
        """Extract function calls within a function."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_func_id = f"FUNCTION:{relative_path}:{node.func.id}"
                    if called_func_id in self.nodes:
                        self.edges.append(Edge(func_id, called_func_id, "FUNCTION_CALLS_FUNCTION"))
    
    def _build_output(self) -> Dict:
        """Build the output graph structure."""
        return {
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges],
            "summary": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "node_types": self._count_node_types(),
                "edge_types": self._count_edge_types()
            }
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type."""
        counts = {}
        for node in self.nodes.values():
            counts[node.type] = counts.get(node.type, 0) + 1
        return counts
    
    def _count_edge_types(self) -> Dict[str, int]:
        """Count edges by type."""
        counts = {}
        for edge in self.edges:
            counts[edge.rel_type] = counts.get(edge.rel_type, 0) + 1
        return counts


def extract_repository_graph(repo_path: str, output_file: str = "graph.json") -> str:
    """
    Extract the repository graph and save to JSON file.
    
    Args:
        repo_path (str): Path to the repository root
        output_file (str): Output JSON file path
    
    Returns:
        str: Path to the generated JSON file
    
    Raises:
        FileNotFoundError: If repo_path doesn't exist
        RuntimeError: If extraction fails
    """
    repo_path = Path(repo_path)
    
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path not found: {repo_path}")
    
    try:
        extractor = RepositoryGraphExtractor(str(repo_path))
        graph = extractor.extract()
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(graph, f, indent=2)
        
        logger.info(f"Graph saved to {output_path}")
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Graph extraction failed: {e}")
        raise RuntimeError(f"Failed to extract repository graph: {e}")


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    if len(sys.argv) < 2:
        print("Usage: python graph_extractor.py <repo_path> [output_file]")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "graph.json"
    
    try:
        result = extract_repository_graph(repo_path, output_file)
        print(f"✓ Graph extracted: {result}")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
