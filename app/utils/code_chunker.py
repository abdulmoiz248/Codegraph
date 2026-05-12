import ast
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a semantic code chunk (function or class)."""
    id: str
    type: str  # "function" or "class"
    name: str
    filepath: str
    source_code: str
    lineno: int
    end_lineno: int
    docstring: str = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class CodeChunker:
    """Extract code chunks by function/class boundaries."""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.chunks: List[CodeChunk] = []
        self.source_lines = []
    
    def chunk(self) -> List[CodeChunk]:
        """Extract all code chunks from the file."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                self.source_lines = f.readlines()
                content = ''.join(self.source_lines)
            
            tree = ast.parse(content)
            self._extract_chunks(tree)
            
            logger.info(f"Extracted {len(self.chunks)} chunks from {self.filepath}")
            return self.chunks
        
        except SyntaxError as e:
            logger.warning(f"Syntax error in {self.filepath}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error chunking {self.filepath}: {e}")
            return []
    
    def _extract_chunks(self, tree: ast.AST, parent_class: str = None) -> None:
        """Extract chunks from AST nodes."""
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                self._process_function(node, parent_class)
            elif isinstance(node, ast.ClassDef):
                self._process_class(node)
    
    def _process_function(self, node: ast.FunctionDef, parent_class: str = None) -> None:
        """Process a function definition."""
        chunk_id = f"{parent_class}:{node.name}" if parent_class else f"{self.filepath.name}:{node.name}"
        
        source_code = self._get_source(node)
        docstring = ast.get_docstring(node)
        dependencies = self._extract_dependencies(node)
        
        chunk = CodeChunk(
            id=chunk_id,
            type="function",
            name=node.name,
            filepath=str(self.filepath),
            source_code=source_code,
            lineno=node.lineno,
            end_lineno=node.end_lineno,
            docstring=docstring,
            dependencies=dependencies
        )
        self.chunks.append(chunk)
    
    def _process_class(self, node: ast.ClassDef) -> None:
        """Process a class definition."""
        chunk_id = f"{self.filepath.name}:{node.name}"
        
        source_code = self._get_source(node)
        docstring = ast.get_docstring(node)
        dependencies = self._extract_dependencies(node)
        
        chunk = CodeChunk(
            id=chunk_id,
            type="class",
            name=node.name,
            filepath=str(self.filepath),
            source_code=source_code,
            lineno=node.lineno,
            end_lineno=node.end_lineno,
            docstring=docstring,
            dependencies=dependencies
        )
        self.chunks.append(chunk)
        
        # Extract methods from class
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self._process_function(item, node.name)
    
    def _get_source(self, node: ast.AST) -> str:
        """Get the source code for an AST node."""
        start = node.lineno - 1
        end = node.end_lineno
        return ''.join(self.source_lines[start:end])
    
    def _extract_dependencies(self, node: ast.AST) -> List[str]:
        """Extract function/variable names referenced in the node."""
        dependencies = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                dependencies.add(child.id)
            elif isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name):
                    dependencies.add(child.value.id)
        
        return list(dependencies)


def chunk_repository(repo_path: str) -> Dict[str, List[CodeChunk]]:
    """
    Chunk all Python files in a repository.
    
    Args:
        repo_path (str): Path to the repository
    
    Returns:
        Dict mapping filepath to list of CodeChunks
    """
    repo_path = Path(repo_path)
    all_chunks = {}
    
    for py_file in repo_path.rglob("*.py"):
        chunker = CodeChunker(str(py_file))
        chunks = chunker.chunk()
        
        if chunks:
            relative_path = str(py_file.relative_to(repo_path))
            all_chunks[relative_path] = chunks
    
    logger.info(f"Total chunks extracted: {sum(len(c) for c in all_chunks.values())}")
    return all_chunks
