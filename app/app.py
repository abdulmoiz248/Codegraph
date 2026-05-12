from pathlib import Path

from config.config import get_settings
from utils.enrichment_pipeline import run_enrichment_pipeline
from utils.graph_extractor import extract_repository_graph
from utils.repo_cloner import clone_github_repo

SETTINGS = get_settings()
REPOS_DIR = Path(SETTINGS.repos_dir)
OUTPUT_DIR = Path(SETTINGS.output_dir)


def main(repo_url: str = None, enrich: bool = None):
    """
    Clone a repository and extract its graph structure with optional LLM enrichment.
    
    Args:
        repo_url (str): GitHub repository URL to clone
        enrich (bool): Whether to run LLM enrichment pipeline (Phase 2)
    """
    # Create directories if they don't exist
    repo_url = repo_url or SETTINGS.default_repo_url
    enrich = SETTINGS.enable_enrichment if enrich is None else enrich

    if enrich and not SETTINGS.gemini_api_key:
        print("! Gemini API key not found in env/.env; skipping enrichment phase.")
        enrich = False

    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract repo name from URL for output file
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    graph_file = OUTPUT_DIR / f"{repo_name}_graph.json"
    enriched_file = OUTPUT_DIR / f"{repo_name}_enriched.json"

    try:
        # Step 1: Clone the GitHub repository
        print("\n=== Phase 1: Repository Cloning ===")
        cloned_repo_path = clone_github_repo(repo_url, str(REPOS_DIR))
        print(f"✓ Repository cloned to: {cloned_repo_path}")

        # Step 2: Extract the repository graph and save to JSON
        print("\n=== Phase 1: Graph Extraction ===")
        graph_json_path = extract_repository_graph(cloned_repo_path, str(graph_file))
        print(f"✓ Graph extracted and saved to: {graph_json_path}")
        
        # Step 3 (Optional): Run LLM enrichment pipeline
        if enrich:
            print("\n=== Phase 2: LLM Enrichment ===")
            enriched_path = run_enrichment_pipeline(cloned_repo_path, str(enriched_file))
            print(f"✓ Enriched graph saved to: {enriched_path}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    repo_url = sys.argv[1] if len(sys.argv) > 1 else None
    enrich = sys.argv[2].lower() != "false" if len(sys.argv) > 2 else None
    
    main(repo_url, enrich)