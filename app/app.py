import os
from pathlib import Path
from utils.repo_cloner import clone_github_repo
from utils.graph_extractor import extract_repository_graph

# Default folders
REPOS_DIR = Path(__file__).parent.parent / "repos"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def main(repo_url: str = "https://github.com/abdulmoiz248/ProtoML"):
    """
    Clone a repository and extract its graph structure.
    
    Args:
        repo_url (str): GitHub repository URL to clone
    """
    # Create directories if they don't exist
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract repo name from URL for output file
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    output_file = OUTPUT_DIR / f"{repo_name}_graph.json"

    try:
        # Step 1: Clone the GitHub repository
        cloned_repo_path = clone_github_repo(repo_url, str(REPOS_DIR))
        print(f"✓ Repository cloned to: {cloned_repo_path}")

        # Step 2: Extract the repository graph and save to JSON
        graph_json_path = extract_repository_graph(cloned_repo_path, str(output_file))
        print(f"✓ Graph extracted and saved to: {graph_json_path}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()