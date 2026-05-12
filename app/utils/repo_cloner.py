import subprocess
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def clone_github_repo(repo_url: str, destination: str = None) -> str:
    """
    Clone a GitHub repository to a local directory.
    
    Args:
        repo_url (str): URL of the GitHub repository to clone
        destination (str, optional): Local path where repo will be cloned.
                                    Defaults to current directory.
    
    Returns:
        str: Path to the cloned repository
    
    Raises:
        ValueError: If repo_url is empty or invalid
        subprocess.CalledProcessError: If git clone fails
        FileNotFoundError: If destination directory doesn't exist
    """
    
    # Validate URL
    if not repo_url or not isinstance(repo_url, str):
        raise ValueError("repo_url must be a non-empty string")
    
    if not repo_url.strip().endswith(".git"):
        if not repo_url.strip().endswith("/"):
            repo_url = repo_url.strip() + ".git"
        else:
            repo_url = repo_url.strip()[:-1] + ".git"
    
    # Set destination
    if destination is None:
        destination = os.getcwd()
    
    destination = Path(destination)
    
    # Validate destination exists
    if not destination.exists():
        raise FileNotFoundError(f"Destination directory does not exist: {destination}")
    
    # Extract repo name from URL
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    clone_path = destination / repo_name
    
    try:
        logger.info(f"Cloning repo: {repo_url} into {clone_path}")
        
        result = subprocess.run(
            ["git", "clone", repo_url, str(clone_path)],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            raise subprocess.CalledProcessError(result.returncode, result.args, error_msg)
        
        logger.info(f"Successfully cloned repo to {clone_path}")
        return str(clone_path)
    
    except subprocess.TimeoutExpired:
        logger.error(f"Clone timeout for {repo_url}")
        raise TimeoutError(f"Repository clone timed out: {repo_url}")
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Git clone failed: {e.stderr}")
        raise RuntimeError(f"Failed to clone repository: {e.stderr}")
    
    except Exception as e:
        logger.error(f"Unexpected error during clone: {str(e)}")
        raise
