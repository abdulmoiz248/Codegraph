# Codegraph

Codegraph is a Python-based **GraphRAG ingestion and retrieval tool** for source repositories.  
It clones a target GitHub repo, builds a structural code graph, optionally enriches nodes with Gemini-generated semantic metadata, detects code communities with Leiden, and lets you query the result through a CLI.

## Core Idea

The project turns code into a graph so repository understanding and Q&A can be driven by relationships between files, classes, and functions instead of raw text only.

## How It Works

1. **Repository cloning**  
   Clones a GitHub repository into a local workspace (`repos/` by default).

2. **Graph extraction (AST-based)**  
   Parses Python files and builds nodes/edges for:
   - files
   - classes
   - functions/methods
   - containment, inheritance, and function-call relationships

3. **Community detection (Leiden)**  
   Adds graph communities to expose architectural clusters (optional dependencies: `python-igraph`, `leidenalg`).

4. **LLM enrichment (optional)**  
   Chunks code by semantic boundaries, asks Gemini for summaries/purpose/tags/relationships, then deduplicates entities.

5. **Retrieval CLI**  
   - **Local search**: node-centric lookup + neighbor context
   - **Global search**: top-community map-reduce style synthesis for architecture-level questions

## Project Structure

```text
app/
  app.py                  # Main interactive menu (ingest + retrieval)
  retrieval_cli.py        # Local/global search over produced graph data
  config/config.py        # Environment-driven settings
  utils/
    repo_cloner.py        # Git clone logic
    graph_extractor.py    # AST graph generation
    community_detector.py # Leiden community detection
    code_chunker.py       # Function/class chunking
    llm_enricher.py       # Gemini enrichment
    enrichment_pipeline.py# End-to-end enrichment + deduplication
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment (`.env` in repo root or shell env vars):

- `GEMINI_API_KEY` (required for enrichment and retrieval LLM answers)
- `GEMINI_MODEL` (default: `gemini-2.5-flash`)
- `GEMINI_REQUEST_DELAY` (default: `30`)
- `REPOS_DIR` (default: `<repo>/repos`)
- `OUTPUT_DIR` (default: `<repo>/output`)

## Run

From the `app` directory:

```bash
cd app
python app.py
```

Menu options:
- **Ingest Repository**: clone + graph extraction (+ optional enrichment)
- **Retrieval CLI**: ask local or global questions against generated artifacts

## Output Artifacts

Generated under `output/` (by default), e.g. for `<repo_name>`:
- `<repo_name>_graph.json`
- `<repo_name>_graph_leiden.json`
- `<repo_name>_enriched.json` (when enrichment is enabled)

