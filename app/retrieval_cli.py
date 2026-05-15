import json
import logging
from pathlib import Path
from google import genai
from config.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

def get_llm_client():
    api_key = settings.gemini_api_key
    if not api_key:
        raise ValueError("Gemini API key is required. Please set GEMINI_API_KEY.")
    return genai.Client(api_key=api_key)

import time

def call_llm(prompt: str) -> str:
    client = get_llm_client()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries - 1:
                    sleep_time = getattr(settings, "gemini_request_delay", 35)
                    print(f"Rate limit hit. Sleeping for {sleep_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(sleep_time)
                    continue
            return f"Error calling LLM: {e}"

def get_output_files(repo_name: str):
    output_dir = Path(settings.output_dir)
    enriched_file = output_dir / f"{repo_name}_enriched.json"
    graph_leiden_file = output_dir / f"{repo_name}_graph_leiden.json"
    return enriched_file, graph_leiden_file

def local_search(query: str, repo_name: str):
    enriched_file, graph_leiden_file = get_output_files(repo_name)
    
    if not enriched_file.exists():
        print(f"Enriched file not found: {enriched_file}. Run ingestion with enrichment first.")
        return
    if not graph_leiden_file.exists():
        print(f"Graph file not found: {graph_leiden_file}. Run ingestion first.")
        return

    print("Loading local graph data...")
    with open(enriched_file, "r") as f:
        enriched_data = json.load(f)
    with open(graph_leiden_file, "r") as f:
        graph_data = json.load(f)

    # 1. Vector Search equivalent: Keyword search on node name/summary
    query_lower = query.lower()
    best_node = None
    
    # Clean query into words for matching
    words = set(query_lower.replace("?", "").replace(".", "").replace(",", "").split())
    
    for node in enriched_data.get("nodes", []):
        name = node.get("name", "").lower()
        if name and (name == query_lower or name in words):
            best_node = node
            break
            
    # If no exact name match, fallback to checking if any word in query matches node name partially
    if not best_node:
        for node in enriched_data.get("nodes", []):
            name = node.get("name", "").lower()
            if name and any(name in word for word in words if len(word) > 3):
                best_node = node
                break

    # Final fallback: check summary or id for the full query string
    if not best_node:
        for node in enriched_data.get("nodes", []):
            if query_lower in node.get("id", "").lower() or query_lower in node.get("summary", "").lower():
                best_node = node
                break

    if not best_node:
        print("Could not identify a specific target node for your query locally.")
        return

    node_id = best_node.get("id")
    print(f"Target node identified: {node_id}")

    # 3. Fetch neighbors
    edges = graph_data.get("edges", [])
    neighbors_ids = set()
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source == node_id:
            neighbors_ids.add(target)
        elif target == node_id:
            neighbors_ids.add(source)
            
    # Fetch neighbor info from enriched data
    neighbor_details = []
    for n in enriched_data.get("nodes", []):
        if n.get("id") in neighbors_ids:
            # exclude raw source code to save context window if large, but keep descriptions
            summary_info = {
                "id": n.get("id"),
                "name": n.get("name"),
                "type": n.get("type"),
                "summary": n.get("description", {}).get("summary", n.get("summary")),
            }
            neighbor_details.append(summary_info)

    # 4. Send all to LLM
    prompt = f"""
    You are an expert developer assistant. 
    A user asked: "{query}"
    
    Here is the primary target node from the codebase:
    {json.dumps(best_node, indent=2)}
    
    Here are the immediate neighbors (dependencies/dependents) of this node:
    {json.dumps(neighbor_details, indent=2)}
    
    Based ONLY on this provided information, answer the user's question. Be concise and accurate.
    """
    print("\n--- Generating Answer ---")
    answer = call_llm(prompt)
    print("\n[Local Search Answer]")
    print(answer)
    return answer


def global_search(query: str, repo_name: str):
    enriched_file, graph_leiden_file = get_output_files(repo_name)
    
    if not graph_leiden_file.exists():
        print(f"Graph file not found: {graph_leiden_file}")
        return

    print("Loading local graph data...")
    with open(graph_leiden_file, "r") as f:
        graph_data = json.load(f)
        
    enriched_nodes = {}
    if enriched_file.exists():
        with open(enriched_file, "r") as f:
            enriched_data = json.load(f)
        enriched_nodes = {n["id"]: n for n in enriched_data.get("nodes", [])}

    communities = graph_data.get("communities", [])
    if not communities:
        print("No communities found in the graph.")
        return
        
    # Sort by size and get top 3
    communities.sort(key=lambda c: c.get("size", 0), reverse=True)
    top_3 = communities[:3]
    
    # Map Phase
    community_summaries = []
    for i, comm in enumerate(top_3):
        print(f"Processing community {i+1}/{len(top_3)}...")
        members = comm.get("members", [])
        
        # Gather nodes
        member_nodes = []
        for m in members:
            if m in enriched_nodes:
                node = enriched_nodes[m]
                # simplify to save context
                member_nodes.append({
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "summary": node.get("description", {}).get("summary", node.get("summary"))
                })
            else:
                member_nodes.append({"id": m})
        
        prompt = f"""
        Analyze the following community of code from a repository.
        Summarize its overall responsibility, architecture, and purpose within the broader system.
        Keep the query in mind to highlight relevant parts: "{query}"
        
        Community Nodes:
        {json.dumps(member_nodes, indent=2)}
        """
        summary = call_llm(prompt)
        community_summaries.append(summary)

    # Reduce Phase
    print("\nReducing summaries for final answer...")
    reduce_prompt = f"""
    You are an expert system architect answering a user's high-level question about a codebase.
    Question: "{query}"
    
    Here are summaries of the top relevant code communities in the repository:
    """
    for i, summary in enumerate(community_summaries):
        reduce_prompt += f"\n\n--- Community {i+1} ---\n{summary}"
        
    reduce_prompt += "\n\nProvide a final, comprehensive answer to the user's question based on these community summaries. Describe the overall architecture if asked."
    
    final_answer = call_llm(reduce_prompt)
    print("\n[Global Search Answer]")
    print(final_answer)
    return final_answer


def run_retrieval_cli():
    print("\n--- Retrieval CLI ---")
    repo_name = input("Enter the repository name (e.g., ProtoML): ").strip()
    if not repo_name:
        print("Repository name is required.")
        return
        
    while True:
        print("\nSearch Options:")
        print("1. Local Search (Specific function, class, or component)")
        print("2. Global Search (High-level architecture, repository-wide context)")
        print("3. Back to Main Menu")
        choice = input("Select an option (1-3): ").strip()
        
        if choice == "3":
            break
            
        query = input("\nEnter your question: ").strip()
        if not query:
            print("Question cannot be empty.")
            continue
            
        if choice == "1":
            local_search(query, repo_name)
        elif choice == "2":
            global_search(query, repo_name)
        else:
            print("Invalid option.")
