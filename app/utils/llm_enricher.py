import json
import logging
import os
import re
import time
from typing import List, Optional

from google import genai

# Load configuration safely in both package and script execution modes
try:
    from app.config.config import get_settings  # when running as package
except Exception:
    try:
        from config.config import get_settings  # when config is top-level module
    except Exception:
        # Fallback: load config module by file path
        import importlib.util
        from pathlib import Path

        cfg_path = Path(__file__).resolve().parents[2] / "config" / "config.py"
        spec = importlib.util.spec_from_file_location("app.config.config", str(cfg_path))
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        get_settings = cfg.get_settings
from .code_chunker import CodeChunk
from .pydantic_models import (
    CodeElementDescription,
    EnrichedCodeChunk,
    ValidationError,
    validate_code_element_description,
)

logger = logging.getLogger(__name__)


class GeminiEnricher:
    """Use Google Gemini to enrich code with semantic descriptions."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, request_delay: Optional[float] = None):
        """
        Initialize Gemini enricher.
        
        Args:
            api_key (str, optional): Google API key. Defaults to GOOGLE_GEMINI_API_KEY env var
            model (str): Model to use (gemini-1.5-pro or gemini-1.5-flash)
        """
        settings = get_settings()

        # prefer explicit api_key, then config, then env
        self.api_key = api_key or settings.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Set GEMINI_API_KEY (preferred) or GOOGLE_GEMINI_API_KEY. "
                "Get one at: https://aistudio.google.com/app/apikeys"
            )

        # model preference: explicit arg -> config -> default to gemini-2.5-flash
        self.model_name = model or settings.gemini_model or "gemini-2.5-flash"

        # request delay between LLM calls (seconds)
        self.request_delay = float(request_delay if request_delay is not None else getattr(settings, "gemini_request_delay", 30.0))

        # Initialize client
        self.client = genai.Client(api_key=self.api_key)
        logger.info(f"Initialized Gemini enricher with model: {self.model_name}; delay={self.request_delay}s")
    
    def enrich_chunk(self, chunk: CodeChunk) -> EnrichedCodeChunk:
        """
        Enrich a single code chunk with semantic description.
        
        Args:
            chunk (CodeChunk): The code chunk to enrich
        
        Returns:
            EnrichedCodeChunk: Enriched chunk with LLM description
        """
        prompt = self._build_prompt(chunk)

        last_error: Optional[Exception] = None
        attempts = 2

        for attempt in range(1, attempts + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )

                # Parse JSON response
                json_str = self._extract_json(response.text or "")
                description_dict = json.loads(json_str)
                description = validate_code_element_description(description_dict)

                logger.info(f"Enriched: {chunk.name}")

                return EnrichedCodeChunk(
                    id=chunk.id,
                    source_code=chunk.source_code,
                    filepath=chunk.filepath,
                    lineno=chunk.lineno,
                    description=description,
                )

            except ValidationError as e:
                logger.error(f"Validation error for {chunk.name}: {e}")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for {chunk.name}: {e}")
                logger.debug(f"Response was: {getattr(response, 'text', '')}")
                raise
            except Exception as e:
                last_error = e
                if not self._is_rate_limit_error(e) or attempt >= attempts:
                    logger.error(f"Error enriching {chunk.name}: {e}")
                    raise

                retry_after = self._extract_retry_delay_seconds(str(e))
                sleep_for = max(self.request_delay, retry_after)
                logger.warning(
                    f"Rate limit for {chunk.name}; sleeping {sleep_for}s before retry {attempt + 1}/{attempts}"
                )
                time.sleep(sleep_for)

        if last_error:
            raise last_error
        raise RuntimeError(f"Failed to enrich chunk {chunk.name}")
    
    def enrich_batch(self, chunks: List[CodeChunk], batch_size: int = 5) -> List[EnrichedCodeChunk]:
        """
        Enrich multiple code chunks.
        
        Args:
            chunks (List[CodeChunk]): Chunks to enrich
            batch_size (int): Process in batches (rate limiting)
        
        Returns:
            List[EnrichedCodeChunk]: Enriched chunks
        """
        enriched = []
        total = len(chunks)
        
        for i, chunk in enumerate(chunks):
            try:
                enriched_chunk = self.enrich_chunk(chunk)
                enriched.append(enriched_chunk)

                if (i + 1) % batch_size == 0:
                    logger.info(f"Progress: {i + 1}/{total} chunks enriched")

                # polite delay between LLM requests to avoid rate limits
                if self.request_delay and (i + 1) < total:
                    logger.debug(f"Sleeping {self.request_delay}s before next LLM request")
                    time.sleep(self.request_delay)

            except Exception as e:
                logger.warning(f"Skipping chunk {chunk.name}: {e}")
                continue
        
        logger.info(f"Enrichment complete: {len(enriched)}/{total} chunks")
        return enriched
    
    def _build_prompt(self, chunk: CodeChunk) -> str:
        """Build the enrichment prompt for a code chunk."""
        docstring_info = f"Docstring: {chunk.docstring}\n" if chunk.docstring else ""
        
        return f"""Analyze this {chunk.type} and provide a semantic description as JSON.

{chunk.type.upper()}: {chunk.name}
FILE: {chunk.filepath}

DOCSTRING/COMMENTS:
{docstring_info}

SOURCE CODE:
```python
{chunk.source_code}
```

DIRECT DEPENDENCIES (imported/referenced):
{', '.join(chunk.dependencies) if chunk.dependencies else 'None'}

Respond with ONLY valid JSON (no markdown, no code blocks) matching this structure:
{{
  "name": "function/class name",
  "type": "function or class",
  "summary": "1-2 sentence description of what it does",
  "purpose": "core responsibility/intent",
  "inputs": "what it takes as input",
  "outputs": "what it returns/produces",
  "side_effects": "any mutations, API calls, DB writes, file I/O, etc. Leave empty if none",
  "hidden_relationships": [
    {{
      "target": "name of other function/class it implicitly depends on",
      "rel_type": "USES|DEPENDS_ON|SHARES_STATE|SHARES_CONFIG|EXTENDS",
      "confidence": 0.9,
      "reason": "why they're related (e.g., 'both access redis key USER_CACHE')"
    }}
  ],
  "tags": ["tag1", "tag2"],
  "complexity": "simple|medium|complex"
}}

Focus on:
1. What the code actually does (not just method names)
2. Implicit dependencies (shared files, config, global state)
3. Performance implications
4. Error handling and edge cases
"""
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from response text."""
        # Remove markdown code blocks if present
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        
        # Try to find JSON object
        start = text.find("{")
        if start != -1:
            # Find matching closing brace
            brace_count = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start:i+1]
        
        raise ValueError(f"Could not extract JSON from response: {text[:200]}")

    def _is_rate_limit_error(self, error: Exception) -> bool:
        text = str(error).lower()
        return "429" in text or "resource_exhausted" in text or "rate limit" in text

    def _extract_retry_delay_seconds(self, text: str) -> float:
        match = re.search(r"retry(?:\s+in)?\s+(\d+(?:\.\d+)?)s", text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return self.request_delay
        return self.request_delay
