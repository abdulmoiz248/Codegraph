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
                logger.error(f"Validation error for {chunk.name}: {self._format_exception(e)}")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for {chunk.name}: {e}")
                logger.debug("Response was: %s", self._shorten_text(getattr(response, 'text', '')))
                raise
            except Exception as e:
                last_error = e
                if not self._is_rate_limit_error(e) or attempt >= attempts:
                    logger.error(f"Error enriching {chunk.name}: {self._format_exception(e)}")
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
    
    def enrich_batch(self, chunks: List[CodeChunk], batch_size: int = 20) -> List[EnrichedCodeChunk]:
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
        done = 0

        # Process in batches so we can send multiple chunks in a single LLM request
        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]

            # Build a single prompt containing all chunks in this batch
            batch_prompt = self._build_batch_prompt(batch)

            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=batch_prompt,
                )

                text = response.text or ""
                json_arr_str = self._extract_json_array(text)
                descriptions = json.loads(json_arr_str)

                if not isinstance(descriptions, list):
                    raise ValueError("Batch response did not contain a JSON array")

                # Match descriptions to chunks by order; validate each
                batch_failed = False
                for i, desc_dict in enumerate(descriptions):
                    try:
                        description = validate_code_element_description(desc_dict)
                    except ValidationError as e:
                        # don't surface full validation errors to console by default
                        logger.debug(
                            "Validation error in batch for chunk index %s: %s",
                            i,
                            self._format_exception(e),
                        )
                        batch_failed = True
                        break
                    except Exception as e:
                        logger.debug(
                            "Unexpected validation error for chunk index %s: %s",
                            i,
                            self._format_exception(e),
                        )
                        batch_failed = True
                        break

                    # If response omitted names/order, try to guard against index errors
                    if i < len(batch):
                        chunk = batch[i]
                    else:
                        # Extra descriptions: skip
                        logger.warning("Received more descriptions than chunks; skipping extras")
                        break

                    enriched.append(
                        EnrichedCodeChunk(
                            id=chunk.id,
                            source_code=chunk.source_code,
                            filepath=chunk.filepath,
                            lineno=chunk.lineno,
                            description=description,
                        )
                    )
                    done += 1

                if batch_failed:
                    # Do not re-raise; perform fallback per-chunk enrichment for this batch
                    logger.warning("Batch validation failed; falling back to per-chunk enrichment for this batch")
                    for chunk in batch:
                        try:
                            enriched_chunk = self.enrich_chunk(chunk)
                            enriched.append(enriched_chunk)
                            done += 1
                        except Exception as e2:
                            logger.warning(f"Skipping chunk {chunk.name} during fallback: {self._format_exception(e2)}")
                    # proceed to next batch
                    logger.info(
                        "Enrichment progress: %s/%s (%s%%) — %s remaining",
                        done,
                        total,
                        int(done / total * 100) if total else 0,
                        max(total - done, 0),
                    )
                    if self.request_delay and (start + batch_size) < total:
                        logger.debug(f"Sleeping {self.request_delay}s before next batch")
                        time.sleep(self.request_delay)
                    continue

            except Exception as e:
                # On any failure parsing the batch, fall back to safe per-chunk enrichment
                logger.warning(f"Batch enrichment failed ({self._format_exception(e)}); falling back to per-chunk for this batch")
                for chunk in batch:
                    try:
                        enriched_chunk = self.enrich_chunk(chunk)
                        enriched.append(enriched_chunk)
                        done += 1
                    except Exception as e2:
                        logger.warning(f"Skipping chunk {chunk.name} during fallback: {self._format_exception(e2)}")
                        continue

            logger.info(
                "Enrichment progress: %s/%s (%s%%) — %s remaining",
                done,
                total,
                int(done / total * 100) if total else 0,
                max(total - done, 0),
            )

            # Sleep between batches to respect rate limits
            if self.request_delay and (start + batch_size) < total:
                logger.debug(f"Sleeping {self.request_delay}s before next batch")
                time.sleep(self.request_delay)

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

    def _build_batch_prompt(self, chunks: List[CodeChunk]) -> str:
        """Build a single prompt that contains multiple chunks.

        The model is instructed to return a JSON array of descriptions in the same
        order as the chunks provided.
        """
        parts = [
            "Analyze the following code chunks and return a JSON array of semantic descriptions",
            "Return ONLY valid JSON (an array) with one description object per chunk, in the same order.",
            "Do NOT include any extra text or markdown."
        ]

        for i, chunk in enumerate(chunks):
            docstring_info = f"Docstring: {chunk.docstring}\n" if chunk.docstring else ""
            chunk_block = (
                f"CHUNK_INDEX: {i}\n"
                f"TYPE: {chunk.type}\n"
                f"NAME: {chunk.name}\n"
                f"FILE: {chunk.filepath}\n"
                f"DOCSTRING/COMMENTS:\n{docstring_info}\n"
                f"SOURCE CODE:\n```python\n{chunk.source_code}\n```\n"
                f"DIRECT DEPENDENCIES: {', '.join(chunk.dependencies) if chunk.dependencies else 'None'}\n"
            )
            parts.append(chunk_block)

        parts.append(
            "Respond with a single JSON array like: [{...}, {...}] where each object matches the expected schema."
        )

        return "\n\n".join(parts)

    def _extract_json_array(self, text: str) -> str:
        """Extract a JSON array or multiple JSON objects from response text.

        Returns a JSON array string (e.g. '[{...}, {...}]').
        """
        # Remove fenced code blocks first
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            candidate = text[start:end].strip()
            if candidate.startswith("["):
                return candidate
            # If it's newline separated objects, fall through to object collection
            text = candidate
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            candidate = text[start:end].strip()
            text = candidate

        text = text.strip()

        # If it already looks like an array, try to extract the matching brackets
        if text.startswith("["):
            brace_count = 0
            start = text.find("[")
            for i in range(start, len(text)):
                if text[i] == "[":
                    brace_count += 1
                elif text[i] == "]":
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start:i+1]

        # Otherwise, collect multiple JSON objects and wrap them as an array
        objs = []
        idx = 0
        while True:
            start = text.find("{", idx)
            if start == -1:
                break
            brace_count = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        objs.append(text[start:i+1])
                        idx = i + 1
                        break
            else:
                break

        if not objs:
            raise ValueError(f"Could not extract JSON array from response: {text[:200]}")

        return "[" + ",".join(objs) + "]"

    def _is_rate_limit_error(self, error: Exception) -> bool:
        text = str(error).lower()
        return "429" in text or "resource_exhausted" in text or "rate limit" in text

    def _format_exception(self, error: Exception) -> str:
        """Return a concise, pretty-printed representation of an exception.

        If the exception carries a dict-like payload as the first arg (common in
        API client errors), pretty-print that JSON for readability. Otherwise,
        return the exception string.
        """
        # Prefer extracting structured payloads when present
        try:
            if error.args:
                first = error.args[0]

                # If it's already a dict/list, try to extract common fields
                if isinstance(first, dict):
                    # Google GenAI style error: contains 'error' dict
                    if "error" in first and isinstance(first["error"], dict):
                        err = first["error"]
                        code = err.get("code")
                        status = err.get("status")
                        message = err.get("message")

                        # Try to extract retry info or quota violations
                        retry = None
                        quota = None
                        for d in err.get("details", []) if isinstance(err.get("details"), list) else []:
                            t = d.get("@type", "") if isinstance(d, dict) else ""
                            if "RetryInfo" in t or d.get("@type", "").endswith("RetryInfo"):
                                retry = d.get("retryDelay") or d.get("retry_delay")
                            if "QuotaFailure" in t or d.get("@type", "").endswith("QuotaFailure"):
                                quota = d.get("violations")

                        parts = []
                        if status:
                            parts.append(f"status={status}")
                        if code:
                            parts.append(f"code={code}")
                        if message:
                            # keep message single-line
                            parts.append(f"message={message.splitlines()[0]}")
                        if quota:
                            parts.append("quota_violation=true")
                        if retry:
                            parts.append(f"retry={retry}")

                        short = ", ".join(parts) if parts else json.dumps(first)
                        return short[:1000]

                    # Otherwise pretty-print but truncate
                    pretty = json.dumps(first, indent=2)
                    return (pretty[:1000] + "...") if len(pretty) > 1000 else pretty

                # sometimes the arg is a string that contains JSON-like content
                if isinstance(first, str):
                    s = " ".join(first.splitlines())

                    # Try to extract key fields using regex to handle Python dict repr
                    code_match = re.search(r"'code'\s*:\s*(\d+)|\"code\"\s*:\s*(\d+)|\\bcode:\s*(\d+)", s)
                    status_match = re.search(r"'status'\s*:\s*'([^']+)'|\"status\"\s*:\s*\"([^\"]+)\"", s)
                    message_match = re.search(r"'message'\s*:\s*'([^']+)'|\"message\"\s*:\s*\"([^\"]+)\"", s)
                    retry_match = re.search(r"'retryDelay'\s*:\s*'([^']+)'|\"retryDelay\"\s*:\s*\"([^\"]+)\"|Please retry in\s+([\d\.]+)s", s)
                    quota_ind = 'quota' if ('QuotaFailure' in s or 'quota' in s.lower() or 'quota_exceeded' in s.lower()) else None

                    parts = []
                    if status_match:
                        status_val = next(g for g in status_match.groups() if g)
                        parts.append(f"status={status_val}")
                    if code_match:
                        code_val = next(g for g in code_match.groups() if g)
                        parts.append(f"code={code_val}")
                    if message_match:
                        msg_val = next(g for g in message_match.groups() if g)
                        parts.append(f"message={msg_val.splitlines()[0]}")
                    if quota_ind:
                        parts.append("quota_violation=true")
                    if retry_match:
                        retry_val = next(g for g in retry_match.groups() if g)
                        parts.append(f"retry={retry_val}")

                    if parts:
                        short = ", ".join(parts)
                        return (short[:300] + "...") if len(short) > 300 else short

                    # last resort: truncate long string
                    return (s[:300] + "...") if len(s) > 300 else s
        except Exception:
            pass
        # As a last resort, return the exception string
        s = str(error)
        return (s[:1000] + "...") if len(s) > 1000 else s

    def _extract_retry_delay_seconds(self, text: str) -> float:
        match = re.search(r"retry(?:\s+in)?\s+(\d+(?:\.\d+)?)s", text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return self.request_delay
        return self.request_delay

    def _shorten_text(self, text: str, limit: int = 300) -> str:
        if not text:
            return ""
        s = " ".join(str(text).splitlines())
        return (s[:limit] + "...") if len(s) > limit else s
