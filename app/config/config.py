import os
import importlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

try:
	_pydantic = importlib.import_module("pydantic")
	BaseModel = _pydantic.BaseModel
	Field = _pydantic.Field
except Exception:  # pragma: no cover - keep working even if pydantic is unavailable
	class BaseModel:
		def __init__(self, **data: Any):
			for key, value in data.items():
				setattr(self, key, value)

		def model_dump(self) -> Dict[str, Any]:
			return dict(self.__dict__)

	def Field(default=None, **_kwargs):
		return default


ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"


def _parse_env_file(env_file: Path) -> Dict[str, str]:
	"""Parse a simple .env file and return key/value pairs."""
	if not env_file.exists():
		return {}

	data: Dict[str, str] = {}
	for raw_line in env_file.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue

		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip().strip('"').strip("'")
		if key:
			data[key] = value
	return data


def _load_env() -> Dict[str, str]:
	"""Load configuration from .env first, then environment variables."""
	data = _parse_env_file(ENV_FILE)

	# Copy into process environment so downstream code can keep using os.getenv.
	for key, value in data.items():
		os.environ.setdefault(key, value)

	return {
		"gemini_api_key": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY") or "",
		"gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
		"gemini_request_delay": float(os.getenv("GEMINI_REQUEST_DELAY", "30")),
		"repos_dir": os.getenv("REPOS_DIR", str(ROOT_DIR / "repos")),
		"output_dir": os.getenv("OUTPUT_DIR", str(ROOT_DIR / "output")),
		"default_repo_url": os.getenv(
			"DEFAULT_REPO_URL",
			"https://github.com/abdulmoiz248/ProtoML",
		),
		"enable_enrichment": os.getenv("ENABLE_ENRICHMENT", "true").strip().lower()
		not in {"0", "false", "no", "off"},
	}


class AppSettings(BaseModel):
	"""Typed application settings loaded from environment variables."""

	gemini_api_key: str = Field(default="", description="Gemini API key")
	gemini_model: str = Field(default="gemini-2.5-flash", description="Gemini model name")
	gemini_request_delay: float = Field(default=30.0, description="Delay between Gemini requests in seconds")
	repos_dir: Path = Field(default=ROOT_DIR / "repos", description="Local clone directory")
	output_dir: Path = Field(default=ROOT_DIR / "output", description="Output directory")
	default_repo_url: str = Field(
		default="https://github.com/abdulmoiz248/ProtoML",
		description="Default repository URL",
	)
	enable_enrichment: bool = Field(default=True, description="Run the LLM enrichment phase")


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
	"""Return cached app settings."""
	raw = _load_env()
	settings = AppSettings(**raw)

	# Normalize Path fields.
	settings.repos_dir = Path(settings.repos_dir)
	settings.output_dir = Path(settings.output_dir)
	return settings

