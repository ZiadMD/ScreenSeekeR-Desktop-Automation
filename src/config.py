import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Provider & Models
    LLM_PROVIDER: Literal["gemini", "openai", "groq", "ollama", "local"] = "gemini"
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    OLLAMA_API_URL: str = "http://localhost:11434"

    # Split provider overrides (None = use LLM_PROVIDER for both)
    # Enables hybrid mode: e.g. Gemini planner + local grounder
    PLANNER_PROVIDER: Optional[Literal["gemini", "openai", "groq", "ollama", "local"]] = None
    GROUNDER_PROVIDER: Optional[Literal["gemini", "openai", "groq", "ollama", "local"]] = None

    # Default planner and grounder models
    PLANNER_MODEL: str = "gemini-2.0-flash"
    GROUNDER_MODEL: str = "gemini-2.0-flash"

    # Local Model Settings (used when provider is "local")
    LOCAL_MODEL_PATH: Optional[str] = None       # Path to local model weights dir
    LOCAL_MODEL_TYPE: str = "gui-actor"           # Model family: "gui-actor" | future types
    LOCAL_DEVICE: str = "cuda:0"                  # Device for local inference
    LOCAL_TORCH_DTYPE: str = "float16"            # Precision: "float16" | "bfloat16" | "float32"
    LOCAL_ATTN_IMPL: str = "sdpa"                 # Attention: "sdpa" | "flash_attention_2"
    LOCAL_MAX_PIXELS: int = 3200 * 1800           # Max image pixels for preprocessing

    # Display settings (DPI Scaling)
    DPI_SCALING: float = 1.00

    # Grounding search settings
    CONFIRMATION_STEP: bool = True
    MAX_SEARCH_DEPTH: int = 3
    MIN_PATCH_SIZE: int = 256
    IoU_THRESHOLD: float = 0.3
    CONFIDENCE_THRESHOLD: float = 0.4

    # API Configuration
    JSONPLACEHOLDER_URL: str = "https://jsonplaceholder.typicode.com/posts"

    # Directories
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    SCREENSHOTS_DIR: Path = PROJECT_ROOT / "screenshots"
    MODELS_DIR: Path = PROJECT_ROOT / "models"

    @property
    def desktop_output_dir(self) -> Path:
        return Path.home() / "Desktop" 

    @property
    def resolved_local_model_path(self) -> Optional[Path]:
        """Resolve local model path — supports both absolute and relative (to MODELS_DIR) paths."""
        if not self.LOCAL_MODEL_PATH:
            return None
        p = Path(self.LOCAL_MODEL_PATH)
        if p.is_absolute():
            return p
        return self.MODELS_DIR / p

    # Logging
    LOG_LEVEL: str = "INFO"

# Instantiate settings
settings = Settings()

# Make sure essential directories exist
settings.SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
settings.desktop_output_dir.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
