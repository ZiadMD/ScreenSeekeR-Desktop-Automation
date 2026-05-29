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
    LLM_PROVIDER: Literal["gemini", "openai", "groq", "ollama"] = "gemini"
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    OLLAMA_API_URL: str = "http://localhost:11434"

    # Default planner and grounder models
    PLANNER_MODEL: str = "gemini-2.0-flash"
    GROUNDER_MODEL: str = "gemini-2.0-flash"

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

    @property
    def desktop_output_dir(self) -> Path:
        return Path.home() / "Desktop" 

    # Logging
    LOG_LEVEL: str = "INFO"

# Instantiate settings
settings = Settings()

# Make sure essential directories exist
settings.SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
settings.desktop_output_dir.mkdir(parents=True, exist_ok=True)
