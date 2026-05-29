import sys
from pathlib import Path
from loguru import logger
from src.config import settings

def setup_logging():
    # Clear any existing handlers
    logger.remove()

    # Path to logs folder
    log_dir = settings.PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "automation.log"

    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
        colorize=True
    )

    # Add file handler
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.LOG_LEVEL,
        rotation="10 MB",
        retention="5 days",
        encoding="utf-8"
    )

    logger.info("Logging initialized successfully!")

# Auto-setup logging when imported
setup_logging()
