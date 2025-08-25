"""Centralized logging configuration for the Vector Database REST API.

Provides consistent logging across all modules with configurable levels,
formats, and handlers.
"""

import logging
import logging.handlers
import sys
from pathlib import Path

from app.core.config import get_settings


def setup_logging(
    log_level: str | None = None,
    log_file: str | None = None,
    enable_console: bool = True,
    enable_file: bool = False,
) -> None:
    """Setup global logging configuration.

    Args:
        log_level: Logging level (debug, info, warning, error, critical)
        log_file: Path to log file (if file logging is enabled)
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
    """
    settings = get_settings()

    # Use provided log_level or fall back to settings
    level = log_level or settings.log_level
    log_level_num = getattr(logging, level.upper(), logging.INFO)

    # Use provided log_file or fall back to settings
    file_path = log_file or settings.log_file
    enable_file_logging = enable_file or settings.enable_file_logging

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_num)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Create formatter using settings
    formatter = logging.Formatter(fmt=settings.log_format, datefmt=settings.log_date_format)

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level_num)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if enable_file_logging and file_path:
        # Ensure log directory exists
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler using settings
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=settings.log_max_size_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level_num)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set specific logger levels for external libraries
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("cohere").setLevel(logging.WARNING)

    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {level.upper()}, Console: {enable_console}, File: {enable_file_logging}")
    if enable_file_logging and file_path:
        logger.info(f"Log file: {file_path}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_startup_info() -> None:
    """Log startup information about the application."""
    logger = logging.getLogger(__name__)
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("Vector Database REST API Starting Up")
    logger.info("=" * 60)
    logger.info(f"Version: {getattr(settings, '__version__', '1.0.0')}")
    logger.info(f"Host: {settings.host}")
    logger.info(f"Port: {settings.port}")
    logger.info(f"Log Level: {settings.log_level.upper()}")
    logger.info(f"Log File: {settings.log_file or 'Not configured'}")
    logger.info(f"File Logging: {'Enabled' if settings.enable_file_logging else 'Disabled'}")
    logger.info(f"Cohere API Enabled: {settings.cohere_api_key is not None}")
    logger.info(f"Persistence Path: {settings.persistence_path}")
    logger.info(f"Default Index Type: {settings.default_index_type}")
    logger.info("=" * 60)


# Default logging setup
def setup_default_logging() -> None:
    """Setup default logging configuration based on settings."""
    settings = get_settings()

    # Setup logging using settings
    setup_logging(
        log_level=settings.log_level,
        log_file=settings.log_file,
        enable_console=True,
        enable_file=settings.enable_file_logging,
    )


# Initialize logging when module is imported
setup_default_logging()
