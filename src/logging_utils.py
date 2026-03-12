"""
Logging configuration utilities for ResearchHelp-AI-anaylsis-system.
Provides centralized logging setup for the application.
"""

import logging
import sys

# Module-level flag to ensure logging is configured only once
_logging_configured = False


def setup_logging(level: str = "ERROR") -> None:
    """
    Configure logging for the application.
    Should be called once at application startup.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global _logging_configured

    if _logging_configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Silence excessively noisy 3rd-party loggers
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("streamlit").setLevel(logging.ERROR)

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    Ensures logging is configured before returning.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    if not _logging_configured:
        setup_logging()

    return logging.getLogger(name)
