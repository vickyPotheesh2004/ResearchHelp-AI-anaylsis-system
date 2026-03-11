"""
Centralized configuration module for ResearchHelp-AI-anaylsis-system AI Document Q&A System.
All configurable settings should be defined here and accessed via environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== PATHS ====================

# Tesseract OCR path - configurable via environment variable
# Default to common Windows installation path
TESSERACT_PATH = os.getenv(
    "TESSERACT_PATH", 
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ChromaDB persistence path
CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH", 
    "./chroma_db"
)

# ==================== LLM MODELS ====================

# Default LLM model for general Q&A
DEFAULT_LLM_MODEL = os.getenv(
    "LLM_MODEL", 
    "google/gemini-2.0-flash-001"
)

# Model for research engine (can be different for specialized tasks)
RESEARCH_LLM_MODEL = os.getenv(
    "RESEARCH_LLM_MODEL",
    "google/gemini-2.0-flash-001"
)

# Model for intent classification (fast, efficient model)
INTENT_CLASSIFIER_MODEL = os.getenv(
    "INTENT_CLASSIFIER_MODEL",
    "google/gemini-2.0-flash-001"
)

# ==================== LLM PARAMETERS ====================

# Topic titler / generator parameters
TOPIC_MAX_TOKENS = int(os.getenv("TOPIC_MAX_TOKENS", "10"))
TOPIC_TEMPERATURE = float(os.getenv("TOPIC_TEMPERATURE", "0.1"))
TOPIC_TEXT_CHUNK_SIZE = int(os.getenv("TOPIC_TEXT_CHUNK_SIZE", "500"))

# Research engine parameters
RESEARCH_OVERVIEW_MAX_TOKENS = int(os.getenv("RESEARCH_OVERVIEW_MAX_TOKENS", "2000"))
RESEARCH_OVERVIEW_TEMPERATURE = float(os.getenv("RESEARCH_OVERVIEW_TEMPERATURE", "0.3"))
RESEARCH_SUGGESTIONS_MAX_TOKENS = int(os.getenv("RESEARCH_SUGGESTIONS_MAX_TOKENS", "1000"))
RESEARCH_SUGGESTIONS_TEMPERATURE = float(os.getenv("RESEARCH_SUGGESTIONS_TEMPERATURE", "0.4"))
RESEARCH_ADDON_MAX_TOKENS = int(os.getenv("RESEARCH_ADDON_MAX_TOKENS", "3000"))
RESEARCH_ADDON_TEMPERATURE = float(os.getenv("RESEARCH_ADDON_TEMPERATURE", "0.3"))

# QA engine parameters
QA_MAX_TOKENS = int(os.getenv("QA_MAX_TOKENS", "3000"))
QA_TEMPERATURE = float(os.getenv("QA_TEMPERATURE", "0.25"))

# Intent classifier parameters
INTENT_MAX_TOKENS = int(os.getenv("INTENT_MAX_TOKENS", "10"))
INTENT_TEMPERATURE = float(os.getenv("INTENT_TEMPERATURE", "0.0"))
INTENT_TIMEOUT = float(os.getenv("INTENT_TIMEOUT", "25.0"))

# ==================== RETRIEVAL SETTINGS ====================

# Hybrid retrieval weights
SEMANTIC_SEARCH_WEIGHT = float(os.getenv("SEMANTIC_SEARCH_WEIGHT", "0.7"))
BM25_SEARCH_WEIGHT = float(os.getenv("BM25_SEARCH_WEIGHT", "0.3"))

# Number of results to retrieve
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "8"))

# Topic segmentation threshold
TOPIC_SIMILARITY_THRESHOLD = float(os.getenv("TOPIC_SIMILARITY_THRESHOLD", "0.78"))

# Topic segment overlap (number of sentences to overlap between segments)
TOPIC_SEGMENT_OVERLAP = int(os.getenv("TOPIC_SEGMENT_OVERLAP", "2"))

# Intent classifier cache size
INTENT_CACHE_MAX_SIZE = int(os.getenv("INTENT_CACHE_MAX_SIZE", "1000"))

# ==================== PERFORMANCE SETTINGS ====================

# Thread pool for concurrent operations
THREAD_POOL_MAX_WORKERS = int(os.getenv("THREAD_POOL_MAX_WORKERS", "2"))

# Retrieval and intent timeout (seconds)
RETRIEVAL_TIMEOUT = float(os.getenv("RETRIEVAL_TIMEOUT", "30"))
INTENT_TIMEOUT_SECONDS = float(os.getenv("INTENT_TIMEOUT_SECONDS", "30"))

# Chat history size to include in context
CHAT_HISTORY_CONTEXT_SIZE = int(os.getenv("CHAT_HISTORY_CONTEXT_SIZE", "6"))

# ==================== EMBEDDINGS ====================

# Sentence transformer model for embeddings
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-mpnet-base-v2"
)

# ==================== FILE LIMITS ====================

# File size limits (in bytes)
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(50 * 1024 * 1024)))  # 50MB default
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", str(10 * 1024 * 1024)))  # 10MB default

# ==================== API SETTINGS ====================

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# ==================== VALIDATION ====================

def validate_config():
    """Validate required configuration settings."""
    errors = []
    
    if not OPENROUTER_API_KEY:
        errors.append("OPENROUTER_API_KEY is not set in environment variables")
    elif "placeholder" in OPENROUTER_API_KEY.lower() or "your_" in OPENROUTER_API_KEY.lower():
        errors.append("OPENROUTER_API_KEY appears to be a placeholder - please set a valid API key")
    
    if errors:
        return False, errors
    return True, []

# ==================== LOGGING ====================

# Log level configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Enable/disable detailed error logging
VERBOSE_ERRORS = os.getenv("VERBOSE_ERRORS", "true").lower() == "true"
