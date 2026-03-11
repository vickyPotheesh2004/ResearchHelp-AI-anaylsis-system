"""
Shared LLM Client module for ResearchHelp-AI-anaylsis-system AI Document Q&A System.
This module provides a singleton client for OpenAI/OpenRouter API access.
"""
import os
import logging
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

# Import logging utility
from src.logging_utils import get_logger

load_dotenv()

# Get logger - this ensures logging is configured
logger = get_logger(__name__)


class LLMClient:
    """
    Singleton class for shared LLM client access.
    Ensures consistent API configuration across the application.
    """
    _instance: Optional['LLMClient'] = None
    _client: Optional[OpenAI] = None
    _api_key: Optional[str] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the client with API configuration."""
        self._api_key = os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            logger.warning("OPENROUTER_API_KEY not found in environment")
            return
            
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self._api_key,
        )
        logger.info("LLMClient initialized successfully")
    
    @property
    def client(self) -> Optional[OpenAI]:
        """Get the OpenAI client instance."""
        if self._client is None:
            self._initialize()
        return self._client
    
    @property
    def api_key(self) -> Optional[str]:
        """Get the API key."""
        if self._api_key is None:
            self._api_key = os.getenv("OPENROUTER_API_KEY")
        return self._api_key
    
    def is_available(self) -> bool:
        """Check if the client is available."""
        return self._client is not None and self._api_key is not None
    
    def reset(self):
        """Reset the singleton instance (useful for testing)."""
        self._instance = None
        self._client = None
        self._api_key = None


def get_llm_client() -> LLMClient:
    """Get the singleton LLM client instance."""
    return LLMClient()
