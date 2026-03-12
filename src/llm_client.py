"""
Shared LLM Client module for ResearchHelp-AI-anaylsis-system AI Document Q&A System.
This module provides a singleton client for OpenAI/OpenRouter API access.
"""

import os
from typing import Optional, Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

# Import logging utility
from src.logging_utils import get_logger

# Import config
from src.config import (
    GLM_45_AIR_MODEL,
    TRINITY_LARGE_MODEL,
    NEMOTRON_3_SUPER_MODEL,
)

load_dotenv()

# Get logger - this ensures logging is configured
logger = get_logger(__name__)


class LLMClient:
    """
    Singleton class for shared LLM client access.
    Ensures consistent API configuration across the application.
    """

    _instance: Optional["LLMClient"] = None
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

    # ==================== Model Helper Properties ====================
    
    @property
    def glm_model(self) -> str:
        """Get GLM 4.5 Air model (fast, for simple tasks)."""
        return GLM_45_AIR_MODEL
    
    @property
    def trinity_model(self) -> str:
        """Get Trinity Large model (reasoning for Q&A)."""
        return TRINITY_LARGE_MODEL
    
    @property
    def nemotron_model(self) -> str:
        """Get Nemotron 3 Super model (best reasoning for complex tasks)."""
        return NEMOTRON_3_SUPER_MODEL

    # ==================== Chat Completion Helpers ====================
    
    def create_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.3,
        enable_reasoning: bool = False,
        **kwargs
    ) -> Any:
        """
        Create a chat completion with optional reasoning support.
        
        Args:
            model: Model identifier
            messages: Chat messages
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            enable_reasoning: Enable reasoning for supported models
            **kwargs: Additional parameters
            
        Returns:
            Chat completion response
        """
        extra_body = {}
        
        # Enable reasoning for models that support it
        if enable_reasoning and model in [TRINITY_LARGE_MODEL, NEMOTRON_3_SUPER_MODEL]:
            extra_body["reasoning"] = {"enabled": True}
        
        # Merge any additional extra_body params
        if "extra_body" in kwargs:
            extra_body.update(kwargs.pop("extra_body"))
        
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body if extra_body else None,
            **kwargs
        )

    def create_fast_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 50,
        temperature: float = 0.0
    ) -> Any:
        """
        Create a fast completion using GLM 4.5 Air (for simple tasks like intent classification).
        """
        return self.create_chat_completion(
            model=self.glm_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_reasoning=False  # No reasoning needed for simple tasks
        )

    def create_qa_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.25
    ) -> Any:
        """
        Create a Q&A completion using Trinity Large (with reasoning).
        """
        return self.create_chat_completion(
            model=self.trinity_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_reasoning=True  # Enable reasoning for Q&A
        )

    def create_research_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 0.3
    ) -> Any:
        """
        Create a research completion using Nemotron 3 Super (best reasoning).
        """
        return self.create_chat_completion(
            model=self.nemotron_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_reasoning=True  # Enable reasoning for complex tasks
        )

    def create_mermaid_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.2
    ) -> Any:
        """
        Create a Mermaid diagram completion using GLM 4.5 Air (fast for structured output).
        GLM is ideal for Mermaid as it's fast and good at structured formatting.
        """
        return self.create_chat_completion(
            model=self.glm_model,  # Use fast GLM for structured output
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_reasoning=False  # No reasoning needed for simple diagrams
        )


def get_llm_client() -> LLMClient:
    """Get the singleton LLM client instance."""
    return LLMClient()
