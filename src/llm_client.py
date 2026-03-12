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
    GEMMA_3_12B_MODEL,
    TRINITY_LARGE_MODEL,
    NEMOTRON_3_SUPER_MODEL,
    REASONING_MODELS,
)

load_dotenv()

# Get logger - this ensures logging is configured
logger = get_logger(__name__)

# OpenRouter site identification for ranking
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "https://github.com")
OPENROUTER_SITE_TITLE = os.getenv("OPENROUTER_SITE_TITLE", "ResearchHelp AI Analysis System")


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
    def gemma_model(self) -> str:
        """Get Gemma 3 12B model (balanced for standard tasks)."""
        return GEMMA_3_12B_MODEL
    
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
        extra_headers = {
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-OpenRouter-Title": OPENROUTER_SITE_TITLE,
        }
        
        extra_body = {}
        
        # Enable reasoning for models that support it
        if enable_reasoning and model in REASONING_MODELS:
            extra_body["reasoning"] = {"enabled": True}
        
        # Merge any additional extra_body params
        if "extra_body" in kwargs:
            extra_body.update(kwargs.pop("extra_body"))
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers=extra_headers,
                extra_body=extra_body if extra_body else None,
                **kwargs
            )
            
            # Post-process for reasoning models that might return null content but valid reasoning
            message = response.choices[0].message
            if not getattr(message, "content", None):
                reasoning = getattr(message, "reasoning", None)
                if not reasoning and hasattr(message, "reasoning_details"):
                    reasoning = message.reasoning_details
                
                if reasoning:
                    logger.info(f"Model {model} returned reasoning without content. Using reasoning as content.")
                    message.content = f"> [Reasoning Mode]\n\n{reasoning}"
            
            return response

        except Exception as e:
            error_str = str(e)
            if "404" in error_str and "guardrail" in error_str.lower():
                logger.error(f"OpenRouter Guardrail/Data Policy 404 Error: {error_str}")
                raise Exception(
                    "OpenRouter API Error (404): No endpoints available matching your privacy settings.\n"
                    "FIX: Go to https://openrouter.ai/settings/privacy and set 'Allow Data Retention' to ENABLED "
                    "to use free models, or add credits to your account."
                )
            raise e

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

    def create_standard_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.3
    ) -> Any:
        """
        Create a standard completion using Gemma 3 12B (balanced accuracy and speed).
        """
        return self.create_chat_completion(
            model=self.gemma_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_reasoning=False
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

    def create_vision_completion(
        self,
        text: str,
        image_url: str,
        max_tokens: int = 1000,
        temperature: float = 0.3
    ) -> Any:
        """
        Create a vision/image understanding completion using Gemma 3 12B.
        Gemma 3 12B supports multimodal inputs (text + images).
        
        Args:
            text: The text prompt/question about the image
            image_url: URL of the image to analyze
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Chat completion response with image understanding
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ]
        
        return self.create_chat_completion(
            model=self.gemma_model,  # Use Gemma 3 12B for vision
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_reasoning=False
        )

    def create_reasoning_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 0.25,
        use_nemotron: bool = False
    ) -> Any:
        """
        Create a reasoning-focused completion using Trinity or Nemotron.
        
        Args:
            messages: Chat messages
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            use_nemotron: If True, use Nemotron (best reasoning), otherwise Trinity
            
        Returns:
            Chat completion response with reasoning details
        """
        model = self.nemotron_model if use_nemotron else self.trinity_model
        return self.create_chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_reasoning=True  # Enable reasoning for both models
        )


def get_llm_client() -> LLMClient:
    """Get the singleton LLM client instance."""
    return LLMClient()
