from dotenv import load_dotenv
from typing import List
from src.llm_client import get_llm_client
from src.config import TOPIC_MAX_TOKENS, TOPIC_TEMPERATURE, TITLER_MODEL
from src.logging_utils import get_logger

load_dotenv()

# Get logger - this ensures logging is configured
logger = get_logger(__name__)


class TopicTitler:
    def __init__(self):
        llm_client = get_llm_client()
        self.client = llm_client.client
        # Use optimized titler model from config
        self.model = TITLER_MODEL

    def generate_title(self, texts: List[str]) -> str:
        """Uses an LLM to generate a concise, 1-2 word conceptual category for the text block."""
        if not texts or not any(text.strip() for text in texts):
            return "General"

        combined_text = " ".join(texts)[:1000]

        system_prompt = (
            "Read the following text and categorize it with a 1 or 2 word conceptual title. "
            "Examples: Introduction, Hardware, Software, Methodology, Conclusion, Specifications. "
            "Reply ONLY with the title. No quotes or punctuation."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": combined_text},
                ],
                max_tokens=TOPIC_MAX_TOKENS,
                temperature=TOPIC_TEMPERATURE,
                # No reasoning needed for simple title generation - faster response
            )
            title = response.choices[0].message.content.strip()
            return title.replace('"', "").replace(".", "").title()

        except Exception as e:
            # Fallback to the first two words if the API is unavailable
            logger.warning(f"Topic titler failed: {e}. Using fallback.")
            words = combined_text.split()
            return " ".join(words[:2]).title()
