import json
from dotenv import load_dotenv
from src.prompt_templates import (
    DOCUMENT_OVERVIEW_PROMPT,
    AUTO_SUGGESTIONS_PROMPT,
)
from src.config import (
    RESEARCH_OVERVIEW_MAX_TOKENS,
    RESEARCH_OVERVIEW_TEMPERATURE,
    RESEARCH_SUGGESTIONS_MAX_TOKENS,
    RESEARCH_SUGGESTIONS_TEMPERATURE,
    RESEARCH_ADDON_MAX_TOKENS,
    RESEARCH_ADDON_TEMPERATURE,
)
from src.llm_client import get_llm_client
from src.logging_utils import get_logger

load_dotenv()

# Get logger - this ensures logging is configured
logger = get_logger(__name__)


class ResearchEngine:
    def __init__(self):
        llm_client = get_llm_client()
        self.client = llm_client.client
        self.llm_client = llm_client  # Use the full client for helper methods
        self.model = llm_client.nemotron_model  # Use Nemotron 3 Super - best reasoning
        logger.info(f"ResearchEngine initialized with model: {self.model}")

    def generate_document_overview(
        self, all_chunks: list, all_metadata: list
    ) -> str:
        logger.info("Generating document overview...")
        combined = []
        for doc, meta in zip(all_chunks, all_metadata):
            combined.append(
                f"[{meta.get('source_file', 'Unknown')} | {meta.get('topic', 'General')}]\n{doc}"
            )

        context = "\n\n".join(combined[:20])

        try:
            # Use create_research_completion for Nemotron 3 Super with reasoning
            response = self.llm_client.create_research_completion(
                messages=[
                    {"role": "system", "content": DOCUMENT_OVERVIEW_PROMPT},
                    {
                        "role": "user",
                        "content": f"Analyze these document segments:\n\n{context}",
                    },
                ],
                max_tokens=RESEARCH_OVERVIEW_MAX_TOKENS,
                temperature=RESEARCH_OVERVIEW_TEMPERATURE
            )
            result = response.choices[0].message.content.strip()
            logger.info(
                f"Document overview generated successfully ({len(result)} chars)"
            )
            return result
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to generate document overview: {error_msg}")
            if "429" in error_msg:
                return "ERROR: API_RATE_LIMIT_EXCEEDED"
            return f"Could not generate overview: {error_msg}"

    def generate_auto_suggestions(
        self, all_chunks: list, all_metadata: list
    ) -> list:
        logger.info("Generating auto suggestions...")
        combined = []
        for doc, meta in zip(all_chunks, all_metadata):
            combined.append(
                f"[{meta.get('source_file', 'Unknown')} | {meta.get('topic', 'General')}]\n{doc[:300]}"
            )

        context = "\n\n".join(combined[:15])

        try:
            # Use create_research_completion for Nemotron 3 Super with reasoning
            response = self.llm_client.create_research_completion(
                messages=[
                    {"role": "system", "content": AUTO_SUGGESTIONS_PROMPT},
                    {
                        "role": "user",
                        "content": f"Document content:\n\n{context}",
                    },
                ],
                max_tokens=RESEARCH_SUGGESTIONS_MAX_TOKENS,
                temperature=RESEARCH_SUGGESTIONS_TEMPERATURE
            )
            raw = response.choices[0].message.content.strip()

            start_idx = raw.find("[")
            end_idx = raw.rfind("]")

            if start_idx != -1 and end_idx != -1:
                raw = raw[start_idx : end_idx + 1]

            try:
                suggestions = json.loads(raw)
            except json.JSONDecodeError as je:
                logger.warning(f"Failed to parse suggestions JSON: {je}. Raw response: {raw[:200]}")
                return []
            
            if isinstance(suggestions, list):
                logger.info(f"Generated {len(suggestions[:5])} suggestions")
                return suggestions[:5]
            return []
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to generate auto suggestions: {error_msg}")
            if "429" in error_msg:
                return ["ERROR: API_RATE_LIMIT_EXCEEDED"]
            return []

    def evaluate_addon_feasibility(self, proposal: str, context: str) -> str:
        from src.prompt_templates import RESEARCH_ADDON_PROMPT

        logger.info(f"Evaluating addon feasibility: {proposal[:30]}...")
        try:
            # Use create_research_completion for Nemotron 3 Super with reasoning
            response = self.llm_client.create_research_completion(
                messages=[
                    {
                        "role": "system",
                        "content": f"{RESEARCH_ADDON_PROMPT}\n\nDOCUMENT CONTEXT:\n{context}",
                    },
                    {"role": "user", "content": proposal},
                ],
                max_tokens=RESEARCH_ADDON_MAX_TOKENS,
                temperature=RESEARCH_ADDON_TEMPERATURE
            )
            result = response.choices[0].message.content.strip()
            logger.info("Addon feasibility evaluation completed")
            return result
        except Exception as e:
            logger.error(f"Failed to evaluate addon feasibility: {e}")
            return f"Could not evaluate proposal: {str(e)}"
