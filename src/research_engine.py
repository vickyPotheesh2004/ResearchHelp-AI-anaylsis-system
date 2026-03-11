import os
import json
import logging
from dotenv import load_dotenv
from src.prompt_templates import DOCUMENT_OVERVIEW_PROMPT, AUTO_SUGGESTIONS_PROMPT
from src.config import RESEARCH_LLM_MODEL, RESEARCH_OVERVIEW_MAX_TOKENS, RESEARCH_OVERVIEW_TEMPERATURE, RESEARCH_SUGGESTIONS_MAX_TOKENS, RESEARCH_SUGGESTIONS_TEMPERATURE, RESEARCH_ADDON_MAX_TOKENS, RESEARCH_ADDON_TEMPERATURE
from src.llm_client import get_llm_client
from src.logging_utils import get_logger

load_dotenv()

# Get logger - this ensures logging is configured
logger = get_logger(__name__)

class ResearchEngine:
    def __init__(self):
        llm_client = get_llm_client()
        self.client = llm_client.client
        self.model = RESEARCH_LLM_MODEL
        logger.info(f"ResearchEngine initialized with model: {self.model}")

    def generate_document_overview(self, all_chunks: list, all_metadata: list) -> str:
        logger.info("Generating document overview...")
        combined = []
        for doc, meta in zip(all_chunks, all_metadata):
            combined.append(f"[{meta.get('source_file', 'Unknown')} | {meta.get('topic', 'General')}]\n{doc}")

        context = "\n\n".join(combined[:20])

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": DOCUMENT_OVERVIEW_PROMPT},
                    {"role": "user", "content": f"Analyze these document segments:\n\n{context}"}
                ],
                max_tokens=RESEARCH_OVERVIEW_MAX_TOKENS,
                temperature=RESEARCH_OVERVIEW_TEMPERATURE
            )
            result = response.choices[0].message.content.strip()
            logger.info(f"Document overview generated successfully ({len(result)} chars)")
            return result
        except Exception as e:
            logger.error(f"Failed to generate document overview: {e}")
            return f"Could not generate overview: {str(e)}"

    def generate_auto_suggestions(self, all_chunks: list, all_metadata: list) -> list:
        logger.info("Generating auto suggestions...")
        combined = []
        for doc, meta in zip(all_chunks, all_metadata):
            combined.append(f"[{meta.get('source_file', 'Unknown')} | {meta.get('topic', 'General')}]\n{doc[:300]}")

        context = "\n\n".join(combined[:15])

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": AUTO_SUGGESTIONS_PROMPT},
                    {"role": "user", "content": f"Document content:\n\n{context}"}
                ],
                max_tokens=RESEARCH_SUGGESTIONS_MAX_TOKENS,
                temperature=RESEARCH_SUGGESTIONS_TEMPERATURE
            )
            raw = response.choices[0].message.content.strip()

            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            suggestions = json.loads(raw)
            if isinstance(suggestions, list):
                logger.info(f"Generated {len(suggestions[:5])} suggestions")
                return suggestions[:5]
            return []
        except Exception as e:
            logger.error(f"Failed to generate auto suggestions: {e}")
            return []

    def evaluate_addon_feasibility(self, proposal: str, context: str) -> str:
        from src.prompt_templates import RESEARCH_ADDON_PROMPT
        logger.info(f"Evaluating addon feasibility: {proposal[:30]}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"{RESEARCH_ADDON_PROMPT}\n\nDOCUMENT CONTEXT:\n{context}"},
                    {"role": "user", "content": proposal}
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
