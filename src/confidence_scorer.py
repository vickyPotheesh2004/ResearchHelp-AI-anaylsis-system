"""
Confidence Scorer Module for ResearchHelp-AI Analysis System.
Evaluates how confidently the system can answer user questions based on retrieved context.
"""

from __future__ import annotations

import re
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── CONFIDENCE SCORING PROMPT ─────────────────────────────────────────────────

CONFIDENCE_SCORING_PROMPT = """You are a strict self-evaluation engine embedded inside a document Q&A system.

Your ONLY job is to output a single JSON object — nothing else.

## Your Task
Evaluate how confidently the system can answer the user question given:
- The retrieved document context chunks
- The detected research domain
- The intent category of the query

## Scoring Criteria (apply ALL of them)

| Factor                          | Effect on Score         |
|---------------------------------|------------------------|
| Irrelevant to documents        | −80 pts (MANDATORY)    |
| Context directly answers query  | +30 to +40 pts         |
| Context partially answers query | +15 to +25 pts         |
| No relevant context found       | −40 pts                |
| Multiple corroborating chunks   | +10 pts                |
| Domain matched to document      | +10 pts                |
| Domain mismatch                 | −15 pts                |
| Ambiguous or vague question     | −20 pts                |

## Inputs

**User Question:**
{user_question}

**Intent Category:**
{intent}

**Detected Domain:**
{domain}

**Retrieved Context Chunks (top {num_chunks}):**
{context_chunks}

## Strict Evaluation Note
If the question is unrelated to the provided context, the score MUST be below 20. Do not reward "sources found" if they are not relevant to the question.

## Output Rules
- Output ONLY valid JSON.
- `score` must be an INTEGER between 0 and 100.
- `reason` must be ONE short sentence explaining the score.
- `level` must match the score mapping.

## Score → Level Mapping
- 85–100 → "Very High"
- 65–84  → "High"
- 45–64  → "Moderate"
- 25–44  → "Low"
- 0–24   → "Very Low"

## Required JSON format:
{{
  "score": <integer 0-100>,
  "level": "<Very High|High|Moderate|Low|Very Low>",
  "reason": "<one sentence>"
}}
"""


# ── CONFIDENCE SCORER CLASS ──────────────────────────────────────────────────

class ConfidenceScorer:
    """
    Calls the fastest available LLM (GLM-4.5-Air) with a structured prompt
    to produce a confidence score BEFORE the main answer is streamed to the user.
    """

    # Criteria-based scoring thresholds
    CHUNK_WEIGHT = 7          # Points per chunk (max 5 chunks = +35)
    LENGTH_WEIGHT = 0.01      # Points per character in context
    SOURCE_BONUS = 15         # Bonus when sources are available
    DOMAIN_MATCH_BONUS = 10   # Bonus when domain is matched
    
    # Base score and bounds
    BASE_SCORE = 20
    MIN_SCORE = 0
    MAX_SCORE = 85  # Never exceed 85 without LLM evaluation

    FALLBACK = {"score": 50, "level": "Moderate", "reason": "Score unavailable — showing default."}

    def __init__(self, llm_client):
        self.client = llm_client

    def score_confidence(
        self,
        user_question: str,
        intent: str,
        domain: str,
        context_chunks: list[str],
        max_context_chars: int = 3000,
        has_sources: bool = True,
    ) -> dict:
        """
        Returns a dict: {"score": int, "level": str, "reason": str}
        """
        # Hard cap for off-topic intent
        if intent == "off_topic":
            return {
                "score": 10,
                "level": "Very Low",
                "reason": "Query is officially classified as off-topic."
            }

        try:
            combined = self._format_chunks(context_chunks, max_context_chars)
            
            # 1. Mandatory Keyword Overlap Guardrail (The "Hallucination Killer")
            # Extract meaningful words (length > 3, alpha only)
            q_words = [w.lower() for w in re.findall(r'\b[a-z]{4,}\b', user_question.lower())]
            combined_ctx_lower = combined.lower()
            
            if q_words:
                match_count = sum(1 for w in q_words if w in combined_ctx_lower)
                relevance_ratio = match_count / len(q_words)
            else:
                # If question is too short/generic (e.g., "tell me more"), assume low direct relevance
                relevance_ratio = 0.05
            
            # 2. Get LLM Score
            prompt = CONFIDENCE_SCORING_PROMPT.format(
                user_question=user_question.strip(),
                intent=intent,
                domain=domain if domain else "General",
                num_chunks=len(context_chunks),
                context_chunks=combined,
            )
            raw_response = self._call_llm(prompt)
            result = self._parse_response(raw_response)
            
            # 3. ABSOLUTE OVERRIDE: If relevance ratio is ultra-low, LLM cannot exceed "Very Low"
            if intent == "document_qa" or intent == "general":
                if relevance_ratio < 0.12:
                    # Non-negotiable override for irrelevance
                    result['score'] = min(result['score'], 12)
                    result['level'] = "Very Low"
                    result['reason'] = "Security override: Near-zero technical overlap with document content."
                elif relevance_ratio < 0.25 and result['score'] > 40:
                    # Moderate override for loose relevance
                    result['score'] = min(result['score'], 35)
                    result['level'] = "Low"
                    result['reason'] = "Score capped: Low keyword matching suggests tangential relevance."

            logger.info(f"[ConfidenceScorer] final_score={result['score']} overlap={relevance_ratio:.2f} intent={intent}")
            return result

        except Exception as e:
            logger.warning(f"[ConfidenceScorer] Failed, using criteria-based scoring. Error: {e}")
            return self._criteria_based_scoring(
                user_question=user_question,
                intent=intent,
                domain=domain,
                context_chunks=context_chunks,
                has_sources=has_sources,
            )

    # ── Criteria-Based Scoring (fallback when LLM fails) ─────────────────────

    def _criteria_based_scoring(
        self,
        user_question: str,
        intent: str,
        domain: str,
        context_chunks: list[str],
        has_sources: bool = True,
    ) -> dict:
        """
        Calculates confidence score based on concrete criteria when LLM fails.
        """
        if intent == "off_topic":
            return {"score": 5, "level": "Very Low", "reason": "Off-topic query."}

        # 1. Start with very low base for security
        score = 15 
        reasons = []
        
        # 2. Key Relevance Check (Crucial Fix)
        # Check if important words from question are in the context
        q_words = [w.lower() for w in re.findall(r'\b\w{4,}\b', user_question)]
        combined_ctx = " ".join(context_chunks).lower()
        match_count = sum(1 for w in q_words if w in combined_ctx)
        
        relevance_ratio = match_count / len(q_words) if q_words else 0
        
        if relevance_ratio < 0.2:
            score -= 10
            reasons.append("low relevance")
        elif relevance_ratio > 0.5:
            score += 25
            reasons.append("high keyword overlap")
        
        # 3. Contextual quantity
        num_chunks = len(context_chunks)
        score += min(num_chunks * self.CHUNK_WEIGHT, 35)
        
        # 4. Sources/Domain
        if has_sources and relevance_ratio > 0.2:
            score += self.SOURCE_BONUS
            
        if domain and domain != "General":
            score += self.DOMAIN_MATCH_BONUS
        
        # Force low score if no matches at all
        if match_count == 0 and intent == "document_qa":
            score = min(score, 15)
            reasons = ["no direct matches found"]

        # Apply bounds
        score = max(self.MIN_SCORE, min(self.MAX_SCORE, score))
        
        reason = ", ".join(reasons[:2]) if reasons else "Context quality check"
        if len(reason) > 50: reason = reason[:47] + "..."
        
        return {
            "score": score,
            "level": self._score_to_level(score),
            "reason": reason,
        }

    def _format_chunks(self, chunks: list[str], max_chars: int) -> str:
        """Formats and truncates chunks for the prompt."""
        parts = []
        total = 0
        for i, chunk in enumerate(chunks, 1):
            chunk_text = f"[Chunk {i}]: {chunk.strip()}"
            if total + len(chunk_text) > max_chars:
                parts.append(f"[Chunk {i}]: ...(truncated for length)")
                break
            parts.append(chunk_text)
            total += len(chunk_text)
        return "\n\n".join(parts) if parts else "No context chunks retrieved."

    def _call_llm(self, prompt: str) -> str:
        """
        Calls the LLM via the existing LLMClient (OpenAI-compatible interface).

        Uses GLM-4.5-Air model for fast scoring.
        """
        # Use create_fast_completion for scoring (it's designed for simple tasks)
        response = self.client.create_fast_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.0,  # deterministic scoring
        )
        
        return response.choices[0].message.content.strip()

    def _parse_response(self, raw: str) -> dict:
        """Extracts and validates JSON from the LLM response."""
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in LLM response.")
            data = json.loads(match.group())

        score = max(0, min(100, int(data.get("score", 50))))

        level = data.get("level", "Moderate")
        if level not in {"Very High", "High", "Moderate", "Low", "Very Low"}:
            level = self._score_to_level(score)

        reason = str(data.get("reason", "")).strip() or "Score based on retrieved context quality."

        return {"score": score, "level": level, "reason": reason}

    @staticmethod
    def _score_to_level(score: int) -> str:
        if score >= 85:
            return "Very High"
        if score >= 65:
            return "High"
        if score >= 45:
            return "Moderate"
        if score >= 25:
            return "Low"
        return "Very Low"
