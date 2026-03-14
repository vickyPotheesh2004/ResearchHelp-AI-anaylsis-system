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
| Context directly answers query  | +30 to +40 pts         |
| Context partially answers query | +15 to +25 pts         |
| No relevant context found       | −40 pts                |
| Multiple corroborating chunks   | +10 pts                |
| Single source only              | −5 pts                 |
| Domain matched to document      | +10 pts                |
| Domain mismatch                 | −10 pts                |
| Query is very specific/niche    | −10 to −20 pts         |
| Query is broad/general          | +5 pts                 |
| Ambiguous or vague question     | −15 pts                |
| Context is recent/complete      | +5 pts                 |
| Context appears truncated/cut   | −10 pts                |

## Inputs

**User Question:**
{user_question}

**Intent Category:**
{intent}

**Detected Domain:**
{domain}

**Retrieved Context Chunks (top {num_chunks}):**
{context_chunks}

## Output Rules
- Output ONLY valid JSON — no explanation, no markdown, no commentary.
- `score` must be an INTEGER between 0 and 100.
- `reason` must be ONE short sentence (max 15 words) explaining the score.
- `level` must be exactly one of: "Very High", "High", "Moderate", "Low", "Very Low"

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
  "reason": "<one sentence, max 15 words>"
}}
"""


# ── CONFIDENCE SCORER CLASS ──────────────────────────────────────────────────

class ConfidenceScorer:
    """
    Calls the fastest available LLM (GLM-4.5-Air) with a structured prompt
    to produce a confidence score BEFORE the main answer is streamed to the user.

    Usage
    -----
    scorer = ConfidenceScorer(llm_client)
    result = scorer.score_confidence(
        user_question="What hardware was used?",
        intent="document_qa",
        domain="IoT",
        context_chunks=["chunk1 text...", "chunk2 text..."]
    )
    # result → {"score": 82, "level": "High", "reason": "Multiple chunks directly address hardware specs."}
    """

    # Criteria-based scoring thresholds
    CHUNK_WEIGHT = 8          # Points per chunk (max 5 chunks = +40)
    LENGTH_WEIGHT = 0.015     # Points per character in context
    SOURCE_BONUS = 15         # Bonus when sources are available
    DOMAIN_MATCH_BONUS = 10   # Bonus when domain is matched
    QUERY_SPECIFICITY_PENALTY = -10  # Penalty for very specific/niche queries
    
    # Base score and bounds
    BASE_SCORE = 25
    MIN_SCORE = 10
    MAX_SCORE = 85  # Never exceed 85 without LLM evaluation

    # Pre-compiled patterns for query specificity detection
    _SPECIFIC_INDICATORS = [
        re.compile(r'\d+'),
        re.compile(r'\b[A-Z][a-z]+\d+\b'),
        re.compile(r'version \d'),
        re.compile(r'\d+\.\d+'),
    ]

    FALLBACK = {"score": 50, "level": "Moderate", "reason": "Score unavailable — showing default."}

    def __init__(self, llm_client):
        """
        Parameters
        ----------
        llm_client : LLMClient
            The existing LLMClient instance from src/llm_client.py.
            Uses glm_model (GLM-4.5-Air) — fastest, cheapest option.
        """
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

        Parameters
        ----------
        user_question     : The raw user query string
        intent            : Classified intent (e.g. "document_qa")
        domain            : Detected domain (e.g. "Cybersecurity")
        context_chunks    : List of retrieved text chunks (strings)
        max_context_chars : Safety cap to avoid token overflow
        has_sources      : Whether sources/citations are available
        """
        try:
            combined = self._format_chunks(context_chunks, max_context_chars)
            prompt = CONFIDENCE_SCORING_PROMPT.format(
                user_question=user_question.strip(),
                intent=intent,
                domain=domain if domain else "General",
                num_chunks=len(context_chunks),
                context_chunks=combined,
            )
            raw_response = self._call_llm(prompt)
            result = self._parse_response(raw_response)
            logger.info(f"[ConfidenceScorer] score={result['score']} level={result['level']}")
            return result

        except Exception as e:
            logger.warning(f"[ConfidenceScorer] Failed, using criteria-based scoring. Error: {e}")
            # Use criteria-based grading instead of fixed 50%
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
        
        Grading Criteria:
        -----------------
        1. Number of context chunks: +8 points per chunk (max 5 chunks = +40)
        2. Total context length: +1.5 points per 100 chars (max ~1500 = +22)
        3. Has sources/citations: +15 bonus
        4. Domain matched: +10 bonus
        5. Query specificity: -10 for very specific/niche queries
        6. Base score: 25
        
        Score Range: 10-85 (never exceeds 85 without LLM evaluation)
        """
        score = self.BASE_SCORE
        reasons = []
        
        # Criterion 1: Number of context chunks
        num_chunks = len(context_chunks)
        chunk_score = min(num_chunks * self.CHUNK_WEIGHT, 5 * self.CHUNK_WEIGHT)
        score += chunk_score
        if num_chunks >= 3:
            reasons.append(f"{num_chunks} relevant chunks found")
        elif num_chunks > 0:
            reasons.append(f"{num_chunks} chunk(s) available")
        
        # Criterion 2: Total context length
        total_length = sum(len(chunk) for chunk in context_chunks)
        length_score = min(int(total_length * self.LENGTH_WEIGHT), 22)
        score += length_score
        if total_length > 500:
            reasons.append("substantial context")
        
        # Criterion 3: Sources available
        if has_sources:
            score += self.SOURCE_BONUS
            reasons.append("sources cited")
        
        # Criterion 4: Domain matched (non-empty domain)
        if domain and domain != "General":
            score += self.DOMAIN_MATCH_BONUS
            reasons.append(f"domain: {domain}")
        
        # Criterion 5: Query specificity check
        # Very specific queries (with technical terms, numbers, proper nouns) get lower base
        is_specific = any(p.search(user_question) for p in self._SPECIFIC_INDICATORS)
        if is_specific:
            score += self.QUERY_SPECIFICITY_PENALTY
            reasons.append("specific query")
        
        # Apply bounds
        score = max(self.MIN_SCORE, min(self.MAX_SCORE, score))
        
        # Generate reason string
        reason = ", ".join(reasons[:2]) if reasons else "Criteria-based score"
        if len(reason) > 50:
            reason = reason[:47] + "..."
        
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
