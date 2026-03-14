"""
Test Suite for Confidence Scorer
================================
Tests the ConfidenceScorer class in src/confidence_scorer.py

Usage:
    python test_confidence_scorer.py

Dependencies:
    - pytest (for running tests)
    - unittest.mock (for mocking LLM client)
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.confidence_scorer import ConfidenceScorer


class TestConfidenceScorer(unittest.TestCase):
    """Test cases for ConfidenceScorer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_client = Mock()
        self.scorer = ConfidenceScorer(self.mock_llm_client)
        
    def test_fallback_values(self):
        """Test that fallback values are correct."""
        fallback = ConfidenceScorer.FALLBACK
        self.assertEqual(fallback["score"], 50)
        self.assertEqual(fallback["level"], "Moderate")
        self.assertIn("reason", fallback)
        
    def test_format_chunks_truncation(self):
        """Test that _format_chunks properly truncates long context."""
        chunks = ["Short chunk", "Another short chunk"]
        result = self.scorer._format_chunks(chunks, max_chars=100)
        self.assertIn("[Chunk 1]: Short chunk", result)
        self.assertIn("[Chunk 2]: Another short chunk", result)
        
    def test_format_chunks_truncation_long(self):
        """Test truncation with very long chunks."""
        long_chunk = "A" * 2000  # 2000 char chunk
        chunks = [long_chunk]
        result = self.scorer._format_chunks(chunks, max_chars=100)
        self.assertIn("truncated", result)
        
    def test_format_chunks_empty(self):
        """Test handling of empty chunks."""
        result = self.scorer._format_chunks([], max_chars=100)
        self.assertIn("No context chunks", result)
        
    def test_score_to_level_very_high(self):
        """Test score to level mapping for Very High (85-100)."""
        self.assertEqual(ConfidenceScorer._score_to_level(85), "Very High")
        self.assertEqual(ConfidenceScorer._score_to_level(100), "Very High")
        self.assertEqual(ConfidenceScorer._score_to_level(90), "Very High")
        
    def test_score_to_level_high(self):
        """Test score to level mapping for High (65-84)."""
        self.assertEqual(ConfidenceScorer._score_to_level(65), "High")
        self.assertEqual(ConfidenceScorer._score_to_level(84), "High")
        self.assertEqual(ConfidenceScorer._score_to_level(75), "High")
        
    def test_score_to_level_moderate(self):
        """Test score to level mapping for Moderate (45-64)."""
        self.assertEqual(ConfidenceScorer._score_to_level(45), "Moderate")
        self.assertEqual(ConfidenceScorer._score_to_level(64), "Moderate")
        self.assertEqual(ConfidenceScorer._score_to_level(55), "Moderate")
        
    def test_score_to_level_low(self):
        """Test score to level mapping for Low (25-44)."""
        self.assertEqual(ConfidenceScorer._score_to_level(25), "Low")
        self.assertEqual(ConfidenceScorer._score_to_level(44), "Low")
        self.assertEqual(ConfidenceScorer._score_to_level(35), "Low")
        
    def test_score_to_level_very_low(self):
        """Test score to level mapping for Very Low (0-24)."""
        self.assertEqual(ConfidenceScorer._score_to_level(0), "Very Low")
        self.assertEqual(ConfidenceScorer._score_to_level(24), "Very Low")
        self.assertEqual(ConfidenceScorer._score_to_level(10), "Very Low")
        
    def test_parse_response_valid_json(self):
        """Test parsing valid JSON response."""
        raw = '{"score": 82, "level": "High", "reason": "Multiple chunks directly address the query."}'
        result = self.scorer._parse_response(raw)
        self.assertEqual(result["score"], 82)
        self.assertEqual(result["level"], "High")
        self.assertIn("reason", result)
        
    def test_parse_response_json_with_markdown(self):
        """Test parsing JSON with markdown code blocks."""
        raw = '```json\n{"score": 75, "level": "High", "reason": "Good context coverage."}\n```'
        result = self.scorer._parse_response(raw)
        self.assertEqual(result["score"], 75)
        self.assertEqual(result["level"], "High")
        
    def test_parse_response_score_bounds(self):
        """Test that scores are bounded between 0 and 100."""
        # Test over 100
        raw = '{"score": 150, "level": "High", "reason": "Test"}'
        result = self.scorer._parse_response(raw)
        self.assertEqual(result["score"], 100)
        
        # Test under 0
        raw = '{"score": -20, "level": "Low", "reason": "Test"}'
        result = self.scorer._parse_response(raw)
        self.assertEqual(result["score"], 0)
        
    def test_parse_response_invalid_level(self):
        """Test handling of invalid level values."""
        raw = '{"score": 70, "level": "InvalidLevel", "reason": "Test"}'
        result = self.scorer._parse_response(raw)
        # Should fall back to computing level from score
        self.assertEqual(result["level"], "High")  # 70 maps to High
        
    def test_parse_response_missing_fields(self):
        """Test handling of missing JSON fields."""
        raw = '{"score": 80}'
        result = self.scorer._parse_response(raw)
        self.assertEqual(result["score"], 80)
        self.assertEqual(result["level"], "Moderate")  # Default
        self.assertIn("reason", result)  # Default reason
        
    def test_parse_response_invalid_json_fallback(self):
        """Test fallback when no JSON found."""
        raw = "This is not JSON at all"
        with self.assertRaises(ValueError):
            self.scorer._parse_response(raw)
            
    def test_score_confidence_success(self):
        """Test successful confidence scoring."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"score": 85, "level": "Very High", "reason": "Direct answer found in multiple chunks."}'
        self.mock_llm_client.create_fast_completion.return_value = mock_response
        
        result = self.scorer.score_confidence(
            user_question="What is machine learning?",
            intent="document_qa",
            domain="AI",
            context_chunks=["ML is a subset of AI", "Machine learning enables computers to learn"]
        )
        
        self.assertEqual(result["score"], 85)
        self.assertEqual(result["level"], "Very High")
        self.mock_llm_client.create_fast_completion.assert_called_once()
        
    def test_score_confidence_empty_chunks(self):
        """Test scoring with empty context chunks."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"score": 20, "level": "Very Low", "reason": "No context available."}'
        self.mock_llm_client.create_fast_completion.return_value = mock_response
        
        result = self.scorer.score_confidence(
            user_question="What is X?",
            intent="document_qa",
            domain="General",
            context_chunks=[]
        )
        
        self.assertEqual(result["score"], 20)
        
    def test_score_confidence_llm_exception(self):
        """Test fallback when LLM call fails - should use criteria-based scoring."""
        self.mock_llm_client.create_fast_completion.side_effect = Exception("API Error")
        
        result = self.scorer.score_confidence(
            user_question="Test question",
            intent="document_qa",
            domain="Test",
            context_chunks=["test"]
        )
        
        # Should return criteria-based scoring (not fixed 50% fallback)
        # With 1 chunk and no sources, score should be: 25 + 8 + 0 + 0 + 10 = 43 (but bounded to min 10)
        # Actually: base 25 + chunk 8 + domain 10 = 43, within bounds 10-85
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertIn('level', result)
        self.assertIn('reason', result)
        # Score should NOT be 50 (the old fixed fallback)
        self.assertNotEqual(result['score'], 50)
        # Score should be within the defined bounds
        self.assertGreaterEqual(result['score'], 10)
        self.assertLessEqual(result['score'], 85)
        
    def test_score_confidence_with_long_context(self):
        """Test that long context is properly truncated."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"score": 70, "level": "High", "reason": "Context provided."}'
        self.mock_llm_client.create_fast_completion.return_value = mock_response
        
        long_context = ["A" * 500] * 10  # 10 chunks of 500 chars each
        result = self.scorer.score_confidence(
            user_question="Test?",
            intent="document_qa",
            domain="Test",
            context_chunks=long_context,
            max_context_chars=500  # Small limit to force truncation
        )
        
        # Should complete without error
        self.assertIn("score", result)


class TestConfidenceScorerIntegration(unittest.TestCase):
    """Integration tests that verify the full scoring pipeline."""
    
    def test_prompt_contains_all_inputs(self):
        """Verify that the prompt template includes all required inputs."""
        from src.confidence_scorer import CONFIDENCE_SCORING_PROMPT
        
        # Check that prompt template has all placeholders
        self.assertIn("{user_question}", CONFIDENCE_SCORING_PROMPT)
        self.assertIn("{intent}", CONFIDENCE_SCORING_PROMPT)
        self.assertIn("{domain}", CONFIDENCE_SCORING_PROMPT)
        self.assertIn("{context_chunks}", CONFIDENCE_SCORING_PROMPT)
        self.assertIn("{num_chunks}", CONFIDENCE_SCORING_PROMPT)
        
    def test_score_level_mapping_complete(self):
        """Verify all score ranges map to correct levels."""
        test_cases = [
            (0, "Very Low"),
            (24, "Very Low"),
            (25, "Low"),
            (44, "Low"),
            (45, "Moderate"),
            (64, "Moderate"),
            (65, "High"),
            (84, "High"),
            (85, "Very High"),
            (100, "Very High"),
        ]
        
        for score, expected_level in test_cases:
            actual_level = ConfidenceScorer._score_to_level(score)
            self.assertEqual(actual_level, expected_level, 
                           f"Score {score} should map to {expected_level}, got {actual_level}")


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConfidenceScorer))
    suite.addTests(loader.loadTestsFromTestCase(TestConfidenceScorerIntegration))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("CONFIDENCE SCORER TEST SUITE")
    print("=" * 70)
    print()
    
    result = run_tests()
    
    print()
    print("=" * 70)
    if result.wasSuccessful():
        print("ALL TESTS PASSED!")
    else:
        print(f"FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 70)
