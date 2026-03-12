"""
Unified Topic Pipeline Wrapper
===============================
Provides a unified interface for topic segmentation that can use:
1. Enhanced pipeline (zero API - sentence embeddings + TF-IDF + TextRank)
2. Legacy pipeline (LLM-based generation)

Configurable via USE_ENHANCED_PIPELINE in config.py

This module is backward-compatible with the existing TopicSegmenter interface.
"""

from typing import List, Dict, Optional
import threading

from src.config import (
    USE_ENHANCED_PIPELINE,
    TOPIC_SIMILARITY_THRESHOLD,
    TOPIC_SEGMENT_OVERLAP,
    EMBEDDING_MODEL,
    TOPIC_SUMMARY_SENTENCES,
    TOPIC_KEYWORDS_COUNT,
    TOPIC_MIN_SEGMENT_WORDS,
    DEFAULT_LLM_MODEL,
)
from src.logging_utils import get_logger

logger = get_logger(__name__)

# Lazy imports for enhanced pipeline
_enhanced_pipeline = None
_enhanced_pipeline_lock = threading.Lock()


def _get_enhanced_pipeline():
    """Get or create the enhanced pipeline singleton."""
    global _enhanced_pipeline
    if _enhanced_pipeline is None:
        with _enhanced_pipeline_lock:
            if _enhanced_pipeline is None:
                from src.enhanced_topic_pipeline import EnhancedTopicPipeline
                _enhanced_pipeline = EnhancedTopicPipeline(
                    num_topics=None,
                    summary_sentences=TOPIC_SUMMARY_SENTENCES,
                    top_keywords=TOPIC_KEYWORDS_COUNT,
                    embedding_model=EMBEDDING_MODEL,
                    min_segment_words=TOPIC_MIN_SEGMENT_WORDS,
                )
                logger.info("Initialized Enhanced Topic Pipeline (zero API)")
    return _enhanced_pipeline


class UnifiedTopicPipeline:
    """
    Unified topic segmentation interface.
    
    Automatically chooses between:
    - Enhanced pipeline: Zero API dependency, uses embeddings + TF-IDF + TextRank
    - Legacy pipeline: Uses LLM for title generation (requires API)
    
    Both pipelines provide:
    - Topic segmentation
    - Keyword extraction
    - Extractive summarization
    - Automatic title generation
    """
    
    def __init__(self, use_enhanced: Optional[bool] = None):
        """
        Initialize unified pipeline.
        
        Args:
            use_enhanced: Override config setting. None = use config value.
        """
        # Determine which pipeline to use
        if use_enhanced is not None:
            self.use_enhanced = use_enhanced
        else:
            self.use_enhanced = USE_ENHANCED_PIPELINE
        
        # Initialize the appropriate pipeline
        if self.use_enhanced:
            self.enhanced = _get_enhanced_pipeline()
            logger.info("Using Enhanced Topic Pipeline (zero API)")
        else:
            self.enhanced = None
            self._init_legacy_pipeline()
            logger.info("Using Legacy Topic Pipeline (LLM-based)")
        
        # Backward compatibility: titler for legacy mode
        if not self.use_enhanced:
            from src.topic_titler import TopicTitler
            self.titler = TopicTitler()
    
    def _init_legacy_pipeline(self):
        """Initialize legacy segmenter."""
        from src.topic_segmenter import TopicSegmenter
        from src.text_preprocessor import smart_sentence_split
        
        self.legacy_segmenter = TopicSegmenter()
        self.smart_split = smart_sentence_split
    
    def segment(self, text: str, num_topics: Optional[int] = None) -> List[Dict]:
        """
        Segment text into topics.
        
        Args:
            text: Input text to segment
            num_topics: Optional exact number of topics (None = auto-detect)
            
        Returns:
            List of topic dictionaries with:
                - topic_id: 0-based index
                - title: Generated title
                - content: Segment text
                - keywords: List of keywords (enhanced) or None (legacy)
                - summary: Extractive summary (enhanced) or None (legacy)
        """
        if self.use_enhanced:
            return self._segment_enhanced(text, num_topics)
        else:
            return self._segment_legacy(text)
    
    def _segment_enhanced(self, text: str, num_topics: Optional[int] = None) -> List[Dict]:
        """Segment using enhanced pipeline."""
        # Update num_topics if specified
        if num_topics is not None:
            self.enhanced.num_topics = num_topics
        
        results = self.enhanced.run(text)
        
        # Convert to legacy format for compatibility
        return [{
            "topic_id": r["topic_number"] - 1,
            "title": r["title"],
            "content": r["text"],
            "keywords": r["keywords"],
            "summary": r["summary"],
            "sentence_range": r["sentence_range"],
            "word_count": r["word_count"],
        } for r in results]
    
    def _segment_legacy(self, text: str) -> List[Dict]:
        """Segment using legacy pipeline."""
        topics = self.legacy_segmenter.segment(text)
        
        # Ensure backward compatibility format
        return [{
            "topic_id": t.get("topic_id", idx),
            "title": t.get("title", "Untitled"),
            "content": t.get("content", t.get("text", "")),
            "keywords": None,  # Not available in legacy
            "summary": None,    # Not available in legacy
        } for idx, t in enumerate(topics)]
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using TF-IDF."""
        if self.use_enhanced:
            from src.enhanced_topic_pipeline import KeywordExtractor
            extractor = KeywordExtractor(top_n=TOPIC_KEYWORDS_COUNT)
            keywords = extractor.extract([text])
            return [kw for kw, _ in keywords[0]] if keywords else []
        else:
            # Fallback: use simple extraction
            return self._simple_keywords(text)
    
    def _simple_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction fallback."""
        from collections import Counter
        import re
        
        # Simple tokenization
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Remove common stopwords
        from src.enhanced_topic_pipeline import STOP_WORDS
        words = [w for w in words if w not in STOP_WORDS]
        
        # Get top keywords
        counter = Counter(words)
        return [w for w, _ in counter.most_common(10)]
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Generate extractive summary."""
        if self.use_enhanced:
            from src.enhanced_topic_pipeline import TextRankSummarizer
            summarizer = TextRankSummarizer()
            return summarizer.summarize(text, num_sentences)
        else:
            # Legacy: return first few sentences as summary
            sentences = self.smart_split(text)
            return ' '.join(sentences[:num_sentences]) if sentences else text


# Singleton instance for convenience
_unified_pipeline = None
_pipeline_lock = threading.Lock()


def get_unified_pipeline() -> UnifiedTopicPipeline:
    """Get or create the singleton unified pipeline."""
    global _unified_pipeline
    if _unified_pipeline is None:
        with _pipeline_lock:
            if _unified_pipeline is None:
                _unified_pipeline = UnifiedTopicPipeline()
    return _unified_pipeline


# Backward compatibility: expose the segment function
def segment_topics(text: str, num_topics: Optional[int] = None) -> List[Dict]:
    """
    Convenience function for topic segmentation.
    
    Args:
        text: Input text
        num_topics: Optional exact number of topics
        
    Returns:
        List of topic dictionaries
    """
    pipeline = get_unified_pipeline()
    return pipeline.segment(text, num_topics)


# Demo / test
if __name__ == "__main__":
    from src.config import USE_ENHANCED_PIPELINE
    
    print("=" * 60)
    print("  Unified Topic Pipeline Demo")
    print(f"  Using enhanced pipeline: {USE_ENHANCED_PIPELINE}")
    print("=" * 60)
    
    SAMPLE_TEXT = """
    Artificial intelligence has emerged as one of the most transformative technologies 
    of the twenty-first century. Machine learning, a core subset of AI, enables computers 
    to learn from data without being explicitly programmed. Deep learning architectures 
    such as convolutional neural networks have revolutionized image recognition tasks. 
    Natural language processing models like BERT and GPT have achieved human-level 
    performance on many text benchmarks. The rapid advancement of AI has been fueled 
    by the availability of massive datasets and powerful GPU hardware.
    
    Climate change represents one of the most pressing challenges facing humanity today. 
    Rising global temperatures are caused primarily by greenhouse gas emissions from 
    burning fossil fuels. The effects include rising sea levels, more frequent extreme 
    weather events, and disruption of ecosystems. International agreements like the 
    Paris Accord aim to limit warming to 1.5 degrees Celsius above pre-industrial levels.
    Renewable energy sources such as solar, wind, and hydroelectric power are critical 
    to reducing carbon emissions.
    
    The global economy operates as an interconnected web of trade, finance, and production. 
    International trade allows countries to specialize in goods and services where they 
    hold comparative advantages. Financial markets channel capital from savers to 
    productive investments across borders. Supply chain disruptions, such as those 
    caused by the COVID-19 pandemic, revealed the fragility of global production networks.
    """
    
    # Initialize
    pipeline = UnifiedTopicPipeline()
    
    # Segment
    print("\n--- Topic Segmentation ---")
    topics = pipeline.segment(SAMPLE_TEXT)
    
    for topic in topics:
        print(f"\nTopic {topic['topic_id'] + 1}: {topic['title']}")
        print(f"  Content: {topic['content'][:100]}...")
        if topic.get('keywords'):
            print(f"  Keywords: {topic['keywords'][:3]}")
        if topic.get('summary'):
            print(f"  Summary: {topic['summary'][:100]}...")
    
    # Extract keywords
    print("\n--- Keyword Extraction ---")
    keywords = pipeline.extract_keywords(SAMPLE_TEXT)
    print(f"Keywords: {keywords[:5]}")
    
    # Summarize
    print("\n--- Extractive Summary ---")
    summary = pipeline.summarize(SAMPLE_TEXT, num_sentences=2)
    print(f"Summary: {summary[:200]}...")
