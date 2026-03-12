"""
Unified Topic Pipeline (Local-Only)
====================================
Provides topic segmentation using only local algorithms:
- Sentence embeddings (sentence-transformers) with TF-IDF fallback
- TF-IDF keyword extraction  
- Template-based title generation

No external API dependencies required.
"""

from typing import List, Dict, Optional
import threading

from src.config import (
    EMBEDDING_MODEL,
    TOPIC_SUMMARY_SENTENCES,
    TOPIC_KEYWORDS_COUNT,
    TOPIC_MIN_SEGMENT_WORDS,
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
                logger.info("Initialized Enhanced Topic Pipeline (local-only)")
    return _enhanced_pipeline


class UnifiedTopicPipeline:
    """
    Local-only topic segmentation interface.
    
    Uses:
    - Sentence embeddings (sentence-transformers) or TF-IDF fallback
    - TF-IDF keyword extraction
    - Template-based title generation
    
    Zero API dependencies - completely offline capable.
    """
    
    def __init__(self):
        """Initialize local-only pipeline."""
        self.pipeline = _get_enhanced_pipeline()
        logger.info("Using Local-Only Topic Pipeline")
    
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
                - keywords: List of (keyword, score) tuples
                - sentence_range: (start, end) 1-indexed
                - word_count: Number of words
        """
        # Update num_topics if specified
        if num_topics is not None:
            self.pipeline.num_topics = num_topics
        
        results = self.pipeline.run(text)
        
        # Convert to format expected by system
        return [{
            "topic_id": r["topic_number"] - 1,
            "title": r["title"],
            "content": r["text"],
            "keywords": r["keywords"],
            "sentence_range": r["sentence_range"],
            "word_count": r["word_count"],
        } for r in results]
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using TF-IDF."""
        from src.enhanced_topic_pipeline import KeywordExtractor
        extractor = KeywordExtractor(top_n=TOPIC_KEYWORDS_COUNT)
        keywords = extractor.extract([text])
        return [kw for kw, _ in keywords[0]] if keywords else []


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


# Convenience function for topic segmentation
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


# Backward compatibility: expose the segment function
def segment(text: str, num_topics: Optional[int] = None) -> List[Dict]:
    """Alias for segment_topics."""
    return segment_topics(text, num_topics)


# Demo / test
if __name__ == "__main__":
    print("=" * 60)
    print("  Unified Topic Pipeline Demo (Local-Only)")
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
    Renewable energy sources such as solar, wind, and hydroelectric power are critical to reducing carbon emissions.
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
    
    # Extract keywords
    print("\n--- Keyword Extraction ---")
    keywords = pipeline.extract_keywords(SAMPLE_TEXT)
    print(f"Keywords: {keywords[:5]}")

