"""
Enhanced Topic Segmentation Pipeline
=====================================
Combines the best of both approaches:
- Current system: Sentence embeddings with depth-score segmentation
- Reference code: TF-IDF keyword extraction, TextRank summarization

This provides a complete zero-API-dependency pipeline for:
- Topic segmentation
- Keyword extraction
- Extractive summarization
- Automatic title generation

Author: AI Analysis System
"""

import re
import math
import heapq
import warnings
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to use sentence embeddings if available, fallback to TF-IDF
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from src.logging_utils import get_logger

logger = get_logger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  STOP WORDS (built-in — no NLTK needed for basic functionality)
# ─────────────────────────────────────────────────────────────────────────────
STOP_WORDS = set("""
a about above after again against all also although am an and any are aren't as at
be because been before being below between both but by can can't cannot could couldn't
did didn't do does doesn't doing don't down during each few for from further get got
had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's
its itself let's me more most mustn't my myself no nor not of off on once only or
other ought our ours ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves then there
there's these they they'd they'll they're they've this those through to too under
until up very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's will with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves also however
thus therefore hence moreover furthermore although though despite nevertheless
""".split())


# ─────────────────────────────────────────────────────────────────────────────
#  TEXT PREPROCESSING (Enhanced)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Normalize whitespace and clean raw text."""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def sentence_tokenize(text: str) -> list:
    """Split text into sentences using regex (no NLTK needed)."""
    # Handle common abbreviations to avoid over-splitting
    abbrev_pattern = r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.'
    
    # First, protect common abbreviations
    protected = re.sub(abbrev_pattern, lambda m: m.group(0).replace('.', '<<<DOT>>>'), text)
    
    # Split on sentence endings
    pattern = r'(?<<!<<<DOT>>>)(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+(?=[A-Z])'
    sentences = re.split(pattern, protected)
    
    # Restore periods in abbreviations
    sentences = [s.replace('<<<DOT>>>', '.') for s in sentences]
    
    # Filter and clean
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences


def word_tokenize(text: str) -> list:
    """Tokenize to lowercase words, strip punctuation."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return [w for w in text.split() if w and w not in STOP_WORDS and len(w) > 2]


def get_clean_words(text: str) -> list:
    """All words (including stop words) for title generation."""
    text = text.lower()
    return re.sub(r'[^a-z\s]', ' ', text).split()


# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDING-BASED SEGMENTER (Primary Method)
# ─────────────────────────────────────────────────────────────────────────────
class EmbeddingSegmenter:
    """
    Uses sentence embeddings + depth scores for topic boundary detection.
    Falls back to TF-IDF if sentence transformers not available.
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize sentence transformer model."""
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded sentence transformer: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.model = None
        else:
            logger.info("Using TF-IDF fallback for embeddings")
    
    def segment(self, sentences: List[str], num_topics: int = None) -> List[Tuple[int, int]]:
        """Segment text into topics using embedding similarity."""
        if len(sentences) < 4:
            return [(0, len(sentences) - 1)] if sentences else []
        
        # Get embeddings
        if self.model is not None:
            try:
                embeddings = self.model.encode(sentences, show_progress_bar=False)
            except Exception as e:
                logger.warning(f"Embedding failed: {e}, using TF-IDF fallback")
                embeddings = self._tfidf_embeddings(sentences)
        else:
            embeddings = self._tfidf_embeddings(sentences)
        
        # Calculate similarity profile
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)
        
        # Calculate depth scores
        depth_scores = self._calculate_depth_scores(similarities)
        
        # Find boundaries
        boundaries = self._find_boundaries(depth_scores, len(sentences), num_topics)
        
        # Convert to segment ranges
        return self._build_segments(boundaries, len(sentences))
    
    def _tfidf_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Create TF-IDF based embeddings as fallback."""
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
            tfidf = vectorizer.fit_transform(sentences).toarray()
            return tfidf
        except Exception:
            # Ultimate fallback: random embeddings
            return np.random.rand(len(sentences), 50)
    
    def _calculate_depth_scores(self, similarities: List[float]) -> List[float]:
        """Calculate depth scores - how much each gap dips below surrounding peaks."""
        if not similarities:
            return []
        
        n = len(similarities)
        depth_scores = []
        
        for i in range(n):
            # Look left for peak
            lpeak = similarities[i]
            for j in range(i, -1, -1):
                if similarities[j] > lpeak:
                    lpeak = similarities[j]
                else:
                    break
            
            # Look right for peak
            rpeak = similarities[i]
            for j in range(i, n):
                if similarities[j] > rpeak:
                    rpeak = similarities[j]
                else:
                    break
            
            # Depth = sum of drops from peaks
            depth = (lpeak - similarities[i]) + (rpeak - similarities[i])
            depth_scores.append(depth)
        
        return depth_scores
    
    def _find_boundaries(self, depth_scores: List[float], n_sentences: int, 
                         num_topics: Optional[int]) -> List[int]:
        """Find topic boundaries from depth scores."""
        if not depth_scores:
            return []
        
        # Statistical threshold
        avg_depth = sum(depth_scores) / len(depth_scores)
        variance = sum((d - avg_depth) ** 2 for d in depth_scores) / len(depth_scores)
        std_dev = variance ** 0.5
        
        # Dynamic sensitivity based on variance
        sensitivity = 1.0 + (0.2 if std_dev < 0.1 else 0)
        threshold = avg_depth + (sensitivity * std_dev)
        
        # Dynamic minimum segment length
        dynamic_min_len = max(2, min(5, n_sentences // 15))
        
        # Find potential boundaries
        potential = [(i, depth_scores[i]) for i in range(len(depth_scores)) 
                     if depth_scores[i] > threshold]
        
        if not potential:
            return []
        
        # Sort by depth score (most prominent first)
        potential.sort(key=lambda x: x[1], reverse=True)
        
        # Select boundaries with minimum spacing
        actual_boundaries = []
        for idx, score in potential:
            if all(abs(idx - b) >= dynamic_min_len for b in actual_boundaries):
                actual_boundaries.append(idx)
        
        # If num_topics specified, limit boundaries
        if num_topics is not None:
            n_boundaries = max(0, num_topics - 1)
            actual_boundaries = sorted(actual_boundaries)[:n_boundaries]
        else:
            actual_boundaries.sort()
        
        return actual_boundaries
    
    def _build_segments(self, boundaries: List[int], n_sentences: int) -> List[Tuple[int, int]]:
        """Convert boundary indices to (start, end) sentence ranges."""
        if not boundaries:
            return [(0, n_sentences - 1)]
        
        segments = []
        prev = 0
        
        for b in boundaries:
            segments.append((prev, b))
            prev = b + 1
        
        # Add final segment
        segments.append((prev, n_sentences - 1))
        
        # Filter invalid segments
        return [s for s in segments if s[0] <= s[1]]


# ─────────────────────────────────────────────────────────────────────────────
#  TF-IDF KEYWORD EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
class KeywordExtractor:
    """
    Extracts top-N keywords per segment using TF-IDF with n-gram support.
    """
    
    def __init__(self, top_n: int = 8, use_bigrams: bool = True):
        self.top_n = top_n
        self.use_bigrams = use_bigrams
    
    def extract(self, segments_text: List[str]) -> List[List[Tuple[str, float]]]:
        """Extract keywords from each segment."""
        if not segments_text:
            return []
        
        # Configure vectorizer
        ngram_range = (1, 2) if self.use_bigrams else (1, 1)
        
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=ngram_range,
                max_features=1000,
                min_df=1,
                sublinear_tf=True
            )
            tfidf_matrix = vectorizer.fit_transform(segments_text)
            feature_names = vectorizer.get_feature_names_out()
        except Exception as e:
            logger.warning(f"TF-IDF extraction failed: {e}")
            return [[] for _ in segments_text]
        
        all_keywords = []
        
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix[i].toarray().flatten()
            seg_text = segments_text[i]
            
            # Get top keywords with validation
            top_indices = row.argsort()[-self.top_n * 3:][::-1]
            keywords = []
            seen_unigrams = set()
            
            for idx in top_indices:
                kw = feature_names[idx]
                score = row[idx]
                
                if score == 0:
                    continue
                
                parts = kw.split()
                
                # Validate bigrams
                if len(parts) == 2:
                    if not self._is_valid_bigram(kw, seg_text):
                        continue
                    seen_unigrams.update(parts)
                    keywords.append((kw, round(float(score), 4)))
                elif kw not in seen_unigrams:
                    keywords.append((kw, round(float(score), 4)))
                
                if len(keywords) >= self.top_n:
                    break
            
            all_keywords.append(keywords)
        
        return all_keywords
    
    def _is_valid_bigram(self, bigram: str, text: str) -> bool:
        """Check if bigram appears in text."""
        return bigram.lower() in text.lower()


# ─────────────────────────────────────────────────────────────────────────────
#  TEXTRANK SUMMARIZER (Graph-based extractive)
# ─────────────────────────────────────────────────────────────────────────────
class TextRankSummarizer:
    """
    PageRank-inspired extractive summarizer.
    Nodes = sentences; edges = cosine similarity (TF-IDF).
    """
    
    def __init__(self, damping: float = 0.85, max_iterations: int = 50):
        self.damping = damping
        self.max_iterations = max_iterations
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Generate extractive summary of text."""
        sentences = sentence_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Build TF-IDF matrix
        try:
            vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
            tfidf = vectorizer.fit_transform(sentences)
            sim_matrix = cosine_similarity(tfidf)
        except Exception as e:
            logger.warning(f"TextRank similarity failed: {e}")
            return ' '.join(sentences[:num_sentences])
        
        # Remove self-similarity
        np.fill_diagonal(sim_matrix, 0)
        
        # Normalize rows
        row_sums = sim_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        sim_matrix = sim_matrix / row_sums
        
        # Power iteration (PageRank)
        n = len(sentences)
        scores = np.ones(n) / n
        
        for _ in range(self.max_iterations):
            scores = (1 - self.damping) / n + self.damping * sim_matrix.T.dot(scores)
        
        # Select top sentences in original order
        top_idxs = sorted(np.argsort(scores)[-num_sentences:].tolist())
        summary = ' '.join(sentences[i] for i in top_idxs)
        
        return summary if summary else sentences[0]


# ─────────────────────────────────────────────────────────────────────────────
#  AUTOMATIC TITLE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
class TitleGenerator:
    """
    Generates human-readable titles using keyword patterns.
    Zero API dependency.
    """
    
    MINOR_WORDS = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor',
                   'on', 'at', 'to', 'by', 'in', 'of', 'up', 'vs', 'via'}
    
    def _title_case(self, text: str) -> str:
        """Apply title case with minor word handling."""
        words = text.split()
        result = []
        
        for i, w in enumerate(words):
            if i == 0 or w.lower() not in self.MINOR_WORDS:
                result.append(w.capitalize())
            else:
                result.append(w.lower())
        
        return ' '.join(result)
    
    def _extract_noun_phrase(self, sentence: str) -> str:
        """Extract descriptive noun phrase from first sentence."""
        sentence = sentence.strip()
        
        # Remove leading articles
        sentence = re.sub(r'^(The|A|An)\s+', '', sentence, flags=re.IGNORECASE)
        
        # Take up to first verb-like boundary
        match = re.split(
            r'\b(is|are|was|were|has|have|had|represents?|refers?|'
            r'describes?|involves?|focuses?|aims?|shows?|discusses?|covers?)\b',
            sentence, maxsplit=1
        )
        candidate = match[0].strip().rstrip(',').strip()
        
        # Cap at 6 words
        words = candidate.split()
        if 2 <= len(words) <= 6:
            return candidate
        
        return ""
    
    def generate(self, keywords: List[Tuple[str, float]], segment_text: str = "") -> str:
        """Generate title from keywords."""
        if not keywords:
            return "Untitled Section"
        
        # Try noun phrase from first sentence
        first_sent = sentence_tokenize(segment_text)[0] if segment_text else ""
        noun_phrase = self._extract_noun_phrase(first_sent)
        
        # Separate bigrams and unigrams
        kws = [kw for kw, _ in keywords]
        bigrams = [k for k in kws if ' ' in k]
        unigrams = [k for k in kws if ' ' not in k and len(k) > 3]
        
        best_bigram = bigrams[0] if bigrams else None
        second_bigram = bigrams[1] if len(bigrams) > 1 else None
        best_unigram = unigrams[0] if unigrams else None
        second_uni = unigrams[1] if len(unigrams) > 1 else None
        
        # Template selection
        if noun_phrase and len(noun_phrase.split()) >= 3:
            title = noun_phrase
        elif best_bigram and second_bigram:
            title = f"{best_bigram} and {second_bigram}"
        elif best_bigram and best_unigram:
            title = f"{best_bigram}: An Overview"
        elif best_bigram:
            title = f"Understanding {best_bigram}"
        elif best_unigram and second_uni:
            title = f"{best_unigram} and {second_uni}"
        elif best_unigram:
            title = f"The Role of {best_unigram}"
        else:
            title = "General Discussion"
        
        return self._title_case(title)


# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED TOPIC PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
class EnhancedTopicPipeline:
    """
    End-to-end topic segmentation pipeline.
    
    Combines:
    - Embedding-based segmentation (sentence transformers or TF-IDF fallback)
    - TF-IDF keyword extraction
    - TextRank extractive summarization
    - Automatic title generation
    
    All with ZERO external API dependencies.
    """
    
    def __init__(self,
                 num_topics: int = None,
                 summary_sentences: int = 3,
                 top_keywords: int = 8,
                 embedding_model: str = "all-mpnet-base-v2",
                 min_segment_words: int = 50):
        """
        Initialize the enhanced topic pipeline.
        
        Args:
            num_topics: Force exact number of topics (None = auto-detect)
            summary_sentences: Sentences per segment summary
            top_keywords: Keywords to extract per segment
            embedding_model: Sentence transformer model name
            min_segment_words: Minimum words per segment
        """
        self.num_topics = num_topics
        self.summary_sentences = summary_sentences
        self.top_keywords = top_keywords
        self.min_segment_words = min_segment_words
        
        # Initialize components
        self.segmenter = EmbeddingSegmenter(embedding_model)
        self.extractor = KeywordExtractor(top_n=top_keywords)
        self.summarizer = TextRankSummarizer()
        self.titler = TitleGenerator()
        
        logger.info(f"EnhancedTopicPipeline initialized (topics={num_topics or 'auto'})")
    
    def _merge_short_segments(self, segments: List[Tuple[int, int]], 
                              sentences: List[str]) -> List[Tuple[int, int]]:
        """Merge segments that are too short."""
        if not segments or len(segments) <= 1:
            return segments
        
        merged = []
        
        for start, end in segments:
            seg_text = ' '.join(sentences[start:end + 1])
            word_count = len(seg_text.split())
            
            # If too short and not first segment, merge with previous
            if word_count < self.min_segment_words and merged:
                prev_start, prev_end = merged[-1]
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        
        return merged
    
    def run(self, text: str) -> List[Dict]:
        """
        Main entry point for topic segmentation.
        
        Args:
            text: Raw input text
            
        Returns:
            List of topic dictionaries with:
                - topic_number: 1-based index
                - title: Generated title
                - text: Segment text
                - keywords: List of (keyword, score) tuples
                - summary: Extractive summary
                - sentence_range: (start, end) 1-indexed
                - word_count: Number of words
        """
        # Preprocess
        text = preprocess_text(text)
        sentences = sentence_tokenize(text)
        
        if not sentences:
            return []
        
        logger.info(f"Processing {len(sentences)} sentences")
        
        # Step 1: Segment
        seg_ranges = self.segmenter.segment(sentences, self.num_topics)
        
        # Step 2: Merge short segments
        seg_ranges = self._merge_short_segments(seg_ranges, sentences)
        
        # Step 3: Build segment texts
        segments_text = []
        for start, end in seg_ranges:
            seg_text = ' '.join(sentences[start:end + 1])
            segments_text.append(seg_text)
        
        # Step 4: Extract keywords
        all_keywords = self.extractor.extract(segments_text)
        
        # Step 5: Build results
        results = []
        
        for idx, ((start, end), seg_text, kws) in enumerate(
                zip(seg_ranges, segments_text, all_keywords)):
            
            title = self.titler.generate(kws, seg_text)
            summary = self.summarizer.summarize(seg_text, self.summary_sentences)
            word_count = len(seg_text.split())
            
            results.append({
                "topic_number": idx + 1,
                "title": title,
                "text": seg_text,
                "keywords": kws,
                "summary": summary,
                "sentence_range": (start + 1, end + 1),  # 1-indexed
                "word_count": word_count,
            })
        
        logger.info(f"Segmented into {len(results)} topics")
        
        return results


# ─────────────────────────────────────────────────────────────────────────────
#  EXPORT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def export_to_dict(results: List[Dict]) -> List[Dict]:
    """Export results as clean dictionary for JSON serialization."""
    return [{
        "topic_number": r["topic_number"],
        "title": r["title"],
        "keywords": [kw for kw, _ in r["keywords"]],
        "summary": r["summary"],
        "word_count": r["word_count"],
        "sentence_range": r["sentence_range"],
    } for r in results]


def format_results(results: List[Dict], show_full_text: bool = False) -> str:
    """Format results as readable string."""
    if not results:
        return "No topics detected."
    
    output = []
    output.append("\n" + "=" * 60)
    output.append(f"  DETECTED {len(results)} TOPIC(S)")
    output.append("=" * 60 + "\n")
    
    for topic in results:
        n = topic["topic_number"]
        total = len(results)
        
        output.append("-" * 60)
        output.append(f"  TOPIC {n} of {total}")
        output.append("-" * 60)
        output.append(f"  Title     : {topic['title']}")
        output.append(f"  Sentences : {topic['sentence_range'][0]}–{topic['sentence_range'][1]}")
        output.append(f"  Words     : {topic['word_count']}")
        
        output.append("\n  Keywords:")
        for kw, score in topic["keywords"]:
            output.append(f"       • {kw:<30} score: {score:.4f}")
        
        output.append(f"\n  Summary:")
        output.append(f"     {topic['summary']}")
        
        if show_full_text:
            output.append(f"\n  Full Text:")
            output.append(f"     {topic['text'][:500]}...")
        
        output.append("")
    
    return "\n".join(output)


# ─────────────────────────────────────────────────────────────────────────────
#  DEMO / TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SAMPLE_TEXT = """
    Artificial intelligence has emerged as one of the most transformative technologies 
    of the twenty-first century. Machine learning, a core subset of AI, enables computers 
    to learn from data without being explicitly programmed. Deep learning architectures 
    such as convolutional neural networks have revolutionized image recognition tasks. 
    Natural language processing models like BERT and GPT have achieved human-level 
    performance on many text benchmarks.
    
    Climate change represents one of the most pressing challenges facing humanity today. 
    Rising global temperatures are caused primarily by greenhouse gas emissions from 
    burning fossil fuels. The effects include rising sea levels, more frequent extreme 
    weather events, and disruption of ecosystems. International agreements like the 
    Paris Accord aim to limit warming to 1.5 degrees Celsius above pre-industrial levels.
    
    The global economy operates as an interconnected web of trade, finance, and production. 
    International trade allows countries to specialize in goods and services where they 
    hold comparative advantages. Financial markets channel capital from savers to 
    productive investments across borders. Supply chain disruptions, such as those 
    caused by the COVID-19 pandemic, revealed the fragility of global production networks.
    
    Modern healthcare systems face mounting pressure from aging populations and chronic 
    diseases. Advances in genomics and personalized medicine are enabling treatments 
    tailored to individual patients. mRNA vaccine technology, proven effective during 
    the COVID-19 pandemic, holds promise for many other diseases. Telemedicine has 
    expanded access to healthcare services in rural and underserved communities.
    """
    
    print("=" * 60)
    print("  Enhanced Topic Segmentation Pipeline Demo")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = EnhancedTopicPipeline(
        num_topics=None,  # Auto-detect
        summary_sentences=2,
        top_keywords=6,
    )
    
    # Run
    results = pipeline.run(SAMPLE_TEXT)
    
    # Print formatted results
    print(format_results(results, show_full_text=False))
    
    # Export as dict
    clean_output = export_to_dict(results)
    print("\nStructured output (first topic):")
    for k, v in clean_output[0].items():
        print(f"  {k:16}: {v}")
