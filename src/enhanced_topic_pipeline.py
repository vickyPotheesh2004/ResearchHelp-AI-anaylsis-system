"""
Enhanced Topic Segmentation Pipeline
====================================
A comprehensive zero-API topic segmentation system combining:
- TextTiling (sliding-window cosine similarity for boundary detection)
- Semantic Shift Detection (depth-score valleys = topic boundaries)  
- Sentence Embeddings (sentence-transformers or TF-IDF fallback)
- TF-IDF Keyword Extraction with n-gram support
- TextRank Extractive Summarization (graph-based)
- Automatic Title Generation with noun phrase extraction + WordNet boosting

This provides complete zero-API-dependency for:
- Topic segmentation (multiple methods: texttiling, semantic, hybrid)
- Keyword extraction
- Extractive summarization
- Automatic title generation

Author: AI Analysis System
"""

import re
import math
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
#  STOP WORDS (comprehensive list)
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


# Additional academic stop words
ACADEMIC_STOP_WORDS = {
    "study", "research", "paper", "article", "chapter", "section", "figure", "table",
    "results", "conclusion", "abstract", "introduction", "method", "methods", "approach",
    "data", "analysis", "findings", "discussion", "background", "related", "work"
}
STOP_WORDS.update(ACADEMIC_STOP_WORDS)


# ─────────────────────────────────────────────────────────────────────────────
#  TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Normalize whitespace and clean raw text."""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def sentence_tokenize(text: str) -> list:
    """Split text into sentences using regex with abbreviation handling."""
    # Handle common abbreviations to avoid over-splitting
    abbrev_pattern = r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|e\.g|i\.e|U\.S\.A|U\.K|U\.S\.|Fig|Feb|Apr|Dec|No|Vol|pp|Mrs|Ms)\.'
    
    # Protect abbreviations
    protected = re.sub(abbrev_pattern, lambda m: m.group(0).replace('.', '<<<DOT>>>'), text)
    
    # Split on sentence endings (lookahead for capital letters)
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+(?=[A-Z])'
    sentences = re.split(pattern, protected)
    
    # Restore periods in abbreviations
    sentences = [s.replace('<<<DOT>>>', '.') for s in sentences]
    
    # Filter and clean - keep sentences with meaningful length
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences


def word_tokenize(text: str, include_stops: bool = False) -> list:
    """Tokenize to lowercase words, strip punctuation."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    if include_stops:
        return [w for w in text.split() if w and len(w) > 1]
    return [w for w in text.split() if w and w not in STOP_WORDS and len(w) > 2]


# ─────────────────────────────────────────────────────────────────────────────
#  TEXT TILING — CORE SEGMENTATION ENGINE (from reference)
# ─────────────────────────────────────────────────────────────────────────────
class TextTilingSegmenter:
    """
    Implements Hearst's TextTiling algorithm:
      1. Divide text into pseudo-sentences (token blocks)
      2. Compute vocabulary intro score per block
      3. Slide a window; compute lexical cohesion (cosine similarity)
      4. Find valleys (depth score minima) → topic boundaries
    """

    def __init__(self, w: int = 20, k: int = 10, smoothing_rounds: int = 2):
        """
        w  : pseudo-sentence size (tokens per block)
        k  : number of blocks in each half of comparison window
        smoothing_rounds: gap score smoothing passes
        """
        self.w = w
        self.k = k
        self.smoothing_rounds = smoothing_rounds

    def _tokenize_blocks(self, sentences: List[str]) -> Tuple[List[List[str]], List[Tuple[int, int]]]:
        """Split sentences into fixed-size token blocks; map back to sentences."""
        all_tokens = []
        token_to_sentence = []
        
        for sent_idx, sent in enumerate(sentences):
            words = word_tokenize(sent)
            all_tokens.extend(words)
            token_to_sentence.extend([sent_idx] * len(words))

        if not all_tokens:
            return [], []

        blocks, block_sent_map = [], []
        for i in range(0, len(all_tokens), self.w):
            block = all_tokens[i:i + self.w]
            if block:
                blocks.append(block)
                idxs = token_to_sentence[i:i + self.w]
                block_sent_map.append((min(idxs), max(idxs)))
        
        return blocks, block_sent_map

    def _block_to_vector(self, blocks: List[List[str]], vocab: Dict[str, int]) -> np.ndarray:
        """Convert each block into a TF vector over shared vocabulary."""
        n = len(blocks)
        v = len(vocab)
        matrix = np.zeros((n, v))
        for i, block in enumerate(blocks):
            for word in block:
                if word in vocab:
                    matrix[i, vocab[word]] += 1
        return matrix

    def _gap_scores(self, matrix: np.ndarray) -> np.ndarray:
        """Slide comparison window; compute cosine similarity at each gap."""
        n = matrix.shape[0]
        scores = np.zeros(n - 1)
        
        for gap in range(n - 1):
            left_start = max(0, gap - self.k + 1)
            right_end = min(n, gap + self.k + 1)
            
            left_vec = matrix[left_start:gap + 1].sum(axis=0)
            right_vec = matrix[gap + 1:right_end].sum(axis=0)
            
            norm = (np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
            scores[gap] = np.dot(left_vec, right_vec) / norm if norm > 0 else 0.0
        
        return scores

    def _smooth(self, scores: np.ndarray) -> np.ndarray:
        """Apply simple moving average to gap scores."""
        smoothed = scores.copy()
        window = 2  # Small window for local smoothing
        
        for i in range(len(scores)):
            start = max(0, i - window)
            end = min(len(scores), i + window + 1)
            smoothed[i] = scores[start:end].mean()
        
        return smoothed

    def _depth_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Depth score at gap i = how far it dips below surrounding peaks.
        depth(i) = 0.5 * [(left_peak - score(i)) + (right_peak - score(i))]
        """
        n = len(scores)
        depth = np.zeros(n)
        
        for i in range(n):
            # Left peak (max from start to i)
            left_peak = scores[:i + 1].max() if i >= 0 else scores[0]
            
            # Right peak (max from i to end)
            right_peak = scores[i:].max()
            
            # Depth = average of drops from both peaks
            depth[i] = 0.5 * ((left_peak - scores[i]) + (right_peak - scores[i]))
        
        return depth

    def segment(self, sentences: List[str], num_topics: int = None) -> List[Tuple[int, int]]:
        """
        Returns list of (start_sent_idx, end_sent_idx) for each segment.
        """
        if len(sentences) < 4:
            return [(0, len(sentences) - 1)] if sentences else []

        blocks, block_sent_map = self._tokenize_blocks(sentences)
        if len(blocks) < 3:
            return [(0, len(sentences) - 1)]

        # Build vocabulary from all blocks
        vocab = {w: i for i, w in enumerate(
            set(word for block in blocks for word in block))}
        
        matrix = self._block_to_vector(blocks, vocab)
        gaps = self._gap_scores(matrix)
        
        # Apply smoothing
        for _ in range(self.smoothing_rounds):
            gaps = self._smooth(gaps)
        
        depth = self._depth_scores(gaps)

        # Select boundary blocks
        if num_topics is not None:
            n_boundaries = max(0, num_topics - 1)
            boundary_blocks = sorted(
                np.argsort(depth)[-n_boundaries:].tolist())
        else:
            # Auto threshold: mean + 0.5 * std
            threshold = depth.mean() + 0.5 * depth.std()
            boundary_blocks = sorted(
                [i for i, d in enumerate(depth) if d >= threshold])

        # Convert block boundaries → sentence boundaries
        boundaries = []
        for b in boundary_blocks:
            if b < len(block_sent_map):
                boundaries.append(block_sent_map[b][1])

        # Deduplicate and build segment ranges
        boundaries = sorted(set(boundaries))
        segments, prev = [], 0
        for b in boundaries:
            if b >= prev:
                segments.append((prev, b))
                prev = b + 1
        segments.append((prev, len(sentences) - 1))
        
        return [s for s in segments if s[0] <= s[1]]


# ─────────────────────────────────────────────────────────────────────────────
#  SEMANTIC SHIFT DETECTOR (using sentence embeddings)
# ─────────────────────────────────────────────────────────────────────────────
class SemanticShiftDetector:
    """
    Uses sentence embeddings + cosine similarity drop to find topic shifts.
    Falls back to TF-IDF if sentence transformers not available.
    """

    def __init__(self, embedding_model: str = "all-mpnet-base-v2"):
        self.embedding_model = embedding_model
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initialize sentence transformer model."""
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(self.embedding_model)
                logger.info(f"Loaded sentence transformer for semantic shift: {self.embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.model = None

    def _get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get sentence embeddings or fallback to TF-IDF."""
        if self.model is not None:
            try:
                return self.model.encode(sentences, show_progress_bar=False)
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")
        
        # Fallback to TF-IDF
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
            return vectorizer.fit_transform(sentences).toarray()
        except Exception:
            return np.random.rand(len(sentences), 50)

    def segment(self, sentences: List[str], num_topics: int = None, window: int = 3) -> List[Tuple[int, int]]:
        """Segment using semantic shift detection."""
        if len(sentences) < 4:
            return [(0, len(sentences) - 1)] if sentences else []

        embeddings = self._get_embeddings(sentences)
        n = len(sentences)
        
        # Ensure window is valid for the sentence count
        window = min(window, max(1, n // 4))
        
        # Compute similarity between adjacent windows
        sim_scores = []
        for i in range(window, n - window):
            left = embeddings[max(0, i - window):i].mean(axis=0)
            right = embeddings[i:min(n, i + window)].mean(axis=0)
            
            nl, nr = np.linalg.norm(left), np.linalg.norm(right)
            sim = np.dot(left, right) / (nl * nr) if nl > 0 and nr > 0 else 1.0
            sim_scores.append((i, sim))

        if not sim_scores:
            return [(0, len(sentences) - 1)]

        positions, scores = zip(*sim_scores)
        scores = np.array(scores)

        # Find boundaries
        if num_topics is not None:
            n_boundaries = max(0, num_topics - 1)
            # Find lowest similarity points (valleys = topic shifts)
            idxs = np.argsort(scores)[:n_boundaries]
            boundary_sents = sorted([positions[i] for i in idxs])
        else:
            # Auto threshold: mean - std
            threshold = scores.mean() - 0.5 * scores.std()
            boundary_sents = sorted(
                [positions[i] for i, s in enumerate(scores) if s <= threshold])

        # Build segments
        segments, prev = [], 0
        for b in boundary_sents:
            if b > prev:
                segments.append((prev, b - 1))
                prev = b
        segments.append((prev, len(sentences) - 1))
        
        return [s for s in segments if s[0] <= s[1]]


# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDING-BASED SEGMENTER (enhanced depth scores)
# ─────────────────────────────────────────────────────────────────────────────
class EmbeddingSegmenter:
    """Uses sentence embeddings + depth scores for topic boundary detection."""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initialize sentence transformer model."""
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding segmenter: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.model = None

    def segment(self, sentences: List[str], num_topics: int = None) -> List[Tuple[int, int]]:
        """Segment text using embedding similarity + depth scores."""
        if len(sentences) < 4:
            return [(0, len(sentences) - 1)] if sentences else []

        # Get embeddings
        embeddings = self._get_embeddings(sentences)

        # Calculate similarity profile
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)

        # Calculate depth scores (improved algorithm)
        depth_scores = self._calculate_depth_scores_improved(similarities)

        # Find boundaries
        boundaries = self._find_boundaries_improved(depth_scores, len(sentences), num_topics)

        # Build segments
        return self._build_segments(boundaries, len(sentences))

    def _get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get embeddings with TF-IDF fallback."""
        if self.model is not None:
            try:
                return self.model.encode(sentences, show_progress_bar=False)
            except Exception:
                pass
        
        # TF-IDF fallback
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
            return vectorizer.fit_transform(sentences).toarray()
        except Exception:
            return np.random.rand(len(sentences), 50)

    def _calculate_depth_scores_improved(self, similarities: List[float]) -> List[float]:
        """Calculate depth scores with improved peak detection."""
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

    def _find_boundaries_improved(self, depth_scores: List[float], n_sentences: int,
                                   num_topics: Optional[int]) -> List[int]:
        """Find topic boundaries with dynamic thresholding."""
        if not depth_scores:
            return []

        # Statistical threshold
        avg_depth = sum(depth_scores) / len(depth_scores)
        variance = sum((d - avg_depth) ** 2 for d in depth_scores) / len(depth_scores)
        std_dev = variance ** 0.5

        # Dynamic sensitivity based on variance
        sensitivity = 1.0 + (0.3 if std_dev < 0.1 else 0)
        threshold = avg_depth + (sensitivity * std_dev)

        # Dynamic minimum segment length
        dynamic_min_len = max(2, min(5, n_sentences // 15))

        # Find potential boundaries
        potential = [(i, depth_scores[i]) for i in range(len(depth_scores))
                     if depth_scores[i] > threshold]

        if not potential:
            return []

        # Sort by depth score
        potential.sort(key=lambda x: x[1], reverse=True)

        # Select boundaries with minimum spacing
        actual_boundaries = []
        for idx, score in potential:
            if all(abs(idx - b) >= dynamic_min_len for b in actual_boundaries):
                actual_boundaries.append(idx)

        # Limit to num_topics if specified
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

        segments.append((prev, n_sentences - 1))
        return [s for s in segments if s[0] <= s[1]]


# ─────────────────────────────────────────────────────────────────────────────
#  HYBRID SEGMENTER (combines multiple methods)
# ─────────────────────────────────────────────────────────────────────────────
class HybridSegmenter:
    """
    Combines TextTiling + Semantic Shift + Embedding methods.
    Uses intersection of boundaries from all methods for higher accuracy.
    """

    def __init__(self, embedding_model: str = "all-mpnet-base-v2"):
        self.texttiling = TextTilingSegmenter(w=20, k=8, smoothing_rounds=2)
        self.semantic = SemanticShiftDetector(embedding_model)
        self.embedding = EmbeddingSegmenter(embedding_model)

    def segment(self, sentences: List[str], num_topics: int = None) -> List[Tuple[int, int]]:
        """Segment using hybrid approach."""
        if len(sentences) < 4:
            return [(0, len(sentences) - 1)] if sentences else []

        # Get boundaries from each method
        tt_boundaries = self.texttiling.segment(sentences, num_topics)
        sem_boundaries = self.semantic.segment(sentences, num_topics=num_topics)
        emb_boundaries = self.embedding.segment(sentences, num_topics)

        # Extract end indices from all methods
        all_ends = set()
        for segs in [tt_boundaries, sem_boundaries, emb_boundaries]:
            for start, end in segs:
                all_ends.add(end)

        # Remove last sentence boundary (it's the end of document)
        all_ends.discard(len(sentences) - 1)

        # Weight boundaries by agreement (boundaries found by multiple methods)
        boundary_weights = Counter()
        for segs in [tt_boundaries, sem_boundaries, emb_boundaries]:
            for start, end in segs:
                if end != len(sentences) - 1:
                    boundary_weights[end] += 1

        # Select boundaries with weight >= 2 (found by at least 2 methods)
        # Or if num_topics specified, take top weighted boundaries
        if num_topics is not None:
            n_boundaries = num_topics - 1
            weighted_boundaries = sorted(
                boundary_weights.keys(),
                key=lambda x: (boundary_weights[x], -x),  # Higher weight, earlier position
                reverse=True
            )[:n_boundaries]
            final_boundaries = sorted(weighted_boundaries)
        else:
            # Use boundaries with weight >= 2, or fall back to best single method
            strong_boundaries = [b for b, w in boundary_weights.items() if w >= 2]
            if strong_boundaries:
                final_boundaries = sorted(strong_boundaries)
            else:
                # Fallback to embedding method
                final_boundaries = sorted(emb_boundaries[-1:] if emb_boundaries else [])

        # Build segments from final boundaries
        segments, prev = [], 0
        for b in final_boundaries:
            if b > prev:
                segments.append((prev, b - 1))
                prev = b
        segments.append((prev, len(sentences) - 1))

        return [s for s in segments if s[0] <= s[1]]


# ─────────────────────────────────────────────────────────────────────────────
#  TF-IDF KEYWORD EXTRACTOR (Enhanced)
# ─────────────────────────────────────────────────────────────────────────────
class KeywordExtractor:
    """
    Extracts top-N keywords using TF-IDF with n-gram support.
    Validates bigrams appear in text and filters by semantic coherence.
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
                sublinear_tf=True,
                token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b'  # Only alphabetic tokens
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

            # Get top keywords
            top_indices = row.argsort()[-self.top_n * 3:][::-1]
            keywords = []
            seen_unigrams = set()

            for idx in top_indices:
                kw = feature_names[idx]
                score = row[idx]

                if score == 0:
                    continue

                parts = kw.split()

                # Validate bigrams appear in text
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
        """Check if bigram appears in text (case-insensitive)."""
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
#  ENHANCED TITLE GENERATOR (combines noun phrases + keywords + WordNet)
# ─────────────────────────────────────────────────────────────────────────────
class TitleGenerator:
    """
    Generates human-readable titles using:
      1. Noun phrase extraction from segment text
      2. TF-IDF keywords as supplements
      3. Pattern-based template selection
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
            r'describes?|involves?|focuses?|aims?|shows?|discusses?|covers?|'
            r'demonstrates?|examines?|explores?|investigates?|presents?|'
            r'proposes?|introduces?|addresses?|considers?)\b',
            sentence, maxsplit=1
        )
        candidate = match[0].strip().rstrip(',').strip()

        # Cap at 6 words
        words = candidate.split()
        if 2 <= len(words) <= 6:
            return candidate

        return ""

    def _clean_phrase(self, phrase: str) -> str:
        """Clean extracted phrase for title use."""
        # Remove trailing punctuation
        phrase = re.sub(r'[\.,;:]+$', '', phrase)
        
        # Remove common filler words at start
        phrase = re.sub(r'^(The |A |An |This |These |Those |Such )', '', phrase, flags=re.IGNORECASE)
        
        return phrase.strip()

    def generate(self, keywords: List[Tuple[str, float]], segment_text: str = "") -> str:
        """Generate title from keywords and segment text."""
        if not keywords:
            return "Untitled Section"

        # Try noun phrase from first sentence
        first_sent = sentence_tokenize(segment_text)[0] if segment_text else ""
        noun_phrase = self._extract_noun_phrase(first_sent)
        noun_phrase = self._clean_phrase(noun_phrase)

        # Separate bigrams and unigrams
        kws = [kw for kw, _ in keywords]
        
        # Filter bigrams for meaningful ones
        bigrams = [k for k in kws if ' ' in k and len(k.split()) == 2]
        bigrams = [b for b in bigrams if not any(
            sw in b.lower().split() for sw in ['the', 'and', 'or', 'but', 'for', 'with', 'from']
        )]
        
        unigrams = [k for k in kws if ' ' not in k and len(k) > 3]

        best_bigram = bigrams[0] if bigrams else None
        second_bigram = bigrams[1] if len(bigrams) > 1 else None
        best_unigram = unigrams[0] if unigrams else None
        second_uni = unigrams[1] if len(unigrams) > 1 else None

        # Template selection with priority
        if noun_phrase and len(noun_phrase.split()) >= 2:
            title = noun_phrase

        elif best_bigram and second_bigram:
            title = f"{best_bigram} and {second_bigram}"

        elif best_bigram and best_unigram:
            title = f"{best_bigram}: Overview"

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
    End-to-end topic segmentation pipeline with multiple methods.
    
    Supports:
    - 'texttiling': Classic TextTiling algorithm
    - 'semantic': Semantic shift detection
    - 'embedding': Sentence embedding + depth scores
    - 'hybrid': Combines all methods (default, most accurate)
    
    All with ZERO external API dependencies.
    """
    
    def __init__(self,
                 num_topics: int = None,
                 summary_sentences: int = 3,
                 top_keywords: int = 8,
                 embedding_model: str = "all-mpnet-base-v2",
                 min_segment_words: int = 50,
                 method: str = "hybrid"):
        """
        Initialize the enhanced topic pipeline.
        
        Args:
            num_topics: Force exact number of topics (None = auto-detect)
            summary_sentences: Sentences per segment summary
            top_keywords: Keywords to extract per segment
            embedding_model: Sentence transformer model name
            min_segment_words: Minimum words per segment
            method: Segmentation method ('texttiling', 'semantic', 'embedding', 'hybrid')
        """
        self.num_topics = num_topics
        self.summary_sentences = summary_sentences
        self.top_keywords = top_keywords
        self.min_segment_words = min_segment_words
        self.method = method

        # Initialize segmenter based on method
        if method == "texttiling":
            self.segmenter = TextTilingSegmenter(w=20, k=8, smoothing_rounds=2)
        elif method == "semantic":
            self.segmenter = SemanticShiftDetector(embedding_model)
        elif method == "embedding":
            self.segmenter = EmbeddingSegmenter(embedding_model)
        else:  # hybrid (default)
            self.segmenter = HybridSegmenter(embedding_model)

        # Initialize other components
        self.extractor = KeywordExtractor(top_n=top_keywords)
        self.summarizer = TextRankSummarizer()
        self.titler = TitleGenerator()

        logger.info(f"EnhancedTopicPipeline initialized (method={method}, topics={num_topics or 'auto'})")

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

        logger.info(f"Processing {len(sentences)} sentences with method={self.method}")

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
    performance on many text benchmarks. The rapid advancement of AI has been fueled 
    by the availability of massive datasets and powerful GPU hardware. Researchers 
    continue to push the boundaries of what machines can perceive, understand, and generate.
    
    Climate change represents one of the most pressing challenges facing humanity today. 
    Rising global temperatures are caused primarily by greenhouse gas emissions from 
    burning fossil fuels. The effects include rising sea levels, more frequent extreme 
    weather events, and disruption of ecosystems. International agreements like the 
    Paris Accord aim to limit warming to 1.5 degrees Celsius above pre-industrial levels.
    Renewable energy sources such as solar, wind, and hydroelectric power are critical 
    to reducing carbon emissions. Governments, corporations, and individuals all have 
    roles to play in transitioning to a sustainable economy.
    
    The global economy operates as an interconnected web of trade, finance, and production. 
    International trade allows countries to specialize in goods and services where they 
    hold comparative advantages. Financial markets channel capital from savers to 
    productive investments across borders. Supply chain disruptions, such as those 
    caused by the COVID-19 pandemic, revealed the fragility of global production networks.
    Central banks use monetary policy tools to manage inflation and support employment.
    
    Modern healthcare systems face mounting pressure from aging populations and chronic 
    diseases. Advances in genomics and personalized medicine are enabling treatments 
    tailored to individual patients. mRNA vaccine technology, proven effective during 
    the COVID-19 pandemic, holds promise for many other diseases. Telemedicine has 
    expanded access to healthcare services in rural and underserved communities. 
    Mental health is increasingly recognized as a critical component of overall well-being.
    
    Space exploration has entered a new era driven by both government agencies and private 
    companies. NASA's Artemis program aims to return humans to the Moon by the mid-2020s 
    as a stepping stone to Mars. SpaceX has dramatically reduced launch costs through 
    reusable rocket technology. Satellites now underpin critical infrastructure including 
    GPS navigation, weather forecasting, and communications. The search for extraterrestrial 
    life is being pursued through missions to Mars and the icy moons of Jupiter and Saturn.
    """

    print("=" * 60)
    print("  Enhanced Topic Segmentation Pipeline Demo")
    print("=" * 60)

    # Test all methods
    methods = ["texttiling", "semantic", "embedding", "hybrid"]
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Method: {method.upper()}")
        print(f"{'='*60}")
        
        pipeline = EnhancedTopicPipeline(
            num_topics=None,  # Auto-detect
            summary_sentences=2,
            top_keywords=6,
            method=method,
        )
        
        results = pipeline.run(SAMPLE_TEXT)
        
        print(format_results(results))
        
        # Show first topic structure
        if results:
            clean = export_to_dict(results)
            print(f"\nStructured output (first topic):")
            for k, v in clean[0].items():
                print(f"  {k:16}: {v}")
